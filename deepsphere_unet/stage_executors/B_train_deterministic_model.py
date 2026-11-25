import logging

from tqdm import tqdm

import torch
from torch import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig

import healpy as hp

from .pytorch_model_base_executor import BaseDeepSphereModelExecutor
from cmbml.core import Split, Asset

from cmbml.core.asset_handlers import Config, AppendingCsvHandler, NumpyMap
from cmbml.torch.pytorch_model_handler import PyTorchModel  # Import for typing hint

from deepsphere_unet.dataset import PreprocessedTrainCMBMapDataset


logger = logging.getLogger(__name__)


class DeterministicTrainingExecutor(BaseDeepSphereModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train")

        self.out_model: Asset = self.assets_out["model"]
        self.out_best_epoch: Asset = self.assets_out["best_epoch"]
        self.out_loss_record: Asset = self.assets_out["loss_record"]
        out_model_handler: PyTorchModel
        best_epoch_handler: Config
        loss_record_handler: AppendingCsvHandler

        self.in_model: Asset = self.assets_in["model"]
        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        in_model_handler: PyTorchModel
        in_cmb_map_handler: NumpyMap
        in_obs_map_handler: NumpyMap

        model_precision = "float"
        self.dtype = self.dtype_mapping[model_precision]
        self.choose_device(cfg.model.deepsphere.train.device)
        if self.device == "mps":  # MPS is not supported for sparse models
            logger.info(f"MPS is not supported for sparse models. Using CPU.")
            self.choose_device("cpu")

        self.gradient_checkpointing = cfg.model.deepsphere.train.gradient_checkpointing
        self.mixed_precision = cfg.model.deepsphere.train.mixed_precision

        self.optimizer_name = cfg.model.deepsphere.train.optimizer
        self.loss_name = cfg.model.deepsphere.train.loss
        self.lr = cfg.model.deepsphere.train.learning_rate
        self.batch_size = cfg.model.deepsphere.train.batch_size
        self.n_epochs = cfg.model.deepsphere.train.n_epochs
        self.start_valid = cfg.model.deepsphere.train.start_valid

        self.checkpoint = cfg.model.deepsphere.train.checkpoint_every
        self.extra_check = cfg.model.deepsphere.train.extra_check
        self.restart_epoch = cfg.model.deepsphere.train.restart_epoch

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        dets_str = ", ".join([str(k) for k in self.instrument.dets.keys()])
        logger.info(f"Creating model using detectors: {dets_str}")

        logger.info("Using static learning rate.")
        logger.info(f"Learning rate is {self.lr}")
        logger.info(f"Number of epochs is {self.n_epochs}")
        logger.info(f"Batch size is {self.batch_size}")
        logger.info(f"Checkpoint every {self.checkpoint} iterations")
        logger.info(f"Extra check is set to {self.extra_check}")

        train_split = None
        valid_split = None
        for split in self.splits:
            if split.name == "Train":
                train_split = split
            elif split.name == "Valid":
                valid_split = split

        assert train_split is not None, (
            "Train split not found, add train split in pipeline configuration file"
        )

        train_dataset = self.set_up_dataset(train_split)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        logger.info(f"Inspecting data for {train_split.name} split: ")
        self.inspect_data(train_dataloader)

        if valid_split is not None:
            valid_dataset = self.set_up_dataset(valid_split)
            valid_dataloader = DataLoader(
                valid_dataset, batch_size=self.batch_size, shuffle=False
            )
            logger.info(f"Inspecting data for {valid_split.name} split: ")
            self.inspect_data(valid_dataloader)
            loss_record_headers = ["epoch", "train_loss", "valid_loss"]
        else:
            logger.info(f"No validation split found. Training without validation.")
            logger.info(
                f"This is not recommended. Consider adding a validation split in the pipeline configuration file."
            )
            valid_dataloader = None
            loss_record_headers = ["epoch", "train_loss"]

        model = self.make_model().to(self.device)

        if self.gradient_checkpointing:
            logger.info(f"Gradient checkpointing is enabled.")
            model.enable_gradient_checkpointing()

        logger.debug(f"Testing model output: ")
        self.try_model(model)

        loss_function = getattr(
            torch.nn, self.loss_name
        )()

        optimizer = getattr(
            torch.optim, self.optimizer_name
        )(
            model.parameters(),
            lr=self.lr,
        )

        logger.info(f"Optimizer: {self.optimizer_name}")
        logger.info(f"Loss function: {self.loss_name}")

        if self.mixed_precision:
            logger.info(f"Using mixed precision training.")
            scaler = GradScaler(device=self.device)
        else:
            logger.info(f"Using full precision training.")
            scaler = None

        best_loss = float("inf")

        if self.restart_epoch is not None:
            logger.info(f"Restarting training at epoch {self.restart_epoch}")

            with self.name_tracker.set_context("epoch", self.restart_epoch):
                model_record = self.in_model.read(
                    model=model,
                    epoch=self.restart_epoch,
                    optimizer=optimizer,
                    scaler=scaler,
                )

            loaded_epoch = model_record["epoch"]
            start_epoch = 0 if loaded_epoch == "init" else int(loaded_epoch)

            restored_best_loss = model_record.get("best_loss")

            if restored_best_loss is not None:
                best_loss = restored_best_loss
            elif start_epoch > 0:
                logger.warning(
                    "Checkpoint does not contain best_loss. "
                    "Best-model tracking will restart."
                )

            self.out_loss_record.start()

        else:
            logger.info("Starting new model.")

            with self.name_tracker.set_context("epoch", "init"):
                self.out_model.write(model=model, epoch="init")

            start_epoch = 0
            self.out_loss_record.write(data=loss_record_headers)

        for epoch in range(start_epoch, self.n_epochs):
            train_loss = self.train(
                model, train_dataloader, optimizer, scaler, loss_function
            )
            if epoch >= self.start_valid:
                if valid_dataloader is not None:
                    valid_loss = self.validate(model, valid_dataloader, loss_function)
                    self.out_loss_record.append(
                        data=[epoch + 1, train_loss, valid_loss]
                    )
                    combined_loss = valid_loss
                    # combined_loss = (
                    #     0.8 * valid_loss + 0.2 * train_loss
                    # )  # Per Adams paper, select best model based on this weighted loss
                    logger.info(
                        f"Epoch {epoch + 1} Train Loss: {train_loss}, Valid Loss: {valid_loss}, Combined Loss: {combined_loss}"
                    )
                else:
                    combined_loss = train_loss
                    logger.info(
                        f"Epoch {epoch + 1} Train Loss: {train_loss}, Combined Loss: {combined_loss}"
                    )
                    self.out_loss_record.append(data=[epoch + 1, train_loss])

                if combined_loss < best_loss:
                    best_loss = combined_loss
                    best_epoch = epoch + 1
                    with self.name_tracker.set_context("epoch", "best"):
                        res = {"best_epoch": best_epoch, "best_loss": best_loss}
                        self.out_best_epoch.write(data=res)
                        self.out_model.write(
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            epoch=best_epoch,
                            best_loss=best_loss,
                        )
            else:
                logger.info(f"Epoch {epoch + 1} Train Loss: {train_loss}")

                if valid_split is None:
                    self.out_loss_record.append(data=[epoch + 1, train_loss])
                else:
                    self.out_loss_record.append(data=[epoch + 1, train_loss, None])

            # Checkpoint every so many epochs
            if (epoch + 1) in self.extra_check or (epoch + 1) % self.checkpoint == 0:
                with self.name_tracker.set_context("epoch", epoch + 1):
                    self.out_model.write(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=epoch + 1,
                        best_loss=best_loss,
                    )

    def one_pass(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: torch.GradScaler,
        loss_function: torch.nn.Module,
        train: bool,
    ) -> float:
        """Runs the training or validation loop for a single epoch.

        Args:
            model (torch.nn.Module): Model to train
            dataloader (DataLoader): Data
            optimizer (torch.optim.Optimizer): Optimizer
            scaler (torch.amp.GradScaler): GradScaler for mixed precision training
            loss_function (torch.nn.Module): Loss
            train (bool): If True, runs the training loop. If False, runs the validation loop.

        Returns:
            float: loss for the epoch
        """
        epoch_loss = 0.0
        n_samples = 0
        batch_n = 0

        if train:
            model.train()
        else:
            model.eval()
        with tqdm(dataloader, postfix={"Loss": 0}) as pbar:
            for features, labels in pbar:
                batch_n += 1

                features = features.to(device=self.device, dtype=self.dtype)
                labels = labels.to(device=self.device, dtype=self.dtype)

                if train:
                    optimizer.zero_grad()
                    if self.mixed_precision:
                        with autocast(device_type=self.device):
                            output = model(features)
                            loss = loss_function(output, labels)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        output = model(features)
                        loss = loss_function(output, labels)
                        loss.backward()
                        optimizer.step()
                else:
                    with torch.no_grad():
                        output = model(features)
                        loss = loss_function(output, labels)

                batch_size = features.size(0)
                loss_value = loss.item()

                epoch_loss += loss_value * batch_size
                n_samples += batch_size

                pbar.set_postfix(
                    {
                        f"Loss for {batch_n}/{len(dataloader)}": loss_value
                    }
                )

        return epoch_loss / n_samples

    def train(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: torch.GradScaler,
        loss_function: torch.nn.Module,
    ) -> float:
        """Runs the training loop for a single epoch.

        Args:
            model (torch.nn.Module): Model to train
            dataloader (DataLoader): Training Data
            optimizer (torch.optim.Optimizer): Optimizer
            scaler (torch.amp.GradScaler): GradScaler for mixed precision training
            loss_function (torch.nn.Module): Loss

        Returns:
            float: training loss for the epoch
        """

        return self.one_pass(
            model, dataloader, optimizer, scaler, loss_function, train=True
        )

    def validate(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        loss_function: torch.nn.Module,
    ) -> float:
        """Runs the validation loop for a single epoch.

        Args:
            model (torch.nn.Module): Model to validate
            dataloader (DataLoader): Validation Data
            loss_function (torch.nn.Module): Loss

        Returns:
            float: validation loss for the epoch
        """
        return self.one_pass(
            model=model,
            dataloader=dataloader,
            optimizer=None,
            scaler=None,
            loss_function=loss_function,
            train=False,
        )

    def set_up_dataset(self, template_split: Split) -> Dataset:
        cmb_path_template = self.make_fn_template(
            template_split, self.in_cmb_asset
        )
        obs_path_template = self.make_fn_template(
            template_split, self.in_obs_assets
        )

        feature_transform, target_transform = self.build_transforms()

        dataset = PreprocessedTrainCMBMapDataset(
            n_sims=template_split.n_sims,
            freqs=self.instrument.dets.keys(),
            label_path_template=cmb_path_template,
            label_handler=NumpyMap(),
            feature_path_template=obs_path_template,
            feature_handler=NumpyMap(),
            feature_transform=feature_transform,
            target_transform=target_transform,
        )

        return dataset

    def inspect_data(self, dataloader):
        train_features, train_labels = next(iter(dataloader))
        logger.info(
            f"{self.__class__.__name__}.inspect_data() Feature batch shape: {train_features.size()}"
        )  # Should be (batch_size, npix, n_map_fields)
        logger.info(
            f"{self.__class__.__name__}.inspect_data() Labels batch shape: {train_labels.size()}"
        )
        npix_data = train_features.size()[1]
        npix_cfg = hp.nside2npix(self.nside)
        assert npix_cfg == npix_data, (
            "Npix for loaded map does not match configuration yamls."
        )
