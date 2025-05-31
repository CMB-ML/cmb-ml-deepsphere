import logging
from typing import Tuple, Union

from tqdm import tqdm

import time
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler
from omegaconf import DictConfig

import healpy as hp

from .pytorch_model_base_executor import BaseDeepSphereModelExecutor
from cmbml.core import Split, Asset

from cmbml.core.asset_handlers import HealpyMap, Config, AppendingCsvHandler

from cmbml.torch.pytorch_model_handler import PyTorchModel  # Import for typing hint

from deepsphere_unet.dataset import TrainCMBMapDataset


from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from deepsphere_unet.model_logger import ModelTracker, timer


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


logger = logging.getLogger(__name__)


class ModelTrainer(BaseDeepSphereModelExecutor):
    def __init__(self, cfg: DictConfig, rank) -> None:
        super().__init__(cfg, stage_str="train")

        self.rank = rank

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
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap

        model_precision = "float"
        self.dtype = self.dtype_mapping[model_precision]
        self.choose_device(cfg.model.deepsphere.train.device)
        if self.device == "mps":  # MPS is not supported for sparse models
            logger.info(f"MPS is not supported for sparse models. Using CPU.")
            self.choose_device("cpu")

        self.gradient_checkpointing = cfg.model.deepsphere.train.gradient_checkpointing
        self.mixed_precision = cfg.model.deepsphere.train.mixed_precision
        self.lr = cfg.model.deepsphere.train.learning_rate
        self.n_epochs = cfg.model.deepsphere.train.n_epochs
        self.batch_size = cfg.model.deepsphere.train.batch_size
        self.checkpoint = cfg.model.deepsphere.train.checkpoint_every
        self.extra_check = cfg.model.deepsphere.train.extra_check

        self.restart_epoch = cfg.model.deepsphere.train.restart_epoch
        self.start_valid = cfg.model.deepsphere.train.start_valid

        self.modelTracker = ModelTracker()
        self.record_header = self.modelTracker.to_track

    def execute(self) -> None:
        pass

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
        n_batches = len(dataloader)

        epoch_loss = 0.0
        batch_n = 0
        batch_loss = 0
        if train:
            model.train()
        else:
            model.eval()

        for features, labels in dataloader:
            batch_n += 1

            with timer(self.modelTracker, "data_load_time"):
                features = features.to(
                    device=self.device, dtype=self.dtype, non_blocking=True
                )
                labels = labels.to(
                    device=self.device, dtype=self.dtype, non_blocking=True
                )

            with timer(self.modelTracker, "batch_process_time"):
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
                    self.modelTracker.update_tracker(
                        "train_loss", loss.item(), features.size(0)
                    )
                else:
                    with torch.no_grad():
                        output = model(features)
                        loss = loss_function(output, labels)
                        self.modelTracker.update_tracker(
                            "val_loss", loss.item(), features.size(0)
                        )

            batch_loss += loss.item()

            epoch_loss += batch_loss / self.batch_size
        epoch_loss /= n_batches
        return epoch_loss

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

    def get_datasets(
        self, distributed=True
    ) -> Tuple[Tuple[DataLoader, Sampler], Tuple[DataLoader, Sampler]]:
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

        if distributed:
            train_sampler = DistributedSampler(train_dataset)
        else:
            train_sampler = RandomSampler(train_dataset)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=train_sampler,
            pin_memory=True,
        )

        logger.info(f"Inspecting data for {train_split.name} split: ")
        self.inspect_data(train_dataloader)

        if valid_split is not None:
            valid_dataset = self.set_up_dataset(valid_split)

            if distributed:
                valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
            else:
                valid_sampler = RandomSampler(valid_dataset)

            valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=valid_sampler,
                pin_memory=True,
            )

            logger.info(f"Inspecting data for {valid_split.name} split: ")
            self.inspect_data(valid_dataloader)
        else:
            logger.info(f"No validation split found. Training without validation.")
            logger.info(
                f"This is not recommended. Consider adding a validation split in the pipeline configuration file."
            )
            valid_sampler = None
            valid_dataloader = None

        return ((train_dataloader, train_sampler), (valid_dataloader, valid_sampler))

    def set_up_dataset(self, template_split: Split) -> Dataset:
        cmb_path_template = self.make_fn_template(template_split, self.in_cmb_asset)
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        dataset = TrainCMBMapDataset(
            n_sims=template_split.n_sims,
            freqs=self.instrument.dets.keys(),
            map_fields=self.map_fields,
            label_path_template=cmb_path_template,
            label_handler=HealpyMap(),
            feature_path_template=obs_path_template,
            feature_handler=HealpyMap(),
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

    def write_model(self, epoch, model, optim, scaler=None):
        with self.name_tracker.set_context("epoch", epoch):
            self.out_model.write(
                model=model, optimizer=optim, scaler=scaler, epoch=epoch
            )


def get_mem_stats(device=None):
    mem = torch.cuda.memory_stats(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "gpu_name": props.name,
        "total_gb": 1e-9 * props.total_memory,
        "curr_alloc_gb": 1e-9 * mem["allocated_bytes.all.current"],
        "peak_alloc_gb": 1e-9 * mem["allocated_bytes.all.peak"],
        "curr_resv_gb": 1e-9 * mem["reserved_bytes.all.current"],
        "peak_resv_gb": 1e-9 * mem["reserved_bytes.all.peak"],
    }


# TODO: clean up type checking, implement error handling
def dist_run(rank, world_size, cfg):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    trainer = ModelTrainer(cfg, rank)

    device = f"cuda:{rank}"
    trainer.device = device

    (train_dataloader, train_sampler), (valid_dataloader, _) = trainer.get_datasets()

    torch.cuda.set_device(rank)
    model = trainer.make_model().cuda(rank)
    sync_model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    print(f"Model created on rank {rank}.")
    ddp_model = DDP(
        sync_model, device_ids=[rank], output_device=rank, broadcast_buffers=False
    )
    print(f"DDP model created on rank {rank}")

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=trainer.lr)

    for epoch in range(trainer.n_epochs):
        train_sampler.set_epoch(epoch)
        with timer(trainer.modelTracker, "epoch_process_time"):
            trainer.train(ddp_model, train_dataloader, optimizer, None, loss_fn)
            trainer.validate(ddp_model, valid_dataloader, loss_fn)

        if rank == 0:
            for tracker in trainer.modelTracker.to_track:
                trainer.modelTracker.all_reduce_tracker(tracker)
            vals_to_log = [
                tracker.running_avg
                for tracker in trainer.modelTracker.trackers.values()
            ]
            trainer.out_loss_record.append(vals_to_log)
            logger.info(
                f"""Epoch: {epoch}\n
                Batch process time: {vals_to_log[0]}\n
                Data load time: {vals_to_log[1]}\n
                Epoch process time: {vals_to_log[2]}\n
                Train loss: {vals_to_log[3]}\n
                Valid loss: {vals_to_log[4]}"""
            )
        if (
            (epoch + 1) in trainer.extra_check or (epoch + 1) % trainer.checkpoint == 0
        ) and rank == 0:
            trainer.write_model(epoch + 1, ddp_model, optimizer)
    cleanup()
    print(f"Finished running DDP on rank {rank}.")


class DistributedDeterministicExecutor(BaseDeepSphereModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train")

        self.cfg = cfg
        self.world_size = cfg.model.deepsphere.train.n_gpus

    def execute(self) -> None:
        mp.spawn(
            dist_run,
            args=(self.world_size, self.cfg),
            nprocs=self.world_size,
            join=True,
        )
