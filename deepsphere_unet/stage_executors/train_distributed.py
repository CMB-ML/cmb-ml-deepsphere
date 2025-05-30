import logging

from tqdm import tqdm

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from omegaconf import DictConfig

import healpy as hp

from .pytorch_model_base_executor import BaseDeepSphereModelExecutor
from cmbml.core import Split, Asset

from cmbml.core.asset_handlers import (
    HealpyMap,
    Config,
    AppendingCsvHandler
    )

from cmbml.torch.pytorch_model_handler import PyTorchModel  # Import for typing hint

from deepsphere_unet.dataset import TrainCMBMapDataset


import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def demo_run(rank, world_size, model_maker, dataset, lr, n_epochs, train_fn):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = model_maker().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=lr)

    dataloader = DataLoader(
        dataset,
        batch_size=5,
        shuffle=False,
        sampler=DistributedSampler(dataset),

    )
    
    for epoch in range(n_epochs):
        loss = train_fn(model, dataloader, optimizer, None, loss_fn)
        print(f"Epoch {epoch}, Loss: {loss}")
    
    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")


logger = logging.getLogger(__name__)


class DeterministicTrainingExecutor(BaseDeepSphereModelExecutor):
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
        # self.in_norm: Asset = self.assets_in["dataset_stats"]  # TODO: Does removing this line break anything?
        in_model_handler: PyTorchModel
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap

        model_precision = 'float'
        self.dtype = self.dtype_mapping[model_precision]
        self.choose_device(cfg.model.deepsphere.train.device)
        if self.device == "mps": # MPS is not supported for sparse models
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

    def execute(self) -> None:
        pass

    def one_pass(self, model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, loss_function: torch.nn.Module, train: bool) -> float:
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
        with tqdm(dataloader, postfix={'Loss': 0}) as pbar:
            for features, labels in pbar:
                batch_n += 1

                features = features.to(device=self.rank, dtype=self.dtype)
                labels = labels.to(device=self.rank, dtype=self.dtype)

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
                        print("forward pass")
                        output = model(features)
                        loss = loss_function(output, labels)
                        print("backward pass")
                        loss.backward()
                        optimizer.step()
                else:
                    with torch.no_grad():
                        output = model(features)
                        loss = loss_function(output, labels)

                batch_loss += loss.item()

                pbar.set_postfix({f'Loss for {batch_n}/{len(dataloader)}': loss.item() / self.batch_size})

                epoch_loss += batch_loss / self.batch_size
            epoch_loss /= n_batches
        return epoch_loss
    
    def train(self, model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, loss_function: torch.nn.Module) -> float:
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

        return self.one_pass(model, dataloader, optimizer, scaler, loss_function, train=True)
    
    def validate(self, model: torch.nn.Module, dataloader: DataLoader, loss_function: torch.nn.Module) -> float:
        """Runs the validation loop for a single epoch.

        Args:
            model (torch.nn.Module): Model to validate
            dataloader (DataLoader): Validation Data
            loss_function (torch.nn.Module): Loss

        Returns:
            float: validation loss for the epoch
        """
        return self.one_pass(model=model, dataloader=dataloader, optimizer=None, scaler=None, loss_function=loss_function, train=False)

    def set_up_dataset(self, template_split: Split) -> None:
        cmb_path_template = self.make_fn_template(template_split, self.in_cmb_asset)
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        dataset = TrainCMBMapDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            label_path_template=cmb_path_template, 
            label_handler=HealpyMap(),
            feature_path_template=obs_path_template,
            feature_handler=HealpyMap()
            )
        return dataset

    def inspect_data(self, dataloader):
        train_features, train_labels = next(iter(dataloader))
        logger.info(f"{self.__class__.__name__}.inspect_data() Feature batch shape: {train_features.size()}") # Should be (batch_size, npix, n_map_fields)
        logger.info(f"{self.__class__.__name__}.inspect_data() Labels batch shape: {train_labels.size()}")
        npix_data = train_features.size()[1]
        npix_cfg  = hp.nside2npix(self.nside)
        assert npix_cfg == npix_data, "Npix for loaded map does not match configuration yamls."

def dist_run(rank, world_size, cfg):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    model_trainer = DeterministicTrainingExecutor(cfg, rank)

    dataset = model_trainer.set_up_dataset(model_trainer.splits[0])
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False, sampler=DistributedSampler(dataset))

    model = model_trainer.make_model().to(rank)
    print(f"Model created on rank {rank}.")
    print(f"On rank {rank}: bottleneck lap device {model.bottleneck_block.cheb1.laplacian.laplacian_0.device}")
    ddp_model = DDP(model, device_ids=[rank], broadcast_buffers=False)
    print(f"On rank {rank}: DDP bottleneck lap device {ddp_model.module.bottleneck_block.cheb1.laplacian.laplacian_0.device}")
    print(f'ddp model created on rank {rank}')
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=model_trainer.lr)
    
    for epoch in range(model_trainer.n_epochs):
        print(f"Epoch {epoch} on rank {rank}.")
        loss = model_trainer.train(model, dataloader, optimizer, None, loss_fn)
        print(f"Epoch {epoch}, Loss: {loss}")
    
    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")

class DistDeterministicTrainingExecutor(BaseDeepSphereModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train")

        self.cfg = cfg
        self.world_size = torch.cuda.device_count()


    def execute(self) -> None:
        
        
        mp.spawn(dist_run, args=(self.world_size, self.cfg), nprocs=self.world_size, join=True)
        