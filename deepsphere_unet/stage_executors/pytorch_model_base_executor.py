import logging

import healpy as hp
from omegaconf import DictConfig
import torch

from deepsphere_unet.deepsphere.unet_model import SphericalUNet
from deepsphere_unet.deepsphere.bayesian_unet import BayesianSphericalUNet

from cmbml.torch.pytorch_executor_base import BasePyTorchModelExecutor


logger = logging.getLogger(__name__)


class BaseDeepSphereModelExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig, stage_str) -> None:
        super().__init__(cfg, stage_str)
        
        # self.nside = cfg.scenario.nside
        npix = hp.nside2npix(self.nside)

        input_channels = cfg.scenario.detector_freqs 
        depth = cfg.model.deepsphere.network.depth 
        laplacian_type = cfg.model.deepsphere.network.laplacian_type
        kernel_size = cfg.model.deepsphere.network.kernel_size

        # input_c = len(input_channels) # Ideally should work, but hardcoded to 9
        input_c = 9

        self.model_dict = dict(N=npix, depth=depth, laplacian_type=laplacian_type, kernel_size=kernel_size)

    def try_model(self, model):
        dummy_input = torch.rand(1, hp.nside2npix(self.nside), 9, device=self.device)
        result = model(dummy_input)
        logger.debug(f"Output result size: {result.size()}")

    def make_model(self):
        return SphericalUNet(**self.model_dict)

class BayesianDeepSphereModelExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig, stage_str) -> None:
        super().__init__(cfg, stage_str)
        
        nside = cfg.scenario.nside
        npix = hp.nside2npix(nside)

        input_channels = cfg.scenario.detector_freqs 
        depth = cfg.model.deepsphere.network.depth 
        laplacian_type = cfg.model.deepsphere.network.laplacian_type
        kernel_size = cfg.model.deepsphere.network.kernel_size
        tau = cfg.model.deepsphere.network.tau
        length_scale = cfg.model.deepsphere.network.length_scale

        if cfg.splits.name == "1-1" or cfg.splits.name == "1-10":
            train_length = cfg.splits.Test.n_sims
        train_length = cfg.splits.Train.n_sims

        
        weight_regularizer = length_scale**2 / (tau * train_length)
        dropout_regularizer = 2 / (tau * train_length)
        # input_c = len(input_channels) # Ideally should work, but hardcoded to 9
        input_c = 9

        self.model_dict = dict(N=npix, depth=depth, laplacian_type=laplacian_type, kernel_size=kernel_size, weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)

    def try_model(self, model):
        dummy_input = torch.rand(1, hp.nside2npix(self.nside), 9, device=self.device)
        mu, sigma = model(dummy_input)

        # logger.debug(f"Output result size: {result.size()}")
        logger.debug(f"Output mu size: {mu.size()}")
        logger.debug(f"Output sigma size: {sigma.size()}")
    def make_model(self):
        return BayesianSphericalUNet(**self.model_dict)
