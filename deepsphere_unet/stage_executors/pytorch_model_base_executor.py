import logging

import healpy as hp
from omegaconf import DictConfig
import torch

from deepsphere_unet.deepsphere.deterministic_unet import SphericalUNet
from deepsphere_unet.deepsphere.bayesian_unet import BayesianSphericalUNet

from cmbml.core.asset_handlers import Config
from cmbml.torch.pytorch_executor_base import BasePyTorchModelExecutor
from cmbml.sim_stats.utils.transforms import make_map_transforms


logger = logging.getLogger(__name__)


def get_precomputed_laplacian_path(executor, cfg):
    if not cfg.model.deepsphere.network.use_precomputed_laplacians:
        return None

    path = executor.assets_in["laplacians"].path

    if not path.is_file():
        raise FileNotFoundError(
            f"Precomputed Laplacians not found: {path}. "
            "Run main_deepsphere_laplacians.py first."
        )

    logger.info(f"Using precomputed Laplacians: {path}")

    return path


class BaseDeepSphereModelExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig, stage_str) -> None:
        super().__init__(cfg, stage_str)

        # self.nside = cfg.scenario.nside
        self.in_scalar_stats = self.assets_in["scalar_stats"]
        in_scalar_stats_handler: Config

        self.scaling = cfg.model.deepsphere.scaling

        input_channels = cfg.scenario.detector_freqs
        depth = cfg.model.deepsphere.network.depth
        encoder_channels = cfg.model.deepsphere.network.encoder_channels
        bottleneck_channels = cfg.model.deepsphere.network.bottleneck_channels
        decoder_channels = cfg.model.deepsphere.network.decoder_channels
        laplacian_type = cfg.model.deepsphere.network.laplacian_type
        kernel_size = cfg.model.deepsphere.network.kernel_size

        input_c = len(input_channels)

        self.model_dict = dict(
            nside=self.nside,
            input_channels=input_c,
            depth=depth,
            encoder_channels=encoder_channels,
            bottleneck_channels=bottleneck_channels,
            decoder_channels=decoder_channels,
            laplacian_type=laplacian_type,
            kernel_size=kernel_size,
            laps=get_precomputed_laplacian_path(self, cfg),
        )

    def build_transforms(self):
        return make_map_transforms(
            scaling=self.scaling,
            freqs=list(self.instrument.dets.keys()),
            scalar_stats=self.in_scalar_stats.read(),
        )

    def try_model(self, model):
        dummy_input = torch.rand(
            1,
            hp.nside2npix(self.nside),
            self.model_dict["input_channels"],
            device=self.device,
        )

        was_training = model.training
        model.eval()
        with torch.no_grad():
            result = model(dummy_input)
        model.train(was_training)

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
        encoder_channels = cfg.model.deepsphere.network.encoder_channels
        bottleneck_channels = cfg.model.deepsphere.network.bottleneck_channels
        decoder_channels = cfg.model.deepsphere.network.decoder_channels
        tau = cfg.model.deepsphere.network.tau
        length_scale = cfg.model.deepsphere.network.length_scale

        if cfg.splits.name == "1-1" or cfg.splits.name == "1-10":
            train_length = cfg.splits.Test.n_sims
        train_length = cfg.splits.Train.n_sims

        weight_regularizer = length_scale**2 / (tau * train_length)
        dropout_regularizer = 2 / (tau * train_length)
        input_c = len(input_channels)

        self.model_dict = dict(
            nside=nside,
            input_channels=input_c,
            depth=depth,
            encoder_channels=encoder_channels,
            bottleneck_channels=bottleneck_channels,
            decoder_channels=decoder_channels,
            laplacian_type=laplacian_type,
            kernel_size=kernel_size,
            weight_regularizer=weight_regularizer,
            dropout_regularizer=dropout_regularizer,
            laps=get_precomputed_laplacian_path(self, cfg),
        )

    def try_model(self, model):
        dummy_input = torch.rand(
            1,
            hp.nside2npix(self.nside),
            self.model_dict["input_channels"],
            device=self.device,
        )

        was_training = model.training
        model.eval()
        with torch.no_grad():
            mu, sigma = model(dummy_input)
        model.train(was_training)

        logger.debug(f"Output mu size: {mu.size()}")
        logger.debug(f"Output sigma size: {sigma.size()}")

    def make_model(self):
        return BayesianSphericalUNet(**self.model_dict)
