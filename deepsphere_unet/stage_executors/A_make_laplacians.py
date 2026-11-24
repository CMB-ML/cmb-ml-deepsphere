import logging
import time

import healpy as hp
import torch
from omegaconf import DictConfig

from cmbml.core import BaseStageExecutor, Asset

from deepsphere_unet.deepsphere.laplacian import get_laplacians


logger = logging.getLogger(__name__)


class MakeLaplaciansExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, "make_laplacians")

        self.nside = cfg.scenario.nside
        self.depth = cfg.model.deepsphere.network.depth
        self.laplacian_type = cfg.model.deepsphere.network.laplacian_type

        self.out_laplacians: Asset = self.assets_out["laplacians"]

        if not hp.isnsideok(self.nside, nest=True):
            raise ValueError(f"Invalid NESTED Nside: {self.nside}")

        if self.depth < 1 or self.nside < 2**self.depth:
            raise ValueError(
                f"Invalid depth {self.depth} for Nside={self.nside}"
            )

        if self.laplacian_type not in ("combinatorial", "normalized"):
            raise ValueError(
                f"Unknown Laplacian type: {self.laplacian_type}"
            )

    def execute(self) -> None:
        output_path = self.out_laplacians.path

        if output_path.exists():
            raise FileExistsError(output_path)

        logger.info(f"Nside: {self.nside}")
        logger.info(f"Network depth: {self.depth}")
        logger.info(f"Laplacian type: {self.laplacian_type}")
        logger.info(f"Output path: {output_path}")

        logger.info("Constructing Laplacians...")
        start = time.perf_counter()

        laps = get_laplacians(
            hp.nside2npix(self.nside),
            self.depth + 1,
            self.laplacian_type,
        )

        elapsed = time.perf_counter() - start
        logger.info(f"Laplacian construction took {elapsed:.2f} seconds.")

        self.validate_laplacians(laps)

        logger.info("Saving Laplacians...")
        self.out_laplacians.write(data=laps)

        size_mib = output_path.stat().st_size / 1024**2
        logger.info(f"Output file size: {size_mib:.2f} MiB")

    def validate_laplacians(self, laps) -> None:
        if len(laps) != self.depth + 1:
            raise ValueError("Unexpected number of Laplacians.")

        for i, lap in enumerate(laps):
            nside = self.nside // 2**(self.depth - i)
            npix = hp.nside2npix(nside)

            if lap.shape != (npix, npix):
                raise ValueError(
                    f"Unexpected Laplacian shape at Nside={nside}: "
                    f"{lap.shape}"
                )

            if lap.layout != torch.sparse_csr:
                raise ValueError("Expected sparse CSR Laplacian.")

            if lap.dtype != torch.float32:
                raise ValueError("Expected float32 Laplacian.")

            if lap.device.type != "cpu":
                raise ValueError("Expected CPU Laplacian.")

            if not torch.isfinite(lap.values()).all().item():
                raise ValueError("Nonfinite Laplacian values detected.")

            logger.info(
                f"Nside={nside}: "
                f"shape={tuple(lap.shape)}, "
                f"nnz={lap._nnz()}"
            )

        logger.info("Laplacian validation passed.")
