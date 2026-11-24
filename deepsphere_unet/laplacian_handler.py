import logging
from pathlib import Path

import torch

from cmbml.core import GenericHandler, make_directories, register_handler


logger = logging.getLogger(__name__)


class LaplacianHandler(GenericHandler):
    def read(self, path: Path):
        logger.debug(f"Reading Laplacians from '{path}'")
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

    def write(self, path: Path, data) -> None:
        if path.exists():
            raise FileExistsError(path)

        make_directories(path)

        logger.debug(f"Writing Laplacians to '{path}'")
        torch.save(data, path)


register_handler("LaplacianHandler", LaplacianHandler)
