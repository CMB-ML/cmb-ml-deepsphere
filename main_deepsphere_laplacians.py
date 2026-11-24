"""
Precompute graph Laplacians for DeepSphere.
"""

import logging

import hydra

from cmbml.core import PipelineContext, LogMaker

from deepsphere_unet import MakeLaplaciansExecutor


logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="cfg",
    config_name="config_deepsphere",
)
def run_deepsphere_laplacians(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(MakeLaplaciansExecutor)

    pipeline_context.prerun_pipeline()
    pipeline_context.run_pipeline()

    logger.info("Pipeline completed.")


if __name__ == "__main__":
    run_deepsphere_laplacians()
