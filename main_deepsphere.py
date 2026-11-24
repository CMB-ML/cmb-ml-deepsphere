"""
This script runs a pipeline for training and prediction of the cleaned
CMB signal using DeepSphere.

Usage:
    python main_deepsphere.py
"""

import logging

import hydra

from cmbml.core import PipelineContext, LogMaker

from deepsphere_unet import (
    DeterministicTrainingExecutor,
    PredictionExecutor,
)


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_deepsphere")
def run_deepsphere(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    pipes = [
        DeterministicTrainingExecutor,
        # PredictionExecutor,
    ]
    run(cfg, pipes)


def run(cfg, pipes):
    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)
    for pipe in pipes:
        pipeline_context.add_pipe(pipe)

    pipeline_context.prerun_pipeline()
    try:
        pipeline_context.run_pipeline()
    except Exception as e:
        logger.exception("An exception occurred during the pipeline.", exc_info=e)
        raise
    finally:
        logger.info("Pipeline completed.")
        log_maker.copy_hydra_run_to_dataset_log()


if __name__ == "__main__":
    run_deepsphere()
