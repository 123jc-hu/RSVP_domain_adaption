import time

from Train.trainer import OptimizedExperimentRunner
from Utils.config import Logger, build_config


def main(config):
    start_time = time.time()
    logger = Logger(config["model"]).get_logger()
    logger.info(f"Starting training for model: {config['model']}")

    experiment = OptimizedExperimentRunner(config, logger)
    logger.info(f"\n================ Running Dataset: {config['dataset']} ================\n")
    experiment.run_experiment()

    logger.info("Finished training and testing!")
    elapsed_minutes = (time.time() - start_time) / 60.0
    logger.info(f"Elapsed time: {elapsed_minutes:.2f} minutes \n")


if __name__ == "__main__":
    config = build_config("EEGNet")
    main(config)
