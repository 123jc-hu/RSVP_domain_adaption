import os
import time
from Utils.config import load_config, Logger, build_config
from Train.trainer import OptimizedExperimentRunner


def main(config):
    start_time = time.time()
    logger = Logger(config["model"]).get_logger()
    logger.info(f'Starting training for model: {config["model"]}')

    # 鍒濆鍖栧疄楠岃繍琛屽櫒
    experiment = OptimizedExperimentRunner(config, logger)

    logger.info(f"\n================ Running Dataset: {config['dataset']} ================\n")

    # 鎵ц瀹屾暣瀹為獙娴佺▼
    experiment.run_experiment()
    
    logger.info("Finished training and testing!")
    logger.info(f"Selected K: {config.get('selector_K', 'N/A')}")
    logger.info(f"Elapsed time: {(time.time() - start_time) / 60:.2f} minutes \n")


if __name__ == '__main__':
    # for i in [5, 10, 15, 20]:
    #     for index in range(10):
    #         config = build_config("Ours_light_reg")  # 榛樿鍙窇 Ours 妯″瀷
    #         config["masked_num_channels"] = i
    #         config["mask_index"] = index
    #         main(config)
    config = build_config("EEGNet")  # 榛樿鍙窇 Ours 妯″瀷
    main(config)
