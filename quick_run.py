from main import main
from Utils.config import build_config


def build_quick_config():
    """
    Fast smoke-test config:
    - keep training path unchanged
    - run only a few held-out subjects
    - reduce source pool size and epochs
    """
    cfg = build_config("EEGNet", "THU")

    cfg["n_fold"] = 2
    cfg["epochs"] = 3
    cfg["early_stop_start_epoch"] = 1
    cfg["patience"] = 2

    cfg["source_selection"] = "Random"
    cfg["source_selection_k"] = 6

    cfg["subject_batching"] = True
    cfg["subject_batch_size"] = 32
    cfg["subjects_per_batch"] = 6
    cfg["epoch_step_reference"] = "all_candidates"
    cfg["batch_size"] = 128
    cfg["num_workers"] = 8
    cfg["pin_memory"] = True
    cfg["persistent_workers"] = True
    cfg["prefetch_factor"] = 2

    cfg["log_runtime"] = False
    cfg["rpt_aug_enable"] = False
    cfg["iahm_enable"] = False
    cfg["lambda_align"] = 0.0
    cfg["use_target_stream"] = False
    return cfg


if __name__ == "__main__":
    main(build_quick_config())
