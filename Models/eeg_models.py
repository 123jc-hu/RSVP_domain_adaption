from Models import (Ours_3, EEGNet, Manor_CNN, Ours_light, PLNet, TTMTN, HCANN, EEGInception, DeepConvNet,
                    Ours, PPNN, EEGDeformer, TSformer_SA, CRE_TSCAE, EEGConformer, AdaptEEG, SDDA, SCLDGN,
                    lightweight_erp, SST_DPN, LMDANet, LENet, OneDCNN_BiLSTM, Ours_light_with_K, Ours_new,
                    TSCAE, MOCNN)
from typing import Any, Dict

def model_dict() -> Dict[str, Any]:
    """
    返回EEG模型字典，映射模型名称到对应的模型类。

    Returns:
        Dict[str, Any]: 模型名称到模型类的映射字典
    """
    return {
        "Ours_3": Ours_3,
        "EEGNet": EEGNet,
        "Manor_CNN": Manor_CNN,
        "DeepConvNet": DeepConvNet,
        "PLNet": PLNet,
        "EEGInception": EEGInception,
        "TTMTN": TTMTN,
        'HCANN': HCANN,
        "Ours": Ours,
        "PPNN": PPNN,
        "EEGDeformer": EEGDeformer,
        "TSformer-SA": TSformer_SA,
        "CRE-TSCAE": CRE_TSCAE,
        "EEGConformer": EEGConformer,
        "AdaptEEG": AdaptEEG,
        "SDDA": SDDA,
        "SCLDGN": SCLDGN,
        "Ours_light": Ours_light,
        "Ours_light_without_K": Ours_light,
        "Ours_light_with_K": Ours_light_with_K,
        "lightweight_erp": lightweight_erp,
        "SST_DPN": SST_DPN,
        "LMDANet": LMDANet,
        "LENet": LENet,
        "1DCNN_BiLSTM": OneDCNN_BiLSTM,
        "Ours_light_reg": Ours_light_with_K,
        "Correlation-based Selection": Ours_light_with_K,
        "Frequency-Domain Weighting": Ours_light_with_K,
        "SparseEA": Ours_light_with_K,
        "Gumbel-Softmax Selector": Ours_light_with_K,
        "EEGNet_OGS": EEGNet,
        "PLNet_OGS": PLNet,
        "PPNN_OGS": PPNN,
        "adaptive_ours_light": Ours_light_with_K,
        "Ours_new": Ours_new,
        "Ours_ZOS": Ours_new,
        "Correlation_new": Ours_new,
        "Frequency_new": Ours_new,
        "SparseEA_new": Ours_new,
        "TSCAE": TSCAE,
        "MOCNN": MOCNN,
        "ABMOHS": Ours_new,
        "GSS": Ours_new,
    }