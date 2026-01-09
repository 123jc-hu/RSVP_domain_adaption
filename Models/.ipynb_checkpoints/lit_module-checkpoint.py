from argparse import Namespace
from typing import Any, Dict, List
import lightning as pl
import torch
import torch.nn as nn
from torch.optim import Optimizer, RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from Models.eeg_models import model_dict


class EEGLitModel(pl.LightningModule):
    """EEG模型Lightning封装
    
    Features:
    - 自动处理训练/验证/测试流程
    - 支持动态参数冻结
    - 集成常用评估指标
    """
    def __init__(self, args: Namespace):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_dict()[args.model].Model(args)
        self.criterion = nn.CrossEntropyLoss()

        # 初始化指标
        self._setup_metrics()
    
    def _setup_metrics(self):
        """初始化评估指标"""
        from Utils.metrics import cal_ba, cal_F1_score, calculate_tpr_tnr  # 延迟导入避免循环依赖
        self.metric_fns = {
            "ba": cal_ba,
            "f1": cal_F1_score,
            "tpr_tnr": calculate_tpr_tnr
        }

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        x, y = batch
        outputs, _ = self.model(x.unsqueeze(1))
        loss = self.criterion(outputs, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        x, y = batch
        outputs, _ = self.model(x.unsqueeze(1))
        loss = self.criterion(outputs, y)
        preds = outputs.argmax(dim=1)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"preds": preds, "targets": y}
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        x, y = batch
        outputs, _ = self.model(x.unsqueeze(1))
        preds = outputs.argmax(dim=1)
        return {"preds": preds, "targets": y}

    def on_validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        """验证周期结束处理"""
        preds = torch.cat([x["preds"] for x in outputs])  # 使用参数outputs
        targets = torch.cat([x["targets"] for x in outputs])
        self._log_metrics(preds, targets, "val")
    
    def on_test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        """测试周期结束处理"""
        preds = torch.cat([x["preds"] for x in outputs])  # 使用参数outputs
        targets = torch.cat([x["targets"] for x in outputs])
        self._log_metrics(preds, targets, "test")
    
    def _log_metrics(self, preds: torch.Tensor, targets: torch.Tensor, prefix: str):
        """统一记录指标"""
        metrics = {
            f"{prefix}/ba": self.metric_fns["ba"](targets.cpu(), preds.cpu()),
            f"{prefix}/f1": self.metric_fns["f1"](targets.cpu(), preds.cpu()),
        }
        tpr, tnr = self.metric_fns["tpr_tnr"](targets.cpu(), preds.cpu())
        metrics.update({f"{prefix}/tpr": tpr, f"{prefix}/tnr": tnr})
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        """配置优化器和学习率调度"""
        optimizer = RAdam(
            self.parameters(),
            lr=self.hparams.args.learning_rate,
            weight_decay=self.hparams.args.weight_decay
        )
        scheduler = {
            "scheduler": CosineAnnealingLR(optimizer, T_max=10),
            "interval": "epoch",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def on_fit_start(self):
        """训练开始前的准备"""
        if self.hparams.args.train_mode == "Fine_tune":
            self._freeze_parameters()

    def _freeze_parameters(self):
        """冻结非分类层参数"""
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False