import torch
import torch.nn as nn
import math
from torchinfo import summary
from Utils.config import load_config


class Patch_Embedding(nn.Module):
    def __init__(self, seq_len, patch_num, patch_len, d_model, d_ff, variate_num):
        super(Patch_Embedding, self).__init__()
        self.pad_num = patch_num * patch_len - seq_len
        self.patch_len = patch_len
        self.linear = nn.Sequential(
            nn.LayerNorm([variate_num, patch_num, patch_len]),
            nn.Linear(patch_len, d_ff),
            nn.LayerNorm([variate_num, patch_num, d_ff]),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm([variate_num, patch_num, d_model]),
            nn.ReLU())

    def forward(self, x):
        x = nn.functional.pad(x, (0, self.pad_num))
        x = x.unfold(2, self.patch_len, self.patch_len)
        x = self.linear(x)
        return x


class De_Patch_Embedding(nn.Module):
    def __init__(self, pred_len, patch_num, d_model, d_ff, variate_num):
        super(De_Patch_Embedding, self).__init__()
        self.linear = nn.Sequential(
            nn.Flatten(2),
            nn.Linear(patch_num * d_model, d_ff),
            nn.LayerNorm([variate_num, d_ff]),
            nn.ReLU(),
            nn.Linear(d_ff, pred_len))

    def forward(self, x):
        x = self.linear(x)
        return x


class Model(nn.Module):
    def __init__(self, config: dict):
        super(Model, self).__init__()
        self.task_name = "long_term_forecast"
        self.ms = False
        self.EPS = 1e-5
        patch_len = 16
        patch_num = math.ceil(128 / patch_len)
        variate_num = 62
        # embedding
        self.alpha = nn.Parameter(torch.ones([1]) * 1)
        self.beta = nn.Parameter(torch.ones([1]) * 1)
        self.correlation_embedding = nn.Conv1d(62, variate_num, 3, padding='same')
        self.value_embedding = Patch_Embedding(128, patch_num, patch_len, 128, 512, variate_num)
        self.pos_embedding = nn.Parameter(torch.randn(1, variate_num, patch_num, 128))
        # head
        self.head = self.head = nn.Sequential(
            nn.Flatten(),                  # [B, C, P, D] → [B, C * P * D]
            nn.Linear(variate_num * patch_num * 128, 512),
            nn.ReLU(),
            nn.Linear(512, 2)   # 输出类别数量
        )
    def forward(self, x_enc):
        x_enc = x_enc.squeeze(dim=1) if x_enc.dim() == 4 else x_enc
        # normalization
        x_obj = x_enc[:, [-1], :] if self.ms else x_enc
        mean = torch.mean(x_obj, dim=-1, keepdim=True)
        std = torch.std(x_obj, dim=-1, keepdim=True)
        x_enc = (x_enc - torch.mean(x_enc, dim=-1, keepdim=True)) / (torch.std(x_enc, dim=-1, keepdim=True) + self.EPS)
        # embedding
        x_obj = x_enc[:, [-1], :] if self.ms else x_enc
        x_obj = self.alpha * x_obj + (1 - self.alpha) * self.correlation_embedding(x_enc)
        x_obj = self.beta * self.value_embedding(x_obj) + (1 - self.beta) * self.pos_embedding
        # head
        y_out = self.head(x_obj)
        return y_out


if __name__ == '__main__':
    config_path = "config.yaml"
    args = load_config(config_path)
    model = Model(args)
    input_shape = (1, 1, 62, 128)  # 输入形状 (batch_size, n_channels, seq_len)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 使用 torchsummary 打印模型信息
    summary(model, input_shape)

    # 手动测试模型
    # x = torch.randn(1, 1, 62, 128).to(device)  # 输入数据
    # output = model(x)
    # print(output[0].shape)  # 检查输出形状