import torch
import torch.nn as nn
from einops import rearrange
from torchsummary import summary


class Model(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.n_channels = 62
        self.n_timepoints = 128
        self.n_classes = 2
        self.dim = 32  # 隐藏层维度
        self.heads = 1  # 注意力头数
        self.num_layers = 1  # Transformer 层数

        # 输入嵌入层
        self.input_embed = nn.Linear(self.n_channels, self.dim)

        # Transformer 编码器层
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.dim,
                nhead=self.heads,
                dim_feedforward=self.dim * 4,
                dropout=0.1
            ),
            num_layers=self.num_layers
        )

        # 分类头
        self.fc = nn.Linear(self.dim, self.n_classes)

    def forward(self, x):
        # 输入形状: (batch, channels, timepoints)
        x = x.squeeze()
        x = x.transpose(1, 2)  # (batch, timepoints, channels)
        x = self.input_embed(x)  # (batch, timepoints, dim)

        # Transformer 编码器
        x = self.transformer(x)  # (batch, timepoints, dim)

        # 全局平均池化
        x = x.mean(dim=1)  # (batch, dim)

        # 分类
        x = self.fc(x)  # (batch, n_classes)
        return x
    

if __name__ == '__main__':
    model = Model()
    input_shape = (1, 62, 128)  # 输入形状 (batch_size, n_channels, seq_len)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 使用 torchsummary 打印模型信息
    summary(model, input_shape)
