from torch import nn
from einops.layers.torch import Rearrange
from Models.EEGNet import calculate_outsize
from Models.EEGNet import Conv2dWithConstraint, DenseWithConstraint
import torch
import torch.nn.functional as F
from torchsummary import summary
from Utils.config import load_config


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_channels = configs.n_channels
        self.fs = configs.fs
        self.n_class = configs.n_class
        # self.dropout_rate = configs.dropout_rate
        # self.t_dropout = configs.dropout
        # self.e_layers = configs.e_layers
        # self.d_model = configs.d_model
        # self.n_heads = configs.n_heads
        # self.d_ff = configs.d_model * 4
        # self.projection_dim = configs.projection_dim

        self.CNN_Block = self.cnn_feature_extract_block()
        self.Transformer_Block = self.transformer_feature_extract_block()
        self.BasicBlockOutputSize = calculate_outsize(self.CNN_Block, self.n_channels, self.fs)  # Transformer不改变shape
        self.ClassifierBlock = self.classifier_block(self.BasicBlockOutputSize)
        # 位置编码
        self.position = create_1d_absolute_sin_cos_embedding(16, 16)
        self.activation = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Flatten(),
        )
        # self.projection_head = nn.Sequential(
        #     DenseWithConstraint(self.BasicBlockOutputSize, self.projection_dim),
        #     nn.GELU(),
        #     # nn.Flatten()
        # )

    def cnn_feature_extract_block(self):
        Block1 = nn.Sequential(
            Rearrange("b k c t -> b c k t"),
            Conv2dWithConstraint(
                self.n_channels,
                16,
                (1, self.fs // 2),
                stride=(1, 2),
                bias=True,
                padding=(0, self.fs // 4),
                max_norm=0.5
            ),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(0.7),
        )
        return Block1

    def transformer_feature_extract_block(self):
        Encoder_layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=4,
            dim_feedforward=16 * 4,
            dropout=0.7,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        Block2 = nn.TransformerEncoder(Encoder_layer, num_layers=2)
        return Block2

    def classifier_block(self, input_size):
        Block3 = nn.Sequential(
            nn.Linear(
                input_size,
                self.n_class,
                bias=True,
            ),
            # nn.Softmax(dim=1)
        )
        return Block3

    def forward(self, x, stage=1):
        x = self.CNN_Block(x)  # (b, 16, 1, 16)
        x = x.squeeze(dim=2)  # (b, 16, 16)
        # 绝对位置嵌入
        x = x.permute(0, 2, 1)
        x = x + self.position.cuda()
        x = self.Transformer_Block(x)
        x = self.activation(x)
        # feature = self.projection_head(x)
        x = self.ClassifierBlock(x)

        return x


def create_1d_absolute_sin_cos_embedding(pos_len, dim):
    assert dim % 2 == 0, "wrong dimension!"
    position_emb = torch.zeros(pos_len, dim, dtype=torch.float)
    # i矩阵
    i_matrix = torch.arange(dim//2, dtype=torch.float)
    i_matrix /= dim / 2
    i_matrix = torch.pow(10000, i_matrix)
    i_matrix = 1 / i_matrix
    i_matrix = i_matrix.to(torch.long)
    # pos矩阵
    pos_vec = torch.arange(pos_len).to(torch.long)
    # 矩阵相乘，pos变成列向量，i_matrix变成行向量
    out = pos_vec[:, None] @ i_matrix[None, :]
    # 奇/偶数列
    emb_cos = torch.cos(out)
    emb_sin = torch.sin(out)
    # 赋值
    position_emb[:, 0::2] = emb_sin
    position_emb[:, 1::2] = emb_cos
    return position_emb


if __name__ == '__main__':
    config_path = "config.yaml"
    args = load_config(config_path)
    model = Model(args)
    input_shape = (1, 62, 128)  # 输入形状 (batch_size, n_channels, seq_len)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 生成随机输入数据
    # input_data = torch.randn(input_shape).to(device)
    # output = model(input_data)
    # print(output[0].shape)

    # 使用 torchsummary 打印模型信息
    summary(model, input_shape)
    # print(model.position)
