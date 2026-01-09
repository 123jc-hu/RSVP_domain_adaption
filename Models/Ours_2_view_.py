import math
from torch import nn
from einops.layers.torch import Rearrange
from Models.EEGNet import calculate_outsize
import torch
from torchsummary import summary
from torch.nn.utils import weight_norm as WeightNorm


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        # 使用 PyTorch 的 MultiheadAttention
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_kv, d_model)
            value: (batch_size, seq_len_kv, d_model)
            mask: (batch_size, seq_len_q, seq_len_kv)
        Returns:
            output: (batch_size, seq_len_q, d_model)
        """
        # 交叉注意力（Pre-LayerNorm）
        norm_query = self.norm1(query)  # 先对 query 进行 LayerNorm
        norm_key, norm_value = self.norm1(key), self.norm1(value)  # 对 key 和 value 进行 LayerNorm
        attn_output, _ = self.multihead_attn(norm_query, norm_key, norm_value, attn_mask=mask)  # (batch_size, seq_len_q, d_model)
        attn_output = self.dropout1(attn_output)
        query = query + attn_output  # 残差连接
        # query = self.norm1(query)  # 对 query 进行 LayerNorm

        # 前馈神经网络（Pre-LayerNorm）
        norm_query = self.norm2(query)  # 先对 query 进行 LayerNorm
        ff_output = self.feedforward(norm_query)  # (batch_size, seq_len_q, d_model)
        ff_output = self.dropout2(ff_output)
        output = query + ff_output  # 残差连接

        return output


class MultiLayerCrossAttention(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_layers, dropout=0.1):
        super(MultiLayerCrossAttention, self).__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_kv, d_model)
            value: (batch_size, seq_len_kv, d_model)
            mask: (batch_size, seq_len_q, seq_len_kv)
        Returns:
            output: (batch_size, seq_len_q, d_model)
        """
        for layer in self.layers:
            query = layer(query, key, value, mask)
        return query


class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, query, key_value):
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key_value: (batch_size, seq_len_kv, d_model)
        Returns:
            attn_output: (batch_size, seq_len_q, d_model)
        """
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        return attn_output


class Model(nn.Module):
    def __init__(self, configs=None, loss_type='dist'):
        super().__init__()
        self.n_channels = 62
        self.fs = 128
        self.n_class = 2

        self.common_Block = nn.Conv2d(1, 32, (1, self.fs // 2), stride=(1, 1))
        self.Temporal_Block = self.temporal_feature_extract_block()
        self.Spatial_Block = self.spatial_feature_extract_block()
        self.self_attention = MultiLayerCrossAttention(d_model=32, nhead=1, d_ff=32 * 4, num_layers=1, dropout=0.3)
        # 交叉注意力模块
        self.cross_attention_t = MultiLayerCrossAttention(d_model=32, nhead=1, d_ff=32 * 4, num_layers=1, dropout=0.3)
        self.cross_attention_s = MultiLayerCrossAttention(d_model=32, nhead=1, d_ff=32 * 4, num_layers=1, dropout=0.3)
        # Cross-attention 模块
        # self.CrossAttn_T_S = CrossAttention(d_model=32, nhead=1)
        # self.CrossAttn_S_T = CrossAttention(d_model=32, nhead=1)

        self.Temporal_BlockOutputSize = calculate_outsize(
            nn.Sequential(self.common_Block, self.Temporal_Block), self.n_channels, self.fs)
        self.Spatial_BlockOutputSize = calculate_outsize(
            nn.Sequential(self.common_Block, self.Spatial_Block), self.n_channels, self.fs)

        if loss_type == 'dist':
            self.ClassifierBlock = DistLinear(self.Temporal_BlockOutputSize + self.Spatial_BlockOutputSize, self.n_class)
        else:
            self.ClassifierBlock = nn.Linear(self.Temporal_BlockOutputSize + self.Spatial_BlockOutputSize, self.n_class)

        # Positional encoding
        self.position = tAPE(d_model=32, max_len=65, scale_factor=1.0)
        # self.position = nn.Parameter(torch.randn(1, 65, 32))
        self.activation = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Flatten(),
        )

    def temporal_feature_extract_block(self):
        Block1 = nn.Sequential(
            # nn.Conv2d(1, 32, (1, self.fs // 2), stride=(1, 1)),
            Rearrange("b k c t -> b c k t"),
            nn.Conv2d(self.n_channels, self.n_channels, (1, self.fs // 2 + 1), groups=self.n_channels),
            nn.BatchNorm2d(self.n_channels),
            nn.SiLU(),
            nn.Dropout(0.3),
        )
        return Block1

    def spatial_feature_extract_block(self):
        Block1 = nn.Sequential(
            # nn.Conv2d(1, 32, (1, self.fs // 2), stride=(1, 1)),
            # nn.AvgPool2d((1, 4)),
            Rearrange("b k c t -> b t c k"),
            nn.Conv2d(self.fs // 2 + 1, self.fs // 2 + 1, (self.n_channels, 1), groups=self.fs // 2 + 1),
            nn.BatchNorm2d(self.fs // 2 + 1),
            nn.SiLU(),
            nn.Dropout(0.3),
        )
        return Block1


    def forward(self, x):
        x = self.common_Block(x)  # (b, 32, 62, 65)
        x_t = self.Temporal_Block(x)  # (b, 32, 62, 1)
        x_s = self.Spatial_Block(x)  # (b, 32, 1, 64)
        # 调整维度
        x_t = x_t.squeeze()  # (b, 62, 32)
        x_s = x_s.squeeze()  # (b, 64, 32)

        # Positional encoding for spatial branch
        x_s = self.position(x_s)
        # x_s = x_s + self.position.cuda()

        x_t = self.self_attention(x_t, x_t, x_t)  # 自注意力
        x_s = self.self_attention(x_s, x_s, x_s)
        # 交叉注意力
        x_t = self.cross_attention_t(x_t, x_s, x_s)  # 时间分支作为 Query，空间分支作为 Key/Value
        x_s = self.cross_attention_s(x_s, x_t, x_t)  # 空间分支作为 Query，时间分支作为 Key/Value

        x = torch.cat((x_t, x_s), dim=1)
        # x = self.hybrid_Transformer_Block(x)

        x = self.activation(x)
        x = self.ClassifierBlock(x)

        return x, None


class tAPE(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)


class DistLinear(nn.Module):
    def __init__(self, indim: int, outdim: int):
        super().__init__()
        self.L = WeightNorm(nn.Linear(indim, outdim, bias=False), name='weight', dim=0)  # 使用 WeightNorm 包装 Linear 层
        self.class_wise_learnable_norm = True  # See the issue#4&8 in the github

        # Set scale factor based on output dimension
        self.scale_factor = 2 if outdim <= 200 else 10  # Fixed scale factor for softmax input scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input features
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).expand_as(x)
        x_normalized = x / (x_norm + 1e-5)  # Add small epsilon to avoid division by zero

        # Compute cosine distance (matrix product with normalized input and weights)
        cos_dist = self.L(x_normalized)

        # Scale the cosine distance
        scores = self.scale_factor * cos_dist

        return scores


if __name__ == '__main__':
    model = Model(loss_type='dist')
    input_shape = (1, 62, 128)  # 输入形状 (batch_size, n_channels, seq_len)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 使用 torchsummary 打印模型信息
    summary(model, input_shape)
