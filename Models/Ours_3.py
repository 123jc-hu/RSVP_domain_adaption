from torch import nn
from einops.layers.torch import Rearrange, Reduce
from einops import repeat, rearrange
from Models.EEGNet import calculate_outsize
import torch
from torchsummary import summary
from Models.Ours_2_view import tAPE
import yaml
from argparse import Namespace
import matplotlib.pyplot as plt
import torch.nn.functional as F
from Models.Ours_4 import LinearWithConstraint
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradientReversalFunction.apply(x, lambd)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        # self.feedforward = nn.Sequential(
        #     nn.ZeroPad2d((d_model // 8 - 1, d_model // 8, 0, 0)),
        #     nn.Conv1d(seq_len, seq_len, d_model//4, bias=False, groups=seq_len),
        #     nn.GELU(),
        #     # nn.Dropout(dropout),
        #     # nn.Linear(d_ff, d_model),
        # )
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
        attn_output, attn_weights = self.multihead_attn(norm_query, norm_key, norm_value, attn_mask=mask)  # (batch_size, seq_len_q, d_model)
        attn_output = self.dropout1(attn_output)
        query = query + attn_output  # 残差连接

        # 前馈神经网络（Pre-LayerNorm）
        norm_query = self.norm2(query)  # 先对 query 进行 LayerNorm
        ff_output = self.feedforward(norm_query)  # (batch_size, seq_len_q, d_model)
        ff_output = self.dropout2(ff_output)
        output = query + ff_output  # 残差连接

        return output, attn_weights


class MultiLayerCrossAttention(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_layers, dropout=0.1):
        super(MultiLayerCrossAttention, self).__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, query, key, value, mask=None, use_last_layer=True):
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_kv, d_model)
            value: (batch_size, seq_len_kv, d_model)
            mask: (batch_size, seq_len_q, seq_len_kv)
        Returns:
            output: (batch_size, seq_len_q, d_model)
        """
        attn_weights_list = []  # 存储所有层的注意力矩阵
        for layer in self.layers:
            query, attn_weights = layer(query, key, value, mask)
            attn_weights_list.append(attn_weights)  # 形状: (batch_size, num_heads, seq_len_q, seq_len_kv)
        if use_last_layer:
            attn_weights_final = attn_weights_list[-1]  # 选择最后一层的注意力权重
        else:
            attn_weights_final = torch.stack(attn_weights_list).mean(dim=0)  # 所有层的平均注意力

        return query, attn_weights_final  # 返回最终输出和注意力权重


def get_important_tokens(attn_weights, top_k=None):
    """
    根据 Multi-Head Attention 计算每个 token 的重要性，并筛选不重要的 token

    Args:
        threshold (float): 若 token 重要性低于该阈值，则去掉
        top_k (int): 仅保留最重要的前 K 个 token（若为 None，则基于阈值筛选）

    Returns:
        important_mask (Tensor): 形状 (batch_size, seq_len_q)，值为 0（去掉）或 1（保留）
    """
    head_importance, _ = attn_weights.max(dim=-1)  # (batch, heads, seq_len_q)
    if head_importance.dim() != 3:
        token_scores = head_importance
    else:
        token_scores = head_importance.mean(dim=1)     # (batch, seq_len_q)

    topk_values, _ = torch.topk(token_scores, top_k, dim=-1, largest=True, sorted=True)
    threshold = topk_values[:, -1]  # 选择第 K 个值作为阈值

    important_mask = (token_scores >= threshold.unsqueeze(-1)).float()  # 1 = 保留, 0 = 去掉
    return important_mask


class GraphPositionEncoder(nn.Module):
    def __init__(self, adj_matrix, d_model):
        super().__init__()
        self.adj = torch.from_numpy(adj_matrix).float()  # [C,C]
        self.gcn = nn.Linear(adj_matrix.shape[0], d_model)  # 可替换为GAT层

    def forward(self, x):
        # x: [b,c,t]
        pe = self.gcn(self.adj.to(x.device))  # [C,d_model]
        pe = pe.unsqueeze(0).repeat_interleave(4, dim=1)  # [1,4C,d_model]
        return x + pe  # 广播到 [b,c,t]


class STSConv(nn.Module):
    def __init__(self, C, F):
        super().__init__()
        # 空间核组
        self.spatial_kernels = nn.ModuleList([
            nn.Conv1d(C, F, 1) for _ in range(4)  # 4个基础空间核
        ])
        # 时间选择器
        self.selector = nn.Sequential(
            nn.Conv1d(C, 4, 25, padding=12),
            nn.GELU(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.squeeze(1)  # (b, 62, 128)
        # x: (b, C, T)
        # 生成选择权重 (b, 4, T)
        weights = self.selector(x)

        # 并行计算各空间核结果
        spatial_features = [conv(x) for conv in self.spatial_kernels]  # 4*(b,F,T)
        stacked = torch.stack(spatial_features, dim=1)  # (b,4,F,T)

        # 动态加权组合
        return torch.einsum('bkt,bkft->bft', weights, stacked)


class Temporal_spatial_network(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_channels = configs.n_channels
        self.fs = configs.fs
        self.n_class = configs.n_class
        self.CNN1 = nn.Sequential(
            Rearrange("b k c t -> b t c k"),
            nn.Conv2d(self.fs, self.fs * 4, (self.n_channels // 4, 1), stride=(1, 1), groups=self.fs),
            nn.BatchNorm2d(self.fs * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.3),
            Rearrange("b (t h) c k -> b t h (c k)", h=4),
        )
        self.CNN2 = nn.Sequential(
            nn.Conv2d(1, 8, (self.n_channels, 1), stride=(self.n_channels, 1), groups=1),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Dropout(0.3),
            Rearrange("b k c t -> b t (c k)"),
        )
    def forward(self, x):
        x1 = self.CNN1(x)  # (b, 128*4, 1, 1)
        x2 = self.CNN2(x)  # (b, 8, 1, 128)
        return torch.cat([x1.squeeze(-1), x2], dim=-1)  # (b, 128*4+8, 12)

class LSAM(nn.Module):
    def __init__(self, feature_dim, num_domains, num_classes=2):
        """
        feature_dim: 输入特征的维度
        num_domains: 源域数量 M
        num_classes: 类别数量（默认二分类）
        """
        super(LSAM, self).__init__()
        self.num_classes = num_classes
        self.domain_classifiers = nn.ModuleList([
            nn.Sequential(
                LinearWithConstraint(feature_dim, num_domains, max_norm=1, doWeightNorm=True),
                # nn.ReLU(),
                # nn.Linear(128, num_domains)
            ) for _ in range(num_classes)
        ])

    def forward(self, features):
        """
        features: shape (B, feature_dim)
        返回所有 class-wise 域分类器的输出 logits: List of (B, M)
        """
        return [clf(features) for clf in self.domain_classifiers]

class Model(nn.Module):
    def __init__(self, configs=None):
        super().__init__()
        self.n_channels = configs.n_channels
        self.fs = configs.fs
        self.n_class = configs.n_class
        self.CNN_Block1 = nn.Sequential(
            Rearrange("b k c t -> b c k t"),
            nn.ZeroPad2d((self.fs // 8 - 1, self.fs // 8, 0, 0)),
            nn.Conv2d(
                self.n_channels,
                self.n_channels * 4,
                (1, self.fs // 4),
                stride=(1, 1),
                groups=self.n_channels,
            ),
            nn.BatchNorm2d(self.n_channels * 4),
            nn.GELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.3),
            Rearrange("b c k t -> b c (k t)"),
        )
        # self.CNN_Block1 = nn.Sequential(
        #     nn.Conv2d(1, 16, (self.n_channels, 1), stride=(self.n_channels, 1), padding=(0, 0), groups=1),
        #     nn.BatchNorm2d(16),
        #     nn.GELU(),
        #     # nn.AvgPool2d((1, 8)),
        #     nn.Dropout(0.3),
        # )
        self.domain_general = nn.Sequential(
            # Rearrange("b (c h) t -> b (h c) t", h=4),
            nn.Conv1d(self.n_channels * 4, 16, 1, groups=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
            # nn.Conv1d(16, 16, 8, padding=4),
            # nn.BatchNorm1d(16),
            # nn.GELU(),
            # nn.AvgPool1d(2),
            nn.Dropout(0.3),
            nn.Flatten(),
        )
        # self.domain_domain = nn.Sequential(
        #     Rearrange("b (c h) t -> b (h c) t", h=4),
        #     nn.Conv1d(self.n_channels * 4, 16, 1, groups=4),
        #     nn.BatchNorm1d(16),
        #     nn.GELU(),
        #     nn.Dropout(0.3),
        #     nn.Flatten(),
        # )

        self.Transformer_Block1 = MultiLayerCrossAttention(d_model=16, nhead=1,
                                                           d_ff=16 * 4, num_layers=1, dropout=0.1)
        # self.Transformer_Block2 = MultiLayerCrossAttention(d_model=16, nhead=2, d_ff=16 * 4, num_layers=1,
        #                                                    dropout=0.1)

        # self.BasicBlockOutputSize1 = calculate_outsize(self.CNN_Block1, self.n_channels, self.fs)
        self.BasicBlockOutputSize2 = calculate_outsize(nn.Sequential(self.CNN_Block1, self.domain_general), self.n_channels, self.fs)
        self.ClassifierBlock_g = self.classifier_block(self.BasicBlockOutputSize2, self.n_class)
        # self.ClassifierBlock_s = nn.ModuleList(self.classifier_block(self.BasicBlockOutputSize2, self.n_class) for _ in range(63))
        # self.ClassifierBlock_d = self.classifier_block(self.BasicBlockOutputSize2, 63)
        self.activation = nn.Sequential(
            # nn.GELU(),
            # nn.Dropout(0.1),
            nn.Flatten(),
        )
        # self.position = tAPE(d_model=32, max_len=32)
        # self.position = create_1d_absolute_sin_cos_embedding(16, 16)
        # self.adj_matrix = np.load(r'E:\learning_projects\few-shot-learning_RSVP\Dataset\THU\eeg_adj_matrix.npy')
        # self.encoder = GraphPositionEncoder(self.adj_matrix, d_model=16)
        # self.latent = nn.Parameter(torch.randn(16, 16))
        # self.temporal_mask = MaskedPrediction(self.n_channels * 4, mask_ratio=0.2)
        # self.temporal_mask = TransformerMaskedPrediction(16, mask_ratio=0.2, nhead=1, num_layers=1)
        # self.projection = nn.Sequential(
        #     nn.Linear(256, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 64),
        # )
        self.projection = nn.Identity()
        # self.lsam = LSAM(self.BasicBlockOutputSize2, 63, 2)
        # self.grl_lambda=1.0

    def classifier_block(self, input_size, n_class):
        Block3 = nn.Sequential(
            # nn.Linear(
            #     input_size,
            #     n_class,
            # ),
            LinearWithConstraint(input_size, n_class, max_norm=1, doWeightNorm=True)
            # nn.Softmax(dim=1),
        )
        return Block3

    def forward(self, x, yd=None, is_train=False):
        x = self.CNN_Block1(x)  # (b, 62*4, 16)
        # chan_ids = torch.arange(0, self.n_channels).to(x.device)
        # chan_position = self.chan_embed(chan_ids.long())
        # x += repeat(chan_position, 'c d -> (c n) d', n=4)
        x, _ = self.Transformer_Block1(x, x, x, use_last_layer=True)  # (b, 62*4, 16)

        xg = self.domain_general(x)  # (b, 256)
        proj = self.projection(xg)
        proj = F.normalize(proj, dim=-1)  # (b, 32)
        # xd = self.domain_domain(x)  # (b, 256)
        logits_g = self.ClassifierBlock_g(xg)  # (b, 2)
        # reversed_features = grad_reverse(xg, self.grl_lambda)
        # logits_d_c = self.lsam(reversed_features)  # (b, 63)
        # logits_d_g = self.ClassifierBlock_d(reversed_features)  # (b, 63)


        return logits_g, xg, proj


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


def load_config(config_path):
    """从 yaml 文件中加载配置"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return Namespace(**config)


def plot_attention(attn_weights, head=0):
    plt.matshow(attn_weights[0, head].detach().cpu().numpy())
    plt.colorbar()
    plt.show()


def remove_domain_info(xc, xd):
    """
    从 xc 中去除 xd 方向上的信息，使 xc 与 xd 正交.

    Args:
        xc: 目标信息特征 (N, 256)
        xd: 域信息特征 (N, 256)

    Returns:
        xc_ortho: 去除域信息的 xc
    """
    xd_norm = F.normalize(xd, p=2, dim=1)  # 归一化 xd，确保投影计算稳定
    proj = (xc * xd_norm).sum(dim=1, keepdim=True) * xd_norm  # 计算投影
    xc_ortho = xc - proj  # 去除投影
    return xc_ortho


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
