import math

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn


def _calculate_feature_dim(feature_extractor: nn.Module, n_channels: int, seq_len: int) -> int:
    with torch.no_grad():
        x = torch.zeros(1, 1, n_channels, seq_len)
        feat = feature_extractor(x)
        return int(feat.reshape(1, -1).shape[1])


class PatchEmbedding(nn.Module):
    def __init__(self, n_channels: int, emb_size: int = 40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 12), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.Conv2d(40, 40, (n_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 38), stride=(1, 7)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e h w -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        return self.projection(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd,bhkd->bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(energy.dtype).min
            energy = energy.masked_fill(~mask, fill_value)

        scaling = math.sqrt(self.emb_size)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhqk,bhkd->bhqd", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int, drop_p: float):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size: int,
        num_heads: int = 10,
        drop_p: float = 0.5,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.5,
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, emb_size: int):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, n_class: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, n_class),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


class Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.depth = 1
        self.n_class = int(config["n_class"])
        self.n_channels = int(config["n_channels"])
        self.seq_len = int(config["fs"])
        self.emb_size = 40

        self.PatchEmbedding = PatchEmbedding(self.n_channels, self.emb_size)
        self.TransformerEncoder = TransformerEncoder(self.depth, self.emb_size)
        feature_dim = _calculate_feature_dim(self._feature_extractor(), self.n_channels, self.seq_len)
        self.ClassificationHead = ClassificationHead(feature_dim, self.n_class)

    def _feature_extractor(self) -> nn.Module:
        return nn.Sequential(self.PatchEmbedding, self.TransformerEncoder)

    def _forward_features(self, x: Tensor) -> Tensor:
        return self._feature_extractor()(x).reshape(x.shape[0], -1)

    def forward(self, x: Tensor, train_stage: int = 2, return_features: bool = False):
        _ = train_stage
        features = self._forward_features(x)
        logits = self.ClassificationHead(features)
        if return_features:
            return logits, {"features": features}
        return logits


if __name__ == "__main__":
    from Models.summary_utils import summarize_model

    summarize_model("EEGConformer", Model)
