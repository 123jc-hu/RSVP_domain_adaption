import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple, List, Dict


# =========================
#    Correlation-based channel selection 
#    from 《Correlation-based channel selection and regularized feature optimization for MI-based BCI》
# =========================
class _CS_ChannelSubsetDataset(Dataset):
    """惰性抽取通道的包装器，避免拷贝整数据到内存。"""
    def __init__(self, base_dataset: Dataset, channels_idx: np.ndarray, keepdim: bool = True):
        self.base = base_dataset
        # 关键修复：将 numpy 索引转为 torch.LongTensor，防止张量索引报错
        self.channels_np = np.asarray(channels_idx, dtype=int)
        self.channels = torch.as_tensor(self.channels_np, dtype=torch.long)
        self.keepdim = keepdim

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        out = self.base[idx]

        # 保证至少有 x, y
        if isinstance(out, (tuple, list)):
            x, y, *rest = out   # 拆出前两个，其余放到 rest
        else:
            raise ValueError(f"Dataset item must be tuple/list, got {type(out)}")

        # ---- 下面是通道筛选逻辑 ----
        if x.ndim == 3:  # (1, C, T) 或 (C, T)
            if x.shape[0] == 1:           # (1, C, T)
                x = x[:, self.channels, :]         # (1, K, T)
            else:                          # (C, T)
                x = x[self.channels, :]            # (K, T)
                if self.keepdim:
                    x = x.unsqueeze(0)             # (1, K, T)
        elif x.ndim == 2:  # (C, T)
            x = x[self.channels, :]
            if self.keepdim:
                x = x.unsqueeze(0)                 # (1, K, T)
        else:
            raise ValueError(f"Unsupported x shape in dataset: {tuple(x.shape)}")

        # ---- 返回时把 rest 拼回去，避免丢失信息 ----
        if rest:
            return (x, y, *rest)
        return (x, y)


class CorrelationBasedChannelSelector:
    """
    Correlation-Based Channel Selection (Cox & Savoy, 2003; Li et al., 2009)
    """
    def __init__(self, dataloader, K: int, device: str = "cpu"):
        self.dataloader = dataloader
        self.K = int(K)
        self.device = device
        self.selected_channels_ = None  # 拟合后填充

    @torch.no_grad()
    def _compute_channel_importance(self):
        channel_votes = []
        n_channels_ref = None

        for batch in self.dataloader:
            x = batch[0]
            # 统一到 CPU，便于 numpy 计算
            if x.device.type != "cpu":
                x = x.cpu()

            # 统一到 (B, C, T)
            if x.ndim == 4:         # (B, 1, C, T)
                x = x.squeeze(1)    # (B, C, T)
            elif x.ndim != 3:
                raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")

            B, C, T = x.shape
            if n_channels_ref is None:
                n_channels_ref = C
            elif C != n_channels_ref:
                raise ValueError(f"Inconsistent channel count across batches: {C} vs {n_channels_ref}")

            if self.K > C:
                raise ValueError(f"K={self.K} exceeds number of channels C={C}")

            x_np = x.numpy()
            for i in range(B):
                trial = x_np[i]  # (C, T)
                R = np.corrcoef(trial)                      # (C, C)
                R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
                np.fill_diagonal(R, 0.0)
                mean_corr = R.mean(axis=1)
                top_idx = np.argsort(mean_corr)[::-1][:self.K]
                channel_votes.extend(top_idx.tolist())

        counts = np.bincount(channel_votes, minlength=n_channels_ref)
        top_channels = np.argsort(counts)[::-1][:self.K]
        return top_channels

    def fit(self):
        self.selected_channels_ = self._compute_channel_importance()
        return self

    def transform(self, dataloader, batch_size=None, shuffle: bool = True):
        if self.selected_channels_ is None:
            raise RuntimeError("You must call fit() before transform().")

        wrapped_ds = _CS_ChannelSubsetDataset(
            base_dataset=dataloader.dataset,
            channels_idx=self.selected_channels_,
            keepdim=True
        )

        # 关键改进：透传性能相关参数
        new_loader = DataLoader(
            wrapped_ds,
            batch_size=batch_size or dataloader.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=getattr(dataloader, "num_workers", 0),
            pin_memory=getattr(dataloader, "pin_memory", False),
            persistent_workers=getattr(dataloader, "persistent_workers", False),
        )
        return new_loader

    def get_new_dataloader(self, batch_size=None, shuffle: bool = True):
        if self.selected_channels_ is None:
            self.fit()
        return self.transform(self.dataloader, batch_size=batch_size, shuffle=shuffle)
    
"""
使用方法：
# 1) 仅用训练集计算所选通道
selector = CorrelationBasedChannelSelector(dataloader=train_loader, K=16, device="cuda").fit()
print("Selected channels:", selector.selected_channels_)

# 2) 分别生成新的 train/val/test dataloader
train_new = selector.transform(train_loader, batch_size=train_loader.batch_size, shuffle=True)
val_new   = selector.transform(val_loader,   batch_size=val_loader.batch_size,   shuffle=False)
test_new  = selector.transform(test_loader,  batch_size=test_loader.batch_size,  shuffle=False)
"""

# =========================
#    SparseEA channel selection 
#    from 《Multi-objective optimization approach for channel selection and cross-subject generalization in RSVP-based BCIs》
# =========================
# -*- coding: utf-8 -*-

# ---------- 小工具 ----------

def _to_BCT(x: torch.Tensor) -> torch.Tensor:
    """兼容 (B,1,C,T) 或 (B,C,T) -> 统一为 (B,C,T)"""
    if x.ndim == 4:
        return x.squeeze(1)
    elif x.ndim == 3:
        return x
    else:
        raise ValueError(f"Unsupported x shape: {tuple(x.shape)}")

def _windows(T: int, fs: int, win_ms: int = 100):
    """按100ms划窗，返回[(s,e), ...]，e为开区间索引"""
    win_len = max(1, int(round(fs * win_ms / 1000.0)))
    K = T // win_len
    idx = [(i * win_len, (i + 1) * win_len) for i in range(K)]
    if K < 1:
        raise ValueError(f"T={T} 太短，无法以 {win_ms}ms 划窗（fs={fs}, win_len={win_len}）")
    return idx, K, win_len

def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    p = max(1, np.sum(y_true == 1))
    n = max(1, np.sum(y_true == 0))
    tpr = tp / p
    tnr = tn / n
    return 0.5 * (tpr + tnr)

def _lda_fit(X: np.ndarray, y: np.ndarray, reg: float = 1e-3):
    """
    Fisher LDA（两类）：返回 w, thresh
    X: (N, d), y in {0,1}
    thresh = 0.5 * w^T (mu1 + mu0)  （等先验）
    """
    y = y.astype(int)
    X0 = X[y == 0]
    X1 = X[y == 1]
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    # pooled covariance
    def _cov(M):
        M = M - M.mean(axis=0, keepdims=True)
        return (M.T @ M) / max(1, (M.shape[0] - 1))
    Sw = _cov(X0) + _cov(X1) + reg * np.eye(X.shape[1], dtype=X.dtype)
    # solve Sw w = (mu1 - mu0)
    w = np.linalg.solve(Sw, (mu1 - mu0))
    thresh = 0.5 * float(w @ (mu1 + mu0))
    return w, thresh

def _lda_decision(X: np.ndarray, w: np.ndarray, thresh: float):
    z = X @ w
    yhat = (z >= thresh).astype(int)
    return z, yhat

def _stack_loader_to_numpy(loader: DataLoader):
    """
    将 DataLoader 全量堆叠到 CPU/NumPy：
      - 兼容 item 形如 (x, y), (x, y, *rest) 或 dict({'x':..., 'y':...})
      - x 支持 (B, 1, C, T) 或 (B, C, T)；输出统一为 X:(N, C, T)
      - y 统一为一维整型数组 (N,)
    """
    xs, ys = [], []

    for item in loader:
        # 解包：支持 tuple/list 或 dict
        if isinstance(item, (tuple, list)):
            if len(item) < 2:
                raise ValueError(f"Dataset item needs at least (x, y), got length {len(item)}")
            x, y = item[0], item[1]
        elif isinstance(item, dict):
            # 常见键名兜底
            x = item.get("x", item.get("inputs", None))
            y = item.get("y", item.get("labels", None))
            if x is None or y is None:
                raise ValueError(f"Dict item must contain 'x' and 'y' (or 'inputs'/'labels'), keys={list(item.keys())}")
        else:
            raise ValueError(f"Unsupported dataset item type: {type(item)}")

        # 搬到 CPU
        if isinstance(x, torch.Tensor) and x.device.type != "cpu":
            x = x.cpu()
        if isinstance(y, torch.Tensor) and y.device.type != "cpu":
            y = y.cpu()

        # 统一 x 形状到 (B, C, T)
        if isinstance(x, torch.Tensor):
            if x.ndim == 4:  # (B,1,C,T)
                x = x.squeeze(1)
            elif x.ndim != 3:
                raise ValueError(f"Unsupported x shape: {tuple(x.shape)} (expect (B,1,C,T) or (B,C,T))")
            xs.append(x)
        else:
            # 若上游返回 numpy，可转成 tensor 再处理
            x = torch.as_tensor(x)
            if x.ndim == 4:
                x = x.squeeze(1)
            elif x.ndim != 3:
                raise ValueError(f"Unsupported x numpy shape: {tuple(x.shape)}")
            xs.append(x)

        # 处理 y -> (B,)
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)
        if y.ndim > 1:
            # 兼容 one-hot 或额外维度的标签
            y = y.view(y.shape[0], -1)
            if y.shape[1] > 1:  # one-hot -> argmax
                y = y.argmax(dim=1)
            else:
                y = y.squeeze(1)
        ys.append(y)

    # 堆叠并转 numpy
    X = torch.cat(xs, dim=0).numpy()                 # (N, C, T)
    y = torch.cat(ys, dim=0).to(torch.int64).numpy() # (N,)
    return X, y


def _apply_channels_numpy(X: np.ndarray, channels: np.ndarray):
    return X[:, channels, :]

# ---------- HDCA 分类器（Algorithm 1） ----------

class HDCAClassifier:
    """
    HDCA (Hierarchical Discriminant Component Analysis) for RSVP/P300
    - 每个100ms窗内做 FLD 得到空间权重 u_k，得到标量 y_k = u_k^T x_k
    - 将 y = [y_1,...,y_K] 用 LDA 得到时间维权重 v 作线性判别

    参数
    ----
    fs : 采样率 (Hz)
    win_ms : 窗宽毫秒（默认100）
    reg_spatial : 每窗FLD的协方差岭化
    reg_temporal: y特征上的LDA岭化
    """
    def __init__(self, fs: int, win_ms: int = 100, reg_spatial: float = 1e-3, reg_temporal: float = 1e-3):
        self.fs = int(fs)
        self.win_ms = int(win_ms)
        self.reg_spatial = float(reg_spatial)
        self.reg_temporal = float(reg_temporal)
        self.us_ = None       # list of u_k, 每个 shape (C,)
        self.v_ = None        # shape (K,)
        self.th_temporal_ = None
        self.win_idx_ = None  # 列表[(s,e),...]
        self.C_ = None
        self.K_ = None

    def _window_mean(self, X: np.ndarray):
        """X:(N,C,T) -> Xm:(N,C,K)  取各窗的平均"""
        N, C, T = X.shape
        idx, K, _ = _windows(T, self.fs, self.win_ms)
        Xm = np.empty((N, C, K), dtype=X.dtype)
        for k, (s, e) in enumerate(idx):
            Xm[:, :, k] = X[:, :, s:e].mean(axis=2)
        return Xm, idx, K

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X:(N,C,T), y:(N,)
        """
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        N, C, T = X.shape
        Xm, idx, K = self._window_mean(X)  # (N,C,K)
        self.us_ = []
        # 窗内FLD：每窗特征 (N,C) -> u_k
        for k in range(K):
            Xk = Xm[:, :, k]  # (N,C)
            u_k, _ = _lda_fit(Xk, y, reg=self.reg_spatial)
            self.us_.append(u_k)
        self.us_ = np.stack(self.us_, axis=0)  # (K,C)

        # 生成 y 特征 (N,K)
        Y = np.empty((N, K), dtype=X.dtype)
        for k in range(K):
            Y[:, k] = Xm[:, :, k] @ self.us_[k]  # y_k

        # 时间维 LDA
        v, th = _lda_fit(Y, y, reg=self.reg_temporal)
        self.v_ = v
        self.th_temporal_ = th
        self.win_idx_ = idx
        self.C_ = C
        self.K_ = K
        return self

    def predict(self, X: np.ndarray):
        """返回 logits(=z), y_pred；X:(N,C,T)"""
        assert self.us_ is not None, "Call fit() first"
        Xm, _, K = self._window_mean(X)  # (N,C,K)
        Y = np.empty((Xm.shape[0], K), dtype=X.dtype)
        for k in range(K):
            Y[:, k] = Xm[:, :, k] @ self.us_[k]
        z, yhat = _lda_decision(Y, self.v_, self.th_temporal_)
        return z, yhat

    def score_balanced_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        z, yhat = self.predict(X)
        return _balanced_accuracy(y, yhat)

# ---------- SparseEA 通道选择（Algorithm 2 框架的工程化实现） ----------

class SparseEAChannelSelector:
    """
    多目标 SparseEA 通道选择
    目标: minimize f1 = 1-BA, minimize f2 = |S|（通道数）
    评价器: HDCAClassifier（可替换）
    """
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fs: int,
        target_K: int | None = None,
        population_size: int = 60,
        generations: int = 40,
        win_ms: int = 100,
        reg_spatial: float = 1e-3,
        reg_temporal: float = 1e-3,
        seed: int = 42,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fs = int(fs)
        self.win_ms = int(win_ms)
        self.reg_spatial = float(reg_spatial)
        self.reg_temporal = float(reg_temporal)
        self.pop_size = int(population_size)
        self.generations = int(generations)
        self.rng = np.random.default_rng(seed)
        self.target_K = int(target_K) if target_K is not None else None

        # 缓存数据（CPU, numpy）
        self.Xtr_, self.ytr_ = _stack_loader_to_numpy(train_loader)  # (N,C,T)
        self.Xva_, self.yva_ = _stack_loader_to_numpy(val_loader)
        self.C_ = self.Xtr_.shape[1]

        self.population_ = None      # list of masks (C,)
        self.objs_ = None            # array (P,2): [1-BA, |S|]
        self.selected_channels_ = None

        # 单通道精度作为引导（论文初始化思想）
        self.channel_scores_ = self._single_channel_scores()

    def _single_channel_scores(self):
        scores = np.zeros(self.C_)
        for c in range(self.C_):
            hdca = HDCAClassifier(fs=self.fs, win_ms=self.win_ms,
                                  reg_spatial=self.reg_spatial, reg_temporal=self.reg_temporal)
            hdca.fit(self.Xtr_[:, [c], :], self.ytr_)
            scores[c] = hdca.score_balanced_accuracy(self.Xva_[:, [c], :], self.yva_)
        return scores  # BA per channel

    def _evaluate_mask(self, mask: np.ndarray) -> tuple[float, float]:
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            # f1=1-BA -> 1.0(最差)，f2=与K的偏差（若无K则子集大小）
            if self.target_K is None:
                return 1.0, 0.0
            else:
                return 1.0, float(self.target_K)
        hdca = HDCAClassifier(fs=self.fs, win_ms=self.win_ms,
                            reg_spatial=self.reg_spatial, reg_temporal=self.reg_temporal)
        hdca.fit(_apply_channels_numpy(self.Xtr_, idx), self.ytr_)
        ba = hdca.score_balanced_accuracy(_apply_channels_numpy(self.Xva_, idx), self.yva_)
        f1 = 1.0 - float(ba)
        k  = float(len(idx))

        f2 = k
        return f1, f2

    # --- NSGA-II 相关 ---

    def _init_population(self):
        pop = []
        C = self.C_
        order = np.argsort(self.channel_scores_)[::-1]

        seeds = [1, 2, 4, 8, max(1, C // 4), max(1, C // 2)]
        if self.target_K is not None:
            seeds.extend([
                max(1, self.target_K - 4),
                self.target_K,
                min(C, self.target_K + 4),
            ])
        seeds = [min(C, s) for s in seeds]

        added = set()
        for s in seeds:
            m = np.zeros(C, dtype=np.uint8)
            m[order[:s]] = 1
            t = tuple(m.tolist())
            if t not in added:
                pop.append(m); added.add(t)

        rng = self.rng
        while len(pop) < self.pop_size:
            # 若有 target_K，随机 k 在 [max(1,K-8), min(C,K+8)] 内均匀采样，有助贴近K
            if self.target_K is not None:
                lo = max(1, self.target_K - 8)
                hi = min(C, self.target_K + 8)
                k = int(rng.integers(low=lo, high=hi+1))
            else:
                k = int(rng.integers(low=1, high=C+1))
            m = np.zeros(C, dtype=np.uint8)
            idx = rng.choice(C, size=k, replace=False)
            m[idx] = 1
            t = tuple(m.tolist())
            if t not in added:
                pop.append(m); added.add(t)
        return pop

    @staticmethod
    def _nondominated_sort(F):
        """F:(P,2) -> ranks(list of fronts as lists of indices)"""
        P = F.shape[0]
        dominates = [set() for _ in range(P)]
        dominated_count = np.zeros(P, dtype=int)
        fronts = [[]]
        for p in range(P):
            for q in range(P):
                if p == q: continue
                if (F[p][0] <= F[q][0] and F[p][1] <= F[q][1]) and (F[p][0] < F[q][0] or F[p][1] < F[q][1]):
                    dominates[p].add(q)
                elif (F[q][0] <= F[p][0] and F[q][1] <= F[p][1]) and (F[q][0] < F[p][0] or F[q][1] < F[p][1]):
                    dominated_count[p] += 1
            if dominated_count[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            Q = []
            for p in fronts[i]:
                for q in dominates[p]:
                    dominated_count[q] -= 1
                    if dominated_count[q] == 0:
                        Q.append(q)
            i += 1
            fronts.append(Q)
        if not fronts[-1]:
            fronts.pop()
        return fronts

    @staticmethod
    def _crowding_distance(F, idxs):
        """F:(P,2), idxs:list -> distances np.array"""
        if len(idxs) == 0:
            return np.array([])
        if len(idxs) == 1:
            return np.array([np.inf])
        subF = F[idxs]
        d = np.zeros(len(idxs))
        for m in range(2):
            order = np.argsort(subF[:, m])
            d[order[0]] = d[order[-1]] = np.inf
            minv, maxv = subF[order[0], m], subF[order[-1], m]
            rng = max(1e-12, maxv - minv)
            for j in range(1, len(idxs) - 1):
                d[order[j]] += (subF[order[j+1], m] - subF[order[j-1], m]) / rng
        return d

    def _tournament(self, pop, F, fronts):
        """二元锦标赛（先比较前沿等级，再比较拥挤距离）"""
        i, j = self.rng.integers(0, len(pop), size=2)
        # 找到 i,j 分别所在的前沿和在该前沿的拥挤距离
        def key(a):
            # 返回 (front_rank, -crowd) 便于最小化
            for r, fr in enumerate(fronts):
                if a in fr:
                    # crowding distance 需要在该前沿内计算
                    cd = self._crowding_distance(F, fr)
                    crowd = cd[fr.index(a)]
                    return (r, -crowd)
            return (np.inf, -np.inf)
        return i if key(i) < key(j) else j

    def _crossover_mutation(self, p1: np.ndarray, p2: np.ndarray, mut_prob: float = 0.2):
        """均匀交叉 + 稀疏变异（以相同概率翻转一个0或1）"""
        C = len(p1)
        mask = self.rng.integers(0, 2, size=C, dtype=np.uint8)
        child = (p1 & mask) | (p2 & (1 - mask))
        if self.rng.random() < mut_prob:
            ones = np.flatnonzero(child == 1)
            zeros = np.flatnonzero(child == 0)
            if len(ones) == 0 and len(zeros) == 0:
                pass
            elif len(ones) == 0:
                # 只能从0->1
                j = self.rng.choice(zeros)
                child[j] = 1
            elif len(zeros) == 0:
                # 只能从1->0
                j = self.rng.choice(ones)
                child[j] = 0
            else:
                if self.rng.random() < 0.5:
                    j = self.rng.choice(zeros)
                    child[j] = 1
                else:
                    j = self.rng.choice(ones)
                    child[j] = 0
        return child

    def fit(self):
        # 初始化种群
        pop = self._init_population()
        F = np.array([self._evaluate_mask(m)[0:2] for m in pop], dtype=float)  # (P,2)

        for gen in range(self.generations):
            fronts = self._nondominated_sort(F)
            # 产生子代
            children = []
            while len(children) < self.pop_size:
                # 选父母（按前沿+拥挤距离）
                fronts = self._nondominated_sort(F)
                p_idx = self._tournament(pop, F, fronts)
                q_idx = self._tournament(pop, F, fronts)
                child = self._crossover_mutation(pop[p_idx], pop[q_idx], mut_prob=0.3)
                children.append(child)

            # 合并 & 选择下一代
            union = pop + children
            F_union = np.array([self._evaluate_mask(m)[0:2] for m in union], dtype=float)
            fronts = self._nondominated_sort(F_union)

            new_pop = []
            new_F = []
            for fr in fronts:
                if len(new_pop) + len(fr) <= self.pop_size:
                    new_pop.extend([union[i] for i in fr])
                    new_F.extend([F_union[i] for i in fr])
                else:
                    # 需要在该前沿里按拥挤距离截断
                    cd = self._crowding_distance(F_union, fr)
                    order = np.argsort(-cd)  # 先选拥挤距离大的
                    for j in order:
                        if len(new_pop) < self.pop_size:
                            new_pop.append(union[fr[j]])
                            new_F.append(F_union[fr[j]])
                        else:
                            break
                    break
            pop, F = new_pop, np.array(new_F)

            # --- 最终选择解：不再“补真实通道”，只做 zero-padding 到 Kmax ---
            self.population_ = pop
            self.objs_ = F  # 每行: [f1=1-BA, f2=|S| or other]

            ks = np.array([int(m.sum()) for m in pop])  # 每个体的通道数
            bas = 1.0 - F[:, 0]  # BA = 1 - f1

            if self.target_K is not None:
                Kmax = self.target_K

                # 1) 优先从 k<=Kmax 的候选里选 BA 最大；并列选更少通道
                feas = np.where(ks <= Kmax)[0]
                if len(feas) > 0:
                    # lexsort: 先按 -BA 排序，再按 k 排序（k越小越优）
                    order = feas[np.lexsort((ks[feas], -bas[feas]))]
                    best_i = order[0]
                else:
                    # 极端兜底：如果种群里全都 k>Kmax，选 BA 最大的，然后后面截断到 Kmax
                    best_i = int(np.argmax(bas))

                best_mask = pop[best_i].astype(bool)
                selected = np.flatnonzero(best_mask)

                # 2) 如果 selected > Kmax（理论上只会发生在兜底分支），截断到 Kmax
                if selected.size > Kmax:
                    # 在已选中里优先保留单通道分数高的（你原本就有 channel_scores_）
                    keep_order = np.argsort(self.channel_scores_[selected])[::-1]
                    selected = selected[keep_order[:Kmax]]

                # 3) 注意：绝不再补真实通道！
                selected = np.unique(selected)  # 升序 + 去重
                self.Kmax_ = Kmax
            else:
                # 未指定 Kmax：选 BA 最大，若并列通道更少者优先
                best_i = np.lexsort((ks, -bas))[0]
                best_mask = pop[best_i].astype(bool)
                selected = np.flatnonzero(best_mask)
                self.Kmax_ = None

            self.selected_channels_ = selected.astype(int)
            return self

    def transform(self, loader: DataLoader, batch_size: int | None = None, shuffle: bool = False,
                  return_mask: bool = False) -> DataLoader:
        if self.selected_channels_ is None:
            raise RuntimeError("Call fit() first.")

        # 如果设置了 target_K（作为 Kmax 槽位数），则输出固定 Kmax 并补零
        if getattr(self, "Kmax_", None) is not None:
            ds = _CS_PaddedChannelDataset(
                loader.dataset,
                channels_idx=self.selected_channels_,
                Kmax=self.Kmax_,
                return_mask=return_mask,
            )
            return DataLoader(
                ds,
                batch_size=batch_size or loader.batch_size,
                shuffle=shuffle,
                drop_last=False,
                num_workers=getattr(loader, "num_workers", 0),
                pin_memory=getattr(loader, "pin_memory", False),
                persistent_workers=getattr(loader, "persistent_workers", False),
            )

        # 否则按原逻辑：只取子集（输出维度是 k 而非固定 K）
        return _ChannelSubsetDataLoader.from_base_loader(loader, self.selected_channels_, batch_size, shuffle)


class _ChannelSubsetDataLoader:
    @staticmethod
    def from_base_loader(base_loader: DataLoader, channels_idx: np.ndarray, batch_size=None, shuffle=False) -> DataLoader:
        ds = _CS_ChannelSubsetDataset(base_loader.dataset, channels_idx, keepdim=True)
        return DataLoader(
            ds,
            batch_size=batch_size or base_loader.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=getattr(base_loader, "num_workers", 0),
            pin_memory=getattr(base_loader, "pin_memory", False),
            persistent_workers=getattr(base_loader, "persistent_workers", False),
        )

class _CS_PaddedChannelDataset(torch.utils.data.Dataset):
    """
    返回固定 Kmax 槽位的输入：
        x_pad: [Kmax, T]  (前 k 个是真实通道，后面全 0)
        y
        (可选) slot_mask: [Kmax]  1=真实通道槽位, 0=零通道槽位
    """
    def __init__(self, base_ds, channels_idx: np.ndarray, Kmax: int, return_mask: bool = False):
        self.base_ds = base_ds
        self.channels_idx = np.asarray(channels_idx, dtype=int)
        self.Kmax = int(Kmax)
        self.return_mask = bool(return_mask)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, i):
        item = self.base_ds[i]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            x, y = item[0], item[1]
        else:
            raise ValueError("Base dataset item should be (x, y) or (x, y, ...).")

        # x: [C, T]
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x)
        else:
            x_t = x

        if x_t.dim() != 2:
            raise ValueError(f"Expected x with shape [C,T], got {tuple(x_t.shape)}")

        device = x_t.device
        dtype = x_t.dtype
        T = x_t.shape[-1]

        k = int(len(self.channels_idx))
        Kmax = self.Kmax

        x_pad = torch.zeros((Kmax, T), device=device, dtype=dtype)
        if k > 0:
            x_pad[:k] = x_t[self.channels_idx]

        if not self.return_mask:
            return x_pad, y

        slot_mask = torch.zeros((Kmax,), device=device, dtype=torch.float32)
        if k > 0:
            slot_mask[:k] = 1.0
        return x_pad, y, slot_mask

"""
使用方法：
# 1) 用 train/val 来进化选择通道（target_K 可设为你想要的 K）
selector = SparseEAChannelSelector(
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    fs=config["fs"],          # 必填
    target_K=16,              # 你的目标通道数；也可设为 None 得到Pareto解里BA最高者
    population_size=60,
    generations=40,
    win_ms=100,
).fit()

print("Selected channels:", selector.selected_channels_)  # numpy array

# 2) 将相同通道集应用到 train/val/test
train_new = selector.transform(train_dataloader, batch_size=train_dataloader.batch_size, shuffle=True)
val_new   = selector.transform(val_dataloader,   batch_size=val_dataloader.batch_size,   shuffle=False)
test_new  = selector.transform(test_dataloader,  batch_size=test_dataloader.batch_size,  shuffle=False)
"""

# =========================
#    EEG channel selection based on frequency domain weighting 
#    from 《Depression detection based on the temporal-spatial-frequency feature fusion of EEG》
# =========================
# -*- coding: utf-8 -*-

# ===================== 基础工具 =====================

def _normalize_cov(trial: np.ndarray) -> np.ndarray:
    """
    trial: (C,T), 返回 归一化协方差 Cx / trace(Cx)
    """
    Cx = trial @ trial.T
    tr = np.trace(Cx)
    if tr <= 1e-12:
        return np.eye(Cx.shape[0], dtype=trial.dtype)
    return Cx / tr

def _mean_cov(X: np.ndarray) -> np.ndarray:
    """
    X: (N,C,T) -> 归一化协方差的均值 (C,C)
    """
    N, C, T = X.shape
    acc = np.zeros((C, C), dtype=X.dtype)
    for i in range(N):
        acc += _normalize_cov(X[i])
    acc /= max(1, N)
    # 对称化
    return 0.5 * (acc + acc.T)

# ===================== 纯 PyTorch FIR 带通（可选） =====================

def _design_fir_bandpass(fs: int, f_lo: float, f_hi: float, numtaps: int = 129) -> torch.Tensor:
    """
    窗函数法设计对称FIR (Hamming)。返回 1D kernel (numtaps,)
    """
    assert 0 < f_lo < f_hi < fs / 2.0, "频带必须在(0, fs/2)内"
    n = torch.arange(numtaps, dtype=torch.float32)
    m = (numtaps - 1) / 2.0
    # 避免除零：sinc(x) = sin(pi x)/(pi x)
    def sinc(x): 
        y = torch.where(x == 0, torch.ones_like(x), torch.sin(np.pi * x) / (np.pi * x))
        return y
    # 理想带通 = 高通(f_lo) + 低通(f_hi) - 全通
    h_lp = 2 * (f_hi / fs) * sinc(2 * (f_hi / fs) * (n - m))
    h_hp = -2 * (f_lo / fs) * sinc(2 * (f_lo / fs) * (n - m))
    h_id = h_lp + h_hp
    # Hamming 窗
    w = 0.54 - 0.46 * torch.cos(2 * np.pi * n / (numtaps - 1))
    h = h_id * w
    # 归一化 DC/能量
    h = h / torch.sum(h)
    return h.float()

def _apply_bandpass_torch(X: np.ndarray, fs: int, band: tuple[float, float], numtaps: int = 129) -> np.ndarray:
    """
    X: (N,C,T) numpy -> 经过带通滤波后的 numpy (N,C,T)
    逐通道逐 trial 用 Conv1d 实现，padding='same'。
    """
    N, C, T = X.shape
    kernel = _design_fir_bandpass(fs, band[0], band[1], numtaps=numtaps)  # (K,)
    k = kernel.view(1, 1, -1)  # (out_ch=1, in_ch=1, K)
    pad = (k.shape[-1] - 1) // 2

    Xt = torch.from_numpy(X).float()  # (N,C,T)
    Xt = Xt.reshape(N * C, 1, T)      # 合并 batch 和通道
    k = k.to(Xt.device)

    Y = torch.nn.functional.conv1d(torch.nn.functional.pad(Xt, (pad, pad), mode="reflect"), k, groups=1)
    Y = Y.reshape(N, C, T - 0)  # 长度保持一致（reflect+same设计）
    return Y.numpy()

# ===================== CSP（白化解法，无需 SciPy） =====================

def _csp_filters(C1: np.ndarray, C2: np.ndarray, n_components: int | None = None):
    """
    输入两类的平均协方差矩阵 C1, C2 (C,C)，返回 CSP 投影矩阵 W (C,C) 及特征值向量 lambda
    白化法：C=C1+C2 -> EΛE^T -> P=Λ^{-1/2}E^T -> S1=P C1 P^T -> S1=UΣU^T
    W = E Λ^{-1/2} U
    """
    C = C1 + C2
    # 对称本征分解
    evals, E = np.linalg.eigh(C)
    # 数值稳定
    evals = np.clip(evals, 1e-12, None)
    P = (E @ np.diag(evals ** -0.5) @ E.T)  # 白化
    S1 = P @ C1 @ P.T
    s, U = np.linalg.eigh(S1)
    # s 从小到大，CSP 常取两端分量
    W = E @ np.diag(evals ** -0.5) @ U
    # 返回所有投影分量（列为滤波器），与 s 对应
    order = np.argsort(s)[::-1]  # 也可以保留未翻转，下面会用到两端
    W = W[:, order]
    s = s[order]
    if n_components is not None:
        W = W[:, :n_components]
        s = s[:n_components]
    return W, s

def _csp_channel_weights_w1w2(W_full: np.ndarray) -> np.ndarray:
    """
    取“最显著”的两端空间滤波器：第一列(最大特征值)作为 w1，最后一列(最小特征值)作为 w2，
    然后按 a_i = sqrt(w1_i^2 + w2_i^2) 计算通道权重（式(3)）。
    """
    w1 = W_full[:, 0]           # 对应一类最有判别力
    w2 = W_full[:, -1]          # 另一端
    a = np.sqrt(w1 ** 2 + w2 ** 2)  # (C,)
    # 归一化为 b_i（式(4)）
    s = np.sum(a)
    if s <= 1e-12:
        b = np.ones_like(a) / len(a)
    else:
        b = a / s
    return b  # (C,)


def _wrap_loader(base_loader: DataLoader, channels_idx: np.ndarray, batch_size=None, shuffle=False) -> DataLoader:
    ds = _CS_ChannelSubsetDataset(base_loader.dataset, channels_idx, keepdim=True)
    return DataLoader(
        ds,
        batch_size=batch_size or base_loader.batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=getattr(base_loader, "num_workers", 0),
        pin_memory=getattr(base_loader, "pin_memory", False),
        persistent_workers=getattr(base_loader, "persistent_workers", False),
    )

# ===================== 频域加权 CSP 通道选择 =====================

class FrequencyWeightedCSPChannelSelector:
    """
    EEG 通道选择（基于频域加权的 CSP）
    - 对每个频带：用 CSP 求两端空间滤波器 w1, w2
    - 按 a_i = sqrt(w1_i^2 + w2_i^2)，b_i = a_i / sum(a_i) 得到通道权重
    - 多频带时对 b_i 做聚合（mean/max/median），选出 Top-K

    参数
    ----
    train_loader : 仅用于估计通道权重（建议 drop_last=True）
    fs           : 采样率 (Hz)
    bands        : List[Tuple[float, float]]，例如 [(1,4),(4,8),(8,13),(13,30)]；None=不分频带
    K            : 选择通道数
    aggregate    : 多频带聚合策略: "mean" | "max" | "median"
    fir_taps     : FIR 滤波器 taps 数（奇数，默认129）
    use_filter   : 是否开启带通滤波（True=对每个频带做 FIR；False=直接在原始宽带做 CSP；若 bands 多个但 use_filter=False，则仅按宽带算一次）
    """

    def __init__(
        self,
        train_loader: DataLoader,
        fs: int,
        K: int,
        bands: list[tuple[float, float]] | None = None,
        aggregate: str = "mean",
        fir_taps: int = 129,
        use_filter: bool = True,
    ):
        self.train_loader = train_loader
        self.fs = int(fs)
        self.K = int(K)
        self.bands = bands if bands is not None and len(bands) > 0 else None
        self.aggregate = aggregate
        self.fir_taps = int(fir_taps)
        self.use_filter = bool(use_filter)

        self.selected_channels_ = None         # np.ndarray (K,)
        self.band_weights_ = {}                # dict: band-> b_i (C,)
        self.global_weights_ = None            # 聚合后的 b_i (C,)

        # 缓存训练数据（numpy）
        self.Xtr_, self.ytr_ = _stack_loader_to_numpy(train_loader)
        self.C_ = self.Xtr_.shape[1]

    def _compute_band_weights(self, X: np.ndarray, y: np.ndarray, band: tuple[float, float] | None):
        """
        计算一个频带的通道权重 b_i；若 band=None，则直接宽带 CSP。
        """
        if band is not None and self.use_filter:
            Xb = _apply_bandpass_torch(X, self.fs, band, numtaps=self.fir_taps)  # (N,C,T)
        else:
            Xb = X

        # 计算两类平均协方差
        cls0 = Xb[y == 0]
        cls1 = Xb[y == 1]
        if len(cls0) == 0 or len(cls1) == 0:
            raise ValueError("需要两类样本以计算 CSP（y=0 和 y=1 均需存在）。")
        C0 = _mean_cov(cls0)  # (C,C)
        C1 = _mean_cov(cls1)  # (C,C)

        # CSP 滤波器
        W, s = _csp_filters(C1, C0)  # 注意：与文中 C2^{-1}C1 一致（这里把C1视为“第一类”）
        # 取两端滤波器，按式(3)(4)计算通道权重
        b = _csp_channel_weights_w1w2(W)  # (C,)
        return b

    def fit(self):
        # 单频带 / 多频带
        if self.bands is None or not self.use_filter:
            # 宽带一次
            b = self._compute_band_weights(self.Xtr_, self.ytr_, band=None)
            self.band_weights_ = {("broadband",): b}
            b_global = b
        else:
            # 对每个频带求 b_i
            per_band = []
            for band in self.bands:
                b = self._compute_band_weights(self.Xtr_, self.ytr_, band=band)
                self.band_weights_[tuple(band)] = b
                per_band.append(b[None, :])  # (1,C)
            B = np.concatenate(per_band, axis=0)  # (nbands, C)

            if self.aggregate.lower() == "mean":
                b_global = B.mean(axis=0)
            elif self.aggregate.lower() == "max":
                b_global = B.max(axis=0)
            elif self.aggregate.lower() == "median":
                b_global = np.median(B, axis=0)
            else:
                raise ValueError(f"Unknown aggregate: {self.aggregate}")

        # 最终归一化 + 选 Top-K
        s = np.sum(b_global)
        if s > 1e-12:
            b_global = b_global / s
        self.global_weights_ = b_global  # (C,)

        order = np.argsort(self.global_weights_)[::-1]
        if self.K > len(order):
            raise ValueError(f"K={self.K} exceeds number of channels C={len(order)}")
        self.selected_channels_ = order[:self.K]
        return self

    def transform(self, loader: DataLoader, batch_size: int | None = None, shuffle: bool = False) -> DataLoader:
        if self.selected_channels_ is None:
            raise RuntimeError("Call fit() first.")
        return _wrap_loader(loader, self.selected_channels_, batch_size=batch_size, shuffle=shuffle)

"""
使用方法：
# 例：按 (theta, alpha, beta) 三个频带求权重并取 Top-K=16
bands = [(4,8), (8,13), (13,30)]  # 自行按数据频谱定义
selector = FrequencyWeightedCSPChannelSelector(
    train_loader=train_dataloader,
    fs=config["fs"],
    K=16,
    bands=bands,
    aggregate="mean",   # "mean"/"max"/"median"
    fir_taps=129,
    use_filter=True,    # 若你预先已分频，可设 False 直接宽带
).fit()

print("Selected channels:", selector.selected_channels_)      # numpy array (K,)
print("Global weights shape:", selector.global_weights_.shape)  # (C,)

# 将相同通道集应用到 train/val/test
train_new = selector.transform(train_dataloader, batch_size=train_dataloader.batch_size, shuffle=True)
val_new   = selector.transform(val_dataloader,   batch_size=val_dataloader.batch_size,   shuffle=False)
test_new  = selector.transform(test_dataloader,  batch_size=test_dataloader.batch_size,  shuffle=False)
"""

# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from typing import Optional, Tuple, List, Dict, Any


# ============================================================
# Utils: stack DataLoader -> numpy (N,C,T), apply channel subset
# ============================================================

def _stack_loader_to_numpy(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stack a DataLoader (or its dataset) into numpy arrays:
        X: (N,C,T) float32
        y: (N,) int64
    Accepts x shapes:
        - [B,1,C,T] or [B,C,T]
        - single sample: [1,C,T] or [C,T]
    """
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("DataLoader must yield (x, y) or (x, y, ...).")
        x, y = batch[0], batch[1]

        if torch.is_tensor(x):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)

        if torch.is_tensor(y):
            y_np = y.detach().cpu().numpy()
        else:
            y_np = np.asarray(y)

        # normalize x shape to [B,C,T]
        if x_np.ndim == 4:      # [B,1,C,T] or [B,?,C,T]
            x_np = x_np[:, 0, :, :]  # assume dim1 is 1
        elif x_np.ndim == 3:    # [B,C,T]
            pass
        elif x_np.ndim == 2:    # [C,T] (single)
            x_np = x_np[None, :, :]
        else:
            raise ValueError(f"Unsupported x shape: {x_np.shape}")

        # normalize y to [B]
        y_np = y_np.reshape(-1)

        xs.append(x_np.astype(np.float32, copy=False))
        ys.append(y_np.astype(np.int64, copy=False))

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


def _apply_channels_numpy(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """X: (N,C,T), idx: (k,) -> (N,k,T)"""
    idx = np.asarray(idx, dtype=int)
    return X[:, idx, :]


def _stratified_kfold_indices(y: np.ndarray, n_splits: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Lightweight Stratified KFold: returns list of valid indices arrays."""
    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    per_class = {c: np.where(y == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(per_class[c])

    folds = [[] for _ in range(n_splits)]
    for c in classes:
        idx = per_class[c]
        for i, ii in enumerate(idx):
            folds[i % n_splits].append(int(ii))

    return [np.array(sorted(f), dtype=int) for f in folds]


# ============================================================
# Fisher initialization score (paper-style): logvar segments + Fisher
# ============================================================

def _fisher_scores_logvar_segments(
    X: np.ndarray, y: np.ndarray, fs: int, win_ms: int, eps: float = 1e-12
) -> np.ndarray:
    """
    Paper-like Fisher init:
      P_{ch,t} = log(var(x_{ch,t})) on segments
      phi = (m1-m2)^2 / (var1+var2)
    score(ch) = mean_t phi(ch,t)
    """
    X = np.asarray(X)  # (N,C,T)
    y = np.asarray(y).astype(int)
    N, C, T = X.shape

    seg_len = max(1, int(round(win_ms / 1000.0 * fs)))
    n_seg = max(1, T // seg_len)

    P = np.zeros((N, C, n_seg), dtype=np.float64)
    for s in range(n_seg):
        a = s * seg_len
        b = a + seg_len
        xseg = X[:, :, a:b]
        var = np.var(xseg, axis=-1, ddof=0) + eps
        P[:, :, s] = np.log(var)

    cls = np.unique(y)
    if len(cls) != 2:
        # optional OVR Fisher (not usually needed for RSVP)
        scores = np.zeros(C, dtype=np.float64)
        for ch in range(C):
            best = 0.0
            for k in cls:
                yk = (y == k).astype(int)
                idx1 = np.where(yk == 1)[0]
                idx0 = np.where(yk == 0)[0]
                if len(idx1) < 2 or len(idx0) < 2:
                    continue
                phi_seg = []
                for s in range(n_seg):
                    p1 = P[idx1, ch, s]
                    p0 = P[idx0, ch, s]
                    m1, m0 = p1.mean(), p0.mean()
                    v1, v0 = p1.var(ddof=0), p0.var(ddof=0)
                    phi_seg.append(((m1 - m0) ** 2) / (v1 + v0 + eps))
                best = max(best, float(np.mean(phi_seg)))
            scores[ch] = best
        return scores

    c0, c1 = cls[0], cls[1]
    idx0 = np.where(y == c0)[0]
    idx1 = np.where(y == c1)[0]

    scores = np.zeros(C, dtype=np.float64)
    for ch in range(C):
        phi_seg = []
        for s in range(n_seg):
            p0 = P[idx0, ch, s]
            p1 = P[idx1, ch, s]
            m0, m1 = p0.mean(), p1.mean()
            v0, v1 = p0.var(ddof=0), p1.var(ddof=0)
            phi_seg.append(((m1 - m0) ** 2) / (v0 + v1 + eps))
        scores[ch] = float(np.mean(phi_seg))
    return scores


# ============================================================
# Evaluator: CSP + FLDA (paper-style structure)
# ============================================================

class CSPFLDAEvaluator:
    """
    ABMOHS paper-style evaluator:
      - CSP filters from training data
      - Features: log-variance of projected signals (normalized)
      - Classifier: FLDA (2-class)
      - Two modes:
          * cv_accuracy(Xtr, ytr, idx_ch): stratified K-fold CV acc (default 10-fold)
          * train_val_accuracy(Xtr, ytr, Xva, yva, idx_ch): fit on train, eval on val (faster)
    Notes:
      - Designed for binary classification (RSVP target vs non-target).
      - precompute_cov=True can speed up repeated evaluations (wrapper search),
        but uses memory ~ N*C*C*4 bytes (float32).
    """
    def __init__(
        self,
        n_filt_pairs: int = 2,     # m pairs -> 2m filters -> 2m-dim feature
        cv_folds: int = 10,
        cov_reg: float = 1e-6,
        lda_reg: float = 1e-6,
        seed: int = 42,
        precompute_cov: bool = False,
    ):
        self.m = int(n_filt_pairs)
        self.K = int(cv_folds)
        self.cov_reg = float(cov_reg)
        self.lda_reg = float(lda_reg)
        self.rng = np.random.default_rng(seed)
        self.precompute_cov = bool(precompute_cov)

        self._covs_full: Optional[np.ndarray] = None  # (N,C,C) float32
        self._X_ref: Optional[np.ndarray] = None
        self._y_ref: Optional[np.ndarray] = None

    @staticmethod
    def _trial_cov(x_ct: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Normalized covariance: cov / trace(cov)."""
        x = x_ct - x_ct.mean(axis=-1, keepdims=True)
        cov = (x @ x.T) / max(1, x.shape[-1])
        tr = np.trace(cov)
        if tr < eps:
            return cov
        return cov / tr

    def bind_train_data(self, X: np.ndarray, y: np.ndarray):
        """Optional: precompute cov matrices for full channels (train set)."""
        self._X_ref = X
        self._y_ref = y
        if not self.precompute_cov:
            self._covs_full = None
            return
        N, C, T = X.shape
        covs = np.zeros((N, C, C), dtype=np.float32)
        for i in range(N):
            covs[i] = self._trial_cov(X[i]).astype(np.float32)
        self._covs_full = covs

    def _stratified_kfold(self, y: np.ndarray) -> List[np.ndarray]:
        return _stratified_kfold_indices(y, n_splits=self.K, rng=self.rng)

    def _get_covs_subset(self, X: np.ndarray, idx_ch: np.ndarray) -> np.ndarray:
        idx_ch = np.asarray(idx_ch, dtype=int)
        if self._covs_full is not None and self._X_ref is X:
            # covs_full: (N,C,C) -> (N,k,k)
            covs = self._covs_full[:, idx_ch][:, :, idx_ch]
            return covs.astype(np.float64, copy=False)

        N = X.shape[0]
        k = len(idx_ch)
        covs = np.zeros((N, k, k), dtype=np.float64)
        for i in range(N):
            covs[i] = self._trial_cov(X[i, idx_ch, :])
        return covs

    def _fit_csp_filters(self, covs: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return CSP filters W: (2m, k)."""
        y = np.asarray(y).astype(int)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSPFLDAEvaluator supports binary classification only.")
        c0, c1 = classes[0], classes[1]

        R0 = covs[y == c0].mean(axis=0)
        R1 = covs[y == c1].mean(axis=0)

        k = R0.shape[0]
        R = R0 + R1 + self.cov_reg * np.eye(k)

        d, E = np.linalg.eigh(R)
        d = np.maximum(d, 1e-12)
        P = E @ np.diag(1.0 / np.sqrt(d)) @ E.T  # whitening

        S1 = P @ R1 @ P.T
        lam, B = np.linalg.eigh(S1)
        order = np.argsort(lam)[::-1]
        B = B[:, order]

        W_full = (B.T @ P)  # (k,k)
        m = self.m
        W = np.concatenate([W_full[:m], W_full[-m:]], axis=0)  # (2m,k)
        return W

    @staticmethod
    def _csp_features(W: np.ndarray, X: np.ndarray, idx_ch: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Return CSP log-variance features: (N,2m)."""
        Xs = X[:, idx_ch, :]  # (N,k,T)
        Z = np.einsum("fk,nkt->nft", W, Xs)  # (N,2m,T)
        var = np.var(Z, axis=-1, ddof=0) + eps
        var_norm = var / (var.sum(axis=1, keepdims=True) + eps)
        feats = np.log(var_norm + eps)
        return feats

    def _fit_flda(self, Xf: np.ndarray, y: np.ndarray):
        """2-class Fisher LDA."""
        y = np.asarray(y).astype(int)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("FLDA supports binary classification only.")
        c0, c1 = classes[0], classes[1]
        X0, X1 = Xf[y == c0], Xf[y == c1]
        if len(X0) < 2 or len(X1) < 2:
            # too few samples
            m0 = X0.mean(axis=0) if len(X0) > 0 else np.zeros(Xf.shape[1])
            m1 = X1.mean(axis=0) if len(X1) > 0 else np.zeros(Xf.shape[1])
            Sw = np.eye(Xf.shape[1]) * self.lda_reg
        else:
            m0, m1 = X0.mean(axis=0), X1.mean(axis=0)
            S0 = np.cov(X0, rowvar=False, bias=True)
            S1 = np.cov(X1, rowvar=False, bias=True)
            if S0.ndim == 0:
                S0 = np.array([[float(S0)]])
            if S1.ndim == 0:
                S1 = np.array([[float(S1)]])
            Sw = S0 + S1 + self.lda_reg * np.eye(Xf.shape[1])

        w = np.linalg.solve(Sw, (m1 - m0))
        b = -0.5 * float(w @ (m1 + m0))
        return (c0, c1, w, b)

    @staticmethod
    def _predict_flda(model, Xf: np.ndarray) -> np.ndarray:
        c0, c1, w, b = model
        s = Xf @ w + b
        return np.where(s > 0, c1, c0)

    def cv_accuracy(self, X: np.ndarray, y: np.ndarray, idx_ch: np.ndarray) -> float:
        """Stratified K-fold CV accuracy on X,y."""
        idx_ch = np.asarray(idx_ch, dtype=int)
        if idx_ch.size == 0:
            return 0.0

        folds = self._stratified_kfold(y)
        accs: List[float] = []
        for vidx in folds:
            tidx = np.setdiff1d(np.arange(len(y)), vidx, assume_unique=False)
            covs_tr = self._get_covs_subset(X[tidx], idx_ch)
            W = self._fit_csp_filters(covs_tr, y[tidx])

            Xf_tr = self._csp_features(W, X[tidx], idx_ch)
            Xf_va = self._csp_features(W, X[vidx], idx_ch)

            flda = self._fit_flda(Xf_tr, y[tidx])
            yhat = self._predict_flda(flda, Xf_va)
            accs.append(float((yhat == y[vidx]).mean()))

        return float(np.mean(accs))

    def train_val_accuracy(self, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, idx_ch: np.ndarray) -> float:
        """Fit on train (CSP+FLDA), evaluate on val (accuracy)."""
        idx_ch = np.asarray(idx_ch, dtype=int)
        if idx_ch.size == 0:
            return 0.0

        covs_tr = self._get_covs_subset(Xtr, idx_ch)
        W = self._fit_csp_filters(covs_tr, ytr)

        Xf_tr = self._csp_features(W, Xtr, idx_ch)
        Xf_va = self._csp_features(W, Xva, idx_ch)

        flda = self._fit_flda(Xf_tr, ytr)
        yhat = self._predict_flda(flda, Xf_va)
        return float((yhat == yva).mean())


# ============================================================
# Data wrapper: subset + pad-to-Kmax with zero channels
# ============================================================

class _CS_ChannelPadDataset(Dataset):
    """
    From base dataset, slice selected real channels and pad to pad_to (zero channels).
    Output keeps (x, y); x is normalized to [1, K, T] when keepdim=True.
    """
    def __init__(self, base_dataset: Dataset, selected_channels: np.ndarray, pad_to: Optional[int], keepdim: bool = True):
        self.base = base_dataset
        self.sel = np.asarray(selected_channels, dtype=int)
        self.pad_to = int(pad_to) if pad_to is not None else None
        self.keepdim = bool(keepdim)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise ValueError("Dataset must yield (x, y) or (x, y, ...).")
        x, y = item[0], item[1]

        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        # to [1,C,T]
        if x.ndim == 2:
            x = x.unsqueeze(0)  # [1,C,T]
        elif x.ndim == 3:
            if x.shape[0] != 1 and self.keepdim:
                # if it's [C,T,?] etc. you should normalize upstream
                pass
        else:
            raise ValueError(f"Unsupported x shape in dataset: {tuple(x.shape)}")

        # slice real channels
        x = x[:, self.sel, :]  # [1,k,T]
        if self.pad_to is not None and x.shape[1] < self.pad_to:
            k, T = x.shape[1], x.shape[2]
            pad = torch.zeros((x.shape[0], self.pad_to - k, T), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=1)  # [1,Kmax,T]
        return x, y


class _ChannelPadDataLoader:
    @staticmethod
    def from_base_loader(base_loader: DataLoader, channels_idx: np.ndarray, pad_to: Optional[int], batch_size=None, shuffle=False) -> DataLoader:
        ds = _CS_ChannelPadDataset(base_loader.dataset, channels_idx, pad_to=pad_to, keepdim=True)
        return DataLoader(
            ds,
            batch_size=batch_size or base_loader.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=getattr(base_loader, "num_workers", 0),
            pin_memory=getattr(base_loader, "pin_memory", False),
            persistent_workers=getattr(base_loader, "persistent_workers", False),
        )


# ============================================================
# ABMOHSChannelSelector (final)
# ============================================================

class ABMOHSChannelSelector:
    """
    ABMOHS wrapper-based channel selection (paper-style):
      - Objectives:
          f1 = 1 - Acc  (CSP+FLDA; CV or train->val)
          f2 = #channels
      - Algorithm:
          * Fisher-score-guided HM initialization
          * Adaptive HMCR(t) + adjustment using random solution from first Pareto front F1
          * Multi-objective selection by nondominated sorting + crowding distance
      - Final decision (paper-like frequency-rank):
          run n_runs times -> collect Pareto front solutions -> channel frequency -> rank
          then choose Ns in [min_ns, min(C,Kmax)] that maximizes evaluator accuracy (on val or CV)

    Interface:
      - fit(): sets selected_channels_ (real channel indices; size Ns <= target_K if set)
      - transform(loader): returns DataLoader where x is padded to target_K (Kmax) with zero channels
    """
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fs: int,
        target_K: Optional[int] = None,   # interpret as Kmax (fixed slot capacity)
        harmony_size: int = 50,           # HMS
        iterations: int = 100,            # NI
        AR: float = 0.2,                  # adjustment rate
        hmcr_C: int = 10,                 # constant C in HMCR formula
        cv_folds: int = 10,               # ten-fold CV (paper)
        n_runs: int = 10,                 # independent runs (paper)
        min_ns: int = 2,                  # Ns=2..N (paper); here clipped by Kmax
        win_ms: int = 100,                # for Fisher init segmentation
        seed: int = 42,
        fitness_mode: str = "val",        # "cv" (paper) or "val" (fast)
        # evaluator hyperparams
        csp_pairs: int = 2,
        cov_reg: float = 1e-6,
        lda_reg: float = 1e-6,
        precompute_cov: bool = False,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fs = int(fs)
        self.win_ms = int(win_ms)

        self.HMS = int(harmony_size)
        self.NI = int(iterations)
        self.AR = float(AR)
        self.hmcr_C = int(hmcr_C)
        self.cv_folds = int(cv_folds)
        self.n_runs = int(n_runs)
        self.min_ns = int(min_ns)
        self.target_K = int(target_K) if target_K is not None else None
        self.fitness_mode = str(fitness_mode).lower()

        self.rng = np.random.default_rng(seed)

        # cache full data to numpy
        self.Xtr_, self.ytr_ = _stack_loader_to_numpy(train_loader)  # (N,C,T)
        self.Xva_, self.yva_ = _stack_loader_to_numpy(val_loader)
        self.C_ = int(self.Xtr_.shape[1])

        # Fisher score for init
        self.fisher_scores_ = _fisher_scores_logvar_segments(self.Xtr_, self.ytr_, fs=self.fs, win_ms=self.win_ms)

        # evaluator: CSP + FLDA
        self.evaluator_ = CSPFLDAEvaluator(
            n_filt_pairs=csp_pairs,
            cv_folds=self.cv_folds,
            cov_reg=cov_reg,
            lda_reg=lda_reg,
            seed=seed,
            precompute_cov=precompute_cov,
        )
        # bind train for optional cov precompute
        self.evaluator_.bind_train_data(self.Xtr_, self.ytr_)

        # output
        self.selected_channels_: Optional[np.ndarray] = None
        self.frequency_rank_: Optional[np.ndarray] = None
        self.channel_freq_: Optional[np.ndarray] = None

        # evaluation cache
        self._eval_cache: Dict[bytes, Tuple[float, float]] = {}

    # ---------- multi-objective utilities ----------
    @staticmethod
    def _nondominated_sort(F: np.ndarray) -> List[List[int]]:
        P = F.shape[0]
        dominates = [set() for _ in range(P)]
        dominated_count = np.zeros(P, dtype=int)
        fronts = [[]]

        for p in range(P):
            for q in range(P):
                if p == q:
                    continue
                if (F[p][0] <= F[q][0] and F[p][1] <= F[q][1]) and (F[p][0] < F[q][0] or F[p][1] < F[q][1]):
                    dominates[p].add(q)
                elif (F[q][0] <= F[p][0] and F[q][1] <= F[p][1]) and (F[q][0] < F[p][0] or F[q][1] < F[p][1]):
                    dominated_count[p] += 1
            if dominated_count[p] == 0:
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            Q = []
            for p in fronts[i]:
                for q in dominates[p]:
                    dominated_count[q] -= 1
                    if dominated_count[q] == 0:
                        Q.append(q)
            i += 1
            fronts.append(Q)

        if not fronts[-1]:
            fronts.pop()
        return fronts

    @staticmethod
    def _crowding_distance(F: np.ndarray, idxs: List[int]) -> np.ndarray:
        if len(idxs) == 0:
            return np.array([])
        if len(idxs) == 1:
            return np.array([np.inf])

        subF = F[idxs]
        d = np.zeros(len(idxs))
        for m in range(2):
            order = np.argsort(subF[:, m])
            d[order[0]] = d[order[-1]] = np.inf
            minv, maxv = subF[order[0], m], subF[order[-1], m]
            rng = max(1e-12, maxv - minv)
            for j in range(1, len(idxs) - 1):
                d[order[j]] += (subF[order[j + 1], m] - subF[order[j - 1], m]) / rng
        return d

    # ---------- ABMOHS specifics ----------
    def _hmcr(self, t: int) -> float:
        """
        Adaptive HMCR(t) in paper:
          HMCR = (1 - C/N) + floor(lnN)/N + (lnN)/N * t/NI
        """
        N = self.C_
        lnN = np.log(max(2, N))
        hmcr = (1.0 - self.hmcr_C / N) + (np.floor(lnN) / N) + (lnN / N) * (t / max(1, self.NI))
        return float(np.clip(hmcr, 0.0, 1.0))

    def _evaluate_mask(self, mask: np.ndarray) -> Tuple[float, float]:
        """Return (f1, f2): f1=1-Acc (CSP+FLDA), f2=#channels."""
        mask = mask.astype(np.uint8, copy=False)
        key = mask.tobytes()
        if key in self._eval_cache:
            return self._eval_cache[key]

        idx = np.flatnonzero(mask)
        k = int(len(idx))
        if k == 0:
            out = (1.0, 0.0)
            self._eval_cache[key] = out
            return out

        if self.fitness_mode == "cv":
            acc = self.evaluator_.cv_accuracy(self.Xtr_, self.ytr_, idx)
        else:
            acc = self.evaluator_.train_val_accuracy(self.Xtr_, self.ytr_, self.Xva_, self.yva_, idx)

        out = (1.0 - float(acc), float(k))
        self._eval_cache[key] = out
        return out

    def _init_hm(self) -> List[np.ndarray]:
        """Fisher-guided HM initialization (binary vectors)."""
        hm: List[np.ndarray] = []
        N = self.C_
        scores = self.fisher_scores_

        for _ in range(self.HMS):
            x = np.zeros(N, dtype=np.uint8)
            n_pick = int(self.rng.random() * N)  # rand × N
            for _ in range(n_pick):
                a = int(self.rng.integers(0, N))
                b = int(self.rng.integers(0, N))
                x[a] = 1 if scores[a] > scores[b] else 0
                x[b] = 1 if scores[b] >= scores[a] else 0
            hm.append(x)
        return hm

    def _improvise_one(self, hm: List[np.ndarray], f1_front: List[int], t: int) -> np.ndarray:
        """
        Improvisation:
          - with prob HMCR: pick bit from random harmony (memory consideration)
              - with prob AR: replace bit by bit from a random harmony in first front F1
          - else random bit
        """
        N = self.C_
        hmcr = self._hmcr(t)
        xnew = np.zeros(N, dtype=np.uint8)

        for j in range(N):
            if self.rng.random() <= hmcr:
                i = int(self.rng.integers(0, len(hm)))
                xnew[j] = hm[i][j]

                if self.rng.random() <= self.AR and len(f1_front) > 0:
                    ridx = int(self.rng.integers(0, len(f1_front)))
                    xnew[j] = hm[f1_front[ridx]][j]
            else:
                xnew[j] = 0 if (self.rng.random() < 0.5) else 1

        return xnew

    def _select_next_hm(self, union: List[np.ndarray], F_union: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Environmental selection: nondominated sort + crowding distance."""
        fronts = self._nondominated_sort(F_union)
        new_hm: List[np.ndarray] = []
        new_F: List[np.ndarray] = []

        for fr in fronts:
            if len(new_hm) + len(fr) <= self.HMS:
                new_hm.extend([union[i] for i in fr])
                new_F.extend([F_union[i] for i in fr])
            else:
                cd = self._crowding_distance(F_union, fr)
                order = np.argsort(-cd)  # desc
                for jj in order:
                    if len(new_hm) < self.HMS:
                        new_hm.append(union[fr[jj]])
                        new_F.append(F_union[fr[jj]])
                    else:
                        break
                break

        return new_hm, np.array(new_F, dtype=float)

    def _run_once(self) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
        """One independent run -> return final HM, objectives F, and indices of first front PF."""
        hm = self._init_hm()
        F = np.array([self._evaluate_mask(x) for x in hm], dtype=float)

        for t in range(1, self.NI):
            fronts = self._nondominated_sort(F)
            f1_front = fronts[0]

            hm2: List[np.ndarray] = []
            for _ in range(self.HMS):
                xnew = self._improvise_one(hm, f1_front=f1_front, t=t)
                hm2.append(xnew)

            F2 = np.array([self._evaluate_mask(x) for x in hm2], dtype=float)

            union = hm + hm2
            F_union = np.vstack([F, F2])

            hm, F = self._select_next_hm(union, F_union)

        fronts = self._nondominated_sort(F)
        return hm, F, fronts[0]

    def fit(self):
        # --- n_runs: collect PF solutions -> frequency rank ---
        freq = np.zeros(self.C_, dtype=np.float64)
        total_pf = 0

        for _ in range(self.n_runs):
            hm, F, pf_idx = self._run_once()
            for i in pf_idx:
                freq += hm[i].astype(np.float64)
                total_pf += 1

        if total_pf > 0:
            freq = freq / float(total_pf)
        else:
            # extreme fallback: use fisher above median
            freq = (self.fisher_scores_ > np.median(self.fisher_scores_)).astype(np.float64)

        self.channel_freq_ = freq

        # rank by freq desc, tie-break by fisher desc
        order = np.lexsort((-self.fisher_scores_, -freq))
        self.frequency_rank_ = order

        # --- choose Ns from top-ranked channels ---
        max_ns = self.target_K if self.target_K is not None else self.C_
        max_ns = min(max_ns, self.C_)
        min_ns = min(max(1, self.min_ns), max_ns)

        best_ns = min_ns
        best_acc = -np.inf

        for ns in range(min_ns, max_ns + 1):
            idx = order[:ns]
            if self.fitness_mode == "cv":
                acc = self.evaluator_.cv_accuracy(self.Xtr_, self.ytr_, idx)
            else:
                acc = self.evaluator_.train_val_accuracy(self.Xtr_, self.ytr_, self.Xva_, self.yva_, idx)

            if (acc > best_acc) or (np.isclose(acc, best_acc) and ns < best_ns):
                best_acc = acc
                best_ns = ns

        self.selected_channels_ = np.array(order[:best_ns], dtype=int)
        return self

    def transform(self, loader: DataLoader, batch_size: Optional[int] = None, shuffle: bool = False) -> DataLoader:
        if self.selected_channels_ is None:
            raise RuntimeError("Call fit() first.")
        return _ChannelPadDataLoader.from_base_loader(
            loader,
            channels_idx=self.selected_channels_,
            pad_to=self.target_K,  # pad-to-Kmax with zero channels
            batch_size=batch_size,
            shuffle=shuffle,
        )


# ============================================================
# Example usage
# ============================================================
"""
selector = ABMOHSChannelSelector(
    train_loader=train_loader,
    val_loader=val_loader,
    fs=config["fs"],
    target_K=32,          # Kmax slots
    harmony_size=50,
    iterations=100,
    n_runs=10,
    cv_folds=10,
    fitness_mode="val",   # fast; use "cv" for strict paper-style
    csp_pairs=2,
    precompute_cov=False, # set True if N not huge and you want speed
).fit()

print("Selected real channels:", selector.selected_channels_)  # Ns <= 32
train_K = selector.transform(train_loader)
val_K   = selector.transform(val_loader)
"""
