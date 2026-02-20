import time, math, json
from contextlib import contextmanager
import copy
import torch
import pandas as pd

from Utils.config import build_config
from Models.eeg_models import model_dict

# =============== 基础工具 / Basic utilities ===============
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def bytes_of(dtype):
    return {
        torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
        torch.uint8: 1, torch.int8: 1
    }[dtype]

def estimate_model_size(params, dtype=torch.float32):
    return params * bytes_of(dtype)

def _get_tensor_shape(out):
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.shape

# =============== MACs 估计 / MACs estimate ===============
def estimate_macs(model, input_shape):
    """
    估计 MACs：
    - 支持 Conv2d / Conv1d / Linear / LSTM
    - 对 LSTM：按 batch_first 自动推断 (B, T, F) 或 (T, B, F)
    - 对 depthwise/grouped 卷积，按 (C_in // groups) 计
    """
    macs = 0
    hooks = []

    def _shape(out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.shape

    # ---- Conv2d ----
    def conv2d_hook(m, inp, out):
        try:
            _, C_out, H_out, W_out = _shape(out)
            C_in  = m.in_channels
            Kh, Kw = m.kernel_size
            groups = max(1, m.groups)
            mac = H_out * W_out * C_out * (C_in // groups) * Kh * Kw
            nonlocal macs; macs += mac
        except Exception:
            pass

    # ---- Conv1d ----
    def conv1d_hook(m, inp, out):
        try:
            # out: (B, C_out, L_out)
            _, C_out, L_out = _shape(out)
            C_in  = m.in_channels
            K = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
            groups = max(1, m.groups)
            mac = L_out * C_out * (C_in // groups) * K
            nonlocal macs; macs += mac
        except Exception:
            pass

    # ---- Linear ----
    def linear_hook(m, inp, out):
        try:
            mac = m.in_features * m.out_features
            nonlocal macs; macs += mac
        except Exception:
            pass

    # ---- LSTM ----
    def lstm_hook(m, inp, out):
        """
        近似计算：每时间步、每层、每方向的 MACs:
            4 * (n_in * n_h + n_h * n_h + n_h)
        乘以 seq_len * num_layers * num_directions * batch_size
        """
        try:
            x = inp[0]  # shape: (T, B, F) or (B, T, F) 取决于 batch_first
            if m.batch_first:
                B, T, F = x.shape
            else:
                T, B, F = x.shape
            n_h = m.hidden_size
            n_l = m.num_layers
            n_d = 2 if m.bidirectional else 1
            per_step = 4 * (F * n_h + n_h * n_h + n_h)
            mac = per_step * T * n_l * n_d * B
            nonlocal macs; macs += mac
        except Exception:
            pass

    # 注册 hook
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv2d_hook))
        elif isinstance(m, torch.nn.Conv1d):
            hooks.append(m.register_forward_hook(conv1d_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, torch.nn.LSTM):
            hooks.append(m.register_forward_hook(lstm_hook))

    # 前向（尝试多种输入形状）
    model.eval()
    tried = []
    with torch.no_grad():
        for ishape in [input_shape,  # 例如 (B,1,C,T)
                       (input_shape[0], input_shape[2], input_shape[3])]:  # (B,C,T) for 1D nets
            try:
                x = torch.randn(*ishape)
                _ = model(x)
                break
            except Exception as e:
                tried.append((ishape, str(e)))
                continue
        else:
            # 都失败，清理 hook 并抛错
            for h in hooks: h.remove()
            raise RuntimeError(f"Forward failed for all candidate input shapes: {tried}")

    for h in hooks:
        h.remove()
    return int(macs)


# =============== 计时 / Timing ===============
@contextmanager
def cuda_sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()
    yield
    if torch.cuda.is_available(): torch.cuda.synchronize()

def time_forward(model, input_shape, device=None, iters=500, warmup=100):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(*input_shape, device=device)
    model = model.to(device).eval()

    times = []
    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            if torch.cuda.is_available(): torch.cuda.synchronize()
            _ = model(x)
            if torch.cuda.is_available(): torch.cuda.synchronize()
        # measure with high-res clock
        for _ in range(iters):
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms
    times.sort()
    p50 = times[len(times)//2]
    p95 = times[int(len(times)*0.95)]
    jitter = p95 - p50
    return p50, p95, jitter

def time_forward_cuda_events(model, input_shape, device=None, iters=500, warmup=100):
    if not torch.cuda.is_available():
        return None, None, None
    device = device or 'cuda'
    x = torch.randn(*input_shape, device=device)
    model = model.to(device).eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    times = []
    with torch.no_grad():
        for _ in range(warmup):
            with cuda_sync(): _ = model(x)
        for _ in range(iters):
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))  # ms
    times.sort()
    p50 = times[len(times)//2]
    p95 = times[int(len(times)*0.95)]
    jitter = p95 - p50
    return p50, p95, jitter

def peak_cuda_memory(model, input_shape):
    if not torch.cuda.is_available(): return None
    device = 'cuda'
    x = torch.randn(*input_shape, device=device)
    model = model.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device=device)
    with torch.no_grad():
        _ = model(x)
    return torch.cuda.max_memory_allocated(device=device)

# =============== NVML 功率/能耗 / Power & Energy via NVML ===============
def measure_idle_power_nvml(samples=100, interval_s=0.01):
    try:
        import pynvml
        if not torch.cuda.is_available(): return None
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        vals = []
        for _ in range(samples):
            p_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            vals.append(p_mw / 1000.0)
            time.sleep(interval_s)
        return sum(vals)/len(vals) if vals else None
    except Exception:
        return None

def measure_energy_nvml_integrated(model, input_shape, repeats=1500, warmup=300):
    """
    在推理循环内读 (t, P)，做梯形积分。
    返回: avg_power_W(raw), energy_per_inf_mJ(raw)
    若 NVML/CUDA 不可用，返回 (None, None)
    """
    try:
        import pynvml, time as _t
        if not torch.cuda.is_available(): return None, None
        device = 'cuda'
        x = torch.randn(*input_shape, device=device)
        model = model.to(device).eval()

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        with torch.no_grad():
            # warmup
            for _ in range(warmup):
                if torch.cuda.is_available(): torch.cuda.synchronize()
                _ = model(x)
                if torch.cuda.is_available(): torch.cuda.synchronize()

            # integrate
            t_start = _t.perf_counter()
            t_prev  = t_start
            p_prev  = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
            energy_j = 0.0

            for _ in range(repeats):
                if torch.cuda.is_available(): torch.cuda.synchronize()
                _ = model(x)
                if torch.cuda.is_available(): torch.cuda.synchronize()

                t_now = _t.perf_counter()
                p_now = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                dt = max(0.0, t_now - t_prev)
                # 梯形法：E += (P_prev + P_now)/2 * dt
                energy_j += (p_prev + p_now) * 0.5 * dt
                t_prev, p_prev = t_now, p_now

            t_end = _t.perf_counter()
            total_time = max(1e-9, t_end - t_start)
            avg_power_w = energy_j / total_time
            energy_per_inf_mJ = (energy_j / repeats) * 1000.0
            return avg_power_w, energy_per_inf_mJ
    except Exception:
        return None, None

# =============== 单模型基准 / Benchmark one model ===============
def resolve_model_class(name, mdict):
    entry = mdict[name]
    return getattr(entry, "Model", entry)

def benchmark_one(model_name, base_config, input_shape):
    cfg = copy.deepcopy(base_config)
    cfg["model"] = model_name
    mdict = model_dict()
    model_cls = resolve_model_class(model_name, mdict)
    model = model_cls(cfg)

    # 静态
    total, trainable = count_params(model)
    size_fp32 = estimate_model_size(total, torch.float32)
    size_fp16 = estimate_model_size(total, torch.float16)
    size_int8 = estimate_model_size(total, torch.int8)
    macs = estimate_macs(model, input_shape)
    flops = macs * 2

    # 动态
    p50, p95, jitter = time_forward(model, input_shape)
    p50_ev, p95_ev, jitter_ev = time_forward_cuda_events(model, input_shape)
    peak_mem = peak_cuda_memory(model, input_shape)

    # Power/Energy
    idle_power = measure_idle_power_nvml(samples=100, interval_s=0.01)
    avg_power_raw, energy_raw_mJ = measure_energy_nvml_integrated(model, input_shape, repeats=1500, warmup=300)
    if avg_power_raw is not None and idle_power is not None:
        dyn_power = max(0.0, avg_power_raw - idle_power)
        dyn_energy_mJ = None if energy_raw_mJ is None else max(0.0, energy_raw_mJ - idle_power * ( (energy_raw_mJ/1000.0) / (avg_power_raw if avg_power_raw>0 else 1.0) )*1000.0)
        # 更简单：用“动态功率 × 平均单次时延（ms）”估一下动态能耗
        est_energy_dyn_mJ = dyn_power * (p50/1000.0) * 1000.0 if dyn_power is not None else None
    else:
        dyn_power, dyn_energy_mJ, est_energy_dyn_mJ = None, None, None

    return {
        "模型 / Model": model_name,
        "输入形状 / Input": str(input_shape),
        "参数量 / Params": total,
        "可训练参数 / Trainable": trainable,
        "体积FP32(MB) / SizeFP32": size_fp32/1e6,
        "体积FP16(MB) / SizeFP16": size_fp16/1e6,
        "体积INT8(MB) / SizeINT8": size_int8/1e6,
        "MACs(M) / MACs(M)": macs/1e6,
        "FLOPs(M) / FLOPs(M)": flops/1e6,
        "前向时延p50(ms) / Latency p50": p50,
        "前向时延p95(ms) / Latency p95": p95,
        "抖动(ms) / Jitter": jitter,
        "CUDA事件p50(ms) / CUDAEvt p50": p50_ev if p50_ev is not None else float("nan"),
        "峰值显存(MB) / Peak CUDA Mem": (peak_mem/1e6) if peak_mem is not None else float("nan"),
        "空闲功率(W) / Idle Power": idle_power if idle_power is not None else float("nan"),
        "平均功率(W) / Avg Power(raw)": avg_power_raw if avg_power_raw is not None else float("nan"),
        "单次能耗(mJ) / Energy per inf(raw)": energy_raw_mJ if energy_raw_mJ is not None else float("nan"),
        "动态功率(W) / Dyn Power(≈raw-idle)": dyn_power if dyn_power is not None else float("nan"),
        "估计动态能耗(mJ) / Est Dyn Energy": est_energy_dyn_mJ if est_energy_dyn_mJ is not None else float("nan"),
    }

# =============== 主程序 / Main ===============
if __name__ == "__main__":
    input_shape = (1, 1, 62, 128)
    base_config = build_config("EEGNet")

    # 你要批量测的模型列表 / models to benchmark
    model_list = [
        "DeepConvNet",
        "EEGNet",
        "EEGInception",
        "PLNet",
        "PPNN",
    ]

    results = []
    for name in model_list:
        try:
            print(f"[Run] {name}")
            res = benchmark_one(name, base_config, input_shape)
            results.append(res)
        except Exception as e:
            print(f"[Error] {name}: {e}")

    if not results:
        print("No Results produced.")
        raise SystemExit(0)

    df = pd.DataFrame(results)

    # ---------- Table 1: Core metrics ----------
    core_cols = [
        "模型 / Model",
        "参数量 / Params",
        "体积FP32(MB) / SizeFP32",
        "MACs(M) / MACs(M)",
        "FLOPs(M) / FLOPs(M)",
        "前向时延p50(ms) / Latency p50",
        "前向时延p95(ms) / Latency p95",
        "峰值显存(MB) / Peak CUDA Mem",
    ]
    df_core = df[core_cols]
    df_core.to_csv("Table1_LightweightEEG_Core.csv", index=False, encoding="utf-8-sig")

    # Markdown 版
    df_core_md = df_core.copy()
    for c in df_core_md.select_dtypes(include=["float"]).columns:
        df_core_md[c] = df_core_md[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NaN")
    with open("Table1_LightweightEEG_Core.md","w",encoding="utf-8") as f:
        f.write(df_core_md.to_markdown(index=False))

    # ---------- Table S1: Extended metrics ----------
    # ext_cols = [
    #     "模型 / Model",
    #     "可训练参数 / Trainable",
    #     "体积FP16(MB) / SizeFP16",
    #     "体积INT8(MB) / SizeINT8",
    #     "抖动(ms) / Jitter",
    #     "CUDA事件p50(ms) / CUDAEvt p50",
    #     "空闲功率(W) / Idle Power",
    #     "平均功率(W) / Avg Power(raw)",
    #     "动态功率(W) / Dyn Power(≈raw-idle)",
    #     "单次能耗(mJ) / Energy per inf(raw)",
    #     "估计动态能耗(mJ) / Est Dyn Energy",
    # ]
    # df_ext = df[ext_cols]
    # df_ext.to_csv("TableS1_LightweightEEG_Extended.csv", index=False, encoding="utf-8-sig")
    #
    # # Markdown 版
    # df_ext_md = df_ext.copy()
    # for c in df_ext_md.select_dtypes(include=["float"]).columns:
    #     df_ext_md[c] = df_ext_md[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NaN")
    # with open("TableS1_LightweightEEG_Extended.md","w",encoding="utf-8") as f:
    #     f.write(df_ext_md.to_markdown(index=False))
