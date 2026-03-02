from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from Data.pccs import (
    _project_spd,
    _spd_exp,
    _spd_invsqrt,
    _spd_log,
    _spd_sqrt,
    estimate_spd_cov,
)


def _normalize_layout(layout: str) -> str:
    return str(layout).strip().lower()


def _to_channel_first_trial(trial: np.ndarray, input_layout: str) -> Tuple[np.ndarray, bool]:
    """
    Convert one trial to [C, T] for covariance/manifold operations.
    Returns (trial_ct, transposed_flag).
    """
    x = np.asarray(trial, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D trial, got shape={x.shape}")

    layout = _normalize_layout(input_layout)
    if layout in ("time_first", "t_c", "time_channel"):
        return x.T, True
    if layout in ("channel_first", "c_t", "channel_time"):
        return x, False
    if layout in ("auto",):
        if x.shape[0] >= 4 * x.shape[1]:
            return x.T, True
        return x, False
    raise ValueError(f"Unknown input_layout: {input_layout}")


def _from_channel_first_trial(trial_ct: np.ndarray, transposed: bool) -> np.ndarray:
    return trial_ct.T if transposed else trial_ct


def _safe_float_score(v: Any, default: float = 1.0) -> float:
    try:
        out = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def spd_log_at(
    Q: np.ndarray,
    base: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute Log_base(Q) under AIRM."""
    p = _project_spd(np.asarray(base, dtype=np.float64), float(eps))
    q = _project_spd(np.asarray(Q, dtype=np.float64), float(eps))
    p_sqrt = _spd_sqrt(p, float(eps))
    p_invsqrt = _spd_invsqrt(p, float(eps))
    inner = p_invsqrt @ q @ p_invsqrt
    inner = _project_spd(inner, float(eps))
    log_inner = _spd_log(inner, float(eps))
    v = p_sqrt @ log_inner @ p_sqrt
    return 0.5 * (v + v.T)


def spd_exp_at(
    V: np.ndarray,
    base: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute Exp_base(V) under AIRM."""
    p = _project_spd(np.asarray(base, dtype=np.float64), float(eps))
    v = np.asarray(V, dtype=np.float64)
    v = 0.5 * (v + v.T)
    p_sqrt = _spd_sqrt(p, float(eps))
    p_invsqrt = _spd_invsqrt(p, float(eps))
    inner = p_invsqrt @ v @ p_invsqrt
    inner = 0.5 * (inner + inner.T)
    exp_inner = _spd_exp(inner)
    out = p_sqrt @ exp_inner @ p_sqrt
    return _project_spd(out, float(eps))


def parallel_transport_airm(
    tangent_vec: np.ndarray,
    base_from: np.ndarray,
    base_to: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Parallel-transport tangent_vec from T_base_from to T_base_to under AIRM:
    PT(V) = E V E^T, E = (Q P^{-1})^{1/2}.
    """
    p = _project_spd(np.asarray(base_from, dtype=np.float64), float(eps))
    q = _project_spd(np.asarray(base_to, dtype=np.float64), float(eps))
    v = np.asarray(tangent_vec, dtype=np.float64)
    v = 0.5 * (v + v.T)

    p_inv = np.linalg.inv(p)
    qp_inv = _project_spd(q @ p_inv, float(eps))
    e = _spd_sqrt(qp_inv, float(eps))
    out = e @ v @ e.T
    return 0.5 * (out + out.T)


def whiten_recolor(
    x: np.ndarray,
    R_old: np.ndarray,
    R_new: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Transform one trial x ([C, T]) so covariance changes from R_old to R_new:
    x_new = R_new^(1/2) R_old^(-1/2) x.
    """
    x_ct = np.asarray(x, dtype=np.float64)
    if x_ct.ndim != 2:
        raise ValueError(f"Expected x shape [C,T], got {x_ct.shape}")
    r_old = _project_spd(np.asarray(R_old, dtype=np.float64), float(eps))
    r_new = _project_spd(np.asarray(R_new, dtype=np.float64), float(eps))
    transform = _spd_sqrt(r_new, float(eps)) @ _spd_invsqrt(r_old, float(eps))
    return transform @ x_ct


def _compute_trial_cov(
    x: np.ndarray,
    eps: float,
    use_correlation: bool,
    correlation_eps: float,
    cov_estimator: str = "sample",
    cov_shrinkage: float = 0.0,
    input_layout: str = "channel_first",
) -> np.ndarray:
    """Covariance helper aligned with Data.pccs.estimate_spd_cov."""
    return estimate_spd_cov(
        trial=x,
        reg_eps=float(eps),
        cov_estimator=str(cov_estimator),
        cov_shrinkage=float(cov_shrinkage),
        use_correlation=bool(use_correlation),
        correlation_eps=float(correlation_eps),
        input_layout=str(input_layout),
    )


def extract_p300_effects(
    source_p300_trials: Dict[str, np.ndarray],
    source_bg_protos: Dict[str, np.ndarray],
    selected_subjects: List[str],
    cov_eps: float = 1e-6,
    use_correlation: bool = True,
    correlation_eps: float = 1e-12,
    cov_estimator: str = "sample",
    cov_shrinkage: float = 0.0,
    input_layout: str = "channel_first",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract per-trial tangent effects:
    delta_j = Log_{P_i^-}(R_j^+).
    """
    effects: Dict[str, List[Dict[str, Any]]] = {}
    for subject in selected_subjects:
        sub = str(subject)
        if sub not in source_p300_trials or sub not in source_bg_protos:
            continue
        p_bg = _project_spd(np.asarray(source_bg_protos[sub], dtype=np.float64), float(cov_eps))
        trials = np.asarray(source_p300_trials[sub], dtype=np.float64)
        if trials.ndim != 3 or int(trials.shape[0]) <= 0:
            continue

        rows: List[Dict[str, Any]] = []
        for trial_idx in range(int(trials.shape[0])):
            r_pos = _compute_trial_cov(
                trials[trial_idx],
                eps=float(cov_eps),
                use_correlation=bool(use_correlation),
                correlation_eps=float(correlation_eps),
                cov_estimator=str(cov_estimator),
                cov_shrinkage=float(cov_shrinkage),
                input_layout=str(input_layout),
            )
            delta = spd_log_at(r_pos, base=p_bg, eps=float(cov_eps))
            rows.append(
                {
                    "subject": sub,
                    "trial_idx": int(trial_idx),
                    "trial_cov": r_pos,
                    "tangent_vec": delta,
                }
            )
        if rows:
            effects[sub] = rows
    return effects


def transport_effects(
    effects: Dict[str, List[Dict[str, Any]]],
    source_bg_protos: Dict[str, np.ndarray],
    target_bg_proto: np.ndarray,
    eps: float = 1e-6,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parallel-transport all extracted source effects to target tangent space.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    p_target = _project_spd(np.asarray(target_bg_proto, dtype=np.float64), float(eps))

    for sub, sub_rows in effects.items():
        if sub not in source_bg_protos:
            continue
        p_from = _project_spd(np.asarray(source_bg_protos[sub], dtype=np.float64), float(eps))
        transported_rows: List[Dict[str, Any]] = []
        for row in sub_rows:
            tangent = row.get("tangent_vec", None)
            if tangent is None:
                continue
            tr = parallel_transport_airm(
                np.asarray(tangent, dtype=np.float64),
                p_from,
                p_target,
                eps=float(eps),
            )
            row_new = dict(row)
            row_new["transported_vec"] = tr
            transported_rows.append(row_new)
        if transported_rows:
            out[str(sub)] = transported_rows
    return out


def inject_into_target_trials(
    transported_effects: Dict[str, List[Dict[str, Any]]],
    target_bg_trials: np.ndarray,
    target_bg_proto: np.ndarray,
    rpcs_scores: Dict[str, float],
    n_synth: int,
    beta: float = 1.0,
    seed: int = 2026,
    cov_eps: float = 1e-6,
    use_correlation: bool = True,
    correlation_eps: float = 1e-12,
    cov_estimator: str = "sample",
    cov_shrinkage: float = 0.0,
    input_layout: str = "channel_first",
    rpcs_weighted_sampling: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inject transported effects into target background trials and synthesize target-style P300.
    """
    _ = target_bg_proto  # reserved for future consistency checks
    n_out = max(0, int(n_synth))
    target = np.asarray(target_bg_trials)
    if n_out <= 0 or target.ndim != 3 or int(target.shape[0]) <= 0:
        shape = (0,) + tuple(target.shape[1:]) if target.ndim >= 3 else (0, 0, 0)
        return np.empty(shape, dtype=np.float64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    pool: List[Tuple[np.ndarray, str, float]] = []
    for sub, rows in transported_effects.items():
        raw_score = _safe_float_score(rpcs_scores.get(str(sub), 1.0), default=1.0)
        score = raw_score if bool(rpcs_weighted_sampling) else 1.0
        score = max(score, 1e-12)
        for row in rows:
            tr = row.get("transported_vec", None)
            if tr is None:
                continue
            pool.append((np.asarray(tr, dtype=np.float64), str(sub), float(raw_score)))

    if not pool:
        empty = np.empty((0,) + tuple(target.shape[1:]), dtype=np.float64)
        return empty, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    sample_weights = np.asarray(
        [max(_safe_float_score(rpcs_scores.get(sub, 1.0), default=1.0), 1e-12) if rpcs_weighted_sampling else 1.0 for _, sub, _ in pool],
        dtype=np.float64,
    )
    if not np.isfinite(sample_weights).all() or float(np.sum(sample_weights)) <= 0.0:
        sample_weights = np.ones(len(pool), dtype=np.float64)
    sample_weights /= float(np.sum(sample_weights))

    rng = np.random.default_rng(int(seed))
    synth_trials: List[np.ndarray] = []
    synth_weights = np.zeros(n_out, dtype=np.float64)

    for idx in range(n_out):
        eff_idx = int(rng.choice(len(pool), p=sample_weights))
        delta_t, _sub, raw_score = pool[eff_idx]
        tgt_idx = int(rng.integers(0, int(target.shape[0])))
        trial_raw = np.asarray(target[tgt_idx], dtype=np.float64)
        trial_ct, was_transposed = _to_channel_first_trial(trial_raw, str(input_layout))

        r_bg = _compute_trial_cov(
            trial_ct,
            eps=float(cov_eps),
            use_correlation=bool(use_correlation),
            correlation_eps=float(correlation_eps),
            cov_estimator=str(cov_estimator),
            cov_shrinkage=float(cov_shrinkage),
            input_layout="channel_first",
        )
        r_synth = spd_exp_at(float(beta) * delta_t, base=r_bg, eps=float(cov_eps))
        x_synth_ct = whiten_recolor(trial_ct, r_bg, r_synth, eps=float(cov_eps))
        x_synth = _from_channel_first_trial(x_synth_ct, was_transposed)

        synth_trials.append(x_synth.astype(np.float64, copy=False))
        synth_weights[idx] = float(max(raw_score, 0.0))

    synth_arr = np.stack(synth_trials, axis=0) if synth_trials else np.empty((0,) + tuple(target.shape[1:]), dtype=np.float64)
    synth_labels = np.ones(int(synth_arr.shape[0]), dtype=np.int64)
    return synth_arr, synth_labels, synth_weights[: int(synth_arr.shape[0])]


def generate_rpt_augmented_batch(
    *,
    source_pos_protos: Dict[str, np.ndarray],
    source_bg_protos: Dict[str, np.ndarray],
    target_bg_proto: np.ndarray,
    rpcs_scores: Dict[str, float],
    selected_subjects: List[str],
    source_p300_trials: Dict[str, np.ndarray],
    target_bg_trials: np.ndarray,
    n_synth: int = 100,
    beta: float = 1.0,
    seed: int = 2026,
    cov_eps: float = 1e-6,
    use_correlation: bool = True,
    correlation_eps: float = 1e-12,
    cov_estimator: str = "sample",
    cov_shrinkage: float = 0.0,
    input_layout: str = "channel_first",
    rpcs_weighted_sampling: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Full RPT-Aug pipeline: extract -> transport -> inject.
    """
    _ = source_pos_protos  # reserved for optional source-positive prototype controls
    effects = extract_p300_effects(
        source_p300_trials=source_p300_trials,
        source_bg_protos=source_bg_protos,
        selected_subjects=selected_subjects,
        cov_eps=float(cov_eps),
        use_correlation=bool(use_correlation),
        correlation_eps=float(correlation_eps),
        cov_estimator=str(cov_estimator),
        cov_shrinkage=float(cov_shrinkage),
        input_layout=str(input_layout),
    )
    transported = transport_effects(
        effects=effects,
        source_bg_protos=source_bg_protos,
        target_bg_proto=target_bg_proto,
        eps=float(cov_eps),
    )
    trials, labels, weights = inject_into_target_trials(
        transported_effects=transported,
        target_bg_trials=target_bg_trials,
        target_bg_proto=target_bg_proto,
        rpcs_scores=rpcs_scores,
        n_synth=int(n_synth),
        beta=float(beta),
        seed=int(seed),
        cov_eps=float(cov_eps),
        use_correlation=bool(use_correlation),
        correlation_eps=float(correlation_eps),
        cov_estimator=str(cov_estimator),
        cov_shrinkage=float(cov_shrinkage),
        input_layout=str(input_layout),
        rpcs_weighted_sampling=bool(rpcs_weighted_sampling),
    )
    return {
        "trials": trials,
        "labels": labels,
        "weights": weights,
    }


class RPTAugmentor:
    """
    Stateful fold-level wrapper:
    - prepare(): precompute extract + transport once
    - sample(n, seed): synthesize n samples for current fold
    """

    def __init__(
        self,
        *,
        rpcs_result: Dict[str, Any],
        source_p300_trials: Dict[str, np.ndarray],
        target_bg_trials: np.ndarray,
        selected_subjects: Optional[List[str]] = None,
        beta: float = 1.0,
        cov_eps: float = 1e-6,
        use_correlation: bool = True,
        correlation_eps: float = 1e-12,
        cov_estimator: str = "sample",
        cov_shrinkage: float = 0.0,
        input_layout: str = "channel_first",
        rpcs_weighted_sampling: bool = True,
    ):
        self.rpcs_result = dict(rpcs_result or {})
        self.source_p300_trials = source_p300_trials
        self.target_bg_trials = np.asarray(target_bg_trials)
        self.selected_subjects = [str(s) for s in (selected_subjects or [])]

        self.beta = float(beta)
        self.cov_eps = float(cov_eps)
        self.use_correlation = bool(use_correlation)
        self.correlation_eps = float(correlation_eps)
        self.cov_estimator = str(cov_estimator)
        self.cov_shrinkage = float(cov_shrinkage)
        self.input_layout = str(input_layout)
        self.rpcs_weighted_sampling = bool(rpcs_weighted_sampling)

        self._prepared = False
        self._transported_effects: Dict[str, List[Dict[str, Any]]] = {}
        self._scores: Dict[str, float] = {}

    def _resolve_prototypes(self) -> Dict[str, Any]:
        protos = dict(self.rpcs_result.get("prototypes", {}) or {})
        required = ("target_bg", "source_pos", "source_bg")
        for key in required:
            if key not in protos:
                raise KeyError(
                    f"Missing '{key}' in rpcs_result['prototypes']; "
                    "ensure compute_rpcs_source_scores(..., return_prototypes=True)."
                )
        return protos

    def _resolve_scores(self) -> Dict[str, float]:
        ranking = list(self.rpcs_result.get("ranking", []) or [])
        if ranking:
            return {str(row.get("subject")): _safe_float_score(row.get("score", 1.0), default=1.0) for row in ranking}
        score_map = self.rpcs_result.get("scores", {})
        if isinstance(score_map, dict):
            return {str(k): _safe_float_score(v, default=1.0) for k, v in score_map.items()}
        return {}

    def _resolve_selected_subjects(self, source_bg_protos: Dict[str, np.ndarray]) -> List[str]:
        if self.selected_subjects:
            return [s for s in self.selected_subjects if s in source_bg_protos and s in self.source_p300_trials]

        ranking = list(self.rpcs_result.get("ranking", []) or [])
        ranked_subjects = [str(row.get("subject")) for row in ranking if row.get("subject") is not None]
        if ranked_subjects:
            return [s for s in ranked_subjects if s in source_bg_protos and s in self.source_p300_trials]
        return [s for s in source_bg_protos.keys() if s in self.source_p300_trials]

    def prepare(self) -> None:
        protos = self._resolve_prototypes()
        source_bg = dict(protos.get("source_bg", {}) or {})
        target_bg = np.asarray(protos.get("target_bg"), dtype=np.float64)
        selected = self._resolve_selected_subjects(source_bg)
        self._scores = self._resolve_scores()

        effects = extract_p300_effects(
            source_p300_trials=self.source_p300_trials,
            source_bg_protos=source_bg,
            selected_subjects=selected,
            cov_eps=self.cov_eps,
            use_correlation=self.use_correlation,
            correlation_eps=self.correlation_eps,
            cov_estimator=self.cov_estimator,
            cov_shrinkage=self.cov_shrinkage,
            input_layout=self.input_layout,
        )
        self._transported_effects = transport_effects(
            effects=effects,
            source_bg_protos=source_bg,
            target_bg_proto=target_bg,
            eps=self.cov_eps,
        )
        self._prepared = True

    def sample(self, n: int, seed: int = 0) -> Dict[str, np.ndarray]:
        if not self._prepared:
            raise RuntimeError("RPTAugmentor.prepare() must be called before sample().")

        protos = self._resolve_prototypes()
        trials, labels, weights = inject_into_target_trials(
            transported_effects=self._transported_effects,
            target_bg_trials=self.target_bg_trials,
            target_bg_proto=np.asarray(protos.get("target_bg"), dtype=np.float64),
            rpcs_scores=self._scores,
            n_synth=int(n),
            beta=self.beta,
            seed=int(seed),
            cov_eps=self.cov_eps,
            use_correlation=self.use_correlation,
            correlation_eps=self.correlation_eps,
            cov_estimator=self.cov_estimator,
            cov_shrinkage=self.cov_shrinkage,
            input_layout=self.input_layout,
            rpcs_weighted_sampling=self.rpcs_weighted_sampling,
        )
        return {"trials": trials, "labels": labels, "weights": weights}

