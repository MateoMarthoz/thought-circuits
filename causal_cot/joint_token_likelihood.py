"""
Token-level joint likelihood for reasoning steps (sum of log P(token | prefix)).

Used for normalized joint metric
``(p - p_base) / (p_clean - p_base)`` with float64 log-space accumulation.
``relative_ppl`` / mean-CE paths are unchanged.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

# Bumped when on-disk joint semantics change (see run_joint_baselines.py).
JOINT_METRIC_VERSION = 2

# Minimum positive denominator for (p_clean - p_base) in linear space.
_DENOM_EPS = 1e-300


def sum_logprob_at_step(
    logits: torch.Tensor,
    labels: torch.Tensor,
    step_start: int,
    step_end: int,
) -> float:
    """
    Sum of log P(x_t | x_{<t}) over prediction positions in one reasoning step.

    Matches ``calculate_step_loss`` span: logits ``[s : e-1]`` predict
    ``labels[s+1 : e]``.
    """
    if step_end <= step_start + 1:
        return 0.0
    logits_d = logits[step_start : step_end - 1].double()
    labs = labels[step_start + 1 : step_end].long()
    if logits_d.numel() == 0:
        return 0.0
    log_probs = F.log_softmax(logits_d, dim=-1)
    gathered = log_probs.gather(1, labs.unsqueeze(-1)).squeeze(-1)
    return float(gathered.sum().item())


def joint_metric_from_mean_ce_delta(
    k: int,
    mean_delta: float,
    step_idx: int,
    baseline: Optional[JointBaselineTable] = None,
) -> float:
    """
    Joint display scalar for one reasoning step from its mean CE delta (mixed minus clean).

    If ``baseline`` has ``log_p_clean`` / ``log_p_base`` for ``step_idx``, returns
    normalized ``(p - p_base) / (p_clean - p_base)`` with
    ``log p_mixed ≈ log p_clean - K * mean_delta``.

    Otherwise returns legacy ``exp(-K * mean_delta)`` (same as historical P/P_clean).
    """
    if k <= 0 or not math.isfinite(mean_delta):
        return 1.0
    if baseline is not None:
        lc = baseline.log_p_clean_for(step_idx)
        lb = baseline.log_p_base_for(step_idx)
        if lc is not None and lb is not None:
            log_p_mixed = lc - float(k) * float(mean_delta)
            jn = normalize_joint_from_logprobs(log_p_mixed, lb, lc)
            if math.isfinite(jn):
                return jn
    return math.exp(-float(k) * float(mean_delta))


def normalize_joint_from_logprobs(
    log_p: float,
    log_p_base: float,
    log_p_clean: float,
) -> float:
    """
    ``(p - p_base) / (p_clean - p_base)`` with p = exp(log_p), float64.

    Returns NaN if denominator ``p_clean - p_base`` is non-positive or tiny.
    """
    p = float(torch.exp(torch.tensor(log_p, dtype=torch.float64)).item())
    pb = float(torch.exp(torch.tensor(log_p_base, dtype=torch.float64)).item())
    pc = float(torch.exp(torch.tensor(log_p_clean, dtype=torch.float64)).item())
    denom = pc - pb
    if not math.isfinite(denom) or denom <= _DENOM_EPS:
        return float("nan")
    num = p - pb
    return float(num / denom)


@dataclass
class JointBaselineTable:
    """Loaded ``joint_step_logprobs.json`` (per reasoning step index, 0-based)."""

    joint_metric_version: int
    num_steps: int
    log_p_clean: Dict[int, float]
    log_p_base: Dict[int, float]
    source_path: Path

    def log_p_clean_for(self, step_idx: int) -> Optional[float]:
        return self.log_p_clean.get(step_idx)

    def log_p_base_for(self, step_idx: int) -> Optional[float]:
        return self.log_p_base.get(step_idx)

    def joint_norm(
        self,
        log_p: float,
        step_idx: int,
    ) -> float:
        lc = self.log_p_clean_for(step_idx)
        lb = self.log_p_base_for(step_idx)
        if lc is None or lb is None:
            return float("nan")
        return normalize_joint_from_logprobs(log_p, lb, lc)


def load_joint_baseline_json(path: Union[str, Path]) -> JointBaselineTable:
    p = Path(path).expanduser().resolve()
    with open(p, encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)

    ver = int(raw.get("joint_metric_version", 0))
    n = int(raw["num_steps"])
    lpc = {int(k): float(v) for k, v in (raw.get("log_p_clean_per_step") or {}).items()}
    lpb = {int(k): float(v) for k, v in (raw.get("log_p_base_per_step") or {}).items()}
    return JointBaselineTable(
        joint_metric_version=ver,
        num_steps=n,
        log_p_clean=lpc,
        log_p_base=lpb,
        source_path=p,
    )


def mix_start_for_predecessor_substitution(
    step_ranges: List[Tuple[int, int]],
    target_step_idx: int,
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Spans for all reasoning steps strictly before ``target_step_idx`` (0-based)
    and ``query_mix_start`` for ``PerturbedKVPostSourceContext``.

    Uses ``query_mix_start_for_kv_substitution`` with candidate step 0.
    """
    from .perturbed_kv_post_source import query_mix_start_for_kv_substitution

    if target_step_idx < 1:
        raise ValueError("target_step_idx must be >= 1 for predecessor substitution")
    subst = set(range(target_step_idx))
    spans = [step_ranges[k] for k in range(target_step_idx)]
    mix = query_mix_start_for_kv_substitution(0, subst, step_ranges)
    return spans, mix
