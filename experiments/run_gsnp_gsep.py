#!/usr/bin/env python3
"""
Stage 1: Same greedy perturb ablation as ``run_gsnp.py`` (including joint maximin /
relative-PPL minimax candidate choice). When a chosen prune fails the threshold test,
that prune is **not** applied (state unchanged); stage 1 then stops. Otherwise continues
until all non-final candidates are pruned.

**Threshold modes** (at most one of the two flags; if neither is passed, relative PPL mode
with threshold 1.1)

- ``--ppl-threshold`` (default 1.1 when omitted): ``relative_ppl`` — stage 1 picks the
  candidate ``i`` with the **lowest maximum** later-step relative PPL
  (``min_i max_{j>i} exp(delta)``); reject that prune if that max exceeds the threshold.
  Stage 2 scores a single target step per candidate (same as min delta on target).

- ``--joint-prob-threshold``: ``joint_prob_ratio`` — stage 1 picks ``i`` with the
  **highest minimum** later-step **normalized joint** (product of token log-probs vs
  ``--joint-baseline-json``); reject if that bottleneck is below the threshold.
  Stage 2 uses the same normalized joint on the fixed target (candidate order still
  matches min mean CE delta on the target).

Stage 2: For each reasoning step *not* pruned in stage 1, from the final step backward,
run KV perturb ablation (same mechanism as ``run_backward_ablation`` perturb): always
include stage-1 pruned steps in the corrupt set; greedily prune among earlier survivors
by the same metric as stage 1 (min delta on target for ``relative_ppl``; normalized
joint from the baseline file on target for ``joint_prob_ratio``). Same threshold rule
as stage 1. Think-inner spans must
align (no gaps);
each forward uses one pert file for all corrupt K/V.

Outputs (stdout): kept indices after stage 1; after each target, the steps that target
"kept" among {steps before target and not pruned in stage 1} minus stage-2 prunes.

By default writes JSON to ``results/gsnp_gsep_ppl.json`` or
``results/gsnp_gsep_joint_prob.json`` from the threshold mode (same paths as
``plot_scripts/plot_gsnp_gsep.py``). Override with
``--output_json``; use ``--no-output-json`` to skip writing. The file includes
``edge_sparsity_ppl_curve.checkpoints``: each entry has ``edge_sparsity_fraction``,
``mean_rel_ppl``, ``max_rel_ppl``, plus ``mean_joint_prob_ratio`` and
``min_joint_prob_ratio`` (mean / **min** normalized joint over kept steps; min pairs
with max relative PPL as the worst-step summary).

**Plot curve (worst-of-record):** each checkpoint also has ``curve_min_joint_worst`` and
``curve_max_rel_ppl_worst``. Stage 1: same as ``min_joint_prob_ratio`` /
``max_rel_ppl`` (full-circuit aggregates). Stage 2: each target starts from a fresh
post–stage-1 KV state; only the current target's rel PPL / joint is refreshed. The
curve fields track the **running** worst metric across **completed** targets: after each
target's GSEP session, that target's final normalized joint (resp. rel PPL) is compared
to the stage-1 end value and to prior targets' finals; ``curve_min_joint_worst`` is the
minimum normalized joint seen so far (lower = worse), ``curve_max_rel_ppl_worst`` the
maximum rel PPL (higher = worse). Checkpoints **within** a target use the running value
**before** that target finished; the **last** checkpoint of each target applies the update.

``threshold_prune_config`` records ``threshold_metric`` (from which threshold flag was
used) and ``threshold_value`` for plot naming. ``prediction_token_count_per_step``
matches gsep ``K``.

Also logs ``edge_sparsity_ppl_curve``: cumulative directed edges in the upper-triangle
(i<j) vs mean / max relative PPL over non–stage-1-pruned steps. Stage 1 adds
``N - r`` edges on round ``r``; each accepted stage-2 prune adds one edge (source→target).
Stage 2 resets the running per-step PPL vector to the post–stage-1 snapshot for each
new target; only the current target's PPL is updated within that target's run.

Stage 1 ``current_ppl`` updates only indices ``j > i_star`` in the winning forward; under
causal attention, a kept step ``k < i_star`` does not read KV from pruned steps
``> k``, so its last measured rel PPL stays valid until some prune with ``i < k``
refreshes it.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

import torch
from tqdm import tqdm

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from causal_cot import paths as rp
from causal_cot.eval_perturbations import load_aligned_perturbed_cots
from causal_cot.interventions.kv_perturbation import (
    PerturbedKVPostSourceContext,
    query_mix_start_for_kv_substitution,
)
from causal_cot.joint_token_likelihood import (
    JOINT_METRIC_VERSION,
    JointBaselineTable,
    load_joint_baseline_json,
    sum_logprob_at_step,
)
from causal_cot.load_model import load_model_and_tokenizer
from causal_cot.metrics import calculate_step_loss
from causal_cot.step_utils import (
    assert_reasoning_token_ranges_match_think_inner,
    extract_think_block,
    map_steps_to_tokens,
    parse_reasoning_steps,
)


@torch.no_grad()
def compute_clean_step_losses(
    model,
    clean_ids: torch.Tensor,
    clean_token_ranges: List[Tuple[int, int]],
) -> List[float]:
    logits = model(input_ids=clean_ids, use_cache=False).logits[0]
    losses = []
    for s, e in clean_token_ranges:
        losses.append(calculate_step_loss(logits, clean_ids[0], s, e).item())
    return losses


def _mean_max_rel_ppl_kept_steps(
    current_ppl: List[float], pruned_set: Set[int]
) -> Tuple[float, float]:
    """Mean and max relative PPL over step indices not in ``pruned_set``."""
    active = [j for j in range(len(current_ppl)) if j not in pruned_set]
    if not active:
        return 1.0, 1.0
    vals = [current_ppl[j] for j in active]
    return statistics.mean(vals), max(vals)


def _k_predict_tokens_per_step(
    clean_token_ranges: List[Tuple[int, int]],
) -> List[int]:
    """``K`` per step: prediction positions in ``calculate_step_loss`` (``e - s - 1``)."""
    out: List[int] = []
    for s, e in clean_token_ranges:
        out.append(max(0, e - s - 1))
    return out


def _joint_prob_ratio_from_mean_delta(delta: float, k: int) -> float:
    """
    Joint token-sequence ratio vs clean for one step:
    ``exp(-K * mean_delta)`` (same convention as gsep / joint P/P_clean).

    If ``K <= 0`` (no scored prediction tokens), defined as ``1.0``.

    Computed in **IEEE float64** (Python ``float`` / JSON). Products can **underflow**
    to ``0.0`` near ``~1e-308``; logits/losses come from the model in reduced precision
    (e.g. bfloat16) before ``.item()`` promotes to float64.
    """
    if k <= 0:
        return 1.0
    return math.exp(-float(k) * float(delta))


def _joint_from_rel_ppl(rel_ppl: float, k: int) -> float:
    """Joint ``P/P_clean`` from relative PPL ``exp(delta)`` on a step with token count ``K``."""
    p = max(float(rel_ppl), 1e-300)
    return _joint_prob_ratio_from_mean_delta(math.log(p), k)


def _mean_min_joint_prob_kept_steps(
    current_ppl: List[float],
    pruned_set: Set[int],
    k_per_step: List[int],
) -> Tuple[float, float]:
    """
    Mean and **min** joint ``P/P_clean`` over kept steps (min = worst step, same role
    as ``max_rel_ppl`` on the PPL side).

    ``current_ppl[j] == exp(mean_delta_j)``; joint_j = ``exp(-K_j * delta_j)``.
    """
    active = [j for j in range(len(current_ppl)) if j not in pruned_set]
    if not active:
        return 1.0, 1.0
    vals: List[float] = []
    for j in active:
        p = max(float(current_ppl[j]), 1e-300)
        delta = math.log(p)
        vals.append(_joint_prob_ratio_from_mean_delta(delta, k_per_step[j]))
    return statistics.mean(vals), min(vals)


def _mean_min_joint_norm_kept_steps(
    current_joint_norm: List[float],
    pruned_set: Set[int],
) -> Tuple[float, float]:
    """Mean and min normalized joint over kept steps (uses stored norms per step)."""
    active = [j for j in range(len(current_joint_norm)) if j not in pruned_set]
    if not active:
        return 1.0, 1.0
    vals = [current_joint_norm[j] for j in active if math.isfinite(current_joint_norm[j])]
    if not vals:
        return 1.0, 1.0
    return statistics.mean(vals), min(vals)


def _prune_accepted_by_threshold(
    *,
    threshold_metric: str,
    threshold_value: float,
    delta: float,
    k_for_joint: int,
) -> Tuple[bool, float, float]:
    """
    Returns (accept_prune, rel_ppl_exp_delta, joint_prob_ratio).

    ``relative_ppl``: accept if ``exp(delta) <= threshold_value``.
    ``joint_prob_ratio``: accept if ``exp(-K*delta) >= threshold_value``.
    """
    rel = math.exp(float(delta))
    joint = _joint_prob_ratio_from_mean_delta(delta, k_for_joint)
    if threshold_metric == "joint_prob_ratio":
        return joint >= threshold_value, rel, joint
    return rel <= threshold_value, rel, joint


def _pick_stage1_prune_candidate(
    step_scores: Dict[int, float],
    step_j_stars: Dict[int, int],
    threshold_metric: str,
    _k_predict_per_step: List[int],
    joint_for_pick: Optional[Dict[int, float]] = None,
) -> int:
    """
    Candidate ``i`` to try pruning this round.

    ``relative_ppl``: minimize **maximum** later-step relative PPL (equivalently min
    over ``i`` of ``max_{j > i} exp(delta_{i,j})``).

    ``joint_prob_ratio``: maximize **minimum** later-step joint (maximin).
    With ``joint_for_pick``, that value is normalized joint from the baseline file;
    otherwise legacy ``exp(-K*delta)``.

    Tie-break (joint): lower δ at bottleneck, then lower ``i``. ``relative_ppl``: lower ``i``.
    """

    if threshold_metric == "joint_prob_ratio" and joint_for_pick is not None:
        def sort_key_joint(i: int) -> Tuple[float, ...]:
            jn = joint_for_pick.get(i, float("nan"))
            delta = float(step_scores[i])
            return (-jn, delta, float(i))

        return min(step_scores.keys(), key=sort_key_joint)

    def sort_key(i: int) -> Tuple[float, ...]:
        if threshold_metric == "joint_prob_ratio":
            j = step_j_stars[i]
            k_j = _k_predict_per_step[j] if j >= 0 else 0
            delta = float(step_scores[i])
            joint = _joint_prob_ratio_from_mean_delta(delta, k_j)
            return (-joint, delta, float(i))
        return (float(step_scores[i]), float(i))

    return min(step_scores.keys(), key=sort_key)


def _pick_stage2_prune_candidate(
    step_deltas: Dict[int, float],
    threshold_metric: str,
    k_tgt: int,
) -> int:
    """
    Candidate ``i`` to try pruning for the current target step.

    ``relative_ppl``: minimize mean CE delta on the target.
    ``joint_prob_ratio``: maximize ``exp(-K_tgt * delta)`` on the target (for fixed
    ``K_tgt``, same ordering as minimizing delta).
    """

    def sort_key(i: int) -> Tuple[float, ...]:
        d = step_deltas[i]
        if threshold_metric == "joint_prob_ratio":
            jt = _joint_prob_ratio_from_mean_delta(d, k_tgt)
            return (-jt, d, float(i))
        return (d, float(i))

    return min(step_deltas.keys(), key=sort_key)


@torch.no_grad()
def run_greedy_until_ppl_threshold(
    model,
    clean_ids: torch.Tensor,
    clean_token_ranges: List[Tuple[int, int]],
    pert_ids_list: List[torch.Tensor],
    pert_token_ranges_list: List[List[Tuple[int, int]]],
    clean_step_losses: List[float],
    threshold_value: float,
    k_predict_per_step: List[int],
    threshold_metric: str,
    *,
    joint_baseline: Optional[JointBaselineTable] = None,
    ppl_curve: List[dict] | None = None,
    edge_accumulator: List[int] | None = None,
) -> Tuple[List[int], Set[int], List[dict], bool, List[float], List[float]]:
    """
    Same structure as ``run_greedy_ablation`` / ``run_gsnp.py`` but if the chosen prune
    fails the threshold test, that prune is skipped and the loop stops.

    Candidate selection matches ``run_gsnp.py``: for ``relative_ppl``, ``i_star``
    minimizes the **maximum** later-step relative PPL; for ``joint_prob_ratio``,
    ``i_star`` maximizes the **minimum** later-step joint ``P/P_clean`` (bottleneck).
    Threshold tests use that same aggregate on the winning candidate.

    ``threshold_metric == "relative_ppl"``: reject if ``max_j rel_ppl_j > threshold``.
    ``threshold_metric == "joint_prob_ratio"``: reject if
    ``min_j joint_j < threshold_value`` (bottleneck joint over later steps).

    If ``ppl_curve`` and ``edge_accumulator`` (length-1 list, mutated) are given, appends
    one checkpoint per successful prune: edge count uses ``+(N - round_num)`` for stage 1,
    and updates running per-step relative PPL (later steps only) from the winning
    candidate's avg diffs.

    Returns
    (pruned_order, pruned_set, rounds, stopped_by_threshold, final_step_ppls,
    final_joint_norms). ``final_joint_norms[j]`` is normalized joint on step ``j`` after
    stage 1 (from the baseline file when ``joint_baseline`` is set and the metric is
    joint; else derived from ``final_step_ppls`` via legacy ``exp(-K*delta)``).
    """
    N = len(clean_token_ranges)
    num_pert = len(pert_ids_list)
    total_possible_edges = N * (N - 1) // 2
    pruned_order: List[int] = []
    pruned_set: Set[int] = set()
    candidates: Set[int] = set(range(N - 1))
    round_records: List[dict] = []
    stopped_by_threshold = False
    current_ppl = [1.0] * N
    current_joint_norm = [1.0] * N
    if joint_baseline is not None:
        for j in range(N):
            lc = joint_baseline.log_p_clean_for(j)
            if lc is None:
                continue
            jn = joint_baseline.joint_norm(lc, j)
            if math.isfinite(jn):
                current_joint_norm[j] = jn

    joint_for_pick: Dict[int, float] = {}

    while candidates:
        round_num = len(pruned_order) + 1
        step_scores: Dict[int, float] = {}
        step_j_stars: Dict[int, int] = {}
        cache_avg_diffs: Dict[int, Dict[int, float]] = {}
        cache_avg_logp: Dict[int, Dict[int, float]] = {}
        joint_for_pick.clear()

        for i in tqdm(sorted(candidates), desc=f"greedy round {round_num}"):
            sum_diffs: Dict[int, float] = {}
            sum_logps: Dict[int, float] = {}

            for pert_idx, pert_ids in enumerate(pert_ids_list):
                pert_ranges = pert_token_ranges_list[pert_idx]
                subst_steps = {i, *pruned_set}
                spans = [pert_ranges[j] for j in sorted(subst_steps)]
                mix_start = query_mix_start_for_kv_substitution(
                    i, subst_steps, clean_token_ranges
                )
                batch = torch.cat([clean_ids, pert_ids], dim=0)
                with PerturbedKVPostSourceContext(model, spans, mix_start):
                    logits = model(input_ids=batch, use_cache=False).logits[0]

                for j in range(i + 1, N):
                    if j in pruned_set:
                        continue
                    s, e = clean_token_ranges[j]
                    loss = calculate_step_loss(logits, clean_ids[0], s, e).item()
                    diff = loss - clean_step_losses[j]
                    sum_diffs[j] = sum_diffs.get(j, 0.0) + diff
                    if threshold_metric == "joint_prob_ratio" and joint_baseline is not None:
                        lp = sum_logprob_at_step(logits, clean_ids[0], s, e)
                        sum_logps[j] = sum_logps.get(j, 0.0) + lp

                del logits, batch

            if not sum_diffs:
                step_scores[i] = 0.0
                step_j_stars[i] = -1
                if threshold_metric == "joint_prob_ratio" and joint_baseline is not None:
                    joint_for_pick[i] = 1.0
                continue

            avg_diffs = {j: v / num_pert for j, v in sum_diffs.items()}
            cache_avg_diffs[i] = avg_diffs
            if threshold_metric == "joint_prob_ratio" and joint_baseline is not None:
                avg_logp = {j: sum_logps[j] / num_pert for j in sum_logps}
                cache_avg_logp[i] = avg_logp
                joints = {
                    j: joint_baseline.joint_norm(avg_logp[j], j)
                    for j in avg_logp
                }
                j_bottleneck = min(joints.keys(), key=lambda jj: joints[jj])
                step_scores[i] = avg_diffs[j_bottleneck]
                step_j_stars[i] = j_bottleneck
                joint_for_pick[i] = joints[j_bottleneck]
            elif threshold_metric == "joint_prob_ratio":
                joints = {
                    j: _joint_prob_ratio_from_mean_delta(
                        avg_diffs[j], k_predict_per_step[j]
                    )
                    for j in avg_diffs
                }
                j_bottleneck = min(joints.keys(), key=lambda jj: joints[jj])
                step_scores[i] = avg_diffs[j_bottleneck]
                step_j_stars[i] = j_bottleneck
            else:
                j_ppl = max(avg_diffs, key=avg_diffs.__getitem__)
                step_scores[i] = avg_diffs[j_ppl]
                step_j_stars[i] = j_ppl

        i_star = _pick_stage1_prune_candidate(
            step_scores,
            step_j_stars,
            threshold_metric,
            k_predict_per_step,
            joint_for_pick
            if threshold_metric == "joint_prob_ratio" and joint_baseline is not None
            else None,
        )
        winner_avg = cache_avg_diffs.get(i_star, {})
        if threshold_metric == "joint_prob_ratio":
            if not winner_avg:
                score_star = 0.0
                j_star = -1
                k_joint = 0
                rel_ppl = 1.0
                joint_r = 1.0
                accept = False
            elif joint_baseline is not None:
                avg_lp_w = cache_avg_logp.get(i_star, {})
                joints_w = {
                    j: joint_baseline.joint_norm(avg_lp_w[j], j)
                    for j in winner_avg
                    if j in avg_lp_w
                }
                if not joints_w:
                    score_star = 0.0
                    j_star = -1
                    k_joint = 0
                    rel_ppl = 1.0
                    joint_r = 1.0
                    accept = False
                else:
                    j_star = min(joints_w.keys(), key=lambda jj: joints_w[jj])
                    joint_r = joints_w[j_star]
                    score_star = winner_avg[j_star]
                    k_joint = k_predict_per_step[j_star]
                    rel_ppl = math.exp(score_star)
                    accept = joint_r >= threshold_value
            else:
                joints_w = {
                    j: _joint_prob_ratio_from_mean_delta(
                        winner_avg[j], k_predict_per_step[j]
                    )
                    for j in winner_avg
                }
                j_star = min(joints_w.keys(), key=lambda jj: joints_w[jj])
                joint_r = joints_w[j_star]
                score_star = winner_avg[j_star]
                k_joint = k_predict_per_step[j_star]
                rel_ppl = math.exp(score_star)
                accept = joint_r >= threshold_value
        else:
            score_star = step_scores[i_star]
            j_star = step_j_stars[i_star]
            k_joint = k_predict_per_step[j_star] if j_star >= 0 else 0
            accept, rel_ppl, joint_r = _prune_accepted_by_threshold(
                threshold_metric=threshold_metric,
                threshold_value=threshold_value,
                delta=score_star,
                k_for_joint=k_joint,
            )

        if not accept:
            stopped_by_threshold = True
            metric_lbl = (
                "joint_norm"
                if threshold_metric == "joint_prob_ratio"
                and joint_baseline is not None
                else (
                    "joint P/P_clean"
                    if threshold_metric == "joint_prob_ratio"
                    else "ppl_ratio"
                )
            )
            val_lbl = f"{joint_r:.6g}" if threshold_metric == "joint_prob_ratio" else f"{rel_ppl:.4f}"
            cmp_lbl = "<" if threshold_metric == "joint_prob_ratio" else ">"
            step_hint = ""
            if threshold_metric == "joint_prob_ratio" and winner_avg:
                if joint_baseline is not None and i_star in cache_avg_logp:
                    lpw = cache_avg_logp[i_star]
                    joints_w = {
                        j: joint_baseline.joint_norm(lpw[j], j)
                        for j in winner_avg
                        if j in lpw
                    }
                else:
                    joints_w = {
                        j: _joint_prob_ratio_from_mean_delta(
                            winner_avg[j], k_predict_per_step[j]
                        )
                        for j in winner_avg
                    }
                if joints_w:
                    jb = min(joints_w.keys(), key=lambda jj: joints_w[jj])
                    step_hint = f" bottleneck_joint_step={jb} "
            elif threshold_metric == "relative_ppl" and j_star >= 0:
                step_hint = f" worst_rel_ppl_step={j_star} "
            print(
                f"  [Stage1] Round {round_num}: candidate prune step {i_star} would give "
                f"{step_hint}{metric_lbl}={val_lbl} {cmp_lbl} threshold {threshold_value}; "
                f"not pruning it (state unchanged). Stopping stage 1."
            )
            break

        # Only j > i_star appear in ``winner_avg``; earlier kept indices keep prior
        # values — valid under causal attention (no read of KV from steps > j at j).
        for j, d in winner_avg.items():
            current_ppl[j] = math.exp(d)
        if joint_baseline is not None and i_star in cache_avg_logp:
            for j, lp in cache_avg_logp[i_star].items():
                jn = joint_baseline.joint_norm(lp, j)
                if math.isfinite(jn):
                    current_joint_norm[j] = jn

        pruned_order.append(i_star)
        pruned_set.add(i_star)
        candidates.remove(i_star)

        rec: dict = {
            "round": round_num,
            "pruned_step": i_star,
            "max_affected_step": j_star,
            "score": score_star,
            "ppl_ratio": rel_ppl,
            "joint_prob_ratio": (
                joint_r
                if threshold_metric == "joint_prob_ratio"
                else _joint_prob_ratio_from_mean_delta(score_star, k_joint)
            ),
        }
        if threshold_metric == "joint_prob_ratio" and winner_avg:
            j_worst = max(winner_avg.keys(), key=lambda jj: winner_avg[jj])
            d_w = winner_avg[j_worst]
            rec["worst_rel_ppl_later_step"] = j_worst
            rec["worst_rel_ppl_ratio"] = math.exp(d_w)
        elif threshold_metric == "relative_ppl" and winner_avg:
            joints_w = {
                j: _joint_prob_ratio_from_mean_delta(
                    winner_avg[j], k_predict_per_step[j]
                )
                for j in winner_avg
            }
            j_bot = min(joints_w.keys(), key=lambda jj: joints_w[jj])
            rec["bottleneck_joint_later_step"] = j_bot
            rec["bottleneck_joint_prob_ratio"] = joints_w[j_bot]
            rec["bottleneck_rel_ppl_ratio"] = math.exp(winner_avg[j_bot])
        round_records.append(rec)

        if ppl_curve is not None and edge_accumulator is not None:
            edge_accumulator[0] += N - round_num
            ec = edge_accumulator[0]
            mean_p, max_p = _mean_max_rel_ppl_kept_steps(current_ppl, pruned_set)
            if threshold_metric == "joint_prob_ratio" and joint_baseline is not None:
                mean_j, min_j = _mean_min_joint_norm_kept_steps(
                    current_joint_norm, pruned_set
                )
            else:
                mean_j, min_j = _mean_min_joint_prob_kept_steps(
                    current_ppl, pruned_set, k_predict_per_step
                )
            ppl_curve.append({
                "edge_count": ec,
                "edge_sparsity_fraction": ec / total_possible_edges,
                "mean_rel_ppl": mean_p,
                "max_rel_ppl": max_p,
                "mean_joint_prob_ratio": mean_j,
                "min_joint_prob_ratio": min_j,
                "curve_min_joint_worst": min_j,
                "curve_max_rel_ppl_worst": max_p,
                "phase": "stage1",
                "round": round_num,
            })

        if threshold_metric == "joint_prob_ratio":
            jtag = (
                "joint_norm"
                if joint_baseline is not None
                else "joint_P/P_clean"
            )
            print(
                f"  [Stage1] Round {round_num}: pruned step {i_star}  "
                f"bottleneck_joint_step={j_star}  {jtag}={joint_r:.6g}"
            )
        else:
            print(
                f"  [Stage1] Round {round_num}: pruned step {i_star}  "
                f"worst_rel_ppl_step={j_star}  ppl_ratio={rel_ppl:.4f}"
            )

    final_joint_norms = (
        list(current_joint_norm)
        if threshold_metric == "joint_prob_ratio" and joint_baseline is not None
        else [
            _joint_from_rel_ppl(current_ppl[j], k_predict_per_step[j])
            for j in range(N)
        ]
    )
    return (
        pruned_order,
        pruned_set,
        round_records,
        stopped_by_threshold,
        list(current_ppl),
        final_joint_norms,
    )


@torch.no_grad()
def _avg_target_loss_delta(
    model,
    clean_ids: torch.Tensor,
    clean_token_ranges: List[Tuple[int, int]],
    pert_ids_list: List[torch.Tensor],
    pert_token_ranges_list: List[List[Tuple[int, int]]],
    fixed_pruned: Set[int],
    dynamic_pruned: Set[int],
    candidate_i: int,
    target_t: int,
    clean_target_loss: float,
) -> Tuple[float, float]:
    """
    Mean over pert files of (mixed loss on target - clean_target_loss) and mean
    sum log P(token) on the target step span (float64 accumulation in
    ``sum_logprob_at_step``).
    """
    subst = {candidate_i, *fixed_pruned, *dynamic_pruned}
    deltas: List[float] = []
    logps: List[float] = []
    t_s, t_e = clean_token_ranges[target_t]
    for pert_idx, pert_ids in enumerate(pert_ids_list):
        pert_ranges = pert_token_ranges_list[pert_idx]
        spans = [pert_ranges[j] for j in sorted(subst)]
        mix_start = query_mix_start_for_kv_substitution(
            candidate_i, subst, clean_token_ranges
        )
        batch = torch.cat([clean_ids, pert_ids], dim=0)
        with PerturbedKVPostSourceContext(model, spans, mix_start):
            logits = model(input_ids=batch, use_cache=False).logits[0]
        loss = calculate_step_loss(logits, clean_ids[0], t_s, t_e).item()
        deltas.append(loss - clean_target_loss)
        logps.append(sum_logprob_at_step(logits, clean_ids[0], t_s, t_e))
        del logits, batch
    n = len(deltas)
    return sum(deltas) / n, sum(logps) / n


@torch.no_grad()
def backward_perturb_until_ppl_on_target(
    model,
    clean_ids: torch.Tensor,
    clean_token_ranges: List[Tuple[int, int]],
    pert_ids_list: List[torch.Tensor],
    pert_token_ranges_list: List[List[Tuple[int, int]]],
    clean_step_losses: List[float],
    stage1_pruned: Set[int],
    target_t: int,
    threshold_value: float,
    k_predict_per_step: List[int],
    threshold_metric: str,
    *,
    base_step_ppls: List[float],
    base_step_joint_norms: List[float],
    joint_baseline: Optional[JointBaselineTable] = None,
    stage2_plot_snapshot: Dict[str, float],
    ppl_curve: List[dict] | None = None,
    edge_accumulator: List[int] | None = None,
) -> Tuple[Set[int], List[int], bool, List[dict], float, float]:
    """
    Greedily prune among indices < target_t not in stage1_pruned: by min mean CE delta on
    the target (``relative_ppl``) or by the same candidate ordering under
    ``joint_prob_ratio`` (threshold uses normalized joint vs ``joint_baseline`` when
    set, else legacy ``exp(-K*delta)``). Always corrupt stage1_pruned ∪ dynamic.
    Threshold test matches ``run_greedy_until_ppl_threshold``.

    ``base_step_ppls`` is a snapshot (post–stage 1) copied at the start of this target;
    only ``target_t``'s relative PPL is updated on each accepted prune. If
    ``ppl_curve`` / ``edge_accumulator`` are set, each accepted prune adds one edge and
    logs mean/max relative and joint metrics over steps not in ``stage1_pruned``.

    ``stage2_plot_snapshot`` holds ``min_joint`` and ``max_rel_ppl`` (running worst **before**
    this target ends); each new checkpoint stamps ``curve_*_worst`` from it until the
    caller patches the last checkpoint after the run.

    Returns (stage2_pruned_for_target, prune_order, hit_threshold, prune_rounds,
    final_target_rel_ppl, final_target_joint_metric).
    Each entry in prune_rounds is
    ``{"pruned_step", "delta", "ppl_ratio", "joint_prob_ratio"}``.
    """
    N = len(clean_token_ranges)
    total_possible_edges = N * (N - 1) // 2
    step_ppl = list(base_step_ppls)
    step_joint = list(base_step_joint_norms)
    clean_target_loss = clean_step_losses[target_t]
    k_tgt = k_predict_per_step[target_t]
    possible = {i for i in range(target_t) if i not in stage1_pruned}
    candidates = set(possible)
    dynamic: Set[int] = set()
    order: List[int] = []
    prune_rounds: List[dict] = []
    hit = False

    while candidates:
        step_deltas: Dict[int, float] = {}
        step_avg_lp: Dict[int, float] = {}
        for i in tqdm(
            sorted(candidates),
            desc=f"stage2 target={target_t} round {len(dynamic) + 1}",
            leave=False,
        ):
            d, lp = _avg_target_loss_delta(
                model,
                clean_ids,
                clean_token_ranges,
                pert_ids_list,
                pert_token_ranges_list,
                stage1_pruned,
                dynamic,
                i,
                target_t,
                clean_target_loss,
            )
            step_deltas[i] = d
            step_avg_lp[i] = lp

        best_i = _pick_stage2_prune_candidate(
            step_deltas, threshold_metric, k_tgt
        )
        delta = step_deltas[best_i]
        rel_ppl = math.exp(delta)
        if threshold_metric == "joint_prob_ratio" and joint_baseline is not None:
            joint_r = joint_baseline.joint_norm(step_avg_lp[best_i], target_t)
            accept = (
                math.isfinite(joint_r) and joint_r >= threshold_value
            )
        else:
            accept, rel_ppl, joint_r = _prune_accepted_by_threshold(
                threshold_metric=threshold_metric,
                threshold_value=threshold_value,
                delta=delta,
                k_for_joint=k_tgt,
            )

        if not accept:
            hit = True
            metric_lbl = (
                "joint_norm"
                if threshold_metric == "joint_prob_ratio"
                and joint_baseline is not None
                else (
                    "joint P/P_clean"
                    if threshold_metric == "joint_prob_ratio"
                    else "ppl_ratio"
                )
            )
            val_lbl = f"{joint_r:.6g}" if threshold_metric == "joint_prob_ratio" else f"{rel_ppl:.4f}"
            cmp_lbl = "<" if threshold_metric == "joint_prob_ratio" else ">"
            print(
                f"    [Stage2] target={target_t}: candidate prune step {best_i} would give "
                f"{metric_lbl}={val_lbl} {cmp_lbl} threshold {threshold_value}; "
                f"not pruning it (state unchanged). Stopping stage 2 for this target."
            )
            break

        step_ppl[target_t] = rel_ppl
        step_joint[target_t] = joint_r

        dynamic.add(best_i)
        candidates.remove(best_i)
        order.append(best_i)
        prune_rounds.append(
            {
                "pruned_step": best_i,
                "delta": delta,
                "ppl_ratio": rel_ppl,
                "joint_prob_ratio": joint_r,
            }
        )

        if ppl_curve is not None and edge_accumulator is not None:
            edge_accumulator[0] += 1
            ec = edge_accumulator[0]
            mean_p, max_p = _mean_max_rel_ppl_kept_steps(step_ppl, stage1_pruned)
            if threshold_metric == "joint_prob_ratio" and joint_baseline is not None:
                mean_j, min_j = _mean_min_joint_norm_kept_steps(
                    step_joint, stage1_pruned
                )
            else:
                mean_j, min_j = _mean_min_joint_prob_kept_steps(
                    step_ppl, stage1_pruned, k_predict_per_step
                )
            ppl_curve.append({
                "edge_count": ec,
                "edge_sparsity_fraction": ec / total_possible_edges,
                "mean_rel_ppl": mean_p,
                "max_rel_ppl": max_p,
                "mean_joint_prob_ratio": mean_j,
                "min_joint_prob_ratio": min_j,
                "curve_min_joint_worst": float(stage2_plot_snapshot["min_joint"]),
                "curve_max_rel_ppl_worst": float(stage2_plot_snapshot["max_rel_ppl"]),
                "phase": "stage2",
                "target_step": target_t,
            })

        if threshold_metric == "joint_prob_ratio":
            jtag = (
                "joint_norm"
                if joint_baseline is not None
                else "joint_P/P_clean"
            )
            print(
                f"    [Stage2] target={target_t}: pruned step {best_i}  "
                f"{jtag}={joint_r:.6g}"
            )
        else:
            print(
                f"    [Stage2] target={target_t}: pruned step {best_i}  "
                f"ppl_ratio={rel_ppl:.4f}"
            )

    return (
        dynamic,
        order,
        hit,
        prune_rounds,
        float(step_ppl[target_t]),
        float(step_joint[target_t]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedy perturb ablation until PPL threshold, then backward KV ablation per survivor."
    )
    parser.add_argument(
        "--clean_cot_path", type=str, default=rp.CLEAN_COT_DEFAULT
    )
    parser.add_argument(
        "--pert_cot_dir", type=str, default=rp.PERT_DIR_DEFAULT
    )
    parser.add_argument(
        "-n",
        "--num_perturbations",
        type=int,
        default=5,
        metavar="K",
        help="First K perturbed_*.txt files (sorted); average loss over them.",
    )
    thr = parser.add_mutually_exclusive_group()
    thr.add_argument(
        "--ppl-threshold",
        type=float,
        default=argparse.SUPPRESS,
        metavar="T",
        help=(
            "Reject prune if exp(mean CE delta) exceeds T. Default 1.1 if neither "
            "threshold option is given. Mutually exclusive with --joint-prob-threshold."
        ),
    )
    thr.add_argument(
        "--joint-prob-threshold",
        type=float,
        default=argparse.SUPPRESS,
        metavar="T",
        help=(
            "Reject prune if normalized joint (from --joint-baseline-json) is below T. "
            "Mutually exclusive with --ppl-threshold."
        ),
    )
    parser.add_argument(
        "--joint-baseline-json",
        type=str,
        default=rp.JOINT_STEP_LOGPROBS_JSON,
        help=(
            "Path to joint_step_logprobs.json (run_joint_baselines.py). "
            "Required to exist when using --joint-prob-threshold."
        ),
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "Path for full results JSON. Default: results/gsnp_gsep_ppl.json or "
            "results/gsnp_gsep_joint_prob.json from threshold mode."
        ),
    )
    parser.add_argument(
        "--no-output-json",
        action="store_true",
        help="Do not write JSON (curve data would only exist in memory).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    )
    parser.add_argument("--flash_attn", action="store_true")
    args = parser.parse_args()
    if args.num_perturbations < 1:
        parser.error("-n must be at least 1.")
    if "joint_prob_threshold" in vars(args):
        threshold_metric = "joint_prob_ratio"
        threshold_value = float(args.joint_prob_threshold)
    else:
        threshold_metric = "relative_ppl"
        threshold_value = float(getattr(args, "ppl_threshold", 1.1))

    output_json_path = args.output_json
    if output_json_path is None:
        output_json_path = (
            rp.GSNP_GSEP_JOINT_JSON
            if threshold_metric == "joint_prob_ratio"
            else rp.GSNP_GSEP_PPL_JSON
        )

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_path,
        use_flash_attention_2=args.flash_attn,
    )
    device = next(model.parameters()).device
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    clean_cot = Path(args.clean_cot_path).read_text(encoding="utf-8")
    think_content = extract_think_block(clean_cot)
    clean_steps = parse_reasoning_steps(think_content)
    if not clean_steps:
        raise ValueError("No reasoning steps in think block.")
    N = len(clean_steps)
    print(f"Parsed {N} reasoning steps.")

    clean_ids = tokenizer(
        clean_cot, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    clean_token_ranges = map_steps_to_tokens(clean_cot, clean_steps, tokenizer)
    try:
        assert_reasoning_token_ranges_match_think_inner(
            clean_cot, tokenizer, clean_token_ranges
        )
    except ValueError as e:
        raise ValueError(f"Clean think/step alignment: {e}") from e
    if clean_token_ranges[-1][1] > clean_ids.shape[1]:
        raise ValueError(
            f"Reasoning end {clean_token_ranges[-1][1]} > seq len {clean_ids.shape[1]}."
        )

    pert_ids_list, pert_token_ranges_list, pert_names = load_aligned_perturbed_cots(
        clean_cot,
        clean_token_ranges,
        tokenizer,
        Path(args.pert_cot_dir),
        args.num_perturbations,
        device,
        clean_ids.shape[1],
    )
    print(f"Loaded {len(pert_ids_list)} perturbation file(s): {pert_names}")

    print("Computing clean baseline losses...")
    clean_step_losses = compute_clean_step_losses(
        model, clean_ids, clean_token_ranges
    )
    k_predict_per_step = _k_predict_tokens_per_step(clean_token_ranges)

    joint_baseline: Optional[JointBaselineTable] = None
    joint_baseline_json_recorded: Optional[str] = None
    if threshold_metric == "joint_prob_ratio":
        jb_path = Path(args.joint_baseline_json).expanduser().resolve()
        if not jb_path.is_file():
            raise FileNotFoundError(
                "Joint threshold mode requires an existing --joint-baseline-json "
                f"({jb_path} missing). Run causal_cot/run_joint_baselines.py first."
            )
        joint_baseline = load_joint_baseline_json(jb_path)
        joint_baseline_json_recorded = str(jb_path)
        if joint_baseline.num_steps != N:
            raise ValueError(
                f"{jb_path}: num_steps={joint_baseline.num_steps} != rollout steps {N}."
            )

    total_possible_edges = N * (N - 1) // 2
    edge_accumulator = [0]
    ppl_curve: List[dict] = [
        {
            "edge_count": 0,
            "edge_sparsity_fraction": 0.0,
            "mean_rel_ppl": 1.0,
            "max_rel_ppl": 1.0,
            "mean_joint_prob_ratio": 1.0,
            "min_joint_prob_ratio": 1.0,
            "curve_min_joint_worst": 1.0,
            "curve_max_rel_ppl_worst": 1.0,
            "phase": "stage1_base",
        }
    ]

    # --- Stage 1 ---
    print(
        f"\n=== Stage 1: greedy ablation (metric={threshold_metric}, "
        f"threshold={threshold_value}) ==="
    )
    (
        pruned_order,
        stage1_pruned,
        stage1_rounds,
        stopped_thr,
        final_step_ppls,
        final_joint_norms,
    ) = run_greedy_until_ppl_threshold(
        model,
        clean_ids,
        clean_token_ranges,
        pert_ids_list,
        pert_token_ranges_list,
        clean_step_losses,
        threshold_value,
        k_predict_per_step,
        threshold_metric,
        joint_baseline=joint_baseline,
        ppl_curve=ppl_curve,
        edge_accumulator=edge_accumulator,
    )
    kept_stage1 = sorted(set(range(N)) - stage1_pruned)
    print(f"\n[Stage1] Kept step indices (not pruned in stage 1): {kept_stage1}")
    if stopped_thr:
        _thr_lbl = "joint prob" if threshold_metric == "joint_prob_ratio" else "PPL"
        print(f"[Stage1] Stopped due to {_thr_lbl} threshold.")
    else:
        print("[Stage1] Finished all non-final candidates (no threshold stop).")

    last_s1_cp = ppl_curve[-1]
    stage2_curve_worst: Dict[str, float] = {
        "min_joint": float(last_s1_cp["min_joint_prob_ratio"]),
        "max_rel_ppl": float(last_s1_cp["max_rel_ppl"]),
    }

    # --- Stage 2: targets = survivors, high index first (final backward) ---
    targets = sorted(kept_stage1, reverse=True)
    stage2_report: List[dict] = []

    print(
        f"\n=== Stage 2: backward KV ablation per kept step "
        f"(metric={threshold_metric}, threshold {threshold_value}) ==="
    )
    for target_t in targets:
        possible_before = {i for i in range(target_t) if i not in stage1_pruned}
        if not possible_before:
            kept_here = []
            print(
                f"\n[Stage2] Target step {target_t}: no earlier non-stage1-pruned steps; "
                f"kept (pool empty): []"
            )
            stage2_report.append({
                "target_step": target_t,
                "kept_relative_to_stage1_before_target": [],
                "stage2_pruned_order": [],
                "stopped_on_ppl_threshold": False,
                "stage2_prune_rounds": [],
            })
            rel_fin = float(final_step_ppls[target_t])
            j_fin = float(final_joint_norms[target_t])
            stage2_curve_worst["min_joint"] = min(stage2_curve_worst["min_joint"], j_fin)
            stage2_curve_worst["max_rel_ppl"] = max(
                stage2_curve_worst["max_rel_ppl"], rel_fin
            )
            continue

        plot_snap = {
            "min_joint": stage2_curve_worst["min_joint"],
            "max_rel_ppl": stage2_curve_worst["max_rel_ppl"],
        }
        idx_before = len(ppl_curve)
        dynamic, s2_order, hit, s2_rounds, rel_fin, j_fin = (
            backward_perturb_until_ppl_on_target(
                model,
                clean_ids,
                clean_token_ranges,
                pert_ids_list,
                pert_token_ranges_list,
                clean_step_losses,
                stage1_pruned,
                target_t,
                threshold_value,
                k_predict_per_step,
                threshold_metric,
                base_step_ppls=list(final_step_ppls),
                base_step_joint_norms=list(final_joint_norms),
                joint_baseline=joint_baseline,
                stage2_plot_snapshot=plot_snap,
                ppl_curve=ppl_curve,
                edge_accumulator=edge_accumulator,
            )
        )
        stage2_curve_worst["min_joint"] = min(stage2_curve_worst["min_joint"], j_fin)
        stage2_curve_worst["max_rel_ppl"] = max(
            stage2_curve_worst["max_rel_ppl"], rel_fin
        )
        if len(ppl_curve) > idx_before:
            ppl_curve[-1]["curve_min_joint_worst"] = stage2_curve_worst["min_joint"]
            ppl_curve[-1]["curve_max_rel_ppl_worst"] = stage2_curve_worst[
                "max_rel_ppl"
            ]
        kept_here = sorted(possible_before - dynamic)
        print(
            f"\n[Stage2] Target step {target_t}: "
            f"kept among {{steps < {target_t} not pruned in stage 1}} = {kept_here}"
        )
        if not hit and dynamic:
            print(
                f"    (exhausted candidates before threshold stop; "
                f"pruned in stage2 for this target: {sorted(dynamic)})"
            )
        stage2_report.append({
            "target_step": target_t,
            "kept_relative_to_stage1_before_target": kept_here,
            "stage2_pruned_order": s2_order,
            "stopped_on_ppl_threshold": hit,
            "stage2_prune_rounds": s2_rounds,
        })

    if not args.no_output_json:
        out_path = Path(output_json_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _num_note = (
            "Checkpoint metrics are IEEE float64 in JSON; model CE uses reduced "
            "precision (e.g. bfloat16) before .item(). "
        )
        if joint_baseline is not None:
            _num_note += (
                f"Joint fields use normalized token likelihood vs joint_baseline_json "
                f"(joint_metric_version={JOINT_METRIC_VERSION}). "
            )
        _num_note += (
            "Legacy CE-based joint exp(-K*delta) can underflow to 0.0 near ~1e-308."
        )
        out_obj: Dict[str, object] = {
            "ppl_threshold": (
                threshold_value if threshold_metric == "relative_ppl" else None
            ),
            "joint_prob_threshold": (
                threshold_value if threshold_metric == "joint_prob_ratio" else None
            ),
            "threshold_prune_config": {
                "threshold_metric": threshold_metric,
                "threshold_value": threshold_value,
                "numeric_precision_note": _num_note,
            },
            "prediction_token_count_per_step": {
                str(t): k_predict_per_step[t] for t in range(N)
            },
            "num_perturbations": len(pert_ids_list),
            "pert_files": pert_names,
            "stage1_pruned_order": pruned_order,
            "stage1_pruned_set": sorted(stage1_pruned),
            "stage1_kept": kept_stage1,
            "stage1_stopped_on_threshold": stopped_thr,
            "stage1_rounds": stage1_rounds,
            "stage2_by_target": stage2_report,
            "edge_sparsity_ppl_curve": {
                "num_reasoning_steps": N,
                "total_possible_directed_edges_i_lt_j": total_possible_edges,
                "checkpoints": ppl_curve,
            },
        }
        if joint_baseline is not None:
            out_obj["joint_metric_version"] = JOINT_METRIC_VERSION
            out_obj["joint_baseline_json"] = joint_baseline_json_recorded
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=2)
        print(f"\nWrote JSON to {out_path}")


if __name__ == "__main__":
    main()
