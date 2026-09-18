#!/usr/bin/env python3
"""
Greedy causal ablation via perturbation, measuring impact on all later steps.

At each round, for every non-pruned candidate step i (steps 0..N-2):
  - For each perturbed CoT file (one forward per file; never mix K/V across files):
      * Batch ``[clean_ids, pert_ids]``. Splice **from that file's** token spans for
        steps {i} ∪ pruned_set (``map_steps_to_tokens`` on the pert text; must match
        clean per-step spans). All corrupt K/V for that forward come from row 1.
      * ``query_mix_start`` from ``query_mix_start_for_kv_substitution``: post-source
        after step ``i`` when there are no prior prunes; else earliest substituted
        span start so no query reads a pruned step with clean K/V.
      * Compute the loss diff vs. the clean CoT for every non-pruned later
        step j > i (loss on clean token ranges and labels).
  - Average the loss diffs over perturbation files to get avg_diff[j].

  - ``relative_ppl``: for each candidate ``i``, let ``j*(i)`` be the later step with
    **largest** mean CE δ (worst relative PPL ``exp(δ)``). Prune the ``i`` that
    **minimizes** that worst-case δ (minimax over later steps).

  - ``joint_prob_ratio``: requires ``--joint-baseline-json`` (from ``run_joint_baselines.py``).
    For each later step ``j``, average sum of log P(token|prefix) over perturbation files,
    then normalized joint ``(p-p_base)/(p_clean-p_base)`` vs that baseline file.
    Bottleneck = ``min_j`` normalized joint; prune ``i`` that **maximizes** it (maximin).

  - ``score`` in JSON/CSV is always mean CE δ at the **primary** aggregate later step
    (worst-δ step for ``relative_ppl``, bottleneck-joint step for ``joint_prob_ratio``).
    Secondary tail aggregates: ``worst_rel_ppl_ratio`` (max later rel PPL for the
    chosen prune) and ``bottleneck_joint_prob_ratio`` (min later joint), in JSON and CSV
    when computable.

Record:
  - x = fraction of non-final steps pruned so far
  - ``ppl_ratio`` = ``exp(score)`` at the **primary** step (max rel PPL at worst-δ
    step for ``relative_ppl``; rel PPL at the **bottleneck** step for ``joint_prob_ratio``).
  - ``worst_rel_ppl_ratio`` / ``bottleneck_joint_prob_ratio`` = cross-metric tail values.
  - Embedded ``gsnp.png`` y = **minimax** max later-step rel PPL (same series as
    ``plot_gsnp`` relative-PPL PNG).

Output (default ``--output_dir`` under ``results/``: ``gsnp_ppl_output`` or
``gsnp_joint_prob_output`` from ``--prune-metric``):
  <output_dir>/gsnp.json
  <output_dir>/gsnp.csv
  <output_dir>/gsnp.png

  Root ``prune_metric`` records which ranking was used; each round includes primary
  and secondary fields for both metrics where applicable.

CLI: use ``-n`` / ``--num_perturbations`` to set how many ``perturbed_*.txt`` files
to load (sorted by name); averages metrics over those files. Use ``--prune-metric``
to choose how each round picks which candidate step to remove.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from causal_cot import paths as rp
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
    assert_matching_reasoning_step_counts,
    assert_reasoning_token_ranges_match_think_inner,
    extract_think_block,
    map_steps_to_tokens,
    parse_reasoning_steps,
)


def _k_predict_tokens_per_step(
    clean_token_ranges: List[Tuple[int, int]],
) -> List[int]:
    """K per step: prediction positions in ``calculate_step_loss`` (``e - s - 1``)."""
    out: List[int] = []
    for s, e in clean_token_ranges:
        out.append(max(0, e - s - 1))
    return out


def _joint_prob_ratio_from_mean_delta(delta: float, k: int) -> float:
    """Legacy CE-based joint ``exp(-K * delta)`` (cross-metric fields on rel-PPL runs)."""
    if k <= 0:
        return 1.0
    return math.exp(-float(k) * float(delta))


def _pick_prune_candidate(
    step_scores: Dict[int, float],
    step_j_stars: Dict[int, int],
    prune_metric: str,
    k_predict_per_step: List[int],
    joint_for_pick: Optional[Dict[int, float]] = None,
) -> int:
    """
    Return candidate index ``i`` to prune this round.

    ``relative_ppl``: ``step_scores[i]`` is max later δ for ``i``; minimize it.
    ``joint_prob_ratio``: maximize ``joint_for_pick[i]`` (normalized joint at bottleneck);
    tie-break: lower ``step_scores[i]`` (δ), then lower ``i``.
    """

    if prune_metric == "joint_prob_ratio" and joint_for_pick is not None:
        def sort_key_joint(i: int) -> Tuple[float, ...]:
            jn = joint_for_pick.get(i, float("nan"))
            delta = step_scores[i]
            return (-jn, delta, float(i))

        return min(step_scores.keys(), key=sort_key_joint)

    def sort_key(i: int) -> Tuple[float, ...]:
        delta = step_scores[i]
        j = step_j_stars[i]
        k_j = k_predict_per_step[j] if j >= 0 else 0
        if prune_metric == "joint_prob_ratio":
            joint = _joint_prob_ratio_from_mean_delta(delta, k_j)
            return (-joint, delta, float(i))
        return (delta, float(i))

    return min(step_scores.keys(), key=sort_key)


# ---------------------------------------------------------------------------
# Clean baseline: per-step losses
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main greedy loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_greedy_ablation(
    model,
    clean_ids: torch.Tensor,
    clean_token_ranges: List[Tuple[int, int]],
    pert_ids_list: List[torch.Tensor],
    pert_token_ranges_list: List[List[Tuple[int, int]]],
    clean_step_losses: List[float],
    *,
    prune_metric: str = "relative_ppl",
    joint_baseline: Optional[JointBaselineTable] = None,
) -> Tuple[List[int], List[float], List[float], List[dict]]:
    """
    Returns:
        pruned_order : step indices in the order they were pruned
        xs           : sparsity values (fraction pruned) for the scatter plot
        ys           : relative perplexity values for the scatter plot
        round_records: list of per-round dicts for CSV/JSON output
    """
    N = len(clean_token_ranges)
    num_pert = len(pert_ids_list)
    if len(pert_token_ranges_list) != num_pert:
        raise ValueError("pert_token_ranges_list length must match pert_ids_list.")

    pruned_order: List[int] = []
    pruned_set: Set[int] = set()
    candidates: Set[int] = set(range(N - 1))  # final step is never a candidate
    k_predict_per_step = _k_predict_tokens_per_step(clean_token_ranges)

    xs: List[float] = [0.0]
    ys: List[float] = [1.0]
    round_records: List[dict] = []

    joint_for_pick: Dict[int, float] = {}

    while candidates:
        round_num = len(pruned_order) + 1
        step_scores: Dict[int, float] = {}
        step_j_stars: Dict[int, int] = {}
        cache_avg_diffs: Dict[int, Dict[int, float]] = {}
        cache_avg_logp: Dict[int, Dict[int, float]] = {}

        for i in tqdm(sorted(candidates), desc=f"round {round_num}"):
            # sum_diffs[j] accumulates loss diffs over perturbation files
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
                    if prune_metric == "joint_prob_ratio" and joint_baseline is not None:
                        lp = sum_logprob_at_step(logits, clean_ids[0], s, e)
                        sum_logps[j] = sum_logps.get(j, 0.0) + lp

                del logits, batch

            if not sum_diffs:
                # i is the penultimate step and all later steps are pruned
                step_scores[i] = 0.0
                step_j_stars[i] = -1
                continue

            avg_diffs = {j: v / num_pert for j, v in sum_diffs.items()}
            cache_avg_diffs[i] = avg_diffs
            if prune_metric == "joint_prob_ratio" and joint_baseline is not None:
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
            elif prune_metric == "joint_prob_ratio":
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
                j_worst = max(avg_diffs, key=avg_diffs.__getitem__)
                step_scores[i] = avg_diffs[j_worst]
                step_j_stars[i] = j_worst

        i_star = _pick_prune_candidate(
            step_scores,
            step_j_stars,
            prune_metric,
            k_predict_per_step,
            joint_for_pick
            if prune_metric == "joint_prob_ratio" and joint_baseline is not None
            else None,
        )
        score_star = step_scores[i_star]
        j_star = step_j_stars[i_star]
        k_at_star = k_predict_per_step[j_star] if j_star >= 0 else 0
        ppl_ratio = math.exp(score_star)
        if (
            prune_metric == "joint_prob_ratio"
            and joint_baseline is not None
            and i_star in joint_for_pick
        ):
            joint_ratio = joint_for_pick[i_star]
        else:
            joint_ratio = _joint_prob_ratio_from_mean_delta(score_star, k_at_star)

        winner_avg = cache_avg_diffs.get(i_star, {})
        round_rec: dict = {
            "round": round_num,
            "pruned_step": i_star,
            "max_affected_step": j_star,
            "score": score_star,
            "ppl_ratio": ppl_ratio,
            "joint_prob_ratio": joint_ratio,
            "prune_metric": prune_metric,
            "all_scores": {str(i): step_scores[i] for i in sorted(step_scores)},
        }
        if prune_metric == "joint_prob_ratio":
            round_rec["bottleneck_joint_prob_ratio"] = joint_ratio
            if winner_avg:
                j_worst = max(winner_avg.keys(), key=lambda jj: winner_avg[jj])
                d_w = winner_avg[j_worst]
                round_rec["worst_rel_ppl_later_step"] = j_worst
                round_rec["worst_rel_ppl_ratio"] = math.exp(d_w)
            else:
                # No later kept steps: minimax rel PPL over an empty tail is undefined;
                # use 1.0 (no extra degradation vs clean).
                round_rec["worst_rel_ppl_later_step"] = -1
                round_rec["worst_rel_ppl_ratio"] = 1.0
        else:
            round_rec["worst_rel_ppl_ratio"] = ppl_ratio
            if winner_avg:
                joints_w = {
                    j: _joint_prob_ratio_from_mean_delta(
                        winner_avg[j], k_predict_per_step[j]
                    )
                    for j in winner_avg
                }
                j_bot = min(joints_w.keys(), key=lambda jj: joints_w[jj])
                d_b = winner_avg[j_bot]
                round_rec["bottleneck_joint_later_step"] = j_bot
                round_rec["bottleneck_joint_prob_ratio"] = joints_w[j_bot]
                round_rec["bottleneck_rel_ppl_ratio"] = math.exp(d_b)
            else:
                round_rec["bottleneck_joint_later_step"] = -1
                round_rec["bottleneck_joint_prob_ratio"] = joint_ratio
                round_rec["bottleneck_rel_ppl_ratio"] = ppl_ratio

        pruned_order.append(i_star)
        pruned_set.add(i_star)
        candidates.remove(i_star)

        xs.append(len(pruned_set) / (N - 1))
        # Embedded PPL curve: always max later-step rel PPL for the chosen prune
        # (minimax aggregate). For joint-prune runs that is ``worst_rel_ppl_ratio``;
        # ``ppl_ratio`` is rel PPL at the bottleneck step instead.
        rel_y_plot = (
            float(round_rec["worst_rel_ppl_ratio"])
            if prune_metric == "joint_prob_ratio"
            else ppl_ratio
        )
        ys.append(rel_y_plot)

        if prune_metric == "joint_prob_ratio":
            jlabel = (
                "joint_norm=(p-pb)/(pc-pb)"
                if joint_baseline is not None
                else "joint_exp(-K*delta)"
            )
            print(
                f"  Round {round_num}: pruned step {i_star}  "
                f"bottleneck_joint_step={j_star}  {jlabel}={joint_ratio:.6g}"
            )
        else:
            print(
                f"  Round {round_num}: pruned step {i_star}  "
                f"worst_rel_ppl_step={j_star}  ppl_ratio={ppl_ratio:.4f}"
            )

        round_records.append(round_rec)

    return pruned_order, xs, ys, round_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "gsnp: greedy causal ablation (perturbation only). Each round removes the "
            "candidate least harmful under --prune-metric (see module docstring)."
        )
    )
    parser.add_argument(
        "--clean_cot_path",
        type=str,
        default=rp.CLEAN_COT_DEFAULT,
        help="Path to the original (clean) CoT text file.",
    )
    parser.add_argument(
        "--pert_cot_dir",
        type=str,
        default=rp.PERT_DIR_DEFAULT,
        help="Directory containing perturbed_0.txt … perturbed_N.txt files.",
    )
    parser.add_argument(
        "-n",
        "--num_perturbations",
        type=int,
        default=5,
        metavar="K",
        help=(
            "Take the first K files matching perturbed_*.txt in --pert_cot_dir "
            "(sorted by name) and average loss deltas over them (default: 5)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory for gsnp.json, gsnp.csv, gsnp.png. Default: "
            "results/gsnp_ppl_output or results/gsnp_joint_prob_output from "
            "--prune-metric."
        ),
    )
    parser.add_argument(
        "--model_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    )
    parser.add_argument(
        "--flash_attn", action="store_true",
        help="Use flash_attention_2 for the model.",
    )
    parser.add_argument(
        "--prune-metric",
        type=str,
        choices=("relative_ppl", "joint_prob_ratio"),
        default="relative_ppl",
        help=(
            "How to rank candidates each round: relative_ppl minimizes max later-step "
            "relative PPL; joint_prob_ratio uses normalized token-likelihood joint "
            "(requires --joint-baseline-json from run_joint_baselines.py)."
        ),
    )
    parser.add_argument(
        "--joint-baseline-json",
        type=str,
        default=rp.JOINT_STEP_LOGPROBS_JSON,
        help=(
            "Path to joint_step_logprobs.json (run_joint_baselines.py). "
            "Required when --prune-metric joint_prob_ratio."
        ),
    )
    args = parser.parse_args()
    if args.num_perturbations < 1:
        parser.error("--num_perturbations / -n must be at least 1.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            rp.GSNP_JOINT_OUTPUT
            if args.prune_metric == "joint_prob_ratio"
            else rp.GSNP_PPL_OUTPUT
        )

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_path,
        use_flash_attention_2=args.flash_attn,
    )
    device = next(model.parameters()).device
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # ------------------------------------------------------------------
    # Parse clean CoT
    # ------------------------------------------------------------------
    clean_cot = Path(args.clean_cot_path).read_text(encoding="utf-8")
    think_content = extract_think_block(clean_cot)
    clean_steps = parse_reasoning_steps(think_content)
    if not clean_steps:
        raise ValueError("No reasoning steps found in <think> block.")
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
        raise ValueError(f"Think / step token alignment: {e}") from e
    if clean_token_ranges[-1][1] > clean_ids.shape[1]:
        raise ValueError(
            f"Reasoning end token {clean_token_ranges[-1][1]} exceeds sequence length "
            f"{clean_ids.shape[1]}."
        )

    # ------------------------------------------------------------------
    # Load perturbation files
    # ------------------------------------------------------------------
    pert_dir = Path(args.pert_cot_dir)
    pert_files = sorted(pert_dir.glob("perturbed_*.txt"))[: args.num_perturbations]
    if not pert_files:
        raise ValueError(f"No perturbed_*.txt files found in {pert_dir}")
    if len(pert_files) < args.num_perturbations:
        print(
            f"Warning: requested {args.num_perturbations} perturbations "
            f"but only found {len(pert_files)}."
        )
    pert_texts = [p.read_text(encoding="utf-8") for p in pert_files]
    for t in pert_texts:
        try:
            assert_matching_reasoning_step_counts(clean_cot, [t])
        except ValueError as e:
            raise ValueError(f"Perturbed CoT alignment: {e}") from e
    pert_ids_list: List[torch.Tensor] = []
    pert_token_ranges_list: List[List[Tuple[int, int]]] = []
    for p, t in zip(pert_files, pert_texts):
        pert_steps = parse_reasoning_steps(extract_think_block(t))
        pert_ranges = map_steps_to_tokens(t, pert_steps, tokenizer)
        try:
            assert_reasoning_token_ranges_match_think_inner(
                t, tokenizer, pert_ranges
            )
        except ValueError as e:
            raise ValueError(f"{p.name}: think / step token alignment: {e}") from e
        if len(pert_ranges) != len(clean_token_ranges):
            raise ValueError(
                f"{p.name}: {len(pert_ranges)} step spans vs clean {len(clean_token_ranges)}."
            )
        for j, (pr_span, cr_span) in enumerate(
            zip(pert_ranges, clean_token_ranges)
        ):
            if pr_span != cr_span:
                raise ValueError(
                    f"{p.name}: step {j} pert token span {pr_span} != clean {cr_span} "
                    "(K/V splice uses one index space; texts must align per step)."
                )
        ids = tokenizer(
            t, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        if pert_ranges[-1][1] > ids.shape[1]:
            raise ValueError(
                f"{p.name}: reasoning end token {pert_ranges[-1][1]} exceeds "
                f"sequence length {ids.shape[1]}."
            )
        if ids.shape[1] != clean_ids.shape[1]:
            raise ValueError(
                f"{p.name}: token length {ids.shape[1]} != clean {clean_ids.shape[1]}."
            )
        pert_ids_list.append(ids)
        pert_token_ranges_list.append(pert_ranges)
    print(f"Loaded {len(pert_ids_list)} perturbation file(s).")

    # ------------------------------------------------------------------
    # Clean baseline: per-step losses
    # ------------------------------------------------------------------
    print("Computing clean baseline losses for all steps...")
    clean_step_losses = compute_clean_step_losses(
        model, clean_ids, clean_token_ranges
    )
    for j, loss in enumerate(clean_step_losses):
        print(f"  step {j}: clean_loss={loss:.4f}")

    # ------------------------------------------------------------------
    # Greedy ablation
    # ------------------------------------------------------------------
    joint_baseline: Optional[JointBaselineTable] = None
    if args.prune_metric == "joint_prob_ratio":
        jb_path = Path(args.joint_baseline_json).expanduser().resolve()
        if not jb_path.is_file():
            raise ValueError(
                f"--prune-metric joint_prob_ratio requires {jb_path} "
                "(run causal_cot/run_joint_baselines.py first)."
            )
        joint_baseline = load_joint_baseline_json(jb_path)
        if joint_baseline.num_steps != N:
            raise ValueError(
                f"Baseline num_steps={joint_baseline.num_steps} != CoT steps {N}."
            )

    print(
        f"\nStarting greedy ablation (prune_metric={args.prune_metric}, "
        f"{N - 1} candidates, {len(pert_ids_list)} perturbation(s) per candidate)..."
    )
    pruned_order, xs, ys, round_records = run_greedy_ablation(
        model=model,
        clean_ids=clean_ids,
        clean_token_ranges=clean_token_ranges,
        pert_ids_list=pert_ids_list,
        pert_token_ranges_list=pert_token_ranges_list,
        clean_step_losses=clean_step_losses,
        prune_metric=args.prune_metric,
        joint_baseline=joint_baseline,
    )

    print(f"\nPruning order: {pruned_order}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "gsnp.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_steps": N,
                "num_perturbations": len(pert_ids_list),
                "prune_metric": args.prune_metric,
                "joint_metric_version": (
                    JOINT_METRIC_VERSION
                    if args.prune_metric == "joint_prob_ratio"
                    and joint_baseline is not None
                    else None
                ),
                "joint_baseline_json": (
                    str(Path(args.joint_baseline_json).expanduser().resolve())
                    if joint_baseline is not None
                    else None
                ),
                "pruned_order": pruned_order,
                "rounds": round_records,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {json_path}")

    csv_path = out_dir / "gsnp.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "pruned_step",
            "max_affected_step",
            "score",
            "ppl_ratio",
            "joint_prob_ratio",
            "prune_metric",
            "worst_rel_ppl_ratio",
            "bottleneck_joint_prob_ratio",
        ])
        for rec in round_records:
            writer.writerow([
                rec["round"],
                rec["pruned_step"],
                rec["max_affected_step"],
                rec["score"],
                rec["ppl_ratio"],
                rec["joint_prob_ratio"],
                rec["prune_metric"],
                rec["worst_rel_ppl_ratio"],
                rec["bottleneck_joint_prob_ratio"],
            ])
    print(f"CSV saved to {csv_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys, s=30, color="steelblue", zorder=3)
    plt.plot(xs, ys, color="steelblue", lw=0.8, alpha=0.5)
    plt.axhline(1.0, color="gray", linestyle="--", lw=0.8)
    plt.xlabel("Sparsity (fraction of non-final steps pruned)")
    plt.ylabel("Max relative PPL over later steps (minimax tail)")
    plt.title(
        f"Greedy Causal Ablation (Perturbation): PPL vs Sparsity "
        f"(prune_metric={args.prune_metric})"
    )
    plt.tight_layout()
    plot_path = out_dir / "gsnp.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
