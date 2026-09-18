#!/usr/bin/env python3
"""
Greedy backward ablation for every reasoning-step target (0 .. N-1) and four
intervention types: attention suppression, resampling (text rebuild from
resampled step strings), perturbation (K/V splice from aligned perturbed sequences),
and **perturb_text** (same text-rebuild mechanism as resample, but step strings
come from ``perturbed_*.txt``). For target T, candidates are steps {0,..,T-1};
each round picks the candidate with lowest mean loss delta on step T (averaged
over corruptions where applicable). No plots — all metrics are saved to JSON.

Output: one JSON with clean baselines, per-target / per-method prune orders, and
per-round candidate mean deltas, PPL ratios, and per-corruption deltas (resample,
perturb, perturb_text). Default write location is ``results/gsep_joint_prob_output``
/ ``gsep_with_joint_prob.json`` when ``--joint-baseline-json`` loads successfully;
otherwise ``results/gsep_ppl_output`` / ``gsep.json`` (override with ``--output_dir``
/ ``--output_json``).

Invariants (no cross-target leakage; cumulative pruning only within one target):
- **Clean slate per target T**: Each method calls ``run_*_for_target`` with fresh
  ``pruned`` / ``candidates``; nothing from target T-1 carries into target T.
- **Within target T, prunes accumulate**: Every trial applies all of ``pruned``
  plus the current candidate ``i`` (attention spans, resample / perturb_text
  rebuild, or K/V substitution set). After each round, ``best_i`` is added to ``pruned`` and
  removed from ``candidates``.
- **Target step T is never the intervention**: Candidates are ``0..T-1`` only, so
  step T’s own tokens / K-V are never replaced or suppressed as a “source” span;
  loss is always at step T (clean indices for attn_supp and perturb; rebuilt
  ``new_ranges[T]`` for resample and perturb_text after alignment checks).
- **Structural alignment**: Clean and perturbed CoTs must share per-step token
  spans (enforced at load). Resample rebuilds use fixed inter-step ``gaps`` from
  the clean CoT; ``len(new_ranges) == num_steps`` and think-inner assertions
  catch tokenizer / parsing drift.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from tqdm import tqdm

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from causal_cot import paths as rp
from causal_cot.interventions.attention_suppression import CumulativeAttnSuppressionContext
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
from causal_cot.load_model import MODEL_ID, load_model_and_tokenizer
from causal_cot.metrics import calculate_step_loss
from causal_cot.step_utils import (
    assert_matching_reasoning_step_counts,
    assert_reasoning_token_ranges_match_think_inner,
    extract_think_block,
    map_steps_to_char_ranges,
    map_steps_to_tokens,
    parse_reasoning_steps,
)
from experiments import run_backward_ablation as rba


def _round_record(
    round_index: int,
    candidates: Set[int],
    step_deltas: Dict[int, float],
    best_i: int,
    step_raw: Optional[Dict[int, List[float]]] = None,
) -> dict:
    cand = sorted(candidates)
    rec: dict = {
        "round_index": round_index,
        "chosen_pruned_step": best_i,
        "mean_delta_chosen": step_deltas[best_i],
        "ppl_ratio_chosen": math.exp(step_deltas[best_i]),
        "candidate_mean_delta": {str(i): step_deltas[i] for i in cand},
        "candidate_ppl_ratio": {str(i): math.exp(step_deltas[i]) for i in cand},
    }
    if step_raw is not None:
        rec["candidate_per_corruption_deltas"] = {
            str(i): step_raw[i] for i in cand
        }
    return rec


def run_attn_supp_for_target(
    model: torch.nn.Module,
    clean_ids: torch.Tensor,
    step_ranges: List[Tuple[int, int]],
    full_kv_cache,
    target_t: int,
    clean_loss_t: float,
    device: torch.device,
) -> dict:
    tgt_start, tgt_end = step_ranges[target_t]
    candidates: Set[int] = set(range(target_t))
    pruned: Set[int] = set()
    rounds: List[dict] = []

    round_idx = 0
    while candidates:
        step_deltas: Dict[int, float] = {}
        for i in tqdm(
            sorted(candidates),
            desc=f"attn_supp target={target_t} round {round_idx + 1}",
            leave=False,
        ):
            spans = [step_ranges[j] for j in pruned] + [step_ranges[i]]
            prefix_len = step_ranges[i][0]
            suffix_ids = clean_ids[:, prefix_len:]
            suffix_len = suffix_ids.shape[1]
            position_ids = torch.arange(
                prefix_len, prefix_len + suffix_len, device=device
            ).unsqueeze(0)
            prefix_cache = rba._slice_kv_cache(full_kv_cache, prefix_len)

            with CumulativeAttnSuppressionContext(model, spans):
                with torch.no_grad():
                    out = model(
                        input_ids=suffix_ids,
                        past_key_values=prefix_cache,
                        position_ids=position_ids,
                        use_cache=False,
                    )
            logits = out.logits[0] if out.logits.dim() == 3 else out.logits
            local_start = tgt_start - prefix_len
            local_end = tgt_end - prefix_len
            suffix_labels = clean_ids[0, prefix_len:]
            step_deltas[i] = (
                calculate_step_loss(
                    logits, suffix_labels, local_start, local_end
                ).item()
                - clean_loss_t
            )
            del out, logits, prefix_cache

        best_i = min(step_deltas, key=step_deltas.get)
        rounds.append(
            _round_record(round_idx, candidates, step_deltas, best_i, None)
        )
        pruned.add(best_i)
        candidates.remove(best_i)
        round_idx += 1

    return {
        "target_step": target_t,
        "clean_baseline_loss": clean_loss_t,
        "num_prunable_steps": target_t,
        "prune_order": [r["chosen_pruned_step"] for r in rounds],
        "rounds": rounds,
    }


def run_resample_for_target(
    method_name: str,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    steps: List[str],
    gaps: List[str],
    corrupt_step_texts: List[List[str]],
    target_t: int,
    clean_loss_t: float,
    num_steps: int,
) -> dict:
    candidates: Set[int] = set(range(target_t))
    pruned: Set[int] = set()
    rounds: List[dict] = []
    round_idx = 0

    while candidates:
        step_deltas: Dict[int, float] = {}
        step_raw: Dict[int, List[float]] = {}
        for i in tqdm(
            sorted(candidates),
            desc=f"{method_name} target={target_t} round {round_idx + 1}",
            leave=False,
        ):
            deltas: List[float] = []
            for variant_steps in corrupt_step_texts:
                working = list(steps)
                working[i] = variant_steps[i]
                for j in pruned:
                    working[j] = variant_steps[j]
                text = rba._rebuild_text(working, gaps)
                ids = tokenizer(
                    text, return_tensors="pt", add_special_tokens=False
                ).input_ids.to(device)
                with torch.no_grad():
                    out = model(input_ids=ids, use_cache=False)
                logits = out.logits[0] if out.logits.dim() == 3 else out.logits
                think_new = extract_think_block(text)
                new_steps = parse_reasoning_steps(think_new)
                new_ranges = map_steps_to_tokens(text, new_steps, tokenizer)
                assert_reasoning_token_ranges_match_think_inner(
                    text, tokenizer, new_ranges
                )
                if new_ranges[-1][1] > ids.shape[1]:
                    raise ValueError(
                        "Rebuilt CoT: reasoning end exceeds tokenized length."
                    )
                if len(new_ranges) != num_steps:
                    raise ValueError(
                        f"Rebuilt CoT has {len(new_ranges)} steps, expected {num_steps}."
                    )
                f_start, f_end = new_ranges[target_t]
                loss = calculate_step_loss(logits, ids[0], f_start, f_end).item()
                deltas.append(loss - clean_loss_t)
                del out, logits, ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            step_deltas[i] = sum(deltas) / len(deltas)
            step_raw[i] = deltas

        best_i = min(step_deltas, key=step_deltas.get)
        rounds.append(
            _round_record(round_idx, candidates, step_deltas, best_i, step_raw)
        )
        pruned.add(best_i)
        candidates.remove(best_i)
        round_idx += 1

    return {
        "target_step": target_t,
        "clean_baseline_loss": clean_loss_t,
        "num_prunable_steps": target_t,
        "prune_order": [r["chosen_pruned_step"] for r in rounds],
        "rounds": rounds,
    }


def run_kv_perturb_for_target(
    method_name: str,
    model: torch.nn.Module,
    clean_ids: torch.Tensor,
    step_ranges: List[Tuple[int, int]],
    pert_ids_list: List[torch.Tensor],
    pert_token_ranges_list: List[List[Tuple[int, int]]],
    target_t: int,
    clean_loss_t: float,
    joint_baseline: Optional[JointBaselineTable] = None,
) -> dict:
    t_start, t_end = step_ranges[target_t]
    candidates: Set[int] = set(range(target_t))
    pruned: Set[int] = set()
    rounds: List[dict] = []
    round_idx = 0

    while candidates:
        step_deltas: Dict[int, float] = {}
        step_raw: Dict[int, List[float]] = {}
        step_joint_norm: Dict[int, float] = {}
        for i in tqdm(
            sorted(candidates),
            desc=f"{method_name} target={target_t} round {round_idx + 1}",
            leave=False,
        ):
            deltas: List[float] = []
            logps: List[float] = []
            for pert_idx, pert_ids in enumerate(pert_ids_list):
                pert_ranges = pert_token_ranges_list[pert_idx]
                subst = {i, *pruned}
                spans = [pert_ranges[j] for j in sorted(subst)]
                mix_start = query_mix_start_for_kv_substitution(
                    i, subst, step_ranges
                )
                batch = torch.cat([clean_ids, pert_ids], dim=0)
                with PerturbedKVPostSourceContext(model, spans, mix_start):
                    with torch.no_grad():
                        out = model(input_ids=batch)
                logits = out.logits[0] if out.logits.dim() == 3 else out.logits
                loss = calculate_step_loss(
                    logits, clean_ids[0], t_start, t_end
                ).item()
                deltas.append(loss - clean_loss_t)
                if joint_baseline is not None:
                    logps.append(
                        sum_logprob_at_step(
                            logits, clean_ids[0], t_start, t_end
                        )
                    )
                del out, logits, batch
            step_deltas[i] = sum(deltas) / len(deltas)
            step_raw[i] = deltas
            if joint_baseline is not None and logps:
                avg_lp = sum(logps) / len(logps)
                step_joint_norm[i] = joint_baseline.joint_norm(avg_lp, target_t)

        best_i = min(step_deltas, key=step_deltas.get)
        rec = _round_record(round_idx, candidates, step_deltas, best_i, step_raw)
        if joint_baseline is not None and step_joint_norm:
            cand = sorted(candidates)
            rec["joint_prob_ratio_normalized_chosen"] = float(
                step_joint_norm[best_i]
            )
            rec["candidate_joint_prob_ratio_normalized"] = {
                str(i): float(step_joint_norm[i]) for i in cand if i in step_joint_norm
            }
        rounds.append(rec)
        pruned.add(best_i)
        candidates.remove(best_i)
        round_idx += 1

    return {
        "target_step": target_t,
        "clean_baseline_loss": clean_loss_t,
        "num_prunable_steps": target_t,
        "prune_order": [r["chosen_pruned_step"] for r in rounds],
        "rounds": rounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="gsep: backward ablation for every target step and four interventions (JSON only)."
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
        help="Directory containing perturbed_*.txt files.",
    )
    parser.add_argument(
        "--resample_cot_dir",
        type=str,
        default=rp.RESAMPLE_DIR_DEFAULT,
        help="Directory containing resampled_*.txt files.",
    )
    parser.add_argument(
        "--num_corruptions",
        type=int,
        default=1,
        help="Number of corruption variants to average (resample / perturb / perturb_text).",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="",
        help=(
            "Hugging Face model id or local path for tokenizer+model "
            f"(default: {MODEL_ID}). Recorded in JSON as model_id."
        ),
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        help="Use flash_attention_2 for the model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory for JSON output. Default: results/gsep_joint_prob_output when "
            "a joint baseline file loads, else results/gsep_ppl_output."
        ),
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "Filename inside output_dir. Default: gsep_with_joint_prob.json with "
            "baseline, else gsep.json."
        ),
    )
    parser.add_argument(
        "--joint-baseline-json",
        type=str,
        default=rp.JOINT_STEP_LOGPROBS_JSON,
        help=(
            "If this file exists and matches num_steps, perturb-method rounds include "
            "normalized joint fields (joint_prob_ratio_normalized_chosen, …)."
        ),
    )
    args = parser.parse_args()

    resolved_model_id = args.model_id.strip() or None
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_id=resolved_model_id,
        use_flash_attention_2=args.flash_attn,
    )
    recorded_model_id = resolved_model_id or MODEL_ID
    device = next(model.parameters()).device

    clean_cot = Path(args.clean_cot_path).read_text(encoding="utf-8")
    think_content = extract_think_block(clean_cot)
    steps = parse_reasoning_steps(think_content)
    if not steps:
        raise ValueError("No reasoning steps found in <think> block.")
    num_steps = len(steps)
    print(f"Parsed {num_steps} reasoning steps.")

    char_ranges = map_steps_to_char_ranges(clean_cot, steps)
    step_ranges = map_steps_to_tokens(clean_cot, steps, tokenizer)
    gaps = rba._compute_gaps(clean_cot, char_ranges)

    clean_ids = tokenizer(
        clean_cot, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    seq_len = clean_ids.shape[1]
    try:
        assert_reasoning_token_ranges_match_think_inner(
            clean_cot, tokenizer, step_ranges
        )
    except ValueError as e:
        raise ValueError(f"Think / step token alignment: {e}") from e
    if step_ranges[-1][1] > seq_len:
        raise ValueError(
            f"Reasoning spans end at token {step_ranges[-1][1]} but seq len is {seq_len}."
        )

    pert_dir = Path(args.pert_cot_dir)
    pert_files = sorted(pert_dir.glob("perturbed_*.txt"))[: args.num_corruptions]
    if not pert_files:
        raise ValueError(f"No perturbed_*.txt found in {pert_dir}")
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
            raise ValueError(f"{p.name}: think / step alignment: {e}") from e
        if len(pert_ranges) != len(step_ranges):
            raise ValueError(
                f"{p.name}: {len(pert_ranges)} step spans vs clean {len(step_ranges)}."
            )
        for j, (pr_span, cr_span) in enumerate(zip(pert_ranges, step_ranges)):
            if pr_span != cr_span:
                raise ValueError(
                    f"{p.name}: step {j} pert span {pr_span} != clean {cr_span}."
                )
        ids = tokenizer(
            t, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        if pert_ranges[-1][1] > ids.shape[1]:
            raise ValueError(f"{p.name}: reasoning end exceeds sequence length.")
        if ids.shape[1] != seq_len:
            raise ValueError(
                f"{p.name}: token length {ids.shape[1]} != clean {seq_len}."
            )
        pert_ids_list.append(ids)
        pert_token_ranges_list.append(pert_ranges)
    print(f"Loaded {len(pert_ids_list)} perturbed variant(s).")
    pert_step_texts = [
        rba._load_corrupt_steps(p, num_steps) for p in pert_files
    ]

    resample_dir = Path(args.resample_cot_dir)
    resample_files = sorted(resample_dir.glob("resampled_*.txt"))[
        : args.num_corruptions
    ]
    if not resample_files:
        raise ValueError(f"No resampled_*.txt found in {resample_dir}")
    resample_step_texts = [
        rba._load_corrupt_steps(p, num_steps) for p in resample_files
    ]
    print(f"Loaded {len(resample_step_texts)} resampled variant(s).")

    print("Clean forward (KV cache for attention suppression)...")
    with torch.no_grad():
        clean_out = model(input_ids=clean_ids, use_cache=True)
    clean_logits = (
        clean_out.logits[0] if clean_out.logits.dim() == 3 else clean_out.logits
    )
    full_kv_cache = clean_out.past_key_values

    clean_losses: Dict[int, float] = {}
    prediction_token_count: Dict[int, int] = {}
    for t in range(num_steps):
        s, e = step_ranges[t]
        prediction_token_count[t] = max(0, e - s - 1)
        clean_losses[t] = calculate_step_loss(
            clean_logits, clean_ids[0], s, e
        ).item()
    del clean_logits
    print(
        "Clean baseline CE per step: "
        + ", ".join(f"{t}={clean_losses[t]:.4f}" for t in range(num_steps))
    )

    joint_baseline: Optional[JointBaselineTable] = None
    joint_baseline_json_used: Optional[str] = None
    _jb = Path(args.joint_baseline_json).expanduser().resolve()
    if _jb.is_file():
        joint_baseline = load_joint_baseline_json(_jb)
        joint_baseline_json_used = str(_jb)
        if joint_baseline.num_steps != num_steps:
            raise ValueError(
                f"{_jb}: num_steps={joint_baseline.num_steps} vs rollout {num_steps}."
            )
        print(f"Joint baseline loaded for perturb rounds: {_jb}")
    else:
        print(
            f"Note: {_jb} not found; perturb rounds omit normalized joint fields."
        )

    out_dir = Path(
        args.output_dir
        or (
            rp.GSEP_JOINT_OUTPUT
            if joint_baseline is not None
            else rp.GSEP_PPL_OUTPUT
        )
    )
    out_name = args.output_json or (
        "gsep_with_joint_prob.json"
        if joint_baseline is not None
        else "gsep.json"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "schema_version": 1,
        "description": (
            "Per-target backward greedy ablation: for target T, prune from "
            "steps 0..T-1 minimizing mean delta on loss at step T."
        ),
        "model_id": recorded_model_id,
        "num_steps": num_steps,
        "num_corruptions": args.num_corruptions,
        "perturbed_files": [str(p) for p in pert_files],
        "resampled_files": [str(p) for p in resample_files],
        "clean_baseline_loss_per_step": {str(t): clean_losses[t] for t in range(num_steps)},
        "prediction_token_count_per_step": {
            str(t): prediction_token_count[t] for t in range(num_steps)
        },
        "methods": {
            "attn_supp": {"by_target": {}},
            "resample": {"by_target": {}},
            "perturb": {"by_target": {}},
            "perturb_text": {"by_target": {}},
        },
    }
    if joint_baseline is not None:
        payload["joint_metric_version"] = JOINT_METRIC_VERSION
        payload["joint_baseline_json"] = joint_baseline_json_used

    for target_t in range(num_steps):
        print(f"\n========== Target step {target_t} ==========")
        cl = clean_losses[target_t]

        print("  attn_supp...")
        payload["methods"]["attn_supp"]["by_target"][str(target_t)] = (
            run_attn_supp_for_target(
                model,
                clean_ids,
                step_ranges,
                full_kv_cache,
                target_t,
                cl,
                device,
            )
        )
        torch.cuda.empty_cache()

        print("  resample...")
        payload["methods"]["resample"]["by_target"][str(target_t)] = (
            run_resample_for_target(
                "resample",
                model,
                tokenizer,
                device,
                steps,
                gaps,
                resample_step_texts,
                target_t,
                cl,
                num_steps,
            )
        )
        torch.cuda.empty_cache()

        print("  perturb...")
        payload["methods"]["perturb"]["by_target"][str(target_t)] = (
            run_kv_perturb_for_target(
                "perturb",
                model,
                clean_ids,
                step_ranges,
                pert_ids_list,
                pert_token_ranges_list,
                target_t,
                cl,
                joint_baseline=joint_baseline,
            )
        )
        torch.cuda.empty_cache()

        print("  perturb_text...")
        payload["methods"]["perturb_text"]["by_target"][str(target_t)] = (
            run_resample_for_target(
                "perturb_text",
                model,
                tokenizer,
                device,
                steps,
                gaps,
                pert_step_texts,
                target_t,
                cl,
                num_steps,
            )
        )
        torch.cuda.empty_cache()

    out_path = out_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
