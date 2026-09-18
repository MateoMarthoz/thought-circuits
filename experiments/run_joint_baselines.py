#!/usr/bin/env python3
"""
Precompute per-step log joint likelihoods for normalized joint metrics.

For each reasoning step j (0-based), records:
  - log_p_clean[j]: sum of log P(token | prefix) on prediction positions in step j
    under a full clean forward.
  - log_p_base[j]: same under KV splice: all earlier reasoning steps use
    ``perturbed_0`` branch; only defined for j >= 1 (step 0 has no predecessors).

Output: ``results/joint_prob_output/joint_step_logprobs.json`` (override with ``--output-json``).

Downstream: ``run_gsnp --prune-metric joint_prob_ratio``, ``run_gsep --joint-baseline-json``, etc.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from causal_cot import paths as rp
from causal_cot.interventions.kv_perturbation import PerturbedKVPostSourceContext
from causal_cot.joint_token_likelihood import (
    JOINT_METRIC_VERSION,
    mix_start_for_predecessor_substitution,
    sum_logprob_at_step,
)
from causal_cot.load_model import load_model_and_tokenizer
from causal_cot.step_utils import (
    assert_matching_reasoning_step_counts,
    assert_reasoning_token_ranges_match_think_inner,
    extract_think_block,
    map_steps_to_tokens,
    parse_reasoning_steps,
)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write joint_step_logprobs.json (log P clean and log P base per step)."
    )
    parser.add_argument(
        "--clean_cot_path",
        type=str,
        default=rp.CLEAN_COT_DEFAULT,
    )
    parser.add_argument(
        "--perturbed_path",
        type=str,
        default=str(Path(rp.PERT_DIR_DEFAULT) / "perturbed_0.txt"),
        help="Single perturbed CoT (default: perturbed_output/perturbed_0.txt).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=rp.JOINT_STEP_LOGPROBS_JSON,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    )
    parser.add_argument("--flash_attn", action="store_true")
    args = parser.parse_args()

    clean_cot = Path(args.clean_cot_path).read_text(encoding="utf-8")
    pert_path = Path(args.perturbed_path)
    pert_text = pert_path.read_text(encoding="utf-8")

    think_c = extract_think_block(clean_cot)
    steps_c = parse_reasoning_steps(think_c)
    if not steps_c:
        raise SystemExit("No reasoning steps in clean CoT.")
    N = len(steps_c)

    try:
        assert_matching_reasoning_step_counts(clean_cot, [pert_text])
    except ValueError as e:
        raise SystemExit(f"Alignment: {e}") from e

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_path,
        use_flash_attention_2=args.flash_attn,
    )
    device = next(model.parameters()).device
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    clean_ids = tokenizer(
        clean_cot, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    step_ranges = map_steps_to_tokens(clean_cot, steps_c, tokenizer)
    try:
        assert_reasoning_token_ranges_match_think_inner(
            clean_cot, tokenizer, step_ranges
        )
    except ValueError as e:
        raise SystemExit(f"Clean think alignment: {e}") from e

    pert_steps = parse_reasoning_steps(extract_think_block(pert_text))
    pert_ranges = map_steps_to_tokens(pert_text, pert_steps, tokenizer)
    try:
        assert_reasoning_token_ranges_match_think_inner(
            pert_text, tokenizer, pert_ranges
        )
    except ValueError as e:
        raise SystemExit(f"Pert think alignment: {e}") from e
    if len(pert_ranges) != len(step_ranges):
        raise SystemExit("Perturbed vs clean step count mismatch.")
    for j, (a, b) in enumerate(zip(pert_ranges, step_ranges)):
        if a != b:
            raise SystemExit(f"Step {j} token span mismatch clean vs pert.")

    pert_ids = tokenizer(
        pert_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    if pert_ids.shape != clean_ids.shape:
        raise SystemExit(
            f"Perturbed length {pert_ids.shape} != clean {clean_ids.shape}."
        )

    labels = clean_ids[0]

    print("Clean forward (all steps log P)...")
    with torch.no_grad():
        clean_out = model(input_ids=clean_ids, use_cache=False)
    cl = clean_out.logits[0] if clean_out.logits.dim() == 3 else clean_out.logits
    log_p_clean: Dict[str, float] = {}
    for j in range(N):
        s, e = step_ranges[j]
        log_p_clean[str(j)] = sum_logprob_at_step(cl, labels, s, e)
    del clean_out, cl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log_p_base: Dict[str, float] = {}
    print("P_base forwards (j >= 1)...")
    for j in range(1, N):
        spans, mix_start = mix_start_for_predecessor_substitution(step_ranges, j)
        batch = torch.cat([clean_ids, pert_ids], dim=0)
        with PerturbedKVPostSourceContext(model, spans, mix_start):
            out = model(input_ids=batch, use_cache=False)
        logits = out.logits[0] if out.logits.dim() == 3 else out.logits
        s, e = step_ranges[j]
        log_p_base[str(j)] = sum_logprob_at_step(logits, labels, s, e)
        del out, logits, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  step {j}: log_p_base={log_p_base[str(j)]:.6f}")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "joint_metric_version": JOINT_METRIC_VERSION,
        "description": (
            "Per reasoning step (0-based): log_p_clean = sum log P(token|prefix) on "
            "prediction slice; log_p_base[j] for j>=1 = same under KV splice from "
            "perturbed_0 on all steps < j."
        ),
        "model_id": args.model_path,
        "clean_cot_path": str(Path(args.clean_cot_path).resolve()),
        "perturbed_path": str(pert_path.resolve()),
        "num_steps": N,
        "log_p_clean_per_step": log_p_clean,
        "log_p_base_per_step": log_p_base,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
