#!/usr/bin/env python3
"""
Train Joint Step Edge Pruning (JSEP) for causal CoT edge discovery.
Loads clean + perturbed CoT sequences, extracts <think> steps, runs run_optimization,
and saves active edges to JSON.

Use ``--no-node-sparsity`` for edge-only training (no node L0 gates or node
regularization), similar in spirit to edge-only schedules in Edge-Pruning.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _ROOT.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from causal_cot import paths as rp
from causal_cot.gradient.hooks_jsep import CausalEdgePrunerJSEP
from causal_cot.gradient.l0_utils import deterministic_z_from_log_alpha
from causal_cot.load_model import load_model_and_tokenizer
from causal_cot.step_utils import (
    assert_matching_reasoning_step_counts,
    get_step_ranges_with_prompt,
)


CONFIGS = {
    "test": {
        "num_training_steps": 50,
        "target_sparsity": 1.2,
        "edge_learning_rate": 0.8,
        "reg_edge_learning_rate": 0.4,
        "target_node_sparsity": 0.70,
        "node_learning_rate": 0.8,
        "reg_node_learning_rate": 0.4,
        "num_sparsity_warmup_steps": 40,
        "warmup_steps": 2,
        "log_alpha_init": 10.0,
        "use_linear_schedule": False,
        "max_perturbed": 2,
    },
    "full": {
        "num_training_steps": 600,
        "target_sparsity": 1.2,
        "edge_learning_rate": 0.8,
        "reg_edge_learning_rate": 0.4,
        "target_node_sparsity": 0.7,
        "node_learning_rate": 0.8,
        "reg_node_learning_rate": 0.4,
        "num_sparsity_warmup_steps": 500,
        "warmup_steps": 20,
        "log_alpha_init": 10.0,
        "use_linear_schedule": True,
        "max_perturbed": 5,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Run JSEP (Joint Step Edge Pruning) on clean + perturbed CoT sequences."
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=list(CONFIGS.keys()),
        default=None,
        help="Preset config: 'test' (~1.5h quick validation) or 'full' (~10h production run). "
             "Individual args override the preset.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        help="Model ID or path (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)",
    )
    parser.add_argument(
        "--clean_cot_path",
        type=str,
        default=rp.CLEAN_COT_DEFAULT,
        help="Path to the original (clean) CoT text file.",
    )
    parser.add_argument(
        "--pert_cots_dir",
        type=str,
        default=rp.PERT_DIR_DEFAULT,
        help="Directory containing perturbed CoT text files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=rp.JSEP_EDGES_JSON,
        help="Output JSON file path for active edges.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=rp.CHECKPOINTS_JSEP,
        help="Directory for interval checkpoints (log_alpha_interval_*.pt).",
    )
    parser.add_argument("--target-edge-sparsity", type=float, default=None, dest="target_sparsity")
    parser.add_argument("--target-node-sparsity", type=float, default=None, dest="target_node_sparsity")
    parser.add_argument("--num_steps", type=int, default=None, dest="num_training_steps")
    parser.add_argument("--edge-learning-rate", type=float, default=None, dest="edge_learning_rate")
    parser.add_argument("--reg-edge-learning-rate", type=float, default=None, dest="reg_edge_learning_rate")
    parser.add_argument("--node-learning-rate", type=float, default=None, dest="node_learning_rate")
    parser.add_argument("--reg-node-learning-rate", type=float, default=None, dest="reg_node_learning_rate")
    parser.add_argument("--num-sparsity-warmup-steps", type=int, default=None, dest="num_sparsity_warmup_steps")
    parser.add_argument("--warmup-steps", type=int, default=None, dest="warmup_steps")
    parser.add_argument("--log-alpha-init", type=float, default=None, dest="log_alpha_init")
    parser.add_argument("--max-perturbed", type=int, default=None, dest="max_perturbed",
                        help="Cap on number of perturbed variants to load.")
    parser.add_argument(
        "--faith-loss-fn",
        type=str,
        choices=["weighted_sum", "squared_sum"],
        default="weighted_sum",
        dest="faith_loss_fn",
        help=(
            "Faithfulness loss: 'weighted_sum' (default) is sum_k U_k * delta_k; "
            "'squared_sum' is sum_k ReLU(delta_k * U_k)^2."
        ),
    )
    parser.add_argument(
        "--no-node-sparsity",
        action="store_true",
        dest="no_node_sparsity",
        help=(
            "Disable node L0 masks and node sparsity regularization (edge-only JSEP). "
            "Overrides config defaults for node targets and node mask parameters."
        ),
    )
    args = parser.parse_args()

    preset = CONFIGS.get(args.config, CONFIGS["full"])
    for key, value in preset.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    use_node_sparsity = not args.no_node_sparsity

    if use_node_sparsity:
        print(
            f"Config: {args.config or 'full (default)'} | "
            f"steps={args.num_training_steps}, log_alpha_init={args.log_alpha_init}, "
            f"edge_lr={args.edge_learning_rate}, reg_edge_lr={args.reg_edge_learning_rate}, "
            f"node_lr={args.node_learning_rate}, reg_node_lr={args.reg_node_learning_rate}, "
            f"target_node_sparsity={args.target_node_sparsity}, "
            f"faith_loss_fn={args.faith_loss_fn}"
        )
    else:
        print(
            f"Config: {args.config or 'full (default)'} | "
            f"steps={args.num_training_steps}, log_alpha_init={args.log_alpha_init}, "
            f"edge_lr={args.edge_learning_rate}, reg_edge_lr={args.reg_edge_learning_rate}, "
            f"node sparsity: off (--no-node-sparsity; edge-only), "
            f"faith_loss_fn={args.faith_loss_fn}"
        )

    clean_path = Path(args.clean_cot_path)
    if not clean_path.is_file():
        raise SystemExit(f"Clean CoT file not found: {clean_path}")
    pert_dir = Path(args.pert_cots_dir)
    if not pert_dir.is_dir():
        raise SystemExit(f"Perturbed CoTs directory not found: {pert_dir}")

    clean_cot = clean_path.read_text(encoding="utf-8")

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    device = next(model.parameters()).device

    for param in model.parameters():
        param.requires_grad = False

    print("Extracting <think> block and mapping steps to tokens...")
    step_ranges = get_step_ranges_with_prompt(clean_cot, tokenizer)
    N = len(step_ranges) - 1
    print(f"Step ranges: {len(step_ranges)} (prompt + {N} reasoning steps).")

    clean_enc = tokenizer(
        clean_cot,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=False,
    )
    clean_input_ids = clean_enc["input_ids"].to(device)
    seq_len = clean_input_ids.shape[1]
    if step_ranges[N][1] > seq_len:
        raise SystemExit(
            f"Reasoning spans end at token {step_ranges[N][1]} but sequence length is {seq_len}."
        )

    pert_files = sorted(
        pert_dir.glob("perturbed_*.txt"),
        key=lambda p: (
            int(p.stem.split("_")[1])
            if p.stem.split("_")[1].isdigit()
            else 0
        ),
    )
    if not pert_files:
        raise SystemExit(f"No perturbed_*.txt files found in {pert_dir}")

    if args.max_perturbed is not None and len(pert_files) > args.max_perturbed:
        pert_files = pert_files[: args.max_perturbed]

    pert_texts = [p.read_text(encoding="utf-8") for p in pert_files]
    try:
        assert_matching_reasoning_step_counts(clean_cot, pert_texts)
    except ValueError as e:
        raise SystemExit(f"Alignment error: {e}") from e

    pert_input_ids_list = []
    for p, pert_text in zip(pert_files, pert_texts):
        pert_enc = tokenizer(
            pert_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
        )
        pert_ids = pert_enc["input_ids"].to(device)
        if pert_ids.shape[1] != seq_len:
            raise SystemExit(
                f"Perturbed file {p.name} has token length {pert_ids.shape[1]}, "
                f"expected {seq_len} (must match clean CoT)."
            )
        pert_input_ids_list.append(pert_ids)

    print(f"Loaded clean CoT and {len(pert_input_ids_list)} perturbed variant(s).")

    pruner = CausalEdgePrunerJSEP(
        model=model,
        N=N,
        step_ranges=step_ranges,
        target_sparsity=args.target_sparsity,
        temperature=2.0 / 3.0,
        log_alpha_init=args.log_alpha_init,
        target_node_sparsity=args.target_node_sparsity,
        use_node_sparsity=use_node_sparsity,
    )

    print("Running JSEP optimization (with gradient accumulation)...")
    pruner.run_optimization(
        clean_input_ids,
        pert_input_ids_list,
        faith_loss_fn=args.faith_loss_fn,
        num_training_steps=args.num_training_steps,
        warmup_steps=args.num_sparsity_warmup_steps,
        scheduler_warmup_steps=args.warmup_steps,
        log_alpha_lr=args.edge_learning_rate,
        sparsity_lambda_lr=args.reg_edge_learning_rate,
        node_log_alpha_lr=args.node_learning_rate,
        sparsity_lambda_nodes_lr=args.reg_node_learning_rate,
        use_linear_schedule=args.use_linear_schedule,
        checkpoint_dir=args.checkpoint_dir,
    )

    final_Z = deterministic_z_from_log_alpha(pruner.log_alpha.detach())
    active_edges = []
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if final_Z[i, j] > 0:
                active_edges.append([int(i), int(j)])

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(active_edges, f, indent=2)
    print(f"Saved {len(active_edges)} active edges to {out_path}")


if __name__ == "__main__":
    main()
