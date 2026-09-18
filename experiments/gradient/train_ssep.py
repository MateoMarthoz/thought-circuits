#!/usr/bin/env python3
"""
Train Single Step Edge Pruning (SSEP) for causal CoT edge discovery.
Loads clean + perturbed CoTs (same reasoning-step count inside think), then runs
``CausalEdgePrunerSSEP.run_single_step`` for each ``--steps`` target k: L0 mix on keys
in steps 1..k-1, faithfulness loss only on step k. Multiple values loop in order (same
as invoking this script once per step with a single ``--steps`` value).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Allow running from repo root or from causal_cot
_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _ROOT.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from causal_cot import paths as rp
from causal_cot.gradient.pruning_ssep import CausalEdgePrunerSSEP
from causal_cot.load_model import load_model_and_tokenizer
from causal_cot.step_utils import (
    assert_matching_reasoning_step_counts,
    extract_think_block,
    get_step_ranges_with_prompt,
)


CONFIGS = {
    "test": {
        "num_optim_steps": 80,
        "target_sparsity": 1.2,
        "edge_learning_rate": 0.8,
        "reg_edge_learning_rate": 0.4,
        "num_sparsity_warmup_steps": 65,
        "warmup_steps": 3,
        "log_alpha_init": 10.0,
        "log_alpha_init_std": 0,
        "use_linear_schedule": False,
        "max_perturbed": 2,
    },
    "full": {
        "num_optim_steps": 500,
        "target_sparsity": 1.2,
        "edge_learning_rate": 0.8,
        "reg_edge_learning_rate": 0.4,
        "num_sparsity_warmup_steps": 420,
        "warmup_steps": 15,
        "log_alpha_init": 10.0,
        "log_alpha_init_std": 0,
        "use_linear_schedule": True,
        "max_perturbed": 5,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Run SSEP (Single Step Edge Pruning) on clean + perturbed CoT sequences."
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=list(CONFIGS.keys()),
        default=None,
        help="Preset: 'test' (quick validation) or 'full' (production). "
        "Individual args override the preset. Default: full.",
    )
    parser.add_argument(
        "input_dir",
        type=str,
        nargs="?",
        default=rp.PERT_DIR_DEFAULT,
        help="Directory containing original.txt and perturbed_0.txt, perturbed_1.txt, ... (default: perturbed_output)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=rp.SSEP_EDGES_JSON,
        help="Output path for discovered edges JSON (default: ssep_edges.json)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=rp.CHECKPOINTS_SSEP,
        help="Directory for interval checkpoints (log_alpha_step_*_interval_*.pt).",
    )
    parser.add_argument(
        "--target-edge-sparsity",
        type=float,
        default=None,
        dest="target_sparsity",
        help="Target edge sparsity τ (default: from --config)",
    )
    parser.add_argument(
        "--num-optim-steps",
        type=int,
        default=None,
        dest="num_optim_steps",
        help="Optimization steps per target step (default: from --config)",
    )
    parser.add_argument(
        "--edge-learning-rate",
        type=float,
        default=None,
        dest="edge_learning_rate",
        help="Edge (log_alpha) learning rate (default: from --config)",
    )
    parser.add_argument(
        "--reg-edge-learning-rate",
        type=float,
        default=None,
        dest="reg_edge_learning_rate",
        help="Regularisation (sparsity λ) learning rate (default: from --config)",
    )
    parser.add_argument(
        "--num-sparsity-warmup-steps",
        type=int,
        default=None,
        dest="num_sparsity_warmup_steps",
        help="Steps to ramp sparsity target 0→τ per node (default: from --config)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        dest="warmup_steps",
        help="LR scheduler warmup steps per node (default: from --config)",
    )
    parser.add_argument(
        "--log-alpha-init",
        type=float,
        default=None,
        dest="log_alpha_init",
        help="Mean for initial log_alpha Normal (default: from --config)",
    )
    parser.add_argument(
        "--log-alpha-init-std",
        type=float,
        default=None,
        dest="log_alpha_init_std",
        help="Std for initial log_alpha Normal (default: from --config)",
    )
    parser.add_argument(
        "--use-linear-schedule",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="use_linear_schedule",
        help="Linear LR decay to end of training (full); omit for constant after warmup (test).",
    )
    parser.add_argument(
        "--max-perturbed",
        type=int,
        default=None,
        dest="max_perturbed",
        help="Max perturbed variants to load (default: from --config)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="*",
        default=None,
        dest="steps",
        metavar="K",
        help="Target reasoning step index/indices for SSEP (required). "
        "Valid range is printed if omitted. Example: --steps 57  or  --steps 3 5 57",
    )
    args = parser.parse_args()

    preset = CONFIGS.get(args.config, CONFIGS["full"])
    for key, value in preset.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    original_path = input_dir / "original.txt"
    if not original_path.is_file():
        raise SystemExit(f"Original CoT not found: {original_path}")

    clean_cot = original_path.read_text(encoding="utf-8")
    perturbed_paths = sorted(
        input_dir.glob("perturbed_*.txt"),
        key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 0,
    )
    if not perturbed_paths:
        raise SystemExit(f"No perturbed_*.txt files found in {input_dir}")

    if args.max_perturbed is not None and len(perturbed_paths) > args.max_perturbed:
        perturbed_paths = perturbed_paths[: args.max_perturbed]

    perturbed_cots = [p.read_text(encoding="utf-8") for p in perturbed_paths]
    try:
        assert_matching_reasoning_step_counts(clean_cot, perturbed_cots)
    except ValueError as e:
        raise SystemExit(f"Alignment error: {e}") from e
    print(
        f"Config: {args.config or 'full (default)'} | "
        f"optim_steps={args.num_optim_steps}, log_alpha_init={args.log_alpha_init} "
        f"(std={args.log_alpha_init_std}), sparsity_warmup={args.num_sparsity_warmup_steps}, "
        f"lr_warmup={args.warmup_steps}, linear_schedule={args.use_linear_schedule}, "
        f"max_perturbed={args.max_perturbed}"
    )
    print(f"Loaded clean CoT and {len(perturbed_cots)} perturbed variant(s).")

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    device = next(model.parameters()).device

    # Freeze LLM parameters
    for param in model.parameters():
        param.requires_grad = False

    # --- CRITICAL FIX: Force Gradient Checkpointing to Activate ---
    # 1. Enable checkpointing with safe reentrance
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 2. Trick the Autograd engine into reaching the checkpoint wrappers
    model.get_input_embeddings().weight.requires_grad_(True)

    # 3. HF silently ignores checkpointing if the model is in eval() mode!
    # Force the model into training mode so the wrappers execute.
    model.train()
    # --------------------------------------------------------------

    print("Extracting <think> block and mapping steps to tokens...")
    step_ranges = get_step_ranges_with_prompt(clean_cot, tokenizer)
    N = len(step_ranges) - 1
    print(f"Step ranges: {len(step_ranges)} (prompt + {N} reasoning steps).")
    if N <= 1:
        raise SystemExit(
            f"Need at least 2 reasoning steps for SSEP (found N={N}); nothing to prune into a target step."
        )

    if not args.steps:
        raise SystemExit(
            "Error: --steps is required. Pass one or more target reasoning step indices "
            f"in the inclusive range 2..{N} (step 1 has no earlier reasoning steps to mix). "
            f"Examples: --steps {N}     --steps 3 {N}     --steps 5 10 {N}"
        )

    for s in args.steps:
        if s < 2 or s > N:
            raise SystemExit(
                f"Error: step index {s} is out of range. "
                f"Valid indices are 2..{N} for this CoT."
            )

    print("Tokenizing perturbed sequences...")
    pert_input_ids_list = []
    for pert_cot in perturbed_cots:
        pert_ids = tokenizer.encode(pert_cot, add_special_tokens=False)
        pert_ids_t = torch.tensor(pert_ids, dtype=torch.long, device=device).unsqueeze(0)
        pert_input_ids_list.append(pert_ids_t)

    labels = tokenizer.encode(clean_cot, add_special_tokens=False)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    clean_input_ids = labels_t.unsqueeze(0)
    clean_len = clean_input_ids.shape[1]
    if step_ranges[N][1] > clean_len:
        raise SystemExit(
            f"Reasoning spans end at token {step_ranges[N][1]} but sequence length is {clean_len}."
        )

    for i, p in enumerate(pert_input_ids_list):
        if p.shape[1] != clean_len:
            raise SystemExit(
                f"Perturbed variant {i} has token length {p.shape[1]}, expected {clean_len} "
                "(must match clean for aligned attention)."
            )

    unique_targets = sorted(set(args.steps))
    print(
        f"Pre-calculating clean baseline CE for target step(s) {unique_targets}..."
    )
    clean_step_losses: dict = {}
    with torch.no_grad():
        clean_outputs = model(input_ids=clean_input_ids)
        clean_logits = clean_outputs.logits[0]
        for k in unique_targets:
            start_idx, end_idx = step_ranges[k]
            c_logits = clean_logits[start_idx : end_idx - 1].contiguous()
            c_labels = clean_input_ids[0, start_idx + 1 : end_idx]
            clean_step_losses[k] = torch.nn.functional.cross_entropy(
                c_logits, c_labels, reduction="mean"
            ).detach()
            del c_logits, c_labels

    # Wipe the massive 2.4 GB logits tensor from global scope
    del clean_outputs, clean_logits
    torch.cuda.empty_cache()

    pruner = CausalEdgePrunerSSEP(
        model=model,
        tokenizer=tokenizer,
        step_ranges=step_ranges,
        target_sparsity=args.target_sparsity,
        temperature=2.0 / 3.0,
        warmup_steps=args.num_sparsity_warmup_steps,
        scheduler_warmup_steps=args.warmup_steps,
        log_alpha_lr=args.edge_learning_rate,
        sparsity_lambda_lr=args.reg_edge_learning_rate,
        log_alpha_init_mean=args.log_alpha_init,
        log_alpha_init_std=args.log_alpha_init_std,
        use_linear_schedule=args.use_linear_schedule,
    )
    pruner.num_optim_steps = args.num_optim_steps

    discovered_edges: list = []
    for k in args.steps:
        print(f"\n=== SSEP: target reasoning step k={k} (loss only on this step) ===")
        edges_k = pruner.run_single_step(
            k,
            clean_input_ids,
            pert_input_ids_list,
            {k: clean_step_losses[k]},
            checkpoint_dir=args.checkpoint_dir,
        )
        print(f"Found {len(edges_k)} edges into step {k}")
        discovered_edges.extend(edges_k)

    print(
        f"\nSSEP complete. Ran {len(args.steps)} target run(s); "
        f"total edges discovered: {len(discovered_edges)}"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"source": s, "target": t} for s, t in discovered_edges],
            f,
            indent=2,
        )
    print(f"Saved {len(discovered_edges)} edges to {out_path}")


if __name__ == "__main__":
    main()
