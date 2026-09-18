#!/usr/bin/env python3
"""
Run MLM perturbation on a CoT rollout to produce structurally identical
perturbed variants for training (SSEP / JSEP). Requires the LLM tokenizer
for alignment checks (same token length as original).
"""

import argparse
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from causal_cot import paths as rp
from causal_cot.load_model import load_tokenizer_only
from causal_cot.perturbations.mlm import generate_perturbed_cots


def main():
    parser = argparse.ArgumentParser(
        description="Generate perturbed CoT variants via MLM (same token length as original)"
    )
    parser.add_argument(
        "input",
        type=str,
        default=rp.CLEAN_COT_DEFAULT,
        nargs="?",
        help="Input CoT file path (default: rollout.txt at repo root)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=rp.PERT_DIR_DEFAULT,
        help="Output directory: writes original.txt and perturbed_0.txt, ...",
    )
    parser.add_argument(
        "--max-replacements",
        type=int,
        default=20,
        dest="max_replacements",
        help="Top-k MLM predictions to consider per word (default: 20)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        dest="max_examples",
        help="Number of perturbed variants to generate (default: 5)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible word order",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    original_cot = input_path.read_text(encoding="utf-8")

    print("Loading LLM tokenizer (for alignment checks)...")
    tokenizer = load_tokenizer_only()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "original.txt").write_text(original_cot, encoding="utf-8")
    print(f"Saved original.txt to {out_dir}")

    print(f"Generating {args.max_examples} perturbed variant(s)...")
    perturbed_list = generate_perturbed_cots(
        original_cot,
        tokenizer,
        num_variants=args.max_examples,
        max_replacements_per_token=args.max_replacements,
        seed=args.seed,
        output_dir=out_dir,
    )

    print(f"Wrote original and {len(perturbed_list)} perturbed CoT(s) to {out_dir}")


if __name__ == "__main__":
    main()
