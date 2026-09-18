# Thought Circuits

Cleaned research code from a final-year dissertation on teacher-forced causal
interventions over reasoning-model chains of thought. It preserves the original
experiment implementations with light path and import cleanup only.

## Layout

- `causal_cot/`: shared loading, parsing, metrics, interventions, and pruning code.
- `experiments/`: rollout, perturbation, greedy-pruning, and gradient-pruning entry points.
- `data/`: one representative rollout plus five aligned perturbation and resampling variants.
- `docs/`: implementation notes and attribution.

Create an environment and install the recorded dependencies with
`pip install -r requirements.txt`. Run entry points from the repository root,
for example `python experiments/run_gsep.py --help`.

Scientific interpretation, limitations, and a fuller reproduction guide will
be documented in a later README pass.
