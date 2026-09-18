"""Repository-relative default paths for experiment arguments."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

CLEAN_COT_DEFAULT = str(DATA_DIR / "rollout.txt")
PERT_DIR_DEFAULT = str(DATA_DIR / "perturbed")
RESAMPLE_DIR_DEFAULT = str(DATA_DIR / "resampled")

GSNP_PPL_OUTPUT = str(RESULTS_DIR / "gsnp_ppl_output")
GSNP_JOINT_OUTPUT = str(RESULTS_DIR / "gsnp_joint_prob_output")
GSEP_PPL_OUTPUT = str(RESULTS_DIR / "gsep_ppl_output")
GSEP_JOINT_OUTPUT = str(RESULTS_DIR / "gsep_joint_prob_output")
BACKWARD_ABLATION_OUTPUT = str(RESULTS_DIR / "backward_ablation_output")
GSNP_GSEP_PPL_JSON = str(RESULTS_DIR / "gsnp_gsep_ppl.json")
GSNP_GSEP_JOINT_JSON = str(RESULTS_DIR / "gsnp_gsep_joint_prob.json")
CHECKPOINTS_JSEP = str(RESULTS_DIR / "checkpoints_jsep")
CHECKPOINTS_SSEP = str(RESULTS_DIR / "checkpoints_ssep")
JSEP_EDGES_JSON = str(RESULTS_DIR / "jsep_edges.json")
SSEP_EDGES_JSON = str(RESULTS_DIR / "ssep_edges.json")
JOINT_STEP_LOGPROBS_JSON = str(RESULTS_DIR / "joint_step_logprobs.json")
