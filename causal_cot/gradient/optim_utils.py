"""
Optimizer and scheduler setup for causal edge pruning.
Adapted from Edge-Pruning/src/prune/fllama_boolean_expressions_fs.py get_optimizers.
Groups: log_alpha (masks, minimized), sparsity_lambda (regulators, maximized).
"""

from torch.optim import AdamW
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup


def get_optimizers(
    model,
    log_alpha_lr: float,
    sparsity_lambda_lr: float,
    num_training_steps: int,
    warmup_steps: int = 0,
    use_linear_schedule: bool = True,
    # Match Edge-Pruning launch script: same scale as edge LRs (ELR / RELR)
    node_log_alpha_lr: float = 0.8,
    sparsity_lambda_nodes_lr: float = 0.4,
    optimize_nodes: bool = True,
):
    edge_log_alpha_group = []
    edge_lambda_group = []
    node_log_alpha_group = []
    node_lambda_group = []

    for n, p in model.named_parameters():
        if not optimize_nodes and (
            "node_log_alpha" in n or "sparsity_lambda_nodes" in n
        ):
            continue
        if "node_log_alpha" in n:
            node_log_alpha_group.append(p)
        elif "log_alpha" in n:
            edge_log_alpha_group.append(p)
        elif "sparsity_lambda_nodes" in n:
            node_lambda_group.append(p)
        elif "sparsity_lambda" in n:
            edge_lambda_group.append(p)

    param_groups = [
        {'params': edge_log_alpha_group, 'lr': log_alpha_lr},
        {'params': edge_lambda_group, 'lr': sparsity_lambda_lr, 'maximize': True},
    ]
    if optimize_nodes and node_log_alpha_group:
        param_groups.append({'params': node_log_alpha_group, 'lr': node_log_alpha_lr})
    if optimize_nodes and node_lambda_group:
        param_groups.append({'params': node_lambda_group, 'lr': sparsity_lambda_nodes_lr, 'maximize': True})

    optimizer = AdamW(param_groups)

    if use_linear_schedule:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )

    return optimizer, scheduler
