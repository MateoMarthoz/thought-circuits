"""
Joint Step Edge Pruning (JSEP) for causal CoT edge discovery.
Steps 6.1–6.3: CausalEdgePrunerJSEP, get_Z_matrix, register_hooks, custom_attention_forward (Z_token).
"""

import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .l0_utils import sample_z_from_log_alpha
from .optim_utils import get_optimizers
from ..rope_utils import apply_rotary_pos_emb, repeat_kv
from ..step_utils import assign_token2step_tensor, validate_step_ranges_contiguous


def _default_jsep_checkpoint_dir() -> str:
    """Repo-root ``results/checkpoints_jsep`` (same layout as ``repo_paths.CHECKPOINTS_JSEP``)."""
    repo = Path(__file__).resolve().parent.parent.parent
    return str(repo / "results" / "checkpoints_jsep")


class CausalEdgePrunerJSEP(nn.Module):
    """
    Global mask over attention: for i<j, Z[i,j] mixes clean vs perturbed keys from
    step i for every query token in step j (and likewise for values via the same weights).
    Matrix (N+1) x (N+1): index 0 = prompt, indices 1..N = reasoning steps.

    ``step_ranges`` must partition think-inner tokens with no gaps; every token in the
    think block shares the same step id as the rest of its step, so Z is constant over
    all key (resp. query) positions in that step.
    """

    def __init__(
        self,
        model: nn.Module,
        N: int,
        step_ranges: List[Tuple[int, int]],
        target_sparsity: float,
        temperature: float,
        log_alpha_init: float = 10.0,
        target_node_sparsity: float = 0.70,
        use_node_sparsity: bool = True,
    ):
        """
        Args:
            model: The LLM (e.g. DeepSeek-R1-Distill-Qwen-14B).
            N: Number of reasoning steps (step indices 1..N). Matrix size is (N+1) x (N+1).
            step_ranges: List of (start_idx, end_idx) token ranges; index 0 = prompt, 1..N = reasoning steps.
            target_sparsity: τ — target edge sparsity for L0 regularization.
            temperature: β — temperature for Hard Concrete sampling.
            log_alpha_init: Initial mean for log_alpha (higher = starts cleaner).
            target_node_sparsity: Target node (step) sparsity when ``use_node_sparsity`` is True.
            use_node_sparsity: If False, no node L0 masks, no node reg, edge-only (cf. edge-pruning).
        """
        super().__init__()
        self.model = model
        self.N = N
        validate_step_ranges_contiguous(step_ranges, N)
        self.step_ranges = step_ranges
        self.target_sparsity = target_sparsity
        self.target_node_sparsity = target_node_sparsity
        self.use_node_sparsity = use_node_sparsity
        self.temperature = temperature
        device = next(model.parameters()).device

        # Edge parameters
        self.log_alpha = nn.Parameter(
            torch.empty(N + 1, N + 1, device=device).normal_(mean=log_alpha_init, std=0.01)
        )
        self.sparsity_lambda_1 = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32, device=device)
        )
        self.sparsity_lambda_2 = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32, device=device)
        )

        # Node parameters: one log_alpha per reasoning step (1..N-1); step N has no outgoing edges
        if use_node_sparsity:
            self.node_log_alpha = nn.Parameter(
                torch.empty(N - 1, device=device).normal_(mean=log_alpha_init, std=0.01)
            )
            self.sparsity_lambda_nodes_1 = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float32, device=device)
            )
            self.sparsity_lambda_nodes_2 = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float32, device=device)
            )
        else:
            self.register_parameter("node_log_alpha", None)
            self.register_parameter("sparsity_lambda_nodes_1", None)
            self.register_parameter("sparsity_lambda_nodes_2", None)

    def get_Z_matrix(self) -> torch.Tensor:
        """
        Sample Z from log_alpha and force prompt (row/col 0) and diagonal to 1.0
        using torch.where to avoid in-place autograd errors.
        Node masks gate outgoing edges for each reasoning step before prompt/diagonal forcing.
        """
        Z_raw = sample_z_from_log_alpha(self.log_alpha)
        Z_upper = torch.triu(Z_raw, diagonal=1)

        if self.use_node_sparsity:
            z_nodes = sample_z_from_log_alpha(self.node_log_alpha)
            self._cached_z_nodes = z_nodes
            node_gate = torch.cat([
                torch.ones(1, device=Z_upper.device, dtype=Z_upper.dtype),
                z_nodes,
                torch.ones(1, device=Z_upper.device, dtype=Z_upper.dtype),
            ])
            Z_upper = Z_upper * node_gate.unsqueeze(1)
        else:
            self._cached_z_nodes = torch.ones(
                self.N - 1, device=Z_upper.device, dtype=Z_upper.dtype
            )

        mask = torch.eye(self.N + 1, device=Z_upper.device, dtype=torch.bool)
        mask[0, :] = True
        mask[:, 0] = True
        Z = torch.where(
            mask,
            torch.tensor(1.0, dtype=Z_upper.dtype, device=Z_upper.device),
            Z_upper,
        )
        return Z

    def calculate_usage_probabilities(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute usage probabilities U_j for reasoning steps 1..N (discard log-probs v in log-space).
        Builds v in a list to avoid in-place assignment and autograd graph corruption.
        """
        v_elements = [
            torch.tensor(-float("inf"), device=Z.device, dtype=Z.dtype)
        ]
        for j in range(self.N - 1, 0, -1):
            v_k = torch.stack(v_elements)
            z_jk = Z[j, j + 1 : self.N + 1]
            term_inside_log = 1.0 - z_jk * (1.0 - torch.exp(v_k))
            term_inside_log = torch.clamp(term_inside_log, min=1e-8)
            v_j = torch.sum(torch.log(term_inside_log))
            v_elements.insert(0, v_j)
        v_tensor = torch.stack(v_elements)
        U = 1.0 - torch.exp(v_tensor)
        return U

    def register_hooks(self) -> None:
        for layer in self.model.model.layers:
            attn_module = layer.self_attn
            if not hasattr(attn_module, "original_forward"):
                attn_module.original_forward = attn_module.forward
            attn_module.forward = self._make_hooked_forward(attn_module)

    def _make_hooked_forward(self, attn_module):
        def custom_forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            **kwargs,
        ):
            return self.custom_attention_forward(
                attn_module, hidden_states, attention_mask,
                position_ids, past_key_value, output_attentions,
                use_cache, cache_position, **kwargs,
            )
        return custom_forward

    def custom_attention_forward(
        self,
        attn_module: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        config = attn_module.config
        device = hidden_states.device
        bsz, q_len, _ = hidden_states.size()
        num_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = config.hidden_size // num_heads
        num_key_value_groups = num_heads // num_key_value_heads

        query_states = attn_module.q_proj(hidden_states)
        key_states = attn_module.k_proj(hidden_states)
        value_states = attn_module.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, num_heads, head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, num_key_value_heads, head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, num_key_value_heads, head_dim
        ).transpose(1, 2)

        if position_ids is None:
            position_ids = torch.arange(
                q_len, dtype=torch.long, device=device
            ).unsqueeze(0).expand(bsz, -1)
        else:
            position_ids = position_ids.to(device)

        rotary_emb = getattr(self.model.model, "rotary_emb", None)
        if rotary_emb is not None and callable(rotary_emb):
            cos, sin = rotary_emb(value_states, position_ids=position_ids)
            cos = cos.to(device)
            sin = sin.to(device)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        if past_key_value is not None:
            kv_seq_len = q_len + past_key_value[0].shape[-2]
            key_states = torch.cat([past_key_value[0].to(device), key_states], dim=2)
            value_states = torch.cat([past_key_value[1].to(device), value_states], dim=2)
        else:
            kv_seq_len = q_len

        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)

        if bsz != 2:
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            else:
                causal = torch.triu(
                    torch.ones(q_len, kv_seq_len, device=device, dtype=torch.bool),
                    diagonal=1,
                )
                attn_weights.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)
            return (attn_module.o_proj(attn_output), None)

        Z = self._cached_Z

        # One step id per token: prompt=0, reasoning 1..N covers think-inner exactly;
        # Z[q_step,k_step] applies uniformly to every token in those steps (no leakage).
        token2step = assign_token2step_tensor(
            self.step_ranges, self.N, q_len, hidden_states.device
        )

        # No full-sized Z_token: build Z mask per chunk inside the loop to avoid ~5 GB VRAM across layers.

        Q_clean = query_states[0:1]
        K_clean = key_states[0:1]
        K_pert = key_states[1:2]
        V_clean = value_states[0:1]
        V_pert = value_states[1:2]

        # --- MEMORY OPTIMIZATION BLOCK START ---
        CHUNK_SIZE = 128
        O_mixed_chunks = []

        for i in range(0, q_len, CHUNK_SIZE):
            Q_c = Q_clean[:, :, i : i + CHUNK_SIZE, :]

            # Just-in-time Z mask chunk to avoid saving a 128 MB tensor per layer.
            # Z[r,c] = mix weight for keys from step r used by queries in step c (edge r→c), r<c in the upper triangle.
            # Z.T[q_step, k_step] == Z[k_step, q_step] is the weight for (query step, key step) pairs.
            token2step_c = token2step[i : i + CHUNK_SIZE]
            Z_c = Z.T[token2step_c.unsqueeze(1), token2step.unsqueeze(0)]
            Z_c = Z_c.unsqueeze(0).unsqueeze(0).to(query_states.dtype)

            S_c = torch.matmul(Q_c, K_clean.transpose(2, 3)) / math.sqrt(head_dim)
            S_p = torch.matmul(Q_c, K_pert.transpose(2, 3)) / math.sqrt(head_dim)

            S_f = S_p + Z_c * (S_c - S_p)
            del S_c, S_p

            if attention_mask is not None:
                S_f = S_f + attention_mask[0:1, :, i : i + CHUNK_SIZE, :]
            else:
                seq_len = S_f.shape[-1]
                curr_chunk_q_len = S_f.shape[2]
                causal_mask = torch.tril(
                    torch.ones(
                        (curr_chunk_q_len, seq_len),
                        device=S_f.device,
                        dtype=torch.bool,
                    ),
                    diagonal=i,
                )
                S_f.masked_fill_(
                    ~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

            A_f = F.softmax(S_f, dim=-1, dtype=torch.float32).to(query_states.dtype)
            del S_f

            A_Z = A_f * Z_c
            O_c = torch.matmul(A_Z, V_clean)
            O_p = torch.matmul(A_f - A_Z, V_pert)

            O_mixed_chunks.append(O_c + O_p)
            del Q_c, Z_c, A_f, A_Z, O_c, O_p

        Output_mixed = torch.cat(O_mixed_chunks, dim=2)
        del O_mixed_chunks  # free the chunk list

        # Use SDPA for the perturbed branch to bypass N*N matrix instantiation entirely
        Q_pert = query_states[1:2]
        attn_mask_pert = attention_mask[1:2] if attention_mask is not None else None

        Output_pert = F.scaled_dot_product_attention(
            Q_pert, K_pert, V_pert,
            attn_mask=attn_mask_pert,
            dropout_p=0.0,
            is_causal=(attn_mask_pert is None),
        )

        attn_output = torch.cat([Output_mixed, Output_pert], dim=0)

        # --- EXPLICITLY CLEAR SCOPE TENSORS BEFORE ENTERING THE MLP LAYER ---
        del Output_mixed, Output_pert
        del Q_clean, K_clean, K_pert, V_clean, V_pert
        del Q_pert, query_states, key_states, value_states
        if "attn_mask_pert" in locals():
            del attn_mask_pert
        # --------------------------------------------------------------------
        # --- MEMORY OPTIMIZATION BLOCK END ---

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = attn_module.o_proj(attn_output)

        return (attn_output, None)

    def run_optimization(
        self,
        clean_input_ids: torch.Tensor,
        pert_input_ids_list: List[torch.Tensor],
        faith_loss_fn: str,
        num_training_steps: int = 600,
        warmup_steps: int = 500,
        scheduler_warmup_steps: int = 20,
        log_alpha_lr: float = 0.2,
        sparsity_lambda_lr: float = 0.1,
        node_log_alpha_lr: Optional[float] = None,
        sparsity_lambda_nodes_lr: Optional[float] = None,
        use_linear_schedule: bool = True,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        if faith_loss_fn not in ("weighted_sum", "squared_sum"):
            raise ValueError(
                f"faith_loss_fn must be 'weighted_sum' or 'squared_sum', got {faith_loss_fn!r}"
            )
        self.register_hooks()
        self.model.eval()

        # Match launch_fllama_fs_prune.sh: LLR=$ELR, RLLR=$RELR when not passed explicitly
        node_la_lr = (
            log_alpha_lr if node_log_alpha_lr is None else node_log_alpha_lr
        )
        node_sl_lr = (
            sparsity_lambda_lr if sparsity_lambda_nodes_lr is None else sparsity_lambda_nodes_lr
        )
        optimize_nodes = self.use_node_sparsity

        clean_step_losses = {}
        print("Calculating clean baseline losses...")
        with torch.no_grad():
            clean_embeds = self.model.get_input_embeddings()(clean_input_ids)
            clean_outputs = self.model(inputs_embeds=clean_embeds)
            clean_logits = clean_outputs.logits[0]
            for k in range(1, self.N + 1):
                start_idx, end_idx = self.step_ranges[k]
                c_logits = clean_logits[start_idx : end_idx - 1].contiguous()
                c_labels = clean_input_ids[0, start_idx + 1 : end_idx]
                clean_step_losses[k] = F.cross_entropy(c_logits, c_labels, reduction="mean").detach()
        print("Baseline calculation complete.")

        del clean_outputs, clean_logits, c_logits, c_labels, clean_embeds

        optimizer, scheduler = get_optimizers(
            self,
            log_alpha_lr=log_alpha_lr,
            sparsity_lambda_lr=sparsity_lambda_lr,
            node_log_alpha_lr=node_la_lr,
            sparsity_lambda_nodes_lr=node_sl_lr,
            num_training_steps=num_training_steps,
            warmup_steps=scheduler_warmup_steps,
            use_linear_schedule=use_linear_schedule,
            optimize_nodes=optimize_nodes,
        )
        num_perts = len(pert_input_ids_list)
        self.model.train()
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.model.get_input_embeddings().weight.requires_grad_(True)

        # Checkpoint targets: node sparsity at i/num_ckpt_steps for i in 1..num_ckpt_steps-1
        # (num_ckpt_steps = N-1: CoT steps excluding prompt and final step)
        num_ckpt_steps = self.N - 1
        ckpt_interval = 1.0 / num_ckpt_steps
        ckpt_tolerance = 0.5 / max(num_ckpt_steps - 1, 1)
        ckpt_targets = [(i, i * ckpt_interval) for i in range(1, num_ckpt_steps)]
        # Stores the lowest faithfulness loss seen so far for each interval
        interval_faith_loss: dict = {i: None for i in range(1, num_ckpt_steps)}

        for optim_step in range(num_training_steps):
            optimizer.zero_grad()
            self.model.zero_grad(set_to_none=True)
            for pert_input_ids in pert_input_ids_list:
                batch_ids = torch.cat(
                    [clean_input_ids, pert_input_ids], dim=0
                )

                self._cached_Z = self.get_Z_matrix()

                inputs_embeds = self.model.get_input_embeddings()(batch_ids)
                inputs_embeds.requires_grad_(True)
                outputs = self.model(inputs_embeds=inputs_embeds)
                mixed_logits = outputs.logits[0]

                Z = self._cached_Z

                U = self.calculate_usage_probabilities(Z)

                raw_deltas = []
                u_weights = []
                for k in range(1, self.N + 1):
                    start_idx, end_idx = self.step_ranges[k]
                    step_logits = mixed_logits[start_idx : end_idx - 1]
                    step_labels = clean_input_ids[0, start_idx + 1 : end_idx]
                    step_loss = F.cross_entropy(step_logits, step_labels, reduction="mean")
                    raw_deltas.append(step_loss - clean_step_losses[k])
                    u_weights.append(U[k - 1])

                deltas_tensor = torch.stack(raw_deltas)
                u_tensor = torch.stack(u_weights)

                if faith_loss_fn == "squared_sum":
                    # sum_k ReLU(delta_k * U_k)^2
                    per_step = F.relu(deltas_tensor * u_tensor)
                    global_step_loss = torch.sum(per_step * per_step)
                else:  # weighted_sum
                    global_step_loss = torch.sum(u_tensor * deltas_tensor)

                # --- Edge sparsity loss ---
                if optim_step < warmup_steps:
                    current_target = self.target_sparsity * (optim_step / warmup_steps)
                else:
                    current_target = self.target_sparsity
                active_z = Z[1 : self.N + 1, 1 : self.N + 1]
                upper_triangle_indices = torch.triu_indices(
                    self.N, self.N, offset=1, device=Z.device
                )
                current_sparsity = 1.0 - torch.mean(
                    active_z[
                        upper_triangle_indices[0], upper_triangle_indices[1]
                    ]
                )
                edge_reg_loss = (
                    self.sparsity_lambda_1
                    * (current_sparsity - current_target)
                    + self.sparsity_lambda_2
                    * (current_sparsity - current_target) ** 2
                )

                # --- Node sparsity loss (optional) ---
                if self.use_node_sparsity:
                    z_nodes = self._cached_z_nodes
                    model_node_sparsity = 1.0 - z_nodes.mean()
                    if optim_step < warmup_steps:
                        current_node_target = self.target_node_sparsity * (
                            optim_step / warmup_steps
                        )
                    else:
                        current_node_target = self.target_node_sparsity
                    node_reg_loss = (
                        self.sparsity_lambda_nodes_1
                        * (model_node_sparsity - current_node_target)
                        + self.sparsity_lambda_nodes_2
                        * (model_node_sparsity - current_node_target) ** 2
                    )
                else:
                    model_node_sparsity = torch.tensor(
                        0.0, device=Z.device, dtype=Z.dtype
                    )
                    current_node_target = 0.0
                    node_reg_loss = torch.tensor(
                        0.0, device=Z.device, dtype=Z.dtype
                    )

                reg_loss = edge_reg_loss + node_reg_loss
                total_loss = global_step_loss + reg_loss
                scaled_loss = total_loss / num_perts
                scaled_loss.backward()

                del outputs, mixed_logits, step_logits, step_labels, batch_ids, inputs_embeds
                del raw_deltas, u_weights, deltas_tensor, u_tensor

            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()

            # Interval checkpointing: by node sparsity (default) or edge sparsity (edge-only mode)
            node_sp_val = model_node_sparsity.item()
            edge_sp_val = current_sparsity.item()
            faith_loss_val = global_step_loss.item()
            ckpt_root = (
                checkpoint_dir
                if checkpoint_dir is not None
                else _default_jsep_checkpoint_dir()
            )
            os.makedirs(ckpt_root, exist_ok=True)
            for ckpt_i, target_interval_sp in ckpt_targets:
                sp_val = node_sp_val if self.use_node_sparsity else edge_sp_val
                if abs(sp_val - target_interval_sp) <= ckpt_tolerance:
                    stored = interval_faith_loss[ckpt_i]
                    if stored is None or faith_loss_val < stored:
                        interval_faith_loss[ckpt_i] = faith_loss_val
                        ckpt_path = os.path.join(
                            ckpt_root, f"log_alpha_interval_{ckpt_i}.pt"
                        )
                        ckpt_payload = {
                            "log_alpha": self.log_alpha.detach().cpu(),
                            "sampled_Z": self._cached_Z.detach().cpu(),
                            "sparsity_lambda_1": self.sparsity_lambda_1.detach().cpu(),
                            "sparsity_lambda_2": self.sparsity_lambda_2.detach().cpu(),
                            "optim_step": optim_step + 1,
                            "edge_sparsity": edge_sp_val,
                            "node_sparsity": node_sp_val,
                            "faithfulness_loss": faith_loss_val,
                            "use_node_sparsity": self.use_node_sparsity,
                        }
                        if self.use_node_sparsity:
                            ckpt_payload["node_log_alpha"] = (
                                self.node_log_alpha.detach().cpu()
                            )
                            ckpt_payload["sampled_z_nodes"] = (
                                self._cached_z_nodes.detach().cpu()
                            )
                            ckpt_payload["sparsity_lambda_nodes_1"] = (
                                self.sparsity_lambda_nodes_1.detach().cpu()
                            )
                            ckpt_payload["sparsity_lambda_nodes_2"] = (
                                self.sparsity_lambda_nodes_2.detach().cpu()
                            )
                        else:
                            ckpt_payload["node_log_alpha"] = torch.full(
                                (self.N - 1,), 20.0
                            )
                            ckpt_payload["sampled_z_nodes"] = torch.ones(self.N - 1)
                            ckpt_payload["sparsity_lambda_nodes_1"] = torch.tensor(
                                0.0
                            )
                            ckpt_payload["sparsity_lambda_nodes_2"] = torch.tensor(
                                0.0
                            )
                        torch.save(ckpt_payload, ckpt_path)
                        grad_norm = (
                            self.log_alpha.grad.norm().item()
                            if self.log_alpha.grad is not None
                            else 0.0
                        )
                        if self.use_node_sparsity:
                            print(
                                f"  [JSEP] -> Saved interval {ckpt_i}/{num_ckpt_steps - 1} checkpoint. "
                                f"Node sparsity: {node_sp_val:.4f} (target {target_interval_sp:.4f}), "
                                f"Edge sparsity: {edge_sp_val:.4f}, Faith loss: {faith_loss_val:.4f}, "
                                f"Grad Norm: {grad_norm:.6f}"
                            )
                        else:
                            print(
                                f"  [JSEP] -> Saved interval {ckpt_i}/{num_ckpt_steps - 1} checkpoint. "
                                f"Edge sparsity: {edge_sp_val:.4f} (target {target_interval_sp:.4f}), "
                                f"Faith loss: {faith_loss_val:.4f}, "
                                f"Grad Norm: {grad_norm:.6f}"
                            )

            if self.use_node_sparsity:
                print(
                    f"Step {optim_step + 1}/{num_training_steps}  "
                    f"loss={scaled_loss.item():.4f}  "
                    f"edge_sp={current_sparsity.item():.4f}  edge_tgt={current_target:.4f}  "
                    f"node_sp={model_node_sparsity.item():.4f}  node_tgt={current_node_target:.4f}"
                )
            else:
                print(
                    f"Step {optim_step + 1}/{num_training_steps}  "
                    f"loss={scaled_loss.item():.4f}  "
                    f"edge_sp={current_sparsity.item():.4f}  edge_tgt={current_target:.4f}  "
                    f"(node sparsity off)"
                )

        for layer in self.model.model.layers:
            if hasattr(layer.self_attn, "original_forward"):
                layer.self_attn.forward = layer.self_attn.original_forward
