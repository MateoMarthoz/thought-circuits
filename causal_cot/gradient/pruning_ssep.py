"""
Single Step Edge Pruning (SSEP) for causal CoT edge discovery.
Interpolates clean vs perturbed keys/values per source reasoning step with L0 masks.

Clean and perturbed full CoTs must token-align; callers should verify the same
``parse_reasoning_steps`` count inside ``think`` (see ``assert_matching_reasoning_step_counts``).

Step token ranges are contiguous (no gaps inside the think block). For each source
reasoning step i in 1..N-1, z[i-1] mixes clean/perturbed K/V at **every** key slot
in step i. That mixing applies to **every** query position q with q >= end_i
(first index after step i), through the end of the sequence. Faithfulness loss
is CE on the chosen target step only (skipping its first boundary token per
``calculate_step_loss``). ``run_single_step(k, ...)`` targets step ``k``;
``run_pruning`` is the special case ``k=N``.
"""

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .l0_utils import deterministic_z_from_log_alpha, sample_z_from_log_alpha
from .optim_utils import get_optimizers
from ..rope_utils import apply_rotary_pos_emb, repeat_kv
from ..step_utils import assign_token2step_tensor, validate_step_ranges_contiguous


def _default_ssep_checkpoint_dir() -> str:
    """Repo-root ``results/checkpoints_ssep``."""
    repo = Path(__file__).resolve().parent.parent.parent
    return str(repo / "results" / "checkpoints_ssep")


class _StepParams(nn.Module):
    """Container for step-k learnable params so get_optimizers can find them by name."""

    def __init__(
        self,
        k: int,
        lambda_1_init: float,
        lambda_2_init: float,
        device: torch.device,
        log_alpha_mean: float = 3.0,
        log_alpha_std: float = 0.5,
    ):
        super().__init__()
        # Edges from preceding steps 1 <= i < k (0-based: indices 1..k-1)
        self.log_alpha = nn.Parameter(
            torch.empty(k - 1, device=device).normal_(mean=log_alpha_mean, std=log_alpha_std)
        )
        self.sparsity_lambda_1 = nn.Parameter(
            torch.tensor(lambda_1_init, dtype=torch.float32, device=device)
        )
        self.sparsity_lambda_2 = nn.Parameter(
            torch.tensor(lambda_2_init, dtype=torch.float32, device=device)
        )


class CausalEdgePrunerSSEP(nn.Module):
    """
    L0 masks z[i-1] on keys/values in reasoning step i (i < N). All key tokens in
    step i use the same z; every query token at positions q >= end(step i) uses that
    mix for those keys (uniform over the post-step query region). Think-inner tokens
    must be fully covered by ``step_ranges[1:N+1]`` with no leakage to prompt id 0.
    Optimisation targets one reasoning step ``k`` per ``run_single_step`` call.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        step_ranges: List[Tuple[int, int]],
        target_sparsity: float,
        temperature: float,
        warmup_steps: int = 75,
        scheduler_warmup_steps: int = 10,
        log_alpha_lr: float = 0.8,
        sparsity_lambda_lr: float = 0.4,
        log_alpha_init_mean: float = 3.0,
        log_alpha_init_std: float = 0.5,
        use_linear_schedule: bool = True,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.N = len(step_ranges) - 1
        validate_step_ranges_contiguous(step_ranges, self.N)
        self.step_ranges = step_ranges
        self.target_sparsity = target_sparsity
        self.temperature = temperature
        self.warmup_steps = warmup_steps
        self.scheduler_warmup_steps = scheduler_warmup_steps
        self.log_alpha_lr = log_alpha_lr
        self.sparsity_lambda_lr = sparsity_lambda_lr
        self.log_alpha_init_mean = log_alpha_init_mean
        self.log_alpha_init_std = log_alpha_init_std
        self.use_linear_schedule = use_linear_schedule
        device = next(model.parameters()).device
        self.sparsity_lambda_1 = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32, device=device)
        )
        self.sparsity_lambda_2 = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32, device=device)
        )
        # Set by run_single_step before each forward pass
        self._cached_Z = None   # (k-1,) z values for source steps 1..k-1
        self._target_k = None   # current target step index

    def register_hooks(self) -> None:
        for layer in self.model.model.layers:
            attn = layer.self_attn
            if not hasattr(attn, "original_forward"):
                attn.original_forward = attn.forward
            attn.forward = self._make_hooked_forward(attn)

    def remove_hooks(self) -> None:
        for layer in self.model.model.layers:
            if hasattr(layer.self_attn, "original_forward"):
                layer.self_attn.forward = layer.self_attn.original_forward

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
            cos, sin = cos.to(device), sin.to(device)
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

        # Standard attention when not in interpolation mode (e.g. baseline loss pass)
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

        # Interpolated attention pass (bsz == 2: index 0 = clean, index 1 = perturbed)
        #
        # z[si-1] mixes K/V at keys in reasoning step si (si < target_k). For query q,
        # key j: apply that z iff j is in a learnable source step and q >= end(step(j))
        # (every token after that step sees the mixed K/V for those keys).
        Z_step = self._cached_Z  # (target_k-1,) — z for source steps 1..target_k-1
        target_k = self._target_k

        token2step = assign_token2step_tensor(
            self.step_ranges, self.N, q_len, device
        )

        # Exclusive end index of the step that contains each position (0 -> 0)
        step_end_at_pos = torch.zeros(
            self.N + 1, dtype=torch.long, device=device
        )
        for s in range(1, self.N + 1):
            step_end_at_pos[s] = self.step_ranges[s][1]
        key_step_end = step_end_at_pos[token2step.clamp(min=0, max=self.N)]

        learnable_key = (token2step >= 1) & (token2step < target_k)
        z_per_key = torch.ones(q_len, device=device, dtype=Z_step.dtype)
        sidx = token2step
        for si in range(1, target_k):
            m = sidx == si
            z_per_key = torch.where(m, Z_step[si - 1], z_per_key)

        Q_clean = query_states[0:1]
        K_clean = key_states[0:1]
        K_pert = key_states[1:2]
        V_clean = value_states[0:1]
        V_pert = value_states[1:2]

        kv_len = K_clean.shape[2]
        padl = kv_len - q_len
        if padl > 0:
            z_per_key = torch.cat(
                [
                    torch.ones(padl, device=device, dtype=Z_step.dtype),
                    z_per_key,
                ]
            )
            key_step_end = torch.cat(
                [
                    torch.zeros(padl, dtype=torch.long, device=device),
                    key_step_end,
                ]
            )
            learnable_key = torch.cat(
                [
                    torch.zeros(padl, dtype=torch.bool, device=device),
                    learnable_key,
                ]
            )

        CHUNK_SIZE = 128
        O_mixed_chunks = []

        for i in range(0, q_len, CHUNK_SIZE):
            Q_c = Q_clean[:, :, i:i + CHUNK_SIZE, :]
            chunk_len = Q_c.shape[2]
            q_row = torch.arange(
                i, i + chunk_len, device=device, dtype=torch.long
            ).view(chunk_len, 1)
            query_in_window = q_row >= key_step_end.unsqueeze(0)
            apply_mix = query_in_window & learnable_key.unsqueeze(0)
            Z_chunk = torch.where(
                apply_mix,
                z_per_key.unsqueeze(0).expand(chunk_len, -1),
                torch.ones(
                    chunk_len,
                    kv_len,
                    device=device,
                    dtype=Z_step.dtype,
                ),
            )
            Z_exp = Z_chunk.view(1, 1, chunk_len, kv_len).to(query_states.dtype)

            S_c = torch.matmul(Q_c, K_clean.transpose(2, 3)) / math.sqrt(head_dim)
            S_p = torch.matmul(Q_c, K_pert.transpose(2, 3)) / math.sqrt(head_dim)
            S_f = S_p + Z_exp * (S_c - S_p)
            del S_c, S_p

            if attention_mask is not None:
                S_f = S_f + attention_mask[0:1, :, i:i + CHUNK_SIZE, :]
            else:
                cur_kv = S_f.shape[-1]
                causal_mask = torch.tril(
                    torch.ones((chunk_len, cur_kv), device=S_f.device, dtype=torch.bool),
                    diagonal=i,
                )
                S_f.masked_fill_(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            A_f = F.softmax(S_f, dim=-1, dtype=torch.float32).to(query_states.dtype)
            del S_f

            A_Z = A_f * Z_exp
            O_c = torch.matmul(A_Z, V_clean)
            O_p = torch.matmul(A_f - A_Z, V_pert)
            O_mixed_chunks.append(O_c + O_p)
            del Q_c, A_f, A_Z, O_c, O_p

        Output_mixed = torch.cat(O_mixed_chunks, dim=2)
        del O_mixed_chunks

        # Perturbed branch: fully-perturbed attention output for batch index 1
        Q_pert = query_states[1:2]
        attn_mask_pert = attention_mask[1:2] if attention_mask is not None else None
        Output_pert = F.scaled_dot_product_attention(
            Q_pert, K_pert, V_pert,
            attn_mask=attn_mask_pert,
            dropout_p=0.0,
            is_causal=(attn_mask_pert is None),
        )

        attn_output = torch.cat([Output_mixed, Output_pert], dim=0)
        del Output_mixed, Output_pert
        del Q_clean, K_clean, K_pert, V_clean, V_pert, Q_pert
        del query_states, key_states, value_states
        if attn_mask_pert is not None:
            del attn_mask_pert

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = attn_module.o_proj(attn_output)
        return (attn_output, None)

    def run_single_step(
        self,
        k: int,
        clean_input_ids: torch.Tensor,
        pert_input_ids_list: List[torch.Tensor],
        clean_step_losses: Dict[int, torch.Tensor],
        checkpoint_dir: Optional[str] = None,
    ) -> List[Tuple[int, int]]:
        """
        Run L0-regularized attention pruning for a single target step k.

        Args:
            k: Target step index to optimize edges for.
            clean_input_ids: Full-sequence token IDs (1, seq_len).
            pert_input_ids_list: List of perturbed token ID tensors (each (1, seq_len)).
            clean_step_losses: Pre-computed baseline CE loss per step.

        Returns:
            List of (source, target) edge tuples discovered for step k.
        """
        if k <= 1:
            return []

        self.model.train()
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.model.get_input_embeddings().weight.requires_grad_(True)
        self.register_hooks()
        self._target_k = k

        device = clean_input_ids.device
        num_training_steps = getattr(self, "num_optim_steps", 100)

        clean_loss = clean_step_losses[k]
        start_idx, end_idx = self.step_ranges[k]
        labels_step = clean_input_ids[0, start_idx + 1 : end_idx]
        logit_start = start_idx
        logit_end = end_idx - 1

        step_params = _StepParams(
            k=k,
            lambda_1_init=self.sparsity_lambda_1.detach().item(),
            lambda_2_init=self.sparsity_lambda_2.detach().item(),
            device=device,
            log_alpha_mean=self.log_alpha_init_mean,
            log_alpha_std=self.log_alpha_init_std,
        )
        optimizer, scheduler = get_optimizers(
            step_params,
            log_alpha_lr=self.log_alpha_lr,
            sparsity_lambda_lr=self.sparsity_lambda_lr,
            num_training_steps=num_training_steps,
            warmup_steps=self.scheduler_warmup_steps,
            use_linear_schedule=self.use_linear_schedule,
        )

        num_perts = len(pert_input_ids_list)

        # Checkpoint targets: edge sparsity at i/num_ckpt_steps for i in 1..num_ckpt_steps-1
        # (num_ckpt_steps = N-1: CoT steps excluding prompt and final step)
        num_ckpt_steps = self.N - 1
        ckpt_interval = 1.0 / num_ckpt_steps
        ckpt_tolerance = 0.5 / max(num_ckpt_steps - 1, 1)
        ckpt_targets = [(i, i * ckpt_interval) for i in range(1, num_ckpt_steps)]
        # Stores the best (lowest) loss seen so far for each interval
        interval_loss: dict = {i: None for i in range(1, num_ckpt_steps)}

        for optim_step in range(num_training_steps):
            optimizer.zero_grad()
            self.model.zero_grad(set_to_none=True)

            for pert_input_ids in pert_input_ids_list:
                z = sample_z_from_log_alpha(step_params.log_alpha)
                self._cached_Z = z

                batch_ids = torch.cat([clean_input_ids, pert_input_ids], dim=0)
                inputs_embeds = self.model.get_input_embeddings()(batch_ids)
                inputs_embeds.requires_grad_(True)
                outputs = self.model(inputs_embeds=inputs_embeds)

                mixed_logits = outputs.logits[0]
                m_logits = mixed_logits[logit_start:logit_end].contiguous()
                if m_logits.shape[0] == 0:
                    m_loss = mixed_logits.sum() * 0.0
                else:
                    m_loss = F.cross_entropy(m_logits, labels_step, reduction="mean")
                delta_loss = m_loss - clean_loss

                current_target = (
                    self.target_sparsity * (optim_step / self.warmup_steps)
                    if optim_step < self.warmup_steps
                    else self.target_sparsity
                )
                current_sparsity = 1.0 - torch.mean(z)
                reg_loss = (
                    step_params.sparsity_lambda_1 * (current_sparsity - current_target)
                    + step_params.sparsity_lambda_2
                    * (current_sparsity - current_target) ** 2
                )

                total_loss = delta_loss + reg_loss
                scaled_loss = total_loss / num_perts
                scaled_loss.backward()

                if optim_step == 0:
                    grad_norm = (
                        step_params.log_alpha.grad.norm().item()
                        if step_params.log_alpha.grad is not None
                        else 0.0
                    )
                    print(
                        f"  [Diagnostic] log_alpha grad norm after first backward: {grad_norm:.6f}"
                    )

                _loss = scaled_loss.item()
                _spr = current_sparsity.item()
                _faith_loss = (total_loss - reg_loss).item()
                _tgt = current_target
                z_last = z.detach().cpu()

                del (
                    outputs, mixed_logits, m_logits, m_loss,
                    delta_loss, reg_loss, total_loss, scaled_loss, z,
                    batch_ids, inputs_embeds,
                )

            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()

            # Sparsity-interval checkpointing: save when edge sparsity hits an interval band
            # and the faithfulness loss is lower than previously stored for that band.
            ckpt_root = (
                checkpoint_dir
                if checkpoint_dir is not None
                else _default_ssep_checkpoint_dir()
            )
            os.makedirs(ckpt_root, exist_ok=True)
            for ckpt_i, target_edge_sp in ckpt_targets:
                if abs(_spr - target_edge_sp) <= ckpt_tolerance:
                    stored = interval_loss[ckpt_i]
                    if stored is None or _faith_loss < stored:
                        interval_loss[ckpt_i] = _faith_loss
                        ckpt_path = os.path.join(
                            ckpt_root, f"log_alpha_step_{k}_interval_{ckpt_i}.pt"
                        )
                        torch.save(
                            {
                                "log_alpha": step_params.log_alpha.detach().cpu(),
                                "sampled_z": z_last,
                                "sparsity_lambda_1": step_params.sparsity_lambda_1.detach().cpu(),
                                "sparsity_lambda_2": step_params.sparsity_lambda_2.detach().cpu(),
                                "optim_step": optim_step + 1,
                                "step_index": k,
                                "edge_sparsity": _spr,
                                "faithfulness_loss": _faith_loss,
                            },
                            ckpt_path,
                        )
                        print(
                            f"  [Step {k}] -> Saved interval {ckpt_i}/{num_ckpt_steps - 1} checkpoint "
                            f"(edge_sparsity={_spr:.4f}, target={target_edge_sp:.4f}, faith_loss={_faith_loss:.4f})"
                        )

            print(
                f"Step {optim_step + 1}/{num_training_steps}  "
                f"loss={_loss:.4f}  sparsity={_spr:.4f}  target={_tgt:.4f}"
            )

        self.model.eval()
        self.model.gradient_checkpointing_disable()
        self.remove_hooks()

        final_z = deterministic_z_from_log_alpha(step_params.log_alpha.detach())
        edges = [(i, k) for i in range(1, k) if final_z[i - 1] > 0]
        return edges

    def run_pruning(
        self,
        clean_input_ids: torch.Tensor,
        pert_input_ids_list: List[torch.Tensor],
        clean_step_losses: Dict[int, torch.Tensor],
        checkpoint_dir: Optional[str] = None,
    ) -> List[Tuple[int, int]]:
        """
        Run SSEP once, optimising L0 masks for keys in steps 1..N-1 with loss only
        on the final reasoning step N.

        Args:
            clean_input_ids: Full-sequence token IDs (1, seq_len).
            pert_input_ids_list: Perturbed sequences (same length as clean).
            clean_step_losses: Must include key ``N`` (final reasoning step baseline CE).

        Returns:
            List of (source_step, target_step) edges with target_step == N.
        """
        k = self.N
        print(f"\n=== SSEP: final reasoning step target k={k} (loss only on this step) ===")
        edges = self.run_single_step(
            k,
            clean_input_ids,
            pert_input_ids_list,
            clean_step_losses,
            checkpoint_dir=checkpoint_dir,
        )
        print(f"Found {len(edges)} edges into step {k}")
        return edges
