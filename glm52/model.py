from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GLM52Config


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        normalized = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dimension: int, theta: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, dimension, 2, dtype=torch.float32) / dimension)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = position_ids.float().unsqueeze(-1) * self.inverse_frequency
        return frequencies.cos().to(dtype), frequencies.sin().to(dtype)


def apply_interleaved_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to x shaped [batch, heads, sequence, rope_dimension]."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
    return rotated.flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TopKRouter(nn.Module):
    """GLM's sigmoid, correction-bias, auxiliary-loss-free token router."""

    def __init__(self, config: GLM52Config) -> None:
        super().__init__()
        self.top_k = config.experts_per_token
        self.num_experts = config.num_routed_experts
        self.scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(self.num_experts, config.hidden_size))
        self.register_buffer(
            "expert_score_correction_bias",
            torch.zeros(self.num_experts, dtype=torch.float32),
            persistent=True,
        )

    @torch.no_grad()
    def _update_correction_bias(self, scores: torch.Tensor, cutoff: torch.Tensor) -> None:
        # Exact local-batch quantiles replace the distributed histogram used at
        # frontier scale. Selection uses this bias; mixture weights do not.
        quantile = 1.0 - self.top_k / self.num_experts
        margins = scores.detach() - cutoff.detach().unsqueeze(-1)
        correction = -torch.quantile(margins, quantile, dim=0)
        correction -= correction.mean()
        self.expert_score_correction_bias.copy_(correction.float())

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, hidden_states.size(-1))
        router_logits = F.linear(flat.float(), self.weight.float())
        scores = router_logits.sigmoid()
        selection_scores = scores + self.expert_score_correction_bias
        top_values, top_indices = torch.topk(selection_scores, self.top_k + 1, dim=-1)
        selected = top_indices[:, : self.top_k]
        weights = scores.gather(1, selected)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        weights = weights * self.scaling_factor
        if self.training:
            self._update_correction_bias(scores, top_values[:, self.top_k])
        return weights, selected


class SparseMoE(nn.Module):
    def __init__(self, config: GLM52Config) -> None:
        super().__init__()
        self.router = TopKRouter(config)
        self.routed_experts = nn.ModuleList(
            SwiGLU(config.hidden_size, config.moe_intermediate_size)
            for _ in range(config.num_routed_experts)
        )
        self.shared_experts = nn.ModuleList(
            SwiGLU(config.hidden_size, config.moe_intermediate_size)
            for _ in range(config.num_shared_experts)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_states.size(-1))
        weights, selected = self.router(flat)
        routed = torch.zeros_like(flat)

        for expert_index, expert in enumerate(self.routed_experts):
            token_indices, slots = torch.where(selected == expert_index)
            if token_indices.numel() == 0:
                continue
            contribution = expert(flat[token_indices])
            contribution = contribution * weights[token_indices, slots, None].to(
                contribution.dtype
            )
            routed.index_add_(
                0,
                token_indices,
                contribution.to(routed.dtype),
            )

        shared = torch.zeros_like(flat)
        for expert in self.shared_experts:
            shared = shared + expert(flat).to(shared.dtype)
        return (routed + shared).view(original_shape)


@dataclass
class IndexerSelection:
    indices: torch.Tensor
    logits: torch.Tensor


class LightningIndexer(nn.Module):
    """Low-rank multi-head ReLU indexer used by DeepSeek Sparse Attention."""

    def __init__(self, config: GLM52Config) -> None:
        super().__init__()
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.top_k = config.index_topk
        self.q_proj = nn.Linear(
            config.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.head_weight_proj = nn.Linear(
            config.hidden_size,
            self.num_heads,
            bias=False,
        )
        self.scale = self.head_dim**-0.5
        self.q_proj.weight._muon_heads = self.num_heads  # type: ignore[attr-defined]

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cached_keys: torch.Tensor | None = None,
    ) -> tuple[IndexerSelection, torch.Tensor]:
        batch, sequence, _ = hidden_states.shape
        query = self.q_proj(q_residual)
        query = query.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        query_rotary, query_pass = torch.split(
            query,
            [self.rope_dim, self.head_dim - self.rope_dim],
            dim=-1,
        )
        query_rotary = apply_interleaved_rope(query_rotary, cos, sin)
        query = torch.cat((query_rotary, query_pass), dim=-1)

        key = self.k_norm(self.k_proj(hidden_states)).unsqueeze(1)
        key_rotary, key_pass = torch.split(
            key,
            [self.rope_dim, self.head_dim - self.rope_dim],
            dim=-1,
        )
        key_rotary = apply_interleaved_rope(key_rotary, cos, sin)
        key = torch.cat((key_rotary, key_pass), dim=-1).squeeze(1)
        all_keys = key if cached_keys is None else torch.cat((cached_keys, key), dim=1)

        per_head_scores = torch.matmul(
            query.float(),
            all_keys.float().transpose(-1, -2).unsqueeze(1),
        )
        per_head_scores = F.relu(per_head_scores * self.scale)
        head_weights = self.head_weight_proj(hidden_states).float() * (self.num_heads**-0.5)
        index_logits = torch.einsum("bsh,bhst->bst", head_weights, per_head_scores)

        key_positions = torch.arange(all_keys.size(1), device=hidden_states.device)
        causal_mask = key_positions.view(1, 1, -1) > position_ids.unsqueeze(-1)
        # Autocast may return BF16 from the einsum even though its operands were
        # promoted to FP32. Negative infinity is representable in every floating
        # dtype; the FP32 finite minimum is not representable in BF16.
        index_logits = index_logits.masked_fill(causal_mask, -float("inf"))
        top_k = min(self.top_k, all_keys.size(1))
        indices = torch.topk(index_logits, top_k, dim=-1).indices
        return IndexerSelection(indices=indices, logits=index_logits), all_keys


class MultiLatentSparseAttention(nn.Module):
    """GLM-5.2 MLA core with DSA selection and optional IndexShare reuse."""

    def __init__(self, config: GLM52Config, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.indexer_type = config.indexer_type(layer_index)
        self.num_heads = config.num_heads
        self.qk_nope_dim = config.qk_nope_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.qk_dim = config.qk_head_dim
        self.value_dim = config.v_head_dim
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            self.num_heads * self.qk_dim,
            bias=False,
        )
        self.kv_a_proj = nn.Linear(
            config.hidden_size,
            config.kv_lora_rank + self.rope_dim,
            bias=False,
        )
        self.kv_a_norm = RMSNorm(config.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (self.qk_nope_dim + self.value_dim),
            bias=False,
        )
        self.out_proj = nn.Linear(
            self.num_heads * self.value_dim,
            config.hidden_size,
            bias=False,
        )
        self.indexer = (
            LightningIndexer(config) if self.indexer_type == "full" else None
        )
        self.scale = self.qk_dim**-0.5
        for projection in (self.q_b_proj, self.kv_b_proj):
            projection.weight._muon_heads = self.num_heads  # type: ignore[attr-defined]

    def _expand_kv(
        self,
        latent_kv: torch.Tensor,
        rotary_key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sequence, _ = latent_kv.shape
        expanded = self.kv_b_proj(latent_kv)
        expanded = expanded.view(
            batch,
            sequence,
            self.num_heads,
            self.qk_nope_dim + self.value_dim,
        ).transpose(1, 2)
        key_nope, values = torch.split(
            expanded,
            [self.qk_nope_dim, self.value_dim],
            dim=-1,
        )
        repeated_rotary = rotary_key.expand(
            batch,
            self.num_heads,
            sequence,
            self.rope_dim,
        )
        return torch.cat((key_nope, repeated_rotary), dim=-1), values

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        previous_selection: IndexerSelection | None,
        cache: dict[str, torch.Tensor] | None = None,
        use_cache: bool = False,
        return_teacher: bool = False,
    ) -> tuple[
        torch.Tensor,
        IndexerSelection,
        torch.Tensor | None,
        dict[str, torch.Tensor] | None,
    ]:
        batch, sequence, _ = hidden_states.shape
        cache = {} if cache is None else cache

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        query = self.q_b_proj(q_residual)
        query = query.view(batch, sequence, self.num_heads, self.qk_dim).transpose(1, 2)
        query_nope, query_rotary = torch.split(
            query,
            [self.qk_nope_dim, self.rope_dim],
            dim=-1,
        )
        query_rotary = apply_interleaved_rope(query_rotary, cos, sin)
        query = torch.cat((query_nope, query_rotary), dim=-1)

        compressed = self.kv_a_proj(hidden_states)
        latent_kv, rotary_key = torch.split(
            compressed,
            [self.config.kv_lora_rank, self.rope_dim],
            dim=-1,
        )
        latent_kv = self.kv_a_norm(latent_kv)
        rotary_key = rotary_key.view(batch, 1, sequence, self.rope_dim)
        rotary_key = apply_interleaved_rope(rotary_key, cos, sin)

        old_latent = cache.get("latent_kv")
        old_rotary = cache.get("rotary_key")
        all_latent = latent_kv if old_latent is None else torch.cat((old_latent, latent_kv), dim=1)
        all_rotary = (
            rotary_key
            if old_rotary is None
            else torch.cat((old_rotary, rotary_key), dim=2)
        )
        key, value = self._expand_kv(all_latent, all_rotary)

        if self.indexer is not None:
            selection, all_index_keys = self.indexer(
                hidden_states,
                q_residual,
                position_ids,
                cos,
                sin,
                cache.get("index_keys"),
            )
        else:
            if previous_selection is None:
                raise ValueError(
                    f"shared indexer layer {self.layer_index} has no preceding full selection"
                )
            selection = previous_selection
            all_index_keys = None

        raw_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        key_positions = torch.arange(key.size(2), device=hidden_states.device)
        causal_mask = key_positions.view(1, 1, 1, -1) > position_ids.view(
            batch,
            1,
            sequence,
            1,
        )
        causal_scores = raw_scores.masked_fill(
            causal_mask,
            torch.finfo(raw_scores.dtype).min,
        )
        teacher = None
        if return_teacher:
            teacher = causal_scores.float().softmax(dim=-1).mean(dim=1).detach()

        selected_mask = torch.zeros(
            batch,
            sequence,
            key.size(2),
            dtype=torch.bool,
            device=hidden_states.device,
        )
        selected_mask.scatter_(-1, selection.indices.long(), True)
        selected_mask = selected_mask.unsqueeze(1) & ~causal_mask
        sparse_scores = raw_scores.masked_fill(
            ~selected_mask,
            torch.finfo(raw_scores.dtype).min,
        )
        attention_weights = sparse_scores.float().softmax(dim=-1).to(query.dtype)
        output = torch.matmul(attention_weights, value)
        output = output.transpose(1, 2).contiguous().view(batch, sequence, -1)
        output = self.out_proj(output)

        next_cache = None
        if use_cache:
            next_cache = {
                "latent_kv": all_latent,
                "rotary_key": all_rotary,
            }
            if all_index_keys is not None:
                next_cache["index_keys"] = all_index_keys
        return output, selection, teacher, next_cache


class DecoderLayer(nn.Module):
    def __init__(self, config: GLM52Config, layer_index: int) -> None:
        super().__init__()
        self.mlp_type = config.mlp_type(layer_index)
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = MultiLatentSparseAttention(config, layer_index)
        self.mlp: nn.Module
        if self.mlp_type == "dense":
            self.mlp = SwiGLU(config.hidden_size, config.dense_intermediate_size)
        else:
            self.mlp = SparseMoE(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        selection: IndexerSelection | None,
        cache: dict[str, torch.Tensor] | None,
        use_cache: bool,
        return_teacher: bool,
    ) -> tuple[
        torch.Tensor,
        IndexerSelection,
        torch.Tensor | None,
        dict[str, torch.Tensor] | None,
    ]:
        attention_output, selection, teacher, next_cache = self.attention(
            self.input_norm(hidden_states),
            position_ids,
            cos,
            sin,
            selection,
            cache=cache,
            use_cache=use_cache,
            return_teacher=return_teacher,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.post_attention_norm(hidden_states))
        return hidden_states, selection, teacher, next_cache


class DenseCausalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("MTP hidden size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        for projection in (self.q_proj, self.k_proj, self.v_proj):
            projection.weight._muon_heads = num_heads  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, hidden = x.shape

        def shape(projection: nn.Linear) -> torch.Tensor:
            return projection(x).view(
                batch,
                sequence,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        query, key, value = shape(self.q_proj), shape(self.k_proj), shape(self.v_proj)
        output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(batch, sequence, hidden)
        return self.out_proj(output)


class SharedMTPPredictor(nn.Module):
    """One auxiliary prediction layer reused for every future-token depth."""

    def __init__(self, config: GLM52Config) -> None:
        super().__init__()
        self.state_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.token_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.fuse = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = DenseCausalAttention(config.hidden_size, config.num_heads)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.mtp_intermediate_size)

    def forward(
        self,
        state: torch.Tensor,
        known_token_embedding: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.fuse(
            torch.cat(
                (self.state_norm(state), self.token_norm(known_token_embedding)),
                dim=-1,
            )
        )
        fused = fused + self.attention(self.attention_norm(fused))
        return fused + self.mlp(self.mlp_norm(fused))


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    main_loss: torch.Tensor | None = None
    mtp_loss: torch.Tensor | None = None
    indexer_loss: torch.Tensor | None = None


class GLM52ForCausalLM(nn.Module):
    def __init__(self, config: GLM52Config) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config, index) for index in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.rotary_embedding = RotaryEmbedding(
            config.qk_rope_head_dim,
            config.rope_theta,
        )
        self.mtp_predictor = (
            SharedMTPPredictor(config) if config.mtp_depth > 0 else None
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, TopKRouter):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def _indexer_distillation_loss(
        logits: torch.Tensor,
        targets: list[torch.Tensor],
    ) -> torch.Tensor:
        average_target = torch.stack(targets, dim=0).mean(dim=0)
        log_probabilities = logits.float().log_softmax(dim=-1)
        # Causal positions have zero teacher probability and -inf student
        # log-probability. Exclude them explicitly to avoid the undefined
        # product 0 * -inf while retaining an exact hard attention mask.
        valid_positions = torch.isfinite(logits)
        average_target = average_target.masked_fill(~valid_positions, 0.0)
        log_probabilities = log_probabilities.masked_fill(~valid_positions, 0.0)
        return -(average_target * log_probabilities).sum(dim=-1).mean()

    def _backbone(
        self,
        input_ids: torch.Tensor,
        caches: list[dict[str, torch.Tensor] | None] | None = None,
        use_cache: bool = False,
        return_indexer_loss: bool = False,
    ) -> tuple[
        torch.Tensor,
        list[dict[str, torch.Tensor] | None] | None,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        if caches is not None and len(caches) != len(self.layers):
            raise ValueError("cache length must match num_layers")
        past_length = 0
        if caches and caches[0] is not None:
            past_length = caches[0]["latent_kv"].size(1)
        batch, sequence = input_ids.shape
        position_ids = (
            torch.arange(past_length, past_length + sequence, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        hidden_states = self.token_embedding(input_ids)
        input_embeddings = hidden_states
        cos, sin = self.rotary_embedding(position_ids, hidden_states.dtype)
        selection: IndexerSelection | None = None
        current_logits: torch.Tensor | None = None
        current_targets: list[torch.Tensor] = []
        distillation_losses: list[torch.Tensor] = []
        next_caches: list[dict[str, torch.Tensor] | None] = []

        for layer_index, layer in enumerate(self.layers):
            if layer.attention.indexer_type == "full" and current_logits is not None:
                if return_indexer_loss and current_targets:
                    distillation_losses.append(
                        self._indexer_distillation_loss(current_logits, current_targets)
                    )
                current_targets = []

            layer_cache = None if caches is None else caches[layer_index]
            hidden_states, selection, teacher, next_cache = layer(
                hidden_states,
                position_ids,
                cos,
                sin,
                selection,
                cache=layer_cache,
                use_cache=use_cache,
                return_teacher=return_indexer_loss,
            )
            next_caches.append(next_cache)

            if layer.attention.indexer_type == "full":
                current_logits = selection.logits
            if teacher is not None:
                current_targets.append(teacher)

        if return_indexer_loss and current_logits is not None and current_targets:
            distillation_losses.append(
                self._indexer_distillation_loss(current_logits, current_targets)
            )

        indexer_loss = None
        if distillation_losses:
            indexer_loss = torch.stack(distillation_losses).mean()
        return (
            self.final_norm(hidden_states),
            next_caches if use_cache else None,
            indexer_loss,
            input_embeddings,
        )

    def _mtp_loss(
        self,
        hidden_states: torch.Tensor,
        input_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.mtp_predictor is None or hidden_states.size(1) < 2:
            return None
        state = hidden_states
        losses: list[torch.Tensor] = []
        for depth in range(1, self.config.mtp_depth + 1):
            if state.size(1) < 2 or depth >= input_embeddings.size(1):
                break
            state = self.mtp_predictor(
                state[:, :-1],
                input_embeddings[:, depth:],
            )
            targets = labels[:, depth:]
            logits = self.lm_head(state)
            losses.append(
                F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-100,
                )
            )
        return torch.stack(losses).mean() if losses else None

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        compute_auxiliary: bool | None = None,
    ) -> CausalLMOutput:
        if compute_auxiliary is None:
            compute_auxiliary = labels is not None and self.training
        hidden_states, _, indexer_loss, input_embeddings = self._backbone(
            input_ids,
            return_indexer_loss=compute_auxiliary,
        )
        logits = self.lm_head(hidden_states)
        main_loss = None
        mtp_loss = None
        total_loss = None
        if labels is not None:
            main_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            total_loss = main_loss
            if compute_auxiliary:
                mtp_loss = self._mtp_loss(hidden_states, input_embeddings, labels)
                if mtp_loss is not None:
                    total_loss = total_loss + self.config.mtp_loss_weight * mtp_loss
                if indexer_loss is not None:
                    total_loss = (
                        total_loss + self.config.indexer_loss_weight * indexer_loss
                    )
        return CausalLMOutput(
            logits=logits,
            loss=total_loss,
            main_loss=main_loss,
            mtp_loss=mtp_loss,
            indexer_loss=indexer_loss,
        )

    def num_parameters(self, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (
                parameter for parameter in parameters if parameter.requires_grad
            )
        return sum(parameter.numel() for parameter in parameters)

    def num_active_parameters(self) -> int:
        """Approximate per-token parameter path (shared + selected experts)."""
        total = self.num_parameters()
        for module in self.modules():
            if not isinstance(module, SparseMoE):
                continue
            expert_sizes = [
                sum(parameter.numel() for parameter in expert.parameters())
                for expert in module.routed_experts
            ]
            total -= sum(expert_sizes)
            total += sum(expert_sizes[: self.config.experts_per_token])
        return total

    def forward_step(
        self,
        input_ids: torch.Tensor,
        caches: list[dict[str, torch.Tensor] | None] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor] | None]]:
        if input_ids.ndim != 2 or input_ids.size(1) != 1:
            raise ValueError("forward_step expects input_ids shaped [batch, 1]")
        if caches is None:
            caches = [None] * len(self.layers)
        hidden_states, next_caches, _, _ = self._backbone(
            input_ids,
            caches=caches,
            use_cache=True,
            return_indexer_loss=False,
        )
        if next_caches is None:
            raise RuntimeError("cache construction failed")
        return self.lm_head(hidden_states), next_caches

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        if input_ids.size(1) == 0:
            raise ValueError("input_ids must contain at least one token")
        self.eval()
        caches: list[dict[str, torch.Tensor] | None] | None = None
        logits = None
        for position in range(input_ids.size(1)):
            logits, caches = self.forward_step(
                input_ids[:, position : position + 1],
                caches,
            )
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1].float() / max(temperature, 1e-5)
            if top_k > 0:
                threshold = torch.topk(
                    next_logits,
                    min(top_k, next_logits.size(-1)),
                ).values[:, [-1]]
                next_logits = next_logits.masked_fill(
                    next_logits < threshold,
                    -float("inf"),
                )
            next_token = torch.multinomial(next_logits.softmax(dim=-1), 1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            logits, caches = self.forward_step(next_token, caches)
        return input_ids
