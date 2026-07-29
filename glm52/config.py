from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class GLM52Config:
    """Small, training-oriented scaling of the public GLM-5.2 architecture."""

    vocab_size: int = 8192
    hidden_size: int = 384
    num_layers: int = 13
    num_heads: int = 6
    q_lora_rank: int = 128
    kv_lora_rank: int = 64
    qk_nope_head_dim: int = 48
    qk_rope_head_dim: int = 16
    v_head_dim: int = 64
    index_n_heads: int = 4
    index_head_dim: int = 32
    index_topk: int = 64
    index_share_frequency: int = 4
    index_share_offset: int = 3
    dense_layers: int = 3
    dense_intermediate_size: int = 1024
    num_routed_experts: int = 16
    num_shared_experts: int = 1
    experts_per_token: int = 2
    moe_intermediate_size: int = 176
    routed_scaling_factor: float = 2.5
    mtp_depth: int = 3
    mtp_intermediate_size: int = 512
    rope_theta: float = 8_000_000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    indexer_loss_weight: float = 0.02
    mtp_loss_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive")
        if self.qk_rope_head_dim <= 0 or self.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be a positive even number")
        if self.index_head_dim < self.qk_rope_head_dim:
            raise ValueError("index_head_dim must include the RoPE dimensions")
        if not 0 < self.dense_layers <= self.num_layers:
            raise ValueError("dense_layers must be between 1 and num_layers")
        if not 0 < self.experts_per_token < self.num_routed_experts:
            raise ValueError("experts_per_token must be between 1 and num_routed_experts-1")
        if self.index_topk <= 0:
            raise ValueError("index_topk must be positive")
        if self.index_share_frequency <= 0:
            raise ValueError("index_share_frequency must be positive")
        if self.mtp_depth < 0:
            raise ValueError("mtp_depth cannot be negative")

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def mlp_type(self, layer_index: int) -> str:
        return "dense" if layer_index < self.dense_layers else "sparse"

    def indexer_type(self, layer_index: int) -> str:
        """Mirror GLM-5.2's offset FFFSSSFSSSFSS... IndexShare schedule."""
        adjusted = max(layer_index - self.index_share_offset + 1, 0)
        return "full" if adjusted % self.index_share_frequency == 0 else "shared"

    @property
    def indexer_types(self) -> list[str]:
        return [self.indexer_type(index) for index in range(self.num_layers)]

    @property
    def mlp_types(self) -> list[str]:
        return [self.mlp_type(index) for index in range(self.num_layers)]

    @classmethod
    def from_json(cls, path: str | Path) -> "GLM52Config":
        with Path(path).open(encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def to_dict(self) -> dict:
        return asdict(self)
