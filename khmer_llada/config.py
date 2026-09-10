import json
from dataclasses import dataclass, asdict


@dataclass
class Config:
    vocab_size: int = 8000
    hidden_size: int = 640
    num_hidden_layers: int = 12
    num_attention_heads: int = 10
    intermediate_size: int = 1707
    max_position_embeddings: int = 512
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    # special tokens (khmer-sp-8k)
    mask_token_id: int = 4
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3
    unk_token_id: int = 1

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0, "hidden_size must divide by heads"
        head_dim = self.hidden_size // self.num_attention_heads
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        known = cls.__dataclass_fields__  # noqa: SLF001
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    def param_count(self) -> dict:
        """Analytic parameter breakdown (no model instantiation needed)."""
        h, L, V = self.hidden_size, self.num_hidden_layers, self.vocab_size
        ffn = self.intermediate_size
        emb = V * h
        per_layer = 4 * h * h + 3 * h * ffn + 2 * h  # attn qkvo + swiglu(w1,w3,w2) + 2 RMSNorm
        non_emb = per_layer * L + h  # + final norm
        head = 0 if self.tie_word_embeddings else V * h
        total = emb + non_emb + head
        return {
            "embedding": emb,
            "non_embedding": non_emb,
            "lm_head": head,
            "total": total,
            "total_millions": round(total / 1e6, 1),
        }
