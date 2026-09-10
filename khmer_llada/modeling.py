"""
Minimal bidirectional LLaMA-style transformer for LLaDA-style masked diffusion.

Cross-checked against modeling_llada.py from HF GSAI-ML/LLaDA-8B-Base:
RMSNorm (pre-norm), RoPE, SwiGLU MLP, no attention bias, NO causal mask.

The single most important property: `forward` is BIDIRECTIONAL. Every position attends to
every other position. tests/test_bidirectional.py guards this.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return h.type_as(x) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    # q, k: (B, H, L, Dh) ; cos, sin: (1, 1, L, Dh)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.num_attention_heads
        self.head_dim = c.head_dim
        self.wq = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.wk = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.wv = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.wo = nn.Linear(c.hidden_size, c.hidden_size, bias=False)

    def forward(self, x, cos, sin):
        B, L, D = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos[:, :, :L], sin[:, :, :L])
        # is_causal=False -> full bidirectional attention. Packed sequences, no padding.
        o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        o = o.transpose(1, 2).reshape(B, L, D)
        return self.wo(o)


class SwiGLU(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.w1 = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)  # gate
        self.w3 = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)  # up
        self.w2 = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)  # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.attn_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.attn = Attention(c)
        self.mlp_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.mlp = SwiGLU(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class KhmerLLaDA(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.config = c
        self.embed = nn.Embedding(c.vocab_size, c.hidden_size)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.num_hidden_layers))
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        if c.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

        # RoPE tables (persistent=False -> recomputed on load, not stored in ckpt)
        inv_freq = 1.0 / (c.rope_theta ** (torch.arange(0, c.head_dim, 2).float() / c.head_dim))
        pos = torch.arange(c.max_position_embeddings).float()
        ang = torch.outer(pos, inv_freq)                 # (L, Dh/2)
        emb = torch.cat([ang, ang], dim=-1)              # (L, Dh)
        self.register_buffer("cos", emb.cos()[None, None], persistent=False)  # (1,1,L,Dh)
        self.register_buffer("sin", emb.sin()[None, None], persistent=False)

        self.apply(self._init_weights)
        # GPT-NeoX style residual scaling
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.num_hidden_layers))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, L) long. Returns logits (B, L, V)."""
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
            if not self.config.tie_word_embeddings:
                n -= self.lm_head.weight.numel()
        return n
