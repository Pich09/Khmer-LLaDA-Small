"""
LLaDA masked discrete diffusion: forward (noising) process + training loss.

This is NOT BERT MLM (fixed mask ratio) and NOT next-token prediction.
Per-sequence noise level t ~ U(0,1], every token independently masked w.p. t,
loss reweighted by 1/t so it is an upper bound on NLL.

Reference: LLaDA paper + GUIDELINES.md in github.com/ML-GSAI/LLaDA
"""
import torch
import torch.nn.functional as F


def forward_process(x0: torch.Tensor, mask_id: int, eps: float = 1e-3, pad_id: int | None = None):
    """
    x0: (B, L) clean token ids from a PACKED sequence (no padding, in the common case).

    pad_id: if given, positions equal to pad_id are never masked and never contribute to the
      loss — used by the ~1% variable-length training sequences (paper §2.2 / Nie et al.),
      where a row is truncated to a random length and the remainder filled with PAD so the
      model also learns to handle generation lengths shorter than the training context.
      None (default) preserves the original no-padding behavior exactly.

    Returns
      xt      : (B, L) ids with masked positions set to mask_id
      masked  : (B, L) bool, True where a token was masked
      p_mask  : (B, L) float, the per-token mask probability (== clamped t), used by the loss
    """
    B, L = x0.shape
    t = torch.rand(B, device=x0.device)
    p = ((1.0 - eps) * t + eps)[:, None].expand(B, L)      # in [eps, 1], avoids 1/t blow-up
    masked = torch.rand(B, L, device=x0.device) < p
    if pad_id is not None:
        masked = masked & (x0 != pad_id)
    xt = torch.where(masked, torch.full_like(x0, mask_id), x0)
    return xt, masked, p


def diffusion_loss(logits: torch.Tensor, x0: torch.Tensor,
                   masked: torch.Tensor, p_mask: torch.Tensor) -> torch.Tensor:
    """
    Canonical LLaDA pretraining loss.

    logits : (B, L, V)
    x0     : (B, L) long   (clean targets)
    masked : (B, L) bool
    p_mask : (B, L) float

    Per masked token: CE / p_mask   (the 1/t weight — DO NOT omit)
    Normalized by B*L (total tokens), NOT by masked.sum().
    CE computed in fp32 for stability under autocast.
    """
    B, L = x0.shape
    if masked.sum() == 0:
        return logits.sum() * 0.0  # keep graph; degenerate batch (t ~ 0)
    tok_ce = F.cross_entropy(
        logits[masked].float(), x0[masked], reduction="none"
    )
    tok_ce = tok_ce / p_mask[masked]
    return tok_ce.sum() / (B * L)
