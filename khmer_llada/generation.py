"""
LLaDA-style sampler: semi-autoregressive block decoding + low-confidence remasking.

Adapted from generate.py in github.com/ML-GSAI/LLaDA for this model's forward(ids)->logits
signature. Do not invent a different sampler — matching the reference keeps results interpretable.

Unconditional Khmer generation: prompt_ids = [[BOS]] (or empty), gen_len masked positions.

Classifier-free guidance (cfg_scale > 0, paper §2.4 / Eq. 16, App. A.2): an extra forward pass
with the prompt itself replaced by <MASK> gives an "unconditional" prediction; the conditional
logits are pushed away from it in log-space. Inference-time only — no retraining needed. The
paper's own ablation (Table 6, LLaDA 8B Base) shows small but consistent gains (+0.3 to +2.0 pts
across ARC-C/HellaSwag/TruthfulQA/GPQA/PIQA, flat on WinoGrande) from this exact mechanism.
"""
import torch
import torch.nn.functional as F

from .constants import MASK_ID, EOS_ID, BOS_ID


def _transfer_counts(n_masked: int, steps: int):
    """How many tokens to commit at each step (even spread; LLaDA linear schedule)."""
    base = n_masked // steps
    rem = n_masked % steps
    return [base + (1 if i < rem else 0) for i in range(steps)]


@torch.no_grad()
def generate(
    model,
    prompt_ids: torch.Tensor,      # (1, P) long, or None for pure unconditional
    gen_len: int = 128,
    steps: int = 128,
    block_length: int | None = None,
    temperature: float = 0.0,
    remasking: str = "low_confidence",   # or "random"
    cfg_scale: float = 0.0,              # 0 = off; paper's ablation used {0.5, 1, 1.5, 2}
    mask_id: int = MASK_ID,
    eos_id: int = EOS_ID,
    device: str = "cuda",
) -> torch.Tensor:
    model.eval()
    if prompt_ids is None:
        prompt_ids = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
    prompt_ids = prompt_ids.to(device)
    P = prompt_ids.shape[1]

    block_length = block_length or gen_len
    assert gen_len % block_length == 0, "gen_len must be divisible by block_length"
    n_blocks = gen_len // block_length
    assert steps % n_blocks == 0, "steps must be divisible by n_blocks"
    steps_per_block = steps // n_blocks

    x = torch.full((1, P + gen_len), mask_id, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids

    for b in range(n_blocks):
        lo = P + b * block_length
        hi = P + (b + 1) * block_length
        n_block_masked = int((x[:, lo:hi] == mask_id).sum().item())
        counts = _transfer_counts(n_block_masked, steps_per_block)

        for i in range(steps_per_block):
            is_mask = x == mask_id
            if not is_mask.any():
                break

            if cfg_scale > 0.0:
                # Eq. 16 / App. A.2: contrast the conditional prediction against an
                # "unconditional" one where the prompt is itself replaced by <MASK>,
                # then push the conditional logits away from the unconditional ones.
                x_uncond = x.clone()
                x_uncond[:, :P] = mask_id
                logp_cond = F.log_softmax(model(x).float(), dim=-1)
                logp_uncond = F.log_softmax(model(x_uncond).float(), dim=-1)
                logits = (1 + cfg_scale) * logp_cond - cfg_scale * logp_uncond
            else:
                logits = model(x)

            if temperature > 0:
                probs = F.softmax(logits.float() / temperature, dim=-1)
                x0 = torch.multinomial(probs[0], num_samples=1).squeeze(-1).unsqueeze(0)
            else:
                x0 = logits.argmax(dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits.float(), dim=-1)
                conf = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)   # (1, P+gen_len)
            elif remasking == "random":
                conf = torch.rand_like(x0, dtype=torch.float)
            else:
                raise ValueError(remasking)

            # only consider currently-masked positions inside the active block
            conf = torch.where(is_mask, conf, torch.full_like(conf, -float("inf")))
            conf[:, :lo] = -float("inf")
            conf[:, hi:] = -float("inf")
            x0 = torch.where(is_mask, x0, x)

            k = counts[i]
            if k > 0:
                sel = torch.topk(conf[0], k=k).indices
                x[0, sel] = x0[0, sel]

        # early stop: freeze everything after the first EOS
        eos_pos = (x[0, P:] == eos_id).nonzero(as_tuple=True)[0]
        if eos_pos.numel() > 0:
            cut = P + int(eos_pos[0].item()) + 1
            x[0, cut:] = eos_id
            break

    return x[:, P:]
