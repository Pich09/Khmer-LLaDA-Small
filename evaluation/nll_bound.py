"""
Monte-Carlo NLL upper bound — the validation metric for a masked-diffusion LM.

Masked-diffusion LMs have no exact perplexity. This averages the LLaDA loss over many
resampled noise levels t; lower is better, exp(bound) is a pseudo-perplexity.

Keep n_samples and the seed FIXED across every checkpoint and against the AR baseline,
or the numbers are not comparable. (Ports the estimator idea from get_log_likelihood.py.)
"""
import torch

from khmer_llada import forward_process, diffusion_loss


@torch.no_grad()
def nll_bound(model, batches, mask_id: int, n_samples: int = 64, eps: float = 1e-3,
              seed: int = 12345, pad_id: int | None = None) -> float:
    model.eval()
    g = torch.Generator(device="cpu")
    total, nb = 0.0, 0
    for batch in batches:
        g.manual_seed(seed + nb)          # deterministic per batch, stable across checkpoints
        s = 0.0
        for _ in range(n_samples):
            # re-seed torch's default RNG path deterministically for forward_process
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g).item()))
            xt, masked, p = forward_process(batch, mask_id, eps, pad_id=pad_id)
            s += diffusion_loss(model(xt), batch, masked, p).item()
        total += s / n_samples
        nb += 1
    return total / max(1, nb)
