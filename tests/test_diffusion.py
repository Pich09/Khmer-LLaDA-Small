import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from khmer_llada import Config, KhmerLLaDA, forward_process, diffusion_loss
from khmer_llada.constants import MASK_ID


def test_mask_rate_matches_t_on_average():
    torch.manual_seed(0)
    x0 = torch.randint(5, 8000, (256, 128))
    fracs = []
    for _ in range(200):
        _, masked, p = forward_process(x0, MASK_ID)
        fracs.append(masked.float().mean().item())
    # E[t] = 0.5 (uniform), so mean masked fraction ~ 0.5
    assert 0.45 < sum(fracs) / len(fracs) < 0.55


def test_masked_positions_are_mask_id():
    x0 = torch.randint(5, 8000, (4, 32))
    xt, masked, _ = forward_process(x0, MASK_ID)
    assert (xt[masked] == MASK_ID).all()
    assert (xt[~masked] == x0[~masked]).all()


def test_loss_at_init_is_near_ln_vocab():
    torch.manual_seed(0)
    cfg = Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                 intermediate_size=128, max_position_embeddings=64, vocab_size=8000)
    model = KhmerLLaDA(cfg).eval()
    x0 = torch.randint(5, 8000, (8, 64))
    xt, masked, p = forward_process(x0, MASK_ID)
    with torch.no_grad():
        loss = diffusion_loss(model(xt), x0, masked, p).item()
    import math
    # per-token CE ~ ln(8000) ~ 8.99; the 1/t weight and /(B*L) roughly cancel to ~ln V
    assert 5.0 < loss < 14.0, loss


def test_pad_positions_never_masked():
    """paper §2.2's 1% variable-length trick: PAD-filled tail must never be masked or lossed."""
    from khmer_llada.constants import PAD_ID
    torch.manual_seed(0)
    x0 = torch.randint(5, 8000, (32, 64))
    x0[:, 40:] = PAD_ID  # simulate a truncated row
    xt, masked, p = forward_process(x0, MASK_ID, pad_id=PAD_ID)
    assert not masked[:, 40:].any(), "PAD positions must never be selected for masking"
    assert (xt[:, 40:] == PAD_ID).all(), "PAD positions must stay PAD in xt"


def test_gradients_flow_to_all_params():
    torch.manual_seed(0)
    cfg = Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                 intermediate_size=128, max_position_embeddings=64, vocab_size=8000)
    model = KhmerLLaDA(cfg)
    x0 = torch.randint(5, 8000, (8, 64))
    xt, masked, p = forward_process(x0, MASK_ID)
    diffusion_loss(model(xt), x0, masked, p).backward()
    missing = [n for n, q in model.named_parameters() if q.requires_grad and q.grad is None]
    # RoPE buffers are not params; tied lm_head shares embed grad
    assert not missing, f"no grad for: {missing}"
