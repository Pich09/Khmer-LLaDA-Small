"""The model MUST be bidirectional. If a causal mask sneaks in, this fails."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from khmer_llada import Config, KhmerLLaDA


def test_future_token_affects_past_logits():
    torch.manual_seed(0)
    cfg = Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                 intermediate_size=128, max_position_embeddings=16, vocab_size=100)
    model = KhmerLLaDA(cfg).eval()

    ids = torch.randint(5, 100, (1, 12))
    with torch.no_grad():
        base = model(ids)

    changed = ids.clone()
    changed[0, -1] = (changed[0, -1] + 7) % 100  # perturb the LAST token
    with torch.no_grad():
        after = model(changed)

    # logits at position 0 must move — only possible if 0 attends to later positions
    delta = (after[0, 0] - base[0, 0]).abs().max().item()
    assert delta > 1e-4, f"position 0 unaffected by last token (delta={delta}); attention is causal"


def test_first_token_affects_last_logits():
    """Symmetric check: perturbing token 0 must move the logits at the final position."""
    torch.manual_seed(0)
    cfg = Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                 intermediate_size=128, max_position_embeddings=16, vocab_size=100)
    model = KhmerLLaDA(cfg).eval()
    ids = torch.randint(5, 100, (1, 12))
    with torch.no_grad():
        base = model(ids)
    changed = ids.clone()
    changed[0, 0] = (changed[0, 0] + 7) % 100
    with torch.no_grad():
        after = model(changed)
    delta = (after[0, -1] - base[0, -1]).abs().max().item()
    assert delta > 1e-4, f"last position unaffected by first token (delta={delta})"
