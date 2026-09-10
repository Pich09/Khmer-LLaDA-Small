"""
MILESTONE-1 GATE.

Train the Tiny model on a few hundred Khmer sequences (data/overfit.npy, seq_len 128).
Masked loss MUST fall below ~0.1 nats/token within a few thousand steps.

If it plateaus high, the bug is in one of: mask construction, loss (missing 1/t, wrong
normalization), attention (accidental causal mask), or the label path. Fix that before
spending any real GPU time. Run tests/ too.

Uses fp32 on purpose — fewer moving parts for a correctness gate.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from khmer_llada import Config, KhmerLLaDA, forward_process, diffusion_loss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/overfit.npy")
    ap.add_argument("--model-config", default="configs/tiny.json")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--target", type=float, default=0.1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    cfg = Config.from_json(args.model_config)
    model = KhmerLLaDA(cfg).to(device)
    print(f"model params: {model.num_parameters() / 1e6:.2f}M")

    data = torch.tensor(np.load(args.data).astype(np.int64), device=device)
    assert data.shape[1] == cfg.max_position_embeddings, (
        f"overfit data seq_len {data.shape[1]} != config max_position_embeddings "
        f"{cfg.max_position_embeddings}"
    )
    print(f"overfit set: {data.shape[0]} sequences of length {data.shape[1]}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    best = float("inf")
    model.train()
    for step in range(args.steps):
        b = data[torch.randint(0, data.shape[0], (args.batch_size,), device=device)]
        xt, masked, p = forward_process(b, cfg.mask_token_id)
        loss = diffusion_loss(model(xt), b, masked, p)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        best = min(best, loss.item())
        if step % 200 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  best {best:.4f}")
        if best < args.target:
            print(f"\nPASS: reached loss {best:.4f} < {args.target} at step {step}")
            return

    print(f"\nFAIL: best loss {best:.4f} did not reach {args.target}. "
          f"Check mask / loss / attention (see module docstring).")
    sys.exit(1)


if __name__ == "__main__":
    main()
