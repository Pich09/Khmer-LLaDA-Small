"""
Dump generations for a fixed prompt set at a checkpoint — same prompts, seed, and steps
across checkpoints so quality is comparable over training.

  python evaluation/generate_eval.py --ckpt checkpoints/last.pt --steps 128
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from khmer_llada import Config, KhmerLLaDA  # noqa: E402
from khmer_llada.generation import generate  # noqa: E402
from khmer_llada.tokenizer import KhmerSPTokenizer  # noqa: E402

DEFAULT_PROMPTS = [
    "",  # pure unconditional
    "ប្រទេសកម្ពុជា",
    "នៅថ្ងៃនេះ",
    "បញ្ញាសិប្បនិម្មិត",
    "ការសិក្សា",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-config", default="configs/small_a.json")
    ap.add_argument("--prompts-file", default=None, help="one prompt per line; overrides defaults")
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--block-length", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cfg-scale", type=float, default=0.0,
                    help="classifier-free guidance scale, 0=off (paper ablated {0.5,1,1.5,2})")
    ap.add_argument("--tokenizer-mode", default="wrapper", choices=["wrapper", "sp_direct"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    cfg = Config.from_json(args.model_config)
    model = KhmerLLaDA(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    tok = KhmerSPTokenizer(mode=args.tokenizer_mode)

    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = [l.rstrip("\n") for l in open(args.prompts_file, encoding="utf-8")]

    results = []
    for prm in prompts:
        pid = None
        if prm:
            ids = [cfg.bos_token_id] + tok.encode(prm)
            pid = torch.tensor([ids], dtype=torch.long, device=device)
        out = generate(model, pid, gen_len=args.gen_len, steps=args.steps,
                       block_length=args.block_length, temperature=args.temperature,
                       cfg_scale=args.cfg_scale,
                       mask_id=cfg.mask_token_id, eos_id=cfg.eos_token_id, device=device)
        text = tok.decode(out[0].tolist())
        results.append({"prompt": prm, "generation": text})
        print(f"[{prm!r}] -> {text}")

    out_path = args.out or (os.path.splitext(args.ckpt)[0] + ".gens.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
