"""Print the analytic + actual parameter breakdown for a model config.

  python scripts/count_params.py configs/small_a.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from khmer_llada import Config, KhmerLLaDA  # noqa: E402


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "configs/small_a.json"
    cfg = Config.from_json(path)
    print(f"config: {path}")
    for k, v in cfg.param_count().items():
        print(f"  {k:16s} {v:,}" if isinstance(v, int) else f"  {k:16s} {v}")
    try:
        import torch  # noqa: F401
        m = KhmerLLaDA(cfg)
        print(f"  actual total     {m.num_parameters():,}")
        print(f"  actual non-emb   {m.num_parameters(non_embedding=True):,}")
    except Exception as e:  # torch missing / no memory — analytic numbers still printed
        print(f"  (skipped actual instantiation: {e})")


if __name__ == "__main__":
    main()
