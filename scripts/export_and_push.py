"""
Export a finished checkpoint into a portable, model-only directory and push it to the
Hugging Face Hub (Panhapich/Khmer-LLaDA-Small, under pretrained/). Mirrors the export cell
in notebooks/train_kaggle.ipynb section 17.

  HF_TOKEN=hf_... python scripts/export_and_push.py \
      --ckpt checkpoints/final.pt --repo Panhapich/Khmer-LLaDA-Small
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="defaults to checkpoints/final.pt, else best.pt, else last.pt")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--export-dir", default="export/khmer-llada-small-pretrained")
    ap.add_argument("--repo", default="Panhapich/Khmer-LLaDA-Small")
    ap.add_argument("--path-in-repo", default="pretrained")
    ap.add_argument("--tokenizer-dir", default="tokenizer")
    ap.add_argument("--push", action="store_true", default=True)
    ap.add_argument("--no-push", dest="push", action="store_false")
    args = ap.parse_args()

    src = args.ckpt
    if not src:
        for name in ("final.pt", "best.pt", "last.pt"):
            p = os.path.join(args.ckpt_dir, name)
            if os.path.exists(p):
                src = p
                break
    assert src and os.path.exists(src), f"no checkpoint found (looked in {args.ckpt_dir})"

    os.makedirs(args.export_dir, exist_ok=True)
    st = torch.load(src, map_location="cpu", weights_only=False)

    torch.save(st["model"], os.path.join(args.export_dir, "model.pt"))
    json.dump(st["model_config"], open(os.path.join(args.export_dir, "config.json"), "w"), indent=2)

    git_commit = ""
    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                                     ).stdout.strip()
    except Exception:
        pass

    json.dump({
        "source_checkpoint": os.path.basename(src),
        "step": st.get("step"),
        "tokens_seen": st.get("tokens_seen"),
        "git_commit": git_commit,
        "tokenizer": "Panhapich/khmer-sp-8k (vocab 8000, specials PAD=0 UNK=1 BOS=2 EOS=3 MASK=4)",
        "param_keys": sorted(st["model"].keys()),
    }, open(os.path.join(args.export_dir, "meta.json"), "w"), indent=2, ensure_ascii=False)

    if os.path.isdir(args.tokenizer_dir):
        shutil.copytree(args.tokenizer_dir, os.path.join(args.export_dir, "tokenizer"),
                         dirs_exist_ok=True)

    print(f"exported {src} -> {args.export_dir}")
    print(f"  step {st.get('step')}  tokens_seen {st.get('tokens_seen'):,}  "
          f"{len(st['model'])} weight tensors")
    for f in sorted(os.listdir(args.export_dir)):
        print("  ", f)

    if not args.push:
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("no HF_TOKEN in env — skipping push. Set HF_TOKEN and re-run with the same args.")
        return

    from huggingface_hub import HfApi, create_repo
    create_repo(args.repo, repo_type="model", exist_ok=True, token=token)
    HfApi().upload_folder(folder_path=args.export_dir, path_in_repo=args.path_in_repo,
                          repo_id=args.repo, repo_type="model", token=token,
                          commit_message=f"pretrain export @ step {st.get('step')}")
    print(f"pushed -> {args.repo}/{args.path_in_repo}")


if __name__ == "__main__":
    main()
