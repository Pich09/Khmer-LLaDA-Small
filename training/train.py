"""
T4 training loop for Khmer-LLaDA-Small.

fp16 + dynamic GradScaler + grad clip + grad accumulation + Warmup-Stable-Decay LR + resume.
Schedule is by TOKENS, not epochs (PLAN.md §8). Validation = Monte-Carlo NLL bound.

LR schedule (paper §2.2, scaled to our token budget instead of their literal 1.2T/0.8T/0.3T):
  1. linear warmup: 0 -> lr over warmup_steps
  2. hold at lr until stable_frac of total_steps        (paper: 1.2/2.3 ~= 52.17%)
  3. step down to mid_lr, hold until decay1_frac         (paper: 2.0/2.3 ~= 86.96%)
  4. linear decay mid_lr -> min_lr for the remainder      (paper: last 0.3/2.3 ~= 13.04%)

Single GPU:
  python training/train.py --model-config configs/small_a.json --train-config configs/train_t4.yaml
  python training/train.py --model-config configs/small_a.json --train-config configs/train_t4.yaml \
      --resume checkpoints/last.pt

Multi-GPU (DistributedDataParallel, one process per GPU via torchrun):
  torchrun --standalone --nproc_per_node=2 training/train.py \
      --model-config configs/small_a.json --train-config configs/train_t4.yaml
  # tokens_per_step scales by world_size automatically; total_steps derived from total_tokens
  # shrinks accordingly, so total_tokens in the yaml stays the actual token budget either way.
"""
import argparse
import contextlib
import datetime
import json
import math
import os
import sys
import time
from dataclasses import asdict

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from khmer_llada import Config, KhmerLLaDA, forward_process, diffusion_loss  # noqa: E402
from training import checkpoint as ckpt  # noqa: E402
from training.dataset import PackedShards, ResumableLoader, fixed_val_batches  # noqa: E402
from evaluation.nll_bound import nll_bound  # noqa: E402


# paper §2.2's own breakpoints as fractions of total tokens: stable through 1.2/2.3, hold at
# mid_lr through 2.0/2.3, decay to min_lr over the last 0.3/2.3.
WSD_STABLE_FRAC = 1.2 / 2.3
WSD_DECAY1_FRAC = 2.0 / 2.3


def lr_at(tokens_seen, tc, tokens_per_step):
    """Schedule keyed on tokens_seen/total_tokens, not step/total_steps.

    step counts optimizer updates, and tokens_per_step scales with world_size — so if a run
    is resumed under a different world_size (e.g. 1 GPU -> 2 GPUs), the same step number no
    longer means the same amount of training progress, and total_steps itself changes too.
    tokens_seen is a real, world_size-invariant quantity (it's summed as tokens actually
    consumed), so anchoring the schedule to it keeps lr continuous across such a resume.
    """
    warmup_tokens = tc["warmup_steps"] * tokens_per_step
    total_tokens = tc["total_tokens"]
    peak_lr = tc["lr"]
    mid_lr = tc.get("mid_lr", peak_lr / 4)   # paper: 4e-4 -> 1e-4, i.e. peak/4
    min_lr = tc["min_lr"]

    if tokens_seen < warmup_tokens:
        return peak_lr * tokens_seen / max(1, warmup_tokens)

    stable_end = WSD_STABLE_FRAC * total_tokens
    decay1_end = WSD_DECAY1_FRAC * total_tokens

    if tokens_seen < stable_end:
        return peak_lr
    if tokens_seen < decay1_end:
        return mid_lr
    prog = (tokens_seen - decay1_end) / max(1, total_tokens - decay1_end)
    prog = min(1.0, prog)
    return mid_lr + (min_lr - mid_lr) * prog   # linear decay, mid_lr -> min_lr


def write_status(ckpt_dir, *, step, total_steps, tokens_seen, total_tokens, lr, train_loss,
                 val_nll_bound, best_val, run_name):
    """Small JSON sidecar next to the .pt files — step/tokens/val at a glance without loading
    the full checkpoint. Written on every checkpoint save; used to resume with --resume and to
    check progress from another process/shell while training is running."""
    os.makedirs(ckpt_dir, exist_ok=True)
    status = {
        "run_name": run_name,
        "step": step, "total_steps": total_steps,
        # progress by tokens, not step/total_steps — total_steps is an estimate that shifts
        # with world_size (see lr_at's docstring), while tokens_seen/total_tokens is exact.
        "tokens_seen": tokens_seen, "total_tokens": total_tokens,
        "progress": round(tokens_seen / max(1, total_tokens), 4),
        "lr": lr, "train_loss": train_loss,
        "val_nll_bound": val_nll_bound, "best_val_nll_bound": best_val,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = os.path.join(ckpt_dir, "status.json.tmp")
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, os.path.join(ckpt_dir, "status.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--train-config", required=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    # DDP: torchrun sets these env vars, one process per GPU. Absent -> single-process as before.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = rank == 0
    ddp = world_size > 1

    if ddp:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=45))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    is_cuda = device.startswith("cuda")

    with open(args.train_config) as f:
        tc = yaml.safe_load(f)
    # each rank processes physical_batch*grad_accum*seq_len tokens/step; DDP syncs gradients
    # every step, so the true tokens consumed per optimizer step scales with world_size.
    tokens_per_step = tc["physical_batch"] * tc["grad_accum"] * tc["seq_len"] * world_size
    tc["total_steps"] = int(tc["total_tokens"] / tokens_per_step)
    torch.manual_seed(tc["seed"] + rank)  # decorrelate init noise (dropout etc.) across ranks

    cfg = Config.from_json(args.model_config)
    assert cfg.max_position_embeddings >= tc["seq_len"]
    raw_model = KhmerLLaDA(cfg).to(device)
    if ddp:
        # broadcast rank 0's init weights to every rank so all replicas start identical
        for p in raw_model.parameters():
            dist.broadcast(p.data, src=0)
        model = DDP(raw_model, device_ids=[local_rank])
    else:
        model = raw_model
    if tc.get("gradient_checkpointing"):
        # simple: wrap each block with checkpoint at call time (see note in PLAN.md §8)
        if is_main:
            print("NOTE: enable gradient checkpointing by wrapping blocks in modeling.py if needed")
    if is_main:
        print(f"params: {raw_model.num_parameters()/1e6:.1f}M total | "
              f"{raw_model.num_parameters(non_embedding=True)/1e6:.1f}M non-embedding")
        print(f"world_size: {world_size}  steps: {tc['total_steps']:,}  "
              f"tokens/step: {tokens_per_step:,}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=tc["lr"], betas=(tc["beta1"], tc["beta2"]),
        weight_decay=tc["weight_decay"], fused=is_cuda,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(tc["precision"] == "fp16" and is_cuda))

    shards = PackedShards(tc["data_glob"], tc["seq_len"])
    # rank-distinct data stream: large odd offset keeps rank 0 identical to the non-DDP seed
    loader = ResumableLoader(shards, tc["physical_batch"], seed=tc["seed"] + rank * 100_003)
    val_batches = [b.to(device) for b in fixed_val_batches(
        tc["val_glob"], tc["seq_len"], tc["physical_batch"], tc["val_batches"], seed=0)]

    step, tokens_seen = 0, 0
    if args.resume:
        # every rank loads the same (rank 0's) file; keep each rank's own seed so post-resume
        # data streams stay distinct instead of collapsing onto rank 0's stream on every rank.
        step, tokens_seen = ckpt.load(args.resume, model=raw_model, optimizer=opt,
                                      scaler=scaler, loader=loader, map_location=device,
                                      restore_loader_seed=not ddp)
        if is_main:
            print(f"resumed at step {step}, tokens_seen {tokens_seen:,}")

    run_dir = os.path.join("experiments", tc.get("run_name", "run"))
    metrics_f = None
    wandb = None
    best_val = float("inf")
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
        # recover best_val from a prior run of the same run_name so best.pt tracking survives
        # a --resume instead of resetting to inf and overwriting a genuinely-best checkpoint.
        if os.path.exists(metrics_path):
            for line in open(metrics_path):
                try:
                    v = json.loads(line).get("val_nll_bound")
                    if v is not None:
                        best_val = min(best_val, v)
                except Exception:
                    pass
            if best_val < float("inf") and is_main:
                print(f"best_val recovered from metrics.jsonl: {best_val:.4f}")
        metrics_f = open(metrics_path, "a")
        with open(os.path.join(run_dir, "manifest.json"), "w") as f:
            json.dump({"model_config": asdict(cfg), "train_config": tc,
                       "world_size": world_size}, f, indent=2)
        if tc.get("wandb"):
            import wandb as _wandb
            wandb = _wandb
            wandb.init(project=tc["wandb_project"], name=tc.get("run_name"),
                       config={"model": asdict(cfg), "train": tc}, dir=run_dir)

    model.train()
    t0 = time.time()
    tokens_seen_at_t0 = tokens_seen  # so tok/s is correct right after --resume, not inflated
    last_train_loss = None
    while tokens_seen < tc["total_tokens"]:
        for g in opt.param_groups:
            g["lr"] = lr_at(tokens_seen, tc, tokens_per_step)

        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for micro in range(tc["grad_accum"]):
            batch = next(loader).to(device)
            # DDP all-reduces gradients on every .backward() by default; with grad_accum
            # micro-steps that's grad_accum all-reduces per optimizer step instead of one.
            # no_sync() defers the all-reduce to only the last micro-step.
            last_micro = micro == tc["grad_accum"] - 1
            sync_ctx = model.no_sync() if (ddp and not last_micro) else contextlib.nullcontext()
            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=(tc["precision"] == "fp16" and is_cuda)):
                    xt, masked, p = forward_process(batch, cfg.mask_token_id, pad_id=cfg.pad_token_id)
                    loss = diffusion_loss(model(xt), batch, masked, p) / tc["grad_accum"]
                scaler.scale(loss).backward()
            loss_acc += loss.item()

        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip_norm"])
        scaler.step(opt)
        scaler.update()

        step += 1
        tokens_seen += tokens_per_step

        # logging, validation, and checkpointing only on rank 0 — tokens_seen/step already
        # match across ranks (deterministic arithmetic), so this is a report, not a reduction.
        if is_main and step % tc["log_every_steps"] == 0:
            tok_s = (tokens_seen - tokens_seen_at_t0) / max(1e-9, time.time() - t0)
            rec = {"step": step, "tokens_seen": tokens_seen, "train_loss": loss_acc,
                   "lr": opt.param_groups[0]["lr"], "grad_norm": float(gnorm),
                   "tok_per_s": round(tok_s)}
            print(rec, flush=True)
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()
            if wandb:
                wandb.log(rec, step=step)
            last_train_loss = loss_acc

        if is_main and step % tc["val_every_steps"] == 0:
            vb = nll_bound(raw_model, val_batches, cfg.mask_token_id,
                           n_samples=tc["val_nll_samples"], pad_id=cfg.pad_token_id)
            rec = {"step": step, "val_nll_bound": vb, "val_pseudo_ppl": math.exp(vb)}
            print(rec, flush=True)
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()
            if wandb:
                wandb.log(rec, step=step)
            raw_model.train()
            if vb < best_val:
                best_val = vb
                ckpt.save(os.path.join(tc["ckpt_dir"], "best.pt"), model=raw_model, optimizer=opt,
                          scaler=scaler, loader=loader, step=step, tokens_seen=tokens_seen,
                          model_config=asdict(cfg), train_config=tc)
                print(f"  new best val_nll_bound {vb:.4f} -> best.pt", flush=True)

        if is_main and step % tc["save_every_steps"] == 0:
            ckpt.save(os.path.join(tc["ckpt_dir"], "last.pt"), model=raw_model, optimizer=opt,
                      scaler=scaler, loader=loader, step=step, tokens_seen=tokens_seen,
                      model_config=asdict(cfg), train_config=tc)
            write_status(tc["ckpt_dir"], step=step, total_steps=tc["total_steps"],
                        tokens_seen=tokens_seen, total_tokens=tc["total_tokens"],
                        lr=opt.param_groups[0]["lr"], train_loss=last_train_loss,
                        val_nll_bound=None, best_val=best_val if best_val < float("inf") else None,
                        run_name=tc.get("run_name", "run"))

        if ddp:
            # validation (Monte-Carlo nll_bound, up to val_batches*val_nll_samples forward
            # passes) and checkpoint saving above run only on rank 0 and can take minutes.
            # Other ranks would otherwise race ahead into the next step's forward pass, which
            # DDP starts with an implicit buffer broadcast from rank 0 — with rank 0 still
            # busy, that collective sits idle past the default 600s NCCL watchdog timeout and
            # the whole process group gets torn down. Barrier here so every rank waits for
            # rank 0's off-loop work before any of them touches the next collective.
            dist.barrier()

    if is_main:
        ckpt.save(os.path.join(tc["ckpt_dir"], "final.pt"), model=raw_model, optimizer=opt,
                  scaler=scaler, loader=loader, step=step, tokens_seen=tokens_seen,
                  model_config=asdict(cfg), train_config=tc)
        write_status(tc["ckpt_dir"], step=step, total_steps=tc["total_steps"],
                    tokens_seen=tokens_seen, total_tokens=tc["total_tokens"],
                    lr=opt.param_groups[0]["lr"], train_loss=locals().get("last_train_loss"),
                    val_nll_bound=None, best_val=best_val if best_val < float("inf") else None,
                    run_name=tc.get("run_name", "run"))
        metrics_f.close()
        print("done.")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
