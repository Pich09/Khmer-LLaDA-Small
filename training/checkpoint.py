"""Full-state checkpointing: model, optimizer, GradScaler, data position, RNG, progress."""
import os

import torch


def save(path: str, *, model, optimizer, scaler, loader, step: int, tokens_seen: int,
         model_config: dict, train_config: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "loader": loader.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "model_config": model_config,
        "train_config": train_config,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load(path: str, *, model, optimizer, scaler, loader, map_location="cuda",
         restore_loader_seed: bool = True):
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ck["optimizer"])
    if scaler is not None and ck.get("scaler") is not None:
        scaler.load_state_dict(ck["scaler"])
    if loader is not None:
        loader.load_state_dict(ck["loader"], restore_seed=restore_loader_seed)
    torch.set_rng_state(ck["torch_rng"].to("cpu", torch.uint8))
    if ck.get("cuda_rng") is not None and torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        states = [s.to("cpu", torch.uint8) for s in ck["cuda_rng"][:n_devices]]
        if states:
            torch.cuda.set_rng_state_all(states)
    return ck["step"], ck["tokens_seen"]
