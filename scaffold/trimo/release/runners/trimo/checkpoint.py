"""Epoch-boundary state for checkpoints written by this trial runner."""
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from boundaries import check_binding, write_json


def capture_rng(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "loader_generator": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader_generator"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(path, model, optimizer, scheduler, generator, binding, epoch,
                    scaling, best_mae, best_epoch, bad_epochs, history):
    path = Path(path)
    state = {"binding": binding, "epoch": epoch, "scaling": scaling,
             "best_mae": best_mae, "best_epoch": best_epoch, "bad_epochs": bad_epochs, "history": history,
             "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "rng": capture_rng(generator)}
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def copy_checkpoint(source, destination):
    """Publish the exact committed last-state bytes without another serialization."""
    destination = Path(destination)
    temporary = destination.with_suffix(".pt.tmp")
    with Path(source).open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer)
    temporary.replace(destination)


def restore_checkpoint(path, model, optimizer, scheduler, generator, binding, scaling):
    # Caller supplies only this run directory's own last.pt.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    check_binding(saved["binding"], binding)
    if saved["scaling"] != scaling:
        raise ValueError("Resume train-only target scaler mismatch")
    if (not 1 <= saved["best_epoch"] <= saved["epoch"]
            or not np.isfinite(saved["best_mae"])
            or not saved["history"] or saved["history"][-1]["epoch"] != saved["epoch"]):
        raise ValueError("Resume checkpoint selection/history metadata is inconsistent")
    best_path = Path(path).with_name("best.pt")
    if saved["epoch"] != saved["best_epoch"]:
        # An older best cannot be reconstructed from the current last weights.
        if not best_path.is_file():
            raise ValueError("Resume best checkpoint is missing and last is not best")
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        check_binding(best["binding"], binding)
        if (best["scaling"] != scaling or best["epoch"] != saved["best_epoch"]
                or best["best_epoch"] != saved["best_epoch"] or best["best_mae"] != saved["best_mae"]):
            raise ValueError("Resume best checkpoint disagrees with authoritative last selection")
        del best
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    if saved["epoch"] == saved["best_epoch"]:
        copy_checkpoint(path, best_path)
    # Repair even if the epoch cap or early stopping prevents another update.
    write_json(Path(path).with_name("history.json"), saved["history"])
    restore_rng(saved["rng"], generator)
    return saved
