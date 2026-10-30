"""One explicit TriMo train/validation trial. No test loader, sweep, or final evaluation."""
import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np

from boundaries import RawMetrics, id_list, select_rows, sha256, training_scaling, write_json

HERE = Path(__file__).resolve().parent
FROZEN = {"Model.py": "f20216dd869e73cced15f909d981199ca39f856b6fa6f8ce5b4ec0e7d7e38662",
          "data.py": "aee090c3296c72ff16aaf56e661e1f1f66da696c5ff75b31f448d9bd39c9b280"}


def import_frozen(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_inputs(cache_dir, split_path, config_path, frozen_dir, retained_path=None, parent_split_path=None):
    cache_dir, split_path, config_path, frozen_dir = map(Path, (cache_dir, split_path, config_path, frozen_dir))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["mode"] not in ("official_candidate", "engineering"):
        raise ValueError("Explicit mode must be official_candidate or engineering")
    if not 1 <= config["max_epochs"] <= (3 if config["mode"] == "engineering" else 200):
        raise ValueError("Epoch limit is 200, or 3 for the engineering split")
    if type(config["early_stop_patience"]) is not int or config["early_stop_patience"] < 1:
        raise ValueError("early_stop_patience must be a positive integer")
    if config["precision"] != "float32" or config["cmcr"] is not False:
        raise ValueError("This trial is FP32 with CMCR disabled")
    architecture = config["architecture"]
    if architecture["fusion_type"] != "gate" or architecture["d_output"] != 1:
        raise ValueError("This runner is single-target TriMo gate")
    if not all(architecture[k] is True for k in ("use_smiles", "use_graph", "use_point_cloud")):
        raise ValueError("All three original modalities are required")
    if config["batch_size"] < 2 or config["loss"] != "normalized_l1":
        raise ValueError("Use batch_size >=2 and normalized_l1 loss")
    if config["scheduler"]["name"] != "ReduceLROnPlateau":
        raise ValueError("Only the explicitly configured ReduceLROnPlateau scheduler is implemented")
    protocols = json.loads((HERE / "protocol_bindings.json").read_text(encoding="utf-8"))
    protocol = protocols[config["dataset"]]
    for name, digest in FROZEN.items():
        if sha256(frozen_dir / name) != digest:
            raise ValueError(f"Frozen source mismatch: {name}")
    metadata_path, cache_path = cache_dir / "metadata.json", cache_dir / "cache.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache_hash, metadata_hash = sha256(cache_path), sha256(metadata_path)
    if cache_hash != protocol["cache_sha256"] or cache_hash != metadata["cache_sha256"]:
        raise ValueError("Cache SHA256 differs from the audited source")
    if metadata_hash != protocol["metadata_sha256"]:
        raise ValueError("Cache metadata SHA256 differs from the audited source")
    if config["target"] not in metadata["target_names"]:
        raise ValueError("Target absent from audited cache")
    if config["target"] not in protocol["target_units"] or config["unit"] != protocol["target_units"][config["target"]]:
        raise ValueError("Target/unit must match the audited explicit unit mapping")
    with np.load(cache_path, allow_pickle=False) as archive:
        cache_ids = archive["ids"]  # No test targets selected for preflight.
    if len(cache_ids) != protocol["cache_count"]:
        raise ValueError("Unexpected cache population")
    retained_hash = None
    if config["dataset"] == "qm9":
        if retained_path is None or sha256(retained_path) != protocol["retained_sha256"]:
            raise ValueError("QM9 official retained-ID hash is required")
        retained_hash = sha256(retained_path)
        retained = json.loads(Path(retained_path).read_text(encoding="utf-8"))
    else:
        retained = cache_ids.tolist()
    if len(retained) != protocol["cohort_count"]:
        raise ValueError("Unexpected official cohort population")
    split_hash = sha256(split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    parent_hash = None
    if config["mode"] == "engineering":
        if parent_split_path is None or sha256(parent_split_path) not in protocol["split_sha256"].values():
            raise ValueError("Engineering split needs an audited parent official split")
        parent_hash = sha256(parent_split_path)
        parent = json.loads(Path(parent_split_path).read_text(encoding="utf-8"))
        select_rows(cache_ids, parent, retained)
        selected = [mid for part in split.values() for mid in part]
        if len(selected) > 4096 or not set(selected) <= set(parent["train"]):
            raise ValueError("Engineering IDs must be at most 4096 IDs from official train only")
        retained = id_list(selected, "engineering cohort")
    elif split_hash not in protocol["split_sha256"].values():
        raise ValueError("Split SHA256 is not one of the audited official candidates")
    selected_rows = select_rows(cache_ids, split, retained)
    if len(selected_rows["train"]) % config["batch_size"] == 1:
        raise ValueError("A singleton final training batch is incompatible with original BatchNorm; choose a different explicit batch size")
    binding = {"source": {name: sha256(HERE / name) for name in ("trial.py", "boundaries.py", "checkpoint.py")},
               "frozen_source": FROZEN, "config_sha256": sha256(config_path), "split_sha256": split_hash,
               "cache_sha256": cache_hash, "metadata_sha256": metadata_hash,
               "protocol_bindings_sha256": sha256(HERE / "protocol_bindings.json"),
               "retained_sha256": retained_hash, "engineering_parent_split_sha256": parent_hash}
    return config, metadata, split, selected_rows, binding


def run_trial(cache_dir, split_path, config_path, out_dir, device="cuda:0", frozen_dir=None,
              retained_path=None, parent_split_path=None, resume=None, engineering_stop_after_epoch=None):
    import torch
    from torch.utils.data import DataLoader, Subset
    from checkpoint import copy_checkpoint, restore_checkpoint, save_checkpoint

    frozen_dir = Path(frozen_dir) if frozen_dir else HERE.parents[1] / "pilot"
    config, metadata, split, rows, binding = prepare_inputs(
        cache_dir, split_path, config_path, frozen_dir, retained_path, parent_split_path)
    if engineering_stop_after_epoch is not None and (config["mode"] != "engineering"
            or not 1 <= engineering_stop_after_epoch <= config["max_epochs"]):
        raise ValueError("Engineering epoch pause is allowed only within an engineering run's configured cap")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested CPU/CUDA device is unavailable")
    out = Path(out_dir)
    if resume not in (None, "last"):
        raise ValueError("Resume only this run directory's own last checkpoint")
    if resume is None:
        out.mkdir(parents=True, exist_ok=False)
    elif not (out / f"{resume}.pt").is_file():
        raise ValueError("Own-run resume checkpoint is missing")
    loader_module = import_frozen(frozen_dir / "data.py", "trimo_frozen_data")
    model_module = import_frozen(frozen_dir / "Model.py", "trimo_frozen_model")
    selected = np.concatenate([rows["train"], rows["val"]])
    dataset = loader_module.QM9Dataset(cache_dir, target=config["target"], indices=selected, verify_hash=False)
    expected_ids = split["train"] + split["val"]
    if dataset.ids.tolist() != expected_ids:
        raise ValueError("Frozen loader ID order differs from explicit train/val mapping")
    n_train = len(rows["train"])
    scaling = training_scaling(dataset.targets[:n_train])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(config["torch_num_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(Subset(dataset, range(n_train)), batch_size=config["batch_size"],
                              shuffle=True, generator=generator, num_workers=0, drop_last=False,
                              collate_fn=loader_module.collate_qm9)
    # One saved generator covers both loaders without consuming model/dropout RNG.
    val_loader = DataLoader(Subset(dataset, range(n_train, len(dataset))), batch_size=config["batch_size"],
                            shuffle=False, generator=generator, num_workers=0, drop_last=False,
                            collate_fn=loader_module.collate_qm9)
    dims = metadata["dims"]
    architecture = {**config["architecture"], "vocab_size": int(dims["vocab_size"]),
                    "smiles_length": int(dims["smiles_length"]), "d_node": int(dims["node_dim"]),
                    "d_edge": int(dims["edge_dim"]), "d_point_input": int(dims["point_dim"])}
    model = model_module.MultimodalModel(**architecture).to(device=device, dtype=torch.float32)
    if model.smiles_model.cnn.kernel_sizes != [1, 2, 3, 4, 5]:
        raise ValueError("Original five CNN kernels changed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    schedule = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=schedule["factor"], patience=schedule["patience"],
        threshold=schedule["threshold"], threshold_mode="abs", min_lr=schedule["min_lr"])
    # Bind runtime because kernels/RNG behavior can differ between execution environments.
    binding["runtime"] = {"torch": str(torch.__version__), "numpy": np.__version__,
                          "cuda": torch.version.cuda, "device": str(device)}
    start_epoch, best_mae, best_epoch, bad_epochs, history = 1, float("inf"), None, 0, []
    if resume:
        saved = restore_checkpoint(out / f"{resume}.pt", model, optimizer, scheduler, generator, binding, scaling)
        start_epoch, best_mae, best_epoch = saved["epoch"] + 1, saved["best_mae"], saved["best_epoch"]
        bad_epochs = saved["bad_epochs"]
        history = saved["history"]
        del saved
    write_json(out / "run.json", {"status": "DRAFT single trial; no test evaluation", "config": config,
        "binding": binding, "architecture": architecture, "scaling": scaling,
        "train_count": n_train, "val_count": len(rows["val"]),
        "engineering_stop_after_epoch": engineering_stop_after_epoch,
        "label_policy": "Frozen loader decodes full arrays; training/validation only index explicit selected IDs. No test loader or test metrics.",
        "train_metric_scope": "per-sample raw-unit errors from online pre-update training predictions"})

    def epoch_pass(loader, part, train):
        model.train(train)
        metrics = RawMetrics(split[part])
        for batch in loader:
            batch_ids = batch["id"].tolist()
            if not set(batch_ids) <= metrics.expected:
                raise ValueError("Batch IDs cross their permitted partition")
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                prediction = model(batch["smiles_ids"].to(device), batch["graph"].to(device),
                                   batch["points"].to(device), batch["point_lengths"].to(device)).reshape(-1)
                truth = batch["y"].to(device).reshape(-1)
                normalized = ((truth.double() - scaling["mean"]) / scaling["std"]).float()
                if prediction.dtype != torch.float32 or prediction.shape != normalized.shape:
                    raise ValueError("Single-target FP32 prediction shape mismatch")
                if not torch.isfinite(prediction).all() or not torch.isfinite(normalized).all():
                    raise FloatingPointError("Nonfinite prediction/selected target")
                if train:
                    loss = (prediction - normalized).abs().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    loss.backward()
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
                        raise FloatingPointError("Missing/nonfinite gradient")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
                    optimizer.step()
            raw_prediction = prediction.detach().double().cpu().numpy() * scaling["std"] + scaling["mean"]
            metrics.update(batch_ids, raw_prediction, truth.detach().double().cpu().numpy())
        return metrics.result()

    epoch_cap = engineering_stop_after_epoch or config["max_epochs"]
    for epoch in range(start_epoch, epoch_cap + 1):
        if bad_epochs >= config["early_stop_patience"]:
            break
        training = epoch_pass(train_loader, "train", True)
        validation = epoch_pass(val_loader, "val", False)
        scheduler.step(validation["mae"])
        improved = validation["mae"] < best_mae
        if improved:
            best_mae, best_epoch, bad_epochs = validation["mae"], epoch, 0
        else:
            bad_epochs += 1
        record = {"epoch": epoch, "unit": config["unit"], "train": training, "validation": validation,
                  "learning_rate_next_epoch": optimizer.param_groups[0]["lr"], "bad_epochs": bad_epochs}
        history.append(record)
        save_checkpoint(out / "last.pt", model, optimizer, scheduler, generator, binding, epoch,
                        scaling, best_mae, best_epoch, bad_epochs, history)
        if improved:
            copy_checkpoint(out / "last.pt", out / "best.pt")
        write_json(out / "history.json", history)
        print(json.dumps(record, allow_nan=False), flush=True)
    result = {"status": "ENGINEERING_ONLY" if config["mode"] == "engineering" else "TRIAL_COMPLETE_PROTOCOL_DRAFT",
              "last_epoch": history[-1]["epoch"], "best_epoch": best_epoch, "best_validation_mae": best_mae,
              "bad_epochs": bad_epochs, "stop_reason": "early_stopping" if bad_epochs >= config["early_stop_patience"]
                  else ("epoch_cap" if history[-1]["epoch"] >= config["max_epochs"] else "engineering_pause"),
              "unit": config["unit"], "test_evaluated": False, "binding": binding}
    write_json(out / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frozen-dir")
    parser.add_argument("--retained-ids")
    parser.add_argument("--engineering-parent-split")
    parser.add_argument("--resume", choices=("last",))
    parser.add_argument("--engineering-stop-after-epoch", type=int)
    args = parser.parse_args()
    run_trial(args.cache, args.split, args.config, args.out, args.device, args.frozen_dir,
              args.retained_ids, args.engineering_parent_split, args.resume, args.engineering_stop_after_epoch)
