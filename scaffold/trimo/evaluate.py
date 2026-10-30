"""Final, test-only evaluation of one completed, predeclared TriMo scaffold fit.

Never calls run_trial/restore_checkpoint, an optimizer, or a train/val loader.
Only this run's trusted native checkpoints may be deserialized (pickle).
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


SOURCE_HASHES = {
    "runners/trimo/trial.py": "3408182c6f0c5b7fc72fd91ba1987aa4a612ec80814d2fecbcab2d5c50a11cf9",
    "runners/trimo/boundaries.py": "b6e9975fdfca0117c39a520d21110a1cc0491a87434453412698e45fbb345621",
    "runners/trimo/checkpoint.py": "3ba5d51d8f2d9abd3d990f6794ea05d8fc56524f4b3825ac174b1fc2f9f45cbf",
    "runners/trimo/protocol_bindings.json": "41bcfad3686afdc85ea08625c9b90fae35d6111b71ab24fb68523a235172e26c",
    "pilot/Model.py": "f20216dd869e73cced15f909d981199ca39f856b6fa6f8ce5b4ec0e7d7e38662",
    "pilot/data.py": "aee090c3296c72ff16aaf56e661e1f1f66da696c5ff75b31f448d9bd39c9b280",
    "pilot/frozen_sources.json": "8967e2f2da436b9e75f03698d0985c43aaf8c46b3d4737a3d2a733dd025f04ee",
}
TRAINING_FILES = ("best.pt", "last.pt", "run.json", "history.json", "result.json")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ordered_ids_sha256(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()


def inside(base, relative):
    path = (base / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(base.resolve()):
        raise ValueError("Release path escapes its plan directory")
    return path


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_run(plan_path, run_id, expected_plan_sha256):
    plan_path = Path(plan_path).resolve()
    if not expected_plan_sha256 or sha256(plan_path) != expected_plan_sha256:
        raise ValueError("Predeclared plan SHA256 mismatch")
    base, plan = plan_path.parent, read_json(plan_path)
    entries = [item for item in plan["runs"] if item["run_id"] == run_id]
    if len(entries) != 1 or plan["fit_count"] != 60:
        raise ValueError("Run must be unique in the predeclared sixty-fit plan")
    entry = entries[0]
    provenance_path = inside(base, plan["provenance_path"])
    if sha256(provenance_path) != plan["provenance_sha256"]:
        raise ValueError("Frozen provenance SHA256 mismatch")
    source_files = read_json(provenance_path)["source_files"]
    indexed = {item["source_path"]: item for item in source_files}
    if len(indexed) != len(source_files):
        raise ValueError("Duplicate provenance source")
    for relative, expected in SOURCE_HASHES.items():
        item = indexed[relative]
        if item["sha256"] != expected or item["path"] != "release/" + relative:
            raise ValueError(f"Unapproved training source: {relative}")
    for item in source_files:
        path = inside(base, item["path"])
        if sha256(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Release source mismatch: {item['path']}")
    for key in ("config", "split", "retained"):
        relative = entry[key + "_path"]
        if relative is not None and sha256(inside(base, relative)) != entry[key + "_sha256"]:
            raise ValueError(f"Predeclared {key} SHA256 mismatch")
    config = read_json(inside(base, entry["config_path"]))
    if (config["mode"] != "official_candidate" or entry["initialize_from"] is not None
            or entry["train_from_scratch"] is not True):
        raise ValueError("Final evaluation requires a predeclared official candidate from scratch")
    for config_key, entry_key in (("dataset", "dataset"), ("target", "target"), ("unit", "unit"),
                                  ("seed", "training_seed"), ("max_epochs", "max_epochs"),
                                  ("batch_size", "batch_size"), ("early_stop_patience", "early_stop_patience")):
        if config[config_key] != entry[entry_key]:
            raise ValueError("Config differs from its predeclared run entry")
    if plan["runner_path"] != "release/runners/trimo/trial.py" or plan["frozen_dir"] != "release/pilot":
        raise ValueError("Unexpected released runner/model path")
    return base, plan, entry, config


def validate_completed_trial(config, split, binding, architecture, run, history, result, last, best):
    """Validate selection metadata without any labels, loaders or state mutation."""
    if config["mode"] != "official_candidate" or not split["test"]:
        raise ValueError("A final official test partition is required")
    for record in (run, result, last, best):
        if record["binding"] != binding:
            raise ValueError("Source/config/split/cache/cohort/runtime binding mismatch")
    if (run["config"] != config or run["architecture"] != architecture
            or run["engineering_stop_after_epoch"] is not None
            or run["train_count"] != len(split["train"]) or run["val_count"] != len(split["val"])):
        raise ValueError("Run configuration/architecture/partition counts differ")
    scaling = run["scaling"]
    if (set(scaling) != {"mean", "std", "ddof", "n_fit"} or scaling["ddof"] != 0
            or scaling["n_fit"] != len(split["train"]) or scaling["std"] <= 0
            or not np.isfinite([scaling["mean"], scaling["std"]]).all()
            or last["scaling"] != scaling or best["scaling"] != scaling):
        raise ValueError("Train-only scaler mismatch or invalid statistics")
    if not history or last["history"] != history or len(history) > config["max_epochs"]:
        raise ValueError("Last checkpoint/history disagreement")
    best_mae, best_epoch, bad_epochs = float("inf"), None, 0
    for epoch, record in enumerate(history, 1):
        if record["epoch"] != epoch or record["unit"] != config["unit"]:
            raise ValueError("History epoch/unit mismatch")
        if bad_epochs >= config["early_stop_patience"]:
            raise ValueError("History continued past the frozen early-stop boundary")
        for part, ids in (("train", split["train"]), ("validation", split["val"])):
            metric = record[part]
            if (metric["n"] != len(ids) or not np.isfinite([metric["mae"], metric["rmse"]]).all()
                    or metric["mae"] < 0 or metric["rmse"] < 0):
                raise ValueError("History metric count/value mismatch")
        if record["validation"]["mae"] < best_mae:
            best_mae, best_epoch, bad_epochs = record["validation"]["mae"], epoch, 0
        else:
            bad_epochs += 1
        if record["bad_epochs"] != bad_epochs:
            raise ValueError("History best-selection counter mismatch")
    stop = ("early_stopping" if bad_epochs >= config["early_stop_patience"] else
            "epoch_cap" if len(history) == config["max_epochs"] else None)
    if stop is None:
        raise ValueError("Training has not reached its frozen terminal condition")
    if (last["epoch"] != len(history) or last["best_epoch"] != best_epoch
            or last["best_mae"] != best_mae or last["bad_epochs"] != bad_epochs
            or best["epoch"] != best_epoch or best["best_epoch"] != best_epoch
            or best["best_mae"] != best_mae or best["bad_epochs"] != 0
            or best["history"] != history[:best_epoch]):
        raise ValueError("Best/last checkpoints disagree with validation-only selection")
    expected = {"status": "TRIAL_COMPLETE_PROTOCOL_DRAFT", "last_epoch": len(history),
                "best_epoch": best_epoch, "best_validation_mae": best_mae, "bad_epochs": bad_epochs,
                "stop_reason": stop, "unit": config["unit"], "test_evaluated": False, "binding": binding}
    if result != expected:
        raise ValueError("Completed training result disagrees with checkpoint/history")
    return {"best_epoch": best_epoch, "best_validation_mae": best_mae,
            "last_epoch": len(history), "stop_reason": stop, "scaling": scaling}


def own_training_files(run_dir):
    run_dir = Path(run_dir).resolve()
    files = {name: run_dir / name for name in TRAINING_FILES}
    for name, path in files.items():
        if not path.is_file() or path.is_symlink() or path.resolve().parent != run_dir:
            raise ValueError(f"Expected this run's own trusted file: {name}")
    return files


def raw_prediction(normalized, scaling):
    return np.asarray(normalized, dtype=np.float64) * scaling["std"] + scaling["mean"]


def assert_model_unchanged(model, saved_state):
    import torch
    current = model.state_dict()
    if set(current) != set(saved_state):
        raise ValueError("Loaded model state keys changed during evaluation")
    for name, value in current.items():
        if not torch.equal(value.detach().cpu(), saved_state[name]):
            raise ValueError(f"Best model parameter/buffer changed: {name}")


def evaluate_run(plan_path, run_id, cache_dir, run_dir, out_dir, device="cuda:0", expected_plan_sha256=None):
    base, plan, entry, config = frozen_run(plan_path, run_id, expected_plan_sha256)
    runner_dir = inside(base, plan["runner_path"]).parent
    boundary_path = runner_dir / "boundaries.py"
    if "boundaries" in sys.modules:
        if Path(sys.modules["boundaries"].__file__).resolve() != boundary_path:
            raise ValueError("An unrelated boundaries module is already imported")
        boundary = sys.modules["boundaries"]
    else:
        boundary = load_module(boundary_path, "boundaries")
        sys.modules["boundaries"] = boundary
    trial = load_module(runner_dir / "trial.py", "trimo_final_trial")
    config_path, split_path = (inside(base, entry[key]) for key in ("config_path", "split_path"))
    frozen_dir = inside(base, plan["frozen_dir"])
    retained = inside(base, entry["retained_path"]) if entry["retained_path"] else None
    checked_config, metadata, split, _, binding = trial.prepare_inputs(
        cache_dir, split_path, config_path, frozen_dir, retained_path=retained)
    if checked_config != config or {part: len(ids) for part, ids in split.items()} != entry["counts"]:
        raise ValueError("Validated inputs differ from the predeclared plan")
    import torch
    from torch.utils.data import DataLoader
    device = torch.device(device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested CPU/CUDA device is unavailable")
    binding["runtime"] = {"torch": str(torch.__version__), "numpy": np.__version__,
                          "cuda": torch.version.cuda, "device": str(device)}
    files = own_training_files(run_dir)
    original_hashes = {name: sha256(path) for name, path in files.items()}
    run, history, result = (read_json(files[name]) for name in ("run.json", "history.json", "result.json"))
    # Check external JSON bindings before deserializing own-run trusted native states.
    boundary.check_binding(run["binding"], binding)
    boundary.check_binding(result["binding"], binding)
    last = torch.load(files["last.pt"], map_location="cpu", weights_only=False)
    best = torch.load(files["best.pt"], map_location="cpu", weights_only=False)
    dims = metadata["dims"]
    architecture = {**config["architecture"], "vocab_size": int(dims["vocab_size"]),
                    "smiles_length": int(dims["smiles_length"]), "d_node": int(dims["node_dim"]),
                    "d_edge": int(dims["edge_dim"]), "d_point_input": int(dims["point_dim"])}
    selected = validate_completed_trial(config, split, binding, architecture, run, history, result, last, best)
    if best["epoch"] == last["epoch"] and original_hashes["best.pt"] != original_hashes["last.pt"]:
        raise ValueError("Native last-is-best checkpoints must be byte-identical")
    del last
    saved_model = best["model"]
    if not saved_model or any(not torch.isfinite(value).all() for value in saved_model.values()):
        raise ValueError("Best model state is empty or nonfinite")
    del best
    # All selection gates above precede construction of the only data loader: test.
    with np.load(Path(cache_dir) / "cache.npz", allow_pickle=False) as archive:
        row_for_id = {int(mid): row for row, mid in enumerate(archive["ids"])}
    test_ids = boundary.id_list(split["test"], "final test")
    test_rows = np.asarray([row_for_id[mid] for mid in test_ids], dtype=np.int64)
    loader_module = trial.import_frozen(frozen_dir / "data.py", "trimo_final_data")
    model_module = trial.import_frozen(frozen_dir / "Model.py", "trimo_final_model")
    torch.set_num_threads(config["torch_num_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = model_module.MultimodalModel(**architecture).to(device=device, dtype=torch.float32)
    model.load_state_dict(saved_model, strict=True)
    if model.smiles_model.cnn.kernel_sizes != [1, 2, 3, 4, 5]:
        raise ValueError("Original five CNN kernels changed")
    model.eval()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    dataset = loader_module.QM9Dataset(cache_dir, target=config["target"], indices=test_rows, verify_hash=False)
    if dataset.ids.tolist() != test_ids:
        raise ValueError("Frozen test loader ID order differs from the official split")
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0,
                        drop_last=False, collate_fn=loader_module.collate_qm9,
                        generator=torch.Generator().manual_seed(config["seed"]))
    metric, offset = boundary.RawMetrics(test_ids), 0
    prediction_path = out / "predictions.csv"
    temporary = prediction_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle, torch.inference_mode():
        writer = csv.writer(handle)
        writer.writerow(["molecule_id", "target", "unit", "target_value", "prediction", "signed_error"])
        for batch in loader:
            ids = batch["id"].tolist()
            if ids != test_ids[offset:offset + len(ids)]:
                raise ValueError("Test loader repeated/reordered/crossed partition IDs")
            prediction = model(batch["smiles_ids"].to(device), batch["graph"].to(device),
                               batch["points"].to(device), batch["point_lengths"].to(device)).reshape(-1)
            if prediction.dtype != torch.float32 or prediction.shape != (len(ids),):
                raise ValueError("Expected one FP32 normalized model prediction per test ID")
            values = raw_prediction(prediction.double().cpu().numpy(), selected["scaling"])
            truth = batch["y"].reshape(-1).double().cpu().numpy()
            metric.update(ids, values, truth)
            writer.writerows([mid, config["target"], config["unit"], float(y), float(p), float(p - y)]
                             for mid, y, p in zip(ids, truth, values))
            offset += len(ids)
    metrics = metric.result()
    assert_model_unchanged(model, saved_model)
    if {name: sha256(path) for name, path in files.items()} != original_hashes:
        raise ValueError("Training/checkpoint files changed during final evaluation")
    temporary.replace(prediction_path)
    receipt = {"status": "PASS_FINAL_TEST_EVALUATION", "run_id": run_id, "dataset": config["dataset"],
        "target": config["target"], "unit": config["unit"], "split_seed": entry["split_seed"],
        "training_seed": config["seed"], "selection": selected, "test": metrics, "binding": binding,
        "plan_sha256": expected_plan_sha256, "provenance_sha256": plan["provenance_sha256"],
        "evaluator_sha256": sha256(__file__), "training_file_sha256": original_hashes,
        "ids_sha256": {part: ordered_ids_sha256(ids) for part, ids in split.items()},
        "ids_hash_encoding": "SHA256 of ordered JSON integer list; separators=(comma,colon); ASCII; no newline",
        "test_ids": test_ids, "prediction_file": prediction_path.name,
        "prediction_sha256": sha256(prediction_path), "model_parameters_and_buffers_unchanged": True,
        "training_files_unchanged": True, "loaders_constructed": ["test"], "optimizer_updates": 0,
        "scaler_check": "Original checkpoint has mean/std/ddof/n_fit, no independent ID hash. Exact split/source/cache binding, training ID hash and n_fit are checked; best/last/run scalers agree. Training labels/statistics are not recomputed.",
        "label_policy": "Frozen NPZ loader decodes full arrays/target column; only explicit official test rows are forwarded and scored. No train/validation loader or new selection is performed.",
        "runtime_scope": "Exact native checkpoint runtime fields: torch/numpy/CUDA/device. Dependency package versions not recorded by the original checkpoint cannot be retroactively certified.",
        "selection_policy": "Predeclared fixed recipe; earliest strict minimum validation MAE; test cannot select configuration, seed, checkpoint or stopping."}
    boundary.write_json(out / "result.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = evaluate_run(args.plan, args.run_id, args.cache_dir, args.run_dir, args.out,
                          args.device, expected_plan_sha256=args.plan_sha256)
    print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                      "test": result["test"], "unit": result["unit"]}, allow_nan=False))
