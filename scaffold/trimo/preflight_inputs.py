"""Fail-closed input preflight; no model construction, fitting or test-label metrics."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = ("42", "2025", "3407")
ATOM_KEYS = ("element", "degree", "total_h", "formal_charge", "hybridization", "chirality", "atom_ring_size")
BOND_KEYS = ("bond_type", "stereo", "bond_ring_size")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_trial(runner_dir):
    runner_dir = Path(runner_dir).resolve()
    sys.path.insert(0, str(runner_dir))
    # A CLI process uses one explicit release. Refuse an already imported foreign boundary module.
    if "boundaries" in sys.modules:
        require(Path(sys.modules["boundaries"].__file__).resolve() == runner_dir / "boundaries.py",
                "A different runner boundaries module is already imported")
    spec = importlib.util.spec_from_file_location("trimo_preflight_trial", runner_dir / "trial.py")
    trial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trial)
    return trial


def check_splits(ids, retained, split_dir, protocol, trial):
    result = {}
    for seed in SEEDS:
        path = Path(split_dir) / f"split_seed_{seed}.json"
        digest = trial.sha256(path)
        require(digest == protocol["split_sha256"][seed], f"Seed {seed}: split hash mismatch")
        split = read(path)
        rows = trial.select_rows(ids, split, retained)
        require(ids[rows["train"]].tolist() == split["train"] and ids[rows["val"]].tolist() == split["val"],
                f"Seed {seed}: molecule ID to cache-row mismatch")
        result[seed] = {"sha256": digest, "counts": {key: len(value) for key, value in split.items()},
                        "unique_complete_cohort_coverage": True}
    return result


def categorical_block(values, widths, boolean_before, trailing_mass, context):
    require(values.dtype == np.float32 and np.isfinite(values).all(), context + ": nonfinite/non-FP32 features")
    offset = 0
    for index, width in enumerate(widths):
        if index == boolean_before:
            require(np.isin(values[:, offset], (0, 1)).all(), context + ": invalid boolean feature")
            offset += 1
        block = values[:, offset:offset + width]
        require(block.shape[1] == width and np.isin(block, (0, 1)).all() and (block.sum(1) == 1).all(),
                context + ": invalid categorical one-hot block")
        offset += width
    if trailing_mass:
        require((values[:, offset] > 0).all(), context + ": invalid atomic mass")
        offset += 1
    require(offset == values.shape[1], context + ": feature layout mismatch")


def check_structure(arrays, metadata, cohort_ids):
    """Inspect every retained molecule's existing numerical model inputs, without targets."""
    dims, ids = metadata["dims"], arrays["ids"]
    require(ids.ndim == 1 and ids.dtype.kind in "iu" and (ids > 0).all() and len(set(ids.tolist())) == len(ids),
            "Cache IDs must be unique positive integers")
    row_for_id = {int(mid): row for row, mid in enumerate(ids)}
    require(len(cohort_ids) == len(set(cohort_ids)) and set(cohort_ids) <= set(row_for_id), "Invalid retained cohort IDs")
    nodes, edges, x, edge_attr = arrays["node_offsets"], arrays["edge_offsets"], arrays["graph_x"], arrays["edge_attr"]
    edge_index, smiles = arrays["edge_index"], arrays["smiles_ids"]
    points, lengths = arrays["points"], arrays["point_lengths"]
    n = len(ids)
    require(x.ndim == 2 and x.shape[1] == dims["node_dim"], "Node feature dimensions mismatch")
    require(edge_attr.ndim == 2 and edge_attr.shape[1] == dims["edge_dim"], "Edge feature dimensions mismatch")
    for offsets, size, label, positive in ((nodes, len(x), "node", True), (edges, len(edge_attr), "edge", False)):
        require(offsets.shape == (n + 1,) and offsets.dtype.kind in "iu" and offsets[0] == 0
                and offsets[-1] == size and (np.diff(offsets) >= (1 if positive else 0)).all(),
                label + " offsets do not delimit cache rows")
    require(edge_index.shape == (2, len(edge_attr)) and edge_index.dtype.kind in "iu", "Invalid edge index array")
    require(smiles.shape == (n, dims["smiles_length"]) and smiles.dtype.kind in "iu", "Invalid SMILES array")
    require(points.ndim == 3 and points.shape[0] == n and points.shape[2] == dims["point_dim"] == 9
            and points.dtype == np.float32, "Invalid point-cloud dimensions/dtype")
    if "max_atoms" in dims:
        require(points.shape[1] == dims["max_atoms"], "Point capacity differs from metadata")
    require(lengths.shape == (n,) and lengths.dtype.kind in "iu", "Invalid point length array")
    vocab = metadata["token_to_idx"]
    require(vocab.get("PAD") == 0 and sorted(vocab.values()) == list(range(dims["vocab_size"])), "Invalid token vocabulary")
    require(dims["smiles_length"] >= 5, "Original five CNN kernels require length >= 5")
    atom_schema, bond_schema = metadata["atom_schema"], metadata["bond_schema"]
    for schema, keys in ((atom_schema, ATOM_KEYS), (bond_schema, BOND_KEYS)):
        require(set(schema) == set(keys) and all(schema[key] and len(set(schema[key])) == len(schema[key]) for key in keys),
                "Invalid categorical schema")
    require(sum(len(atom_schema[key]) for key in ATOM_KEYS) + 2 == dims["node_dim"], "Atom schema width mismatch")
    require(sum(len(bond_schema[key]) for key in BOND_KEYS) + 1 == dims["edge_dim"], "Bond schema width mismatch")
    point_elements = [name.removeprefix("element_") for name in metadata["point_features"][4:]]
    require(metadata["point_features"][:4] == ["centered_x_angstrom", "centered_y_angstrom", "centered_z_angstrom", "rho_angstrom"]
            and len(point_elements) == 5 and len(set(point_elements)) == 5
            and set(atom_schema["element"]) <= set(point_elements), "Unsupported point feature schema")
    empty_graph_ids, min_nodes, max_nodes, max_points, token_ids = [], None, 0, 0, set()
    for mid in cohort_ids:
        row = row_for_id[mid]
        n0, n1 = nodes[row:row + 2]
        e0, e1 = edges[row:row + 2]
        node_count, length = int(n1 - n0), int(lengths[row])
        context = f"Molecule {mid}"
        categorical_block(x[n0:n1], [len(atom_schema[key]) for key in ATOM_KEYS], 6, True, context + " atoms")
        categorical_block(edge_attr[e0:e1], [len(bond_schema[key]) for key in BOND_KEYS], 1, False, context + " bonds")
        graph_edges = edge_index[:, e0:e1]
        require((graph_edges >= 0).all() and (graph_edges < node_count).all(), context + ": nonlocal edge indices")
        require(graph_edges.shape[1] % 2 == 0 and np.array_equal(graph_edges[:, ::2], graph_edges[::-1, 1::2])
                and np.array_equal(edge_attr[e0:e1:2], edge_attr[e0 + 1:e1:2]), context + ": missing native reverse edge pair")
        require(not (graph_edges[0] == graph_edges[1]).any()
                and len(set(map(tuple, graph_edges.T.tolist()))) == graph_edges.shape[1], context + ": duplicate/self edges")
        if not graph_edges.shape[1]:
            empty_graph_ids.append(mid)
        tokens = smiles[row]
        require((tokens >= 0).all() and (tokens < dims["vocab_size"]).all() and (tokens != 0).any(), context + ": invalid/empty token sequence")
        require(not ((tokens[:-1] == 0) & (tokens[1:] != 0)).any(), context + ": nontrailing sequence padding")
        token_ids.update(tokens.tolist())
        require(1 <= length <= points.shape[1], context + ": point length outside capacity")
        cloud = points[row, :length]
        require(np.isfinite(points[row]).all() and (points[row, length:] == 0).all(), context + ": nonfinite points or nonzero padding")
        require(np.isin(cloud[:, 4:], (0, 1)).all() and (cloud[:, 4:].sum(1) == 1).all(), context + ": invalid point element category")
        require(np.allclose(cloud[:, 3], np.linalg.norm(cloud[:, :3], axis=1), atol=1e-5, rtol=1e-5), context + ": inconsistent point radius")
        for index, element in enumerate(atom_schema["element"]):
            require(x[n0:n1, index].sum() == cloud[:, 4 + point_elements.index(element)].sum(),
                    context + ": graph/point element counts differ")
        min_nodes = node_count if min_nodes is None else min(min_nodes, node_count)
        max_nodes, max_points = max(max_nodes, node_count), max(max_points, length)
    return {"cohort_molecules_checked": len(cohort_ids), "cache_molecules": n,
            "excluded_cache_rows_not_content_checked": n - len(cohort_ids),
            "all_retained_structure_rows_checked": True, "categorical_blocks_checked": True,
            "local_bidirectional_edges_checked": True, "graph_point_element_counts_checked": True,
            "zero_edge_molecules": len(empty_graph_ids), "zero_edge_example_ids": empty_graph_ids[:10],
            "min_graph_nodes": min_nodes, "max_graph_nodes": max_nodes, "max_valid_points": max_points,
            "observed_token_ids": sorted(token_ids), "dims": dims}


def check_labels(targets, target_index, rows, training_scaling):
    train = targets[rows["train"], target_index].astype(np.float64)
    val = targets[rows["val"], target_index].astype(np.float64)
    require(np.isfinite(val).all(), "Nonfinite active validation labels")
    return {"train_count": len(train), "validation_count": len(val), "scaling": training_scaling(train),
            "active_test_target_rows_indexed": False, "statistics_scope": "active split train only; ddof=0"}


def native_collate(cache_dir, frozen_dir, config, split, rows, trial):
    import torch
    loader = trial.import_frozen(Path(frozen_dir) / "data.py", "trimo_preflight_native_data")
    selected = np.concatenate([rows["train"], rows["val"]])
    dataset = loader.QM9Dataset(cache_dir, target=config["target"], indices=selected, verify_hash=False)
    expected = split["train"] + split["val"]
    require(dataset.ids.tolist() == expected, "Native loader ID order mismatch")
    batches = 0
    n_train = len(rows["train"])
    for begin, end in ((0, n_train), (n_train, len(dataset))):
        for start in range(begin, end, config["batch_size"]):
            stop = min(start + config["batch_size"], end)
            samples = [dataset[index] for index in range(start, stop)]
            batch = loader.collate_qm9(samples)
            require(batch["id"].tolist() == expected[start:stop], "Native collate ID mismatch")
            graph = batch["graph"]
            require(graph.num_graphs == len(samples) and graph.ptr[-1].item() == graph.x.shape[0], "Native graph boundaries mismatch")
            require(graph.edge_index.dtype == torch.long and batch["smiles_ids"].dtype == torch.long
                    and batch["point_lengths"].dtype == torch.long, "Native index dtype mismatch")
            for value in (graph.x, graph.edge_attr, batch["points"]):
                require(value.dtype == torch.float32 and torch.isfinite(value).all().item(), "Native nonfinite/non-FP32 features")
            require(batch["y"].shape == (len(samples), 1) and batch["y"].dtype == torch.float64
                    and torch.isfinite(batch["y"]).all().item()
                    and torch.equal(batch["y"].reshape(-1), torch.from_numpy(dataset.targets[start:stop])),
                    "Native selected-label shape/value/finite mismatch")
            require(graph.edge_index.numel() == 0 or (graph.edge_index.min().item() >= 0 and graph.edge_index.max().item() < len(graph.x)
                    and torch.equal(graph.batch[graph.edge_index[0]], graph.batch[graph.edge_index[1]])),
                    "Native batched edge index outside its molecule")
            batches += 1
    return {"performed": True, "molecules_checked": len(dataset), "batches": batches,
            "all_active_train_validation_rows_checked": True, "active_test_samples_loaded": False,
            "model_forward_or_training_performed": False, "torch": str(torch.__version__)}


def preflight(cache_dir, split_path, split_dir, config_path, frozen_dir, runner_dir, retained_path=None, run_native=False):
    trial = load_trial(runner_dir)
    config, metadata, split, rows, binding = trial.prepare_inputs(
        cache_dir, split_path, config_path, frozen_dir, retained_path)
    require(config["mode"] == "official_candidate", "This gate requires an official candidate split")
    protocol = read(Path(runner_dir) / "protocol_bindings.json")[config["dataset"]]
    with np.load(Path(cache_dir) / "cache.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files if key != "targets"}
        retained = read(retained_path) if config["dataset"] == "qm9" else arrays["ids"].tolist()
        split_audit = check_splits(arrays["ids"], retained, split_dir, protocol, trial)
        require(binding["split_sha256"] in [value["sha256"] for value in split_audit.values()], "Active split absent from three split set")
        structure = check_structure(arrays, metadata, retained)
        targets = archive["targets"]  # NPZ decompresses the matrix; indexing below is only active train/val.
        require(targets.shape == (len(arrays["ids"]), len(metadata["target_names"])) and targets.dtype.kind == "f", "Invalid target matrix shape/dtype")
        labels = check_labels(targets, metadata["target_names"].index(config["target"]), rows, trial.training_scaling)
    del arrays, targets
    native = native_collate(cache_dir, frozen_dir, config, split, rows, trial) if run_native else {"performed": False, "required_before_training": True}
    return {"status": "PASS_NATIVE_INPUT_PREFLIGHT" if run_native else "PASS_STRUCTURE_NATIVE_PENDING",
            "dataset": config["dataset"], "target": config["target"], "unit": config["unit"],
            "preflight_source_sha256": trial.sha256(__file__), "binding": binding,
            "all_splits": split_audit, "structure": structure, "selected_labels": labels, "native_collate": native,
            "label_policy": "Full compressed target matrix is decoded. Only active train/validation target rows are indexed or checked; no active test metrics or test statistics. The unchanged native loader also materializes its full target-column view internally.",
            "limitations": "Numerical input/schema and native loading coverage only; no chemical identity rederivation, model forward, gradient or scientific-performance claim.",
            "training_started": False, "test_evaluated": False, "cohort_modified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "split", "split-dir", "config", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--runner-dir", type=Path, default=HERE / "release/runners/trimo")
    parser.add_argument("--frozen-dir", type=Path, default=HERE / "release/pilot")
    parser.add_argument("--retained-ids", type=Path)
    parser.add_argument("--native-collate", action="store_true")
    args = parser.parse_args()
    try:
        result = preflight(args.cache, args.split, args.split_dir, args.config, args.frozen_dir,
                           args.runner_dir, args.retained_ids, args.native_collate)
    except Exception as error:
        result = {"status": "FAILED_INPUT_PREFLIGHT", "error": f"{type(error).__name__}: {error}",
                  "training_started": False, "cohort_modified": False, "test_evaluated": False}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "receipt": str(args.out), "error": result.get("error")}))
    raise SystemExit(0 if result["status"].startswith("PASS_") else 1)
