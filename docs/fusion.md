# Controlled fusion experiments

This release contains the verified six-fusion runs for **QM9 dipole moment
(mu, Debye)** and **QM8 E1-CC2 (hartree)**. Their small reference CSVs are retained
under each experiment's `reference_results/`. These controlled experiment
records do not establish the provenance of every value in the manuscript's
all-property fusion tables. They must not be substituted for the geometry
A/B/C/D experiments, which have different cohorts.

`fusion/Model.py` is byte-identical to the shared model used in both reference
experiments. It implements `concat`, `mean`, `weighted`, `attention`,
`gate_no_interaction`, and `gate`. All branches produce 256-dimensional
embeddings. Encoders and common output-head initialization are fingerprinted
and compared across fusion methods within each seed. Fusion modules retain
their natural parameter counts; this is not a parameter-matched experiment.
No CMCR or hyperparameter search is run in these scripts.

The separate `benchmark/Model.py` also supports these six choices through
`benchmark/run.py --fusion`, with `gate` as the default; see the
[benchmark guide](benchmark.md#fusion-choices). It retains the original benchmark
encoders, including five SMILES CNN kernels `[1, 2, 3, 4, 5]`, whereas
`fusion/Model.py` uses six `[1, 2, 3, 4, 5, 6]`. The benchmark fusion options
follow the benchmark protocol and are not interchangeable with these
controlled fusion experiments.

The two trainers/loaders/summarizers remain separate because QM9 uses a fixed
random split and QM8 uses the audited conservative structural-group split.
Their seeds are 42, 2025, and 3407; split seed is 20260927. The protocol uses
FP32, batch 128, AdamW at 0.0002 with weight decay 0.0001, gradient clipping 0.5,
at most 200 epochs, at least 50 epochs, and early stopping patience 30.
Target scaling is fitted on training labels only. Validation MAE selects the
checkpoint. The queue waits for **all 18 training runs** before held-out
finalization. Results report mean and sample standard deviation across seeds.
The per-run cumulative time bound is 420 minutes; the queue also has a bounded
wall-time. Slower hardware can hit these limits, which is a retained-checkpoint
failure, not a completed result.

## Running

Run these commands from the repository root in an environment with PyTorch,
PyTorch Geometric, torch-scatter, NumPy, RDKit, and SmilesPE installed. Select a
torch-scatter build compatible with the PyTorch/CUDA installation.

```sh
python fusion/qm9_mu/run_queue.py --data /data/qm9_cache --runs /results/qm9_mu/runs --output /results/qm9_mu/queue --dry-run
python fusion/qm8_e1cc2/run_queue.py --data /data/qm8_cache --runs /results/qm8_e1cc2/runs --output /results/qm8_e1cc2/queue --dry-run
```

Remove `--dry-run` only to start the planned experiments. A dry run constructs
commands and does not load data or start training. The queue uses up to four
concurrent training processes on its selected GPU and serial finalization.
It does not rent machines, connect by SSH, or shut down a server. Use separate
empty output directories. Source-archive checkpoints and runs using these
portable scripts have distinct source fingerprints and must not be mixed.
Each run can resume its own checkpoints through the integrity checks.

For model/optimizer/checkpoint CPU checks without the datasets:

```sh
python fusion/qm9_mu/train_fusion.py --mode self-test --device cpu
python fusion/qm8_e1cc2/train_fusion.py --mode self-test --device cpu
python -m unittest discover -s tests -p test_fusion_release.py
```

The original protocol/queue/summarization tests are also kept beside each
trainer. They test splitting, absence of label leakage, resume checks, metric
aggregation, and the train/finalize barrier. Lightweight tests do not establish
GPU performance or full numerical replication.

## Cache contract and preparation

Each data directory contains `cache.npz` (loaded with `allow_pickle=False`) and
`metadata.json`. Both loaders validate the cache SHA256. Shared array keys are
`ids`, `targets`, `graph_x`, `edge_index`, `edge_attr`, `node_offsets`,
`edge_offsets`, `smiles_ids`, `points`, and `point_lengths`. Graph edge indices
are local to each molecule, and offsets identify concatenated graph slices.
`points` is float32 with shape `(N, max_atoms, 9)`; its channels are centered
x/y/z in angstrom, radius, and the one-hot elements C/F/H/N/O. Real atom counts
mask padding. Metadata contains `dims`, `target_names`, and `cache_sha256`.
QM8 additionally requires integer `group_ids`, the explicit target units, and
the author-source verification fields produced by metadata finalization.

The fusion cohorts are 133885 QM9 records (107108/13388/13389)
and 21747 QM8 records (17398/2174/2175). The official QM9 geometry-exclusion
list was not applied. The prepared QM8 input omits 39 of the 21786 raw records;
the reason for those omissions was not recovered.
These are not the common-success geometry cohorts.

QM9 cache preparation uses the original spreadsheet and original XYZ files:

```sh
python fusion/cache_tools/prepare_data.py --table /data/QM9.xlsx --xyz-dir /data/QM9 --output /data/qm9_cache
```

QM8 preparation uses a **multi-stage identity-verification procedure**.
A public dataset download alone does not satisfy its input requirements. It
requires the original `QM8.xlsx`, `QM8_clean.xlsx`, local `qm8_valid_*.xyz` files, and an audit directory
containing `qm8_row_mapping.csv`, `qm8_xyz_identity_manifest.csv`, and
`qm8_table_summary.json`. These original input files and full data caches are
not bundled in this code-only repository. The builder verifies their recorded
identities and hashes; do not replace them with guessed mappings or recompute
the retained cohort silently. If those inputs are unavailable, the exact
cache cannot be reconstructed from this repository alone.

```sh
python fusion/cache_tools/prepare_qm8.py --raw-table /data/QM8.xlsx --clean-table /data/QM8_clean.xlsx --xyz-dir /data/QM8 --audit-dir /data/qm8_audit --output /data/qm8_cache
python fusion/cache_tools/fetch_author_sources.py --output /data/qm8_cache/official_source
python fusion/cache_tools/compare_official.py --directory /data/qm8_cache --official-directory /data/qm8_cache/official_source
python fusion/cache_tools/replace_points_from_official.py --directory /data/qm8_cache
python fusion/cache_tools/finalize_metadata.py --directory /data/qm8_cache
```

The comparison verifies author labels, molecular identities, and coordinate
mapping. The next step preserves the input local-coordinate cache and
replaces its point coordinates with the verified author DFT XYZ, leaving other
arrays unchanged. The finalizer binds source comparison and point validation
to metadata. The strict cache/source hash checks require a mismatch to be
investigated rather than bypassed. The source downloader uses the recorded
immutable Git blob IDs and checks size, SHA256, and Git blob SHA1; it does not follow a moving branch.

## Source traceability

`fusion/SOURCE_PROVENANCE.json` records relative source names, source SHA256,
distributed-file SHA256, and each portability adaptation. These adaptations
cover shared model loading, its source-hash path, explicit cache-input
arguments, and omission of rental-cost metadata. Model equations, splits,
normalization, optimization, checkpoint selection, and metric definitions
match the experiment sources. Cache paths and script hashes contribute to
metadata identity, so reconstructed metadata need not have the source
archive's metadata-file hash. The reference CSVs preserve the recorded
experimental values.
