# Principal benchmark, modality ablations, and CMCR diagnostics

This directory contains the principal fixed-hyperparameter regression loop, all seven
nonempty combinations of SMILES (S), graphs (G), and point clouds (P), and the
three-branch CMCR training loop with gate and alignment exports. It is distinct
from the controlled fusion and geometry experiments elsewhere in this
repository. The six fusion choices on this benchmark backbone
use its own protocol; the controlled fusion experiments use different seeds
and selection protocols.

## Manuscript scope

The principal Section 4.1 tables report random-split comparisons, including literature-reported external baselines. Scaffold evaluation is documented in SI Section 5 / Table S6 and the [scaffold guide](scaffold.md); it uses a separate protocol and entry point from the commands in this benchmark guide.

## Installation

Use Python 3.11 and install PyTorch, PyTorch Geometric (`torch-geometric`), and
`torch-scatter` with mutually compatible builds. The scatter extension must
match the exact installed PyTorch and CPU/CUDA build; do not reuse an extension
wheel built for another PyTorch version. Then install:

```sh
python -m pip install -r environments/benchmark-requirements.txt
python -B benchmark/test_release.py
python -B benchmark/test_fusion.py
```

The requirements file declares compatible dependencies; it does not lock the
complete execution environment of every reported run. Local validation and its
limits are recorded in `benchmark/validation.json`.

## Inputs and alignment

Supply a CSV or XLSX label table, its exact SMILES and target column names,
a directory of dataset-provided DFT XYZ files, and an explicit CSV mapping:

```csv
row_index,smiles,xyz_file
0,C,dsgdb9nsd_000001.xyz
1,N,dsgdb9nsd_000002.xyz
```

`row_index` is zero-based table order. The `smiles` field must exactly match
that label-table record, and `xyz_file` is relative to the XYZ directory. The
map must contain exactly one distinct existing file for every row, in table
order. Construct it from verified dataset identifiers or a retained row-to-file
mapping; do not infer identity from alphabetic file order. If a prepared QM8
export uses `qm8_valid_00001.xyz`, its numbering must refer to the same jointly
filtered label-table rows. The example above demonstrates format, not a supplied
scientific dataset or a substitute for checking molecule IDs.

For example, when a prepared table already contains a verified `xyz_file`
column, the mapping can be exported without changing record order:

```python
import numpy as np
import pandas as pd

table = pd.read_csv("data/qm9.csv")
pd.DataFrame({
    "row_index": np.arange(len(table)),
    "smiles": table["smiles"],
    "xyz_file": table["xyz_file"],
}).to_csv("data/qm9_xyz_map.csv", index=False)
```

Invalid SMILES, nonfinite labels/coordinates, missing files, mapping mismatches,
and atom-composition disagreements fail immediately. No loader silently removes
a record. XYZ atom compositions (including H) are checked against the mapped
SMILES. This does not reconstruct bonding from coordinates or prove that two
same-formula isomers are identical: correct dataset identifiers in the map
remain required. Exact QM8 exclusion/XYZ preparation cannot be reconstructed
from the source loaders alone; use the documented prepared
input and preserve its IDs rather than silently filtering an official download.

The same table feeds graph, SMILES, target, and coordinate loading. Graph
canonicalization and atom/bond features, dataset-wide sorted feature
vocabularies, atom-wise SMILES tokenization, padding, and point-feature
construction are retained from the supplied code. The full benchmark contains
C/F/H/N/O, giving nine point features: centered xyz, radial distance from the
centroid, and five atom-type indicators. Coordinates are expected in angstrom.
Vocabulary construction uses the complete supplied dataset, not only the
training partition.

## Commands

From the repository root, adapt the column names to your prepared table:

```sh
python -B benchmark/run.py --dataset qm9 --table data/qm9.csv --smiles-column smiles --target mu --unit Debye --xyz-dir data/qm9_xyz --xyz-map data/qm9_xyz_map.csv --output results/qm9_mu --check-inputs
python -B benchmark/run.py --dataset qm9 --table data/qm9.csv --smiles-column smiles --target mu --unit Debye --xyz-dir data/qm9_xyz --xyz-map data/qm9_xyz_map.csv --output results/qm9_mu --modalities S G P
python -B benchmark/run.py --dataset qm8 --table data/qm8_clean.xlsx --smiles-column smiles --target E1-CC2 --unit hartree --xyz-dir data/qm8_xyz --xyz-map data/qm8_xyz_map.csv --output results/qm8_e1_cc2 --modalities S G P
```

For modality ablations, use `--modalities S`, `G`, `P`, `S G`, `S P`, `G P`, or
`S G P`. Inputs and the jointly valid population remain the same for all seven
conditions. Active single branches use the supplied single-modality head;
fusion is bypassed. Multi-branch runs use the selected fusion method, defaulting
to the interaction-augmented feature-level gate (`--fusion gate`). Single-branch
commands reject non-default fusion choices because no fusion is performed.

Add `--cmcr` to a full `S G P` command to train with CMCR and export per-molecule
and per-channel gates, embeddings, and alignment statistics. The diagnostic
export code assumes all three branches and the interaction-augmented gate; this
entry point requires `--fusion gate` and rejects partial-branch CMCR. Use a
different output parent for another target. Existing nonempty run directories
are protected from overwrite.

Labels are used exactly as supplied, with train-only target standardization and
inverse transformation before evaluation. `--unit` documents the unit; it does
not convert values. In particular, U0 atomization energies are not interchangeable
with raw total U0, and energy labels in hartree must not be called eV.

## Fusion choices

`benchmark/Model.py` provides the following choices for `--fusion`. The fixed
trainer, CMCR trainer, and optional Bayesian-optimization trainer all load this
model; CMCR supports only `gate` for its gate-export protocol.

| `--fusion` | Method |
| --- | --- |
| `concat` | Concat: concatenation |
| `mean` | Avg: average fusion |
| `weighted` | SW: scalar-weighted fusion |
| `attention` | Attn: attention fusion |
| `gate_no_interaction` | FLG: feature-level gate without interaction terms |
| `gate` | IA-FLG: interaction-augmented feature-level gate (default) |

For example, run average fusion with all three modalities:

```sh
python -B benchmark/run.py --dataset qm9 --table data/qm9.csv --smiles-column smiles --target mu --unit Debye --xyz-dir data/qm9_xyz --xyz-map data/qm9_xyz_map.csv --output results/qm9_mu --modalities S G P --fusion mean
```

Non-default fusion choices add `_fusion_<kind>` before `_Results` in
run-directory names; `gate` retains the existing names. Each run's
`run_config.json` records `fusion`.
The benchmark encoders use SMILES CNN kernels `[1, 2, 3, 4, 5]`.
The separate `fusion/Model.py` uses six CNN kernels. The benchmark fusion options
follow the benchmark protocol and are not interchangeable with the controlled
QM9 mu or QM8 E1-CC2 experiments described in the [fusion guide](fusion.md).

The six-fusion module and gate-only benchmark source use compatible parameter
names, shapes and evaluation arithmetic for gate checkpoints. Their layer
initialization order differs, so equal seeds do not guarantee equal initial
weights or training trajectories across the two implementations.

## Fixed protocol

- Seeds 42, 44, 46; independently seeded random 80/10/10 splits, with sorted row
  indices within each split. No structural grouping is introduced here.
- Batch size 128; 256-dimensional branch outputs; six SMILES Transformer layers
  with eight heads; six graph message-passing layers; point hidden width 1024.
- AdamW, learning rate 2e-4, weight decay 1e-4, betas (0.9, 0.999), supervised L1
  loss on standardized targets, gradient norm clipping at 0.5.
- ReduceLROnPlateau factor 0.5/patience 8; at most 200 epochs; early-stopping
  patience 10/min_delta 1e-4. The lowest validation L1 checkpoint is evaluated.
  The geometry protocol's minimum 50 epochs/patience 30 is not used here.
- CMCR weight 0.03, temperature 0.2, projection width 64. Supervised validation
  loss selects the checkpoint. In-batch training contrastive loss and global
  test retrieval are different procedures.

The scripts use nondeterministic cuDNN settings. Exact numerical agreement
across hardware/library versions is not promised. Aggregate metric/export
conventions differ between scripts; calculate explicitly stated sample SD from
the per-seed rows rather than assume every summary uses the same convention.

## CMCR diagnostic definitions

Encoder and projected embeddings are L2-normalized for cosine calculations.
The negative-pair routine samples one seeded permutation with
`numpy.random.default_rng(seed)` and replaces fixed-point entries using the
one-step-rolled permutation. It returns N pairs, not all N(N-1) pairs; this repair
can duplicate target indices and is not a uniform derangement. The same
seed-induced mapping is used for encoder and projection spaces. Different rows
are treated as different records without additional chemical-identity filtering.

Top-1 retrieval ranks against the complete test embedding pool in each direction.
Only queries are chunked (1024); the key pool is never limited to the minibatch.
`numpy.argmax` selects the first position in an exact tie. The reported full-input
runs contain 13,389 QM9 and 2,176 QM8 test records per seed. The pool size is
determined by the supplied population; these numbers are not hard-coded.

## Optional Bayesian optimization and provenance limits

`--bayes-opt` runs the separate supplied `Train_and_evaluation.py`: 15 Optuna
trials per seed followed by retraining of the selected configuration. It is
mutually exclusive with `--cmcr`. Its search space remains in that source file.
`--fusion` also selects the fusion method for this optional BO run; this is not
the controlled six-fusion experiment protocol.
This is an available implementation, not evidence that Bayesian optimization
produced the reported tables. The fixed scripts above do not run BO. Exact
reproduction of a Bayesian-optimization result requires its trial records and
selected settings; these records are not supplied here.

`benchmark/source_provenance.json` records source-file hashes using relative
names and identifies portability differences. The distributed interface provides:

1. One CLI configuration for dataset/table/target/XYZ mapping/output/modalities/fusion.
2. Explicit dataset paths, shared row alignment and fail-fast input checks.
3. Six fusion choices in `Model.py`, with `gate` as the benchmark default.
4. Explicit modality flags, independent of output-directory names.

The supplied `Model_1.py` source snapshot selects `ConcatFusion`, whereas the
benchmark entry point defaults to `gate`. Source availability and checkpoint
compatibility alone do not establish which snapshot generated every reported
table entry. Match source snapshots to their result records using the provenance
information.

The non-CMCR script retains best weights in memory but does not export
model checkpoints; it exports split membership, per-seed metrics, predictions,
training histories, and plots. CMCR also exports diagnostic arrays and tables.
Weights, BO logs and split records are not provided unless explicitly included
in the corresponding experiment collection.

No upstream license or third-party source declaration was found in the supplied
original code directory. This is a provenance limitation, not a declaration that
the code is free of third-party contributions. Dependency licenses are separate.
