# Supplementary scaffold-based evaluation

This collection documents the additional TriMo evaluation referenced in
Sections 3.2 and 4.1 of the revised manuscript, with methods in SI Section 5
and the random/scaffold comparison in Table S6. It contains 60 completed
single-target fits: eight QM9 properties and twelve QM8 tasks, each evaluated
on three frozen scaffold partitions. The model uses all three modalities and
the interaction-augmented gate, with CMCR disabled.

The principal manuscript tables still contain historical random-split results
and literature-reported baselines. The supplementary scaffold study, the
historical benchmark, the controlled fusion runs, and the geometry A/B/C/D
experiments are separate protocols. Their scores are not interchangeable.

## Published contents

- `scaffold/trimo/`: training, checkpoint, input-checking and final-evaluation
  source; 60 configurations; six split files; QM9 retained IDs; the plan and
  provenance records. No scheduler, SSH or deployment scripts are included.
- `scaffold/protocol.json`: concise machine-readable protocol description.
- `scaffold/summaries/`: 60 per-split final-test metrics, 20 property summaries,
  historical random-split MAEs, sample SDs and source hashes. See the
  [compact comparison table](../scaffold/summaries/README.md).
- `scaffold/publication_manifest.json`: original and published file hashes,
  including the four limited publication adaptations described below.

Required audited feature caches, metadata and input manifests are **external**.
This release includes exact molecule partitions, but it is not a self-contained
data/weights bundle or a converter from arbitrary QM8/QM9 CSV files. Checkpoints,
per-molecule prediction files, full server logs and private manuscript files are
not distributed. Source-receipt hashes are retained for provenance; their
presence does not mean the omitted artifacts are included.

## Populations and partitions

| Dataset | Retained molecules | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| QM9 | 130,831 | 104,664 | 13,083 | 13,084 |
| QM8 | 21,747 | 17,397 | 2,174 | 2,176 |

QM9 starts from the 133,885-record cache and excludes the 3,054 officially
uncharacterized records through the audited ID mapping. QM8 uses the existing
21,747-record population with its original, nonconsecutive molecule IDs and
raw-input SMILES. Do not substitute row positions for IDs, replace the SMILES
serialization, or silently remove records. These populations differ from the
geometry common-valid subsets of 132,486 QM9 and 21,689 QM8 molecules.

The split construction used RDKit 2026.03.3 and
`MurckoScaffoldSmiles(..., includeChirality=False)` on sanitized molecules from
the original input SMILES. Molecules sharing a scaffold stay in one partition.
For QM9, scaffold groups are additionally joined when a valid original or
relaxed representation shares a normalized identity, including matches across
the two columns. Identity normalization removes stereochemistry and explicit
hydrogens and uses nonchiral canonical SMILES; it does not neutralize molecules
or normalize tautomers. This joins 15,661 scaffold groups into 15,637 allocation
groups. The 612 unparseable relaxed representations are retained using their
available original grouping, so this is not a claim of comprehensive
DFT-identity verification. QM8 uses 2,932 raw-input scaffold groups; the
alternative cleaned representation is not used to join groups.

Groups larger than the target validation/test capacity, `floor(0.1 * N)`, are
placed intact in training. Remaining group keys are sorted and then shuffled
with `random.Random(split_seed)`; whole groups fill training, then validation,
with the remainder assigned to test according to the frozen allocation code.
The exact published ID lists are the authority for these experiments. The
empty-scaffold groups (13,390 QM9 and 3,620 QM8 molecules) are entirely in
training. Consequently, validation and test exclude the acyclic molecules in
those groups; generalization claims must reflect that coverage limitation.

Split seeds are **42, 2025 and 3407**. Each fit uses training seed **42**.
The reported sample SD (`ddof=1`) therefore measures variation across three
scaffold partitions, not repeated training seeds on one fixed partition.
Three partitions alone do not establish statistical significance.

The historical split builders depend on workspace-specific prepared caches,
official ID mappings and identity-audit inputs. They are not distributed here
as a purported portable reconstruction command. The frozen split files and
their hashes allow direct reuse of the actual partitions without guessing
those prerequisites.

## Fixed training and final testing

The published five-kernel SMILES encoder, graph encoder and point-cloud encoder
are used with their default three-branch interaction-augmented gate. Point
features are centered xyz, radial distance and five element indicators. The
frozen model is separate from the six-kernel controlled-fusion snapshot.

- Each target/split starts from scratch. There is no HPO or best-seed search.
- FP32, batch 128, AdamW learning rate 2e-4 and weight decay 1e-4;
  supervised L1 loss on train-standardized targets; gradient norm limit 0.5.
- Train-only target mean and population SD (`ddof=0`) are fitted separately
  for each training partition. Validation and test metrics use native units.
- ReduceLROnPlateau and architecture settings are frozen in the individual
  configs. At most 200 epochs, early-stopping patience 20, and the earliest
  strict minimum of raw validation MAE select the best checkpoint.
- Interrupted continuation may restore only the same run's source-bound
  checkpoint, optimizer, scheduler and RNG state.
- The separate final evaluator checks selection, source/data/partition hashes
  and the selected checkpoint before constructing the test loader. Test
  outcomes do not select hyperparameters, seed, checkpoint or stopping time.
  The unchanged NPZ loader can decode full arrays internally; only the
  designated partition is forwarded and scored at each stage.

HOMO, gap, ZPVE and QM8 excitation energies remain in **hartree**; QM8 oscillator
strengths are dimensionless. QM9 dipole moment uses Debye, polarizability a0^3,
spatial extent a0^2 and Cv cal/(mol K). U0 is the negative atomization quantity
`E_molecule - sum(n_A * E_A)` in kcal/mol, not raw total molecular U0. Labels
already stored in kcal/mol must not be converted a second time.

The verified final receipts record PyTorch 2.7.1+cu128, NumPy 2.4.6 and CUDA
12.8 on the RTX 4090 execution environment. The deployment runtime check
required Linux/Python 3.12, PyG 2.6.1 and torch-scatter 2.1.2+pt27cu128.
These compiled builds must match; `environments/scaffold-requirements.txt`
records them but is not a portable lockfile or a default-PyPI-only installer.
Install the matching CUDA/PyG wheels from their official distributions first.
Exact checkpoint runtime fields are checked during evaluation; dependency
versions absent from those fields are not retroactively inferred from weights.

## Reusing the frozen TriMo code

Run commands from the repository root in the compatible Python 3.12 environment.
First place the **matching audited cache**, `metadata.json` and `manifest.csv`
in an external directory such as `data/qm9_cache`. Required byte hashes are
listed in `scaffold/trimo/plan.json` and `protocol_bindings.json`. Arbitrary
reprocessed inputs will be rejected; do not remove hash checks to force them
through. This guide does not claim the missing caches can be regenerated
exactly from any standard dataset download.

Check inputs before fitting; the output parent directory must exist:

```sh
mkdir -p results/scaffold_qm9_mu
python -B scaffold/trimo/preflight_inputs.py --cache data/qm9_cache --split scaffold/trimo/release/data_prep/official_candidates/qm9/split_seed_42.json --split-dir scaffold/trimo/release/data_prep/official_candidates/qm9 --config scaffold/trimo/configs/qm9/dipole_moment/split_42.json --retained-ids scaffold/trimo/release/cohort_audit/qm9_official_retained_ids.json --native-collate --out results/scaffold_qm9_mu/input_check.json
```

Train the corresponding predeclared fit; `train` must not already exist:

```sh
python -B scaffold/trimo/release/runners/trimo/trial.py --cache data/qm9_cache --split scaffold/trimo/release/data_prep/official_candidates/qm9/split_seed_42.json --config scaffold/trimo/configs/qm9/dipole_moment/split_42.json --retained-ids scaffold/trimo/release/cohort_audit/qm9_official_retained_ids.json --frozen-dir scaffold/trimo/release/pilot --out results/scaffold_qm9_mu/train --device cuda:0
```

Only after this fit has completed, evaluate its selected checkpoint using the
**published** plan hash (the historical plan hash is different):

```sh
python -B scaffold/trimo/evaluate.py --plan scaffold/trimo/plan.json --plan-sha256 e842a7640b6f3b62c11502fdc84e4c1e4c5e4618e1ee699144f99c0caeb0e50a --run-id qm9__dipole_moment__split_42 --cache-dir data/qm9_cache --run-dir results/scaffold_qm9_mu/train --out results/scaffold_qm9_mu/evaluation --device cuda:0
```

For QM8, use its cache, config and split paths and omit `--retained-ids`.
All other run IDs and paths are explicitly listed in the plan. The evaluator
requires the runtime bound to the training checkpoint. Only load trusted
checkpoints; PyTorch checkpoint deserialization may execute code.

## Historical random comparison and publication adaptations

Historical random runs use seeds 42, 44 and 46, changing both the random split
and training RNG. Their QM9 population is 133,885, whereas the new scaffold
study uses 130,831. QM8 population counts agree, but equal counts alone do not
prove identical historical molecule membership. Training/selection settings
also differ, and the historical BO search logs have not been independently
recovered. Therefore the comparison reports a change between two experimental
protocols, not the isolated causal effect of scaffold splitting or an
equal-budget HPO comparison.

The observed scaffold MAEs increase on all twelve QM8 tasks (25.3–74.9%) and
on three QM9 targets; five QM9 targets have lower MAEs. These are descriptive
observations, not universal generalization claims or significance tests.

The original `frozen_sources.json` contained two private absolute source paths.
The publication copy replaces them with relative source names and updates the
dependent expected metadata hash in `evaluate.py`, the provenance entry, and
the plan's provenance hash. Those four files are marked as adapted. All model,
data-loader, training/checkpoint, target-binding, config and split bytes are
unchanged. The original and public SHA256 values are listed in
`scaffold/publication_manifest.json`. Historical numerical summaries retain
the original plan and evaluator hashes; they are not results regenerated with
the public adapter. The repository's `* -text` Git attribute preserves these
recorded bytes across operating systems.

Some unchanged internal identifiers still say `pilot`, `official_candidate`,
`DRAFT`, or `PREPARED_FIXED_RECIPE_NO_FITS_EXECUTED`. They describe the original
preparation snapshot and are preserved for provenance. The separately verified
final receipts establish completion of the 60 scientific fits. Similarly, the
old pilot QM9 candidate hash is historical; current split authority is
`release/runners/trimo/protocol_bindings.json` and the frozen plan.

## Separate follow-up baselines

The last audited completion snapshot (2026-10-24, Beijing time) records:

| Follow-up | Completed scope | Public result coverage in this update |
| --- | --- | --- |
| KRR on QM8 | Twelve targets on three frozen splits, 36 final metrics | Completion status only; numerical receipts were audited locally |
| AttentiveFP on QM9 | Eight targets on three frozen splits, 24 final tests | Completion status only; detailed metric receipts were not yet collected into this release |

These follow-ups are not substituted for the literature-baseline values in
the supplied manuscript's principal tables, and their runners are not part of
this TriMo scaffold source package. GC/DTNN, MPNN and PointGAT are not reported
as completed formal comparisons; PaiNN compatibility checks are not formal
accuracy results. No new training or GPU allocation was performed for this
publication update.
