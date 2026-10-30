# Validation scope

## Implementation and packaging checks

| Component | Completed checks |
| --- | --- |
| Benchmark | Eight release checks and four fusion checks on CPU: CLI selection and output isolation, all 42 fusion/modality combinations, forward/backward computation, strict checkpoint reload, encoder initialization, missing-modality gates, scalar-weight/mean equivalence, input-map failures, QM8/QM9 table and XYZ loading, split handling and retrieval ties |
| Fusion | Five release tests including protocol tests; both trainers' six-fusion CPU self-tests; both 18-job queue dry-runs; six cache-tool help commands |
| Geometry | Twenty target/unit/source/split configurations and default queue dry-runs; conformer functions compared against source outputs on three real molecules per dataset; a real-cache one-step diagnostic training/resume/finalization; runtime preparation on a matched cache |
| Packaging | Python syntax, JSON parsing, a file-type/size inventory, and scans for machine-specific addresses and common credential patterns |

The benchmark's eight encoder definitions match the source implementation.
The six-fusion module and gate-only benchmark source pass strict gate-checkpoint
loading with exactly equal evaluation predictions and gates on seven synthetic
modality combinations. Their layer initialization order differs; checkpoint
compatibility does not establish identical training trajectories from a seed
alone. Source manifests identify portability differences and their scope.

The CPU packaging checks used Python 3.11.4, PyTorch 2.6.0+cu124, PyG 2.5.3,
torch-scatter 2.1.2+pt26cu124, NumPy 2.1.2 and RDKit 2026.03.3. This environment
is distinct from the experiment execution environments documented in the
component guides and runtime records.

## Manuscript metadata and result consistency

The manuscript title, five-person author order, affiliation, corresponding-author
details and repository URL were checked against the manuscript. All 20
random-split means agree with its principal tables at the reported precision.
The CFF preferred citation lists the manuscript authors separately from software
attribution; it does not assert a DOI, publication venue or acceptance status.

## Scaffold source and result verification

- All 78 distributed source/configuration files were checked against the source
  and public hashes. Seventy-four match byte for byte; four differ only in
  private-path removal and dependent metadata hashes recorded in the publication
  manifest. Model, training/checkpoint code, all 60 configs, six partitions and
  retained molecule IDs match the experiment sources byte for byte.
- The evaluator's source/configuration bindings were checked for all 60 planned
  runs, without loading datasets or fitting a model, together with all three
  documented command-line help entry points.
- Nine evaluator boundary tests cover invalid bindings, incomplete runs, scaler
  mismatch, tied checkpoint selection, partition integrity and unchanged model
  state.
- Sixty final metrics and 20 mean/sample-SD summaries were independently
  extracted and recomputed from hash-verified evaluation receipts. File scans
  check for private paths, server addresses, credentials and unexpected files.

## Limits

Unit, boundary and packaging checks do not establish full numerical
reproduction. They do not cover formal training, full-population conformer
generation, optional Bayesian search, or regeneration of all manuscript figures
and Excel exports. Scaffold result evidence comes from the completed fits and
their final-test receipts, independently of these packaging checks. Verification
of the distributed source also does not establish the provenance of every
manuscript table entry or ownership of third-party code. Diagnostic caches,
logs and checkpoints are excluded from the repository.
