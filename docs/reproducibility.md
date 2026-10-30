# Reproducibility notes

## Experiment families

The benchmark/modality ablations, the controlled six-fusion runs, the geometry study, and the supplementary scaffold evaluation are distinct experiments. Differences in seeds, splits, data filtering, checkpoint selection and implementations must remain visible. A result is bound to its recorded implementation and protocol; code availability alone does not establish that provenance.

The six-fusion source collection includes controlled QM9 dipole-moment and QM8 E1-CC2 runs. These records do not by themselves establish the provenance of every cell of a broader manuscript fusion table. The code exposes all six mechanisms; available runs and manuscript-table provenance are separate claims.

The geometry collection covers eight QM9 properties and twelve QM8 targets. Some property-specific local source folders are preparation snapshots rather than independently recovered final server archives. Consult the geometry guide and manifests before describing a source snapshot as byte-identical to a final deployed run.

## Supplementary scaffold evaluation

The `scaffold/` collection contains the 60-run fixed-recipe TriMo evaluation and its explicit molecule partitions. This uses official-filtered QM9 (130,831) and the QM8 benchmark population (21,747), not the geometry comparison populations (132,486 and 21,689). It has no HPO. The three split seeds vary the scaffold partitions while training seed 42 is fixed; sample SD is across partitions. See [the scaffold protocol](scaffold.md) for empty-scaffold handling, identity checks and limitations.

The supplementary result summary is extracted from hash-verified final-test receipts. `scaffold/publication_manifest.json` records experiment-source and distributed-file hashes. Four files differ only in private-path removal and dependent metadata hashes; core model, training/checkpoint code, configurations and molecule IDs match the experiment sources byte for byte. Result receipts retain the plan/evaluator hashes of their actual execution.

## Data identity and units

Never independently filter the SMILES, graph, point-cloud and target arrays. Validate molecule correspondence before constructing a combined dataset. The publication adapters make file locations explicit and fail on mismatches instead of silently pairing different records.

QM9 dipole moment is in Debye; polarizability is in a0^3; electronic spatial extent is in a0^2. HOMO, gap and ZPVE retain their original hartree units. The selected U0 target is **atomization energy** in kcal/mol, defined as molecular energy minus the sum of atomic reference energies, not raw molecular U0. Cv is in cal/(mol K). QM8 excitation energies are in hartree, and oscillator strengths are dimensionless.

If reporting eV, multiply both the mean and its standard deviation by the stated hartree-to-eV factor. Do not recompute the QM9 gap from separately rounded HOMO and LUMO values when the original gap column is available.

## Validation scope

Tests validate code behavior, input correspondence, command construction and source integrity. They do not cover full training, hyperparameter optimization or regeneration of all prediction files. A successful unit or packaging test is not a full numerical reproduction of the paper.

Small reference summaries are provided only where their origin is known. Large datasets and checkpoints are intentionally excluded from Git. Obtaining those inputs and checking the recorded source/data hashes remain part of reproducing a reported run.
