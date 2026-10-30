# Verified TriMo summaries

`trimo_per_split.json` contains 60 final-test metrics and provenance hashes. `random_vs_scaffold.json` contains 20 property summaries, full-precision means and sample SDs, and the historical per-seed MAEs. The compact table below shows mean MAE; use the JSON for SD and unrounded calculations.

**This is a descriptive comparison of two protocols, not a split-only ablation.** Cohort filtering, seeds and training/selection settings differ. Positive change means higher scaffold MAE. Do not pool MAEs across different units.

| Dataset | Target | Unit | Random MAE | Scaffold MAE | Change |
| --- | --- | --- | ---: | ---: | ---: |
| QM9 | dipole_moment | Debye | 0.264273 | 0.271084 | +2.6% |
| QM9 | isotropic_polarizability | a0^3 | 0.224984 | 0.183108 | -18.6% |
| QM9 | energy_HOMO | hartree | 0.00239881 | 0.00321613 | +34.1% |
| QM9 | gap | hartree | 0.00344213 | 0.0044203 | +28.4% |
| QM9 | electronic_spatial_extent | a0^2 | 8.37737 | 6.54856 | -21.8% |
| QM9 | zero_point_vibrational_energy | hartree | 0.00035285 | 0.000192426 | -45.5% |
| QM9 | internal_energy_0K_atom_kcal_mol | kcal/mol | 2.67557 | 1.48958 | -44.3% |
| QM9 | heat_capacity_298K | cal/mol/K | 0.0894342 | 0.0809176 | -9.5% |
| QM8 | E1-CC2 | hartree | 0.0036327 | 0.00534541 | +47.1% |
| QM8 | E2-CC2 | hartree | 0.00511683 | 0.00662923 | +29.6% |
| QM8 | f1-CC2 | dimensionless | 0.00987659 | 0.0135821 | +37.5% |
| QM8 | f2-CC2 | dimensionless | 0.0239572 | 0.0300134 | +25.3% |
| QM8 | E1-PBE0 | hartree | 0.00357274 | 0.00506076 | +41.6% |
| QM8 | E2-PBE0 | hartree | 0.00459832 | 0.00607878 | +32.2% |
| QM8 | f1-PBE0 | dimensionless | 0.00838466 | 0.0128635 | +53.4% |
| QM8 | f2-PBE0 | dimensionless | 0.0177881 | 0.0257617 | +44.8% |
| QM8 | E1-CAM | hartree | 0.00335912 | 0.00472224 | +40.6% |
| QM8 | E2-CAM | hartree | 0.0042152 | 0.00572915 | +35.9% |
| QM8 | f1-CAM | dimensionless | 0.00819073 | 0.0143256 | +74.9% |
| QM8 | f2-CAM | dimensionless | 0.0199889 | 0.0298819 | +49.5% |
