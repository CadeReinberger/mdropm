# Validation of the clean copy

Checked with the existing `fenicsx-env` environment recorded in
`environment.yml`. The conda environment specification was taken from that
working installation; creating a fresh environment was not repeated.

- All 13 retained numerical tests passed.
- A fresh `canting_refine.py --ratios 0` run reached t=3 with 80 nodes and
  301 saved states. Its times and states matched the archived production run
  exactly (maximum absolute state difference: 0).
- `main.py --ratio 1 --duration 0.04 --segment-duration 0.02` completed two
  integration segments. The constant, x-linear, and height-dependent vertical
  manufactured pressure checks passed on both reference meshes.
- The 24 primary and five validation rows in the canting CSV reproduced the
  original numeric values. The retained legacy data produced nine primary
  and one validation point.
- The regenerated 1000 × 1000 map matched the archived array exactly,
  including its NaN mask. Prediction and experimental RIV colormaps and
  line-scan CSVs were generated successfully.
- Contact-line and pressure-gradient animations both produced readable
  GIFs and H.264 MP4s. The checks used four and three frames respectively,
  including the start and end of the archived t=24 trajectory.
- The pressure animator also rendered a canted archive using the compact
  `setup.pkl` plus its full parameter manifest.
- A copy outside the original repository regenerated the canting plot and
  continued the fresh t=3 run to t=3.02 using `main.py --seed`.
- Repeating the completed canted-run command returned its saved results
  without additional integration.
- Checksums confirmed that every copied archive/data file remained identical
  to its source and that original source files were untouched.

The full 24-height sweep and the full t=24 simulation were not recomputed as
part of this cleanup; their completed trajectories and prior convergence
records are included in `archived/`.
