This source snapshot preserves the solver dependency layout.

Plotting and map regeneration use NumPy, SciPy, and Matplotlib. Simulation
also requires the existing FEniCSx 0.9 / PETSc 3.21 environment, Gmsh, mpi4py,
SymPy, Shapely, and PyVista. See ../environment.json for tested versions.

From this directory:

    python -B scripts/chirality_plots/flat_heightfield_run.py --duration 24 --output-dir new_run
    python -B scripts/chirality_plots/flat_heightfield_map.py --results new_run/results.pkl --output-dir new_map

Use the same command to resume an interrupted simulation. Do not use mpirun.
For a smoke run add --duration 0.25 --nodes 20 --lc 0.6 and use a new output
directory. To refine the production run use --nodes 160 or --lc 0.2, each
in a separate output directory. The shipped data use 80 nodes and lc=0.3.

The bundled checkpoint is from a seeded continuation. Resuming it with a
changed manifest is intentionally rejected; use a new output directory for
a new run. All saved trajectory data needed for analysis are in ../results.pkl.

The earlier long-run driver (before adding support for seeded long-run
extensions and source snapshots) is preserved in ../provenance/. Its hash
matches the nested time-21 seed manifest.
