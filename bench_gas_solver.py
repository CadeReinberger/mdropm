"""
Benchmark PETSc linear solver + preconditioner options for the gas-phase
concentration FEM (weighted Laplacian, SPD system).

Run from the project root:
    python bench_gas_solver.py

Sweeps over droplet radii to show how each method scales with problem size.
All iterative solvers use PETSc's default tolerances (rtol=1e-5).
"""

import time
import numpy as np
import gmsh

import heightscape
import tapescape
from physical_params import physical_params
from solver_params import solver_params
from gas_fem import make_gmesh_mesh
from tapescape import BC_TYPES

from dolfinx.fem import (Constant, functionspace, dirichletbc,
                          locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
from mpi4py import MPI
from ufl import SpatialCoordinate, TestFunction, TrialFunction, inner, grad, dx
from petsc4py.PETSc import ScalarType


# ── solver configs ────────────────────────────────────────────────────────────
# The gas-phase PDE is h·∇u·∇v dx with h > 0, so the system is SPD.
# CG / MINRES are the right Krylov methods; direct solvers are the baseline.

CONFIGS = [
    ('preonly + lu  (baseline)',   {'ksp_type': 'preonly', 'pc_type': 'lu'}),
    ('preonly + cholesky',         {'ksp_type': 'preonly', 'pc_type': 'cholesky'}),
    ('cg + gamg',                  {'ksp_type': 'cg',     'pc_type': 'gamg'}),
    ('cg + hypre boomeramg',       {'ksp_type': 'cg',     'pc_type': 'hypre',
                                    'pc_hypre_type': 'boomeramg'}),
    ('cg + icc',                   {'ksp_type': 'cg',     'pc_type': 'icc'}),
    ('cg + sor',                   {'ksp_type': 'cg',     'pc_type': 'sor'}),
    ('cg + jacobi',                {'ksp_type': 'cg',     'pc_type': 'jacobi'}),
    ('minres + hypre boomeramg',   {'ksp_type': 'minres', 'pc_type': 'hypre',
                                    'pc_hypre_type': 'boomeramg'}),
    ('gmres + ilu',                {'ksp_type': 'gmres',  'pc_type': 'ilu'}),
]

# Sweep over these droplet radii — larger radius → finer effective mesh → more DOFs
RADII = [1.5, 3.0, 4.0]

N_WARMUP = 2   # discarded runs to let JIT / PETSc internals warm up
N_TRIALS = 6   # timed runs


# ── helpers ───────────────────────────────────────────────────────────────────

def build_mesh(dr_x, dr_y, ts, sps):
    make_gmesh_mesh(dr_x, dr_y, ts, sps)
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return mesh, facet_tags


def time_solve(mesh, facet_tags, hs, pps, sps, opts):
    """Build and solve the LinearProblem from scratch; return wall time in ms."""
    V = functionspace(mesh, ("CG", sps.FEM_ELT_ORDER))
    u, v = TrialFunction(V), TestFunction(V)

    x = SpatialCoordinate(mesh)
    ufl_h = eval(hs.ufl_str)
    a = ufl_h * inner(grad(u), grad(v)) * dx
    L = inner(Constant(mesh, ScalarType(0)), v) * dx

    dofs_hd = locate_dofs_topological(V, 1, facet_tags.find(BC_TYPES.HOMOGENEOUS_DIRICHLET))
    dofs_id = locate_dofs_topological(V, 1, facet_tags.find(BC_TYPES.INHOMOGENEOUS_DIRICHLET))
    bcs = [
        dirichletbc(value=ScalarType(-1),       dofs=dofs_hd, V=V),
        dirichletbc(value=ScalarType(pps.w_eq), dofs=dofs_id, V=V),
    ]

    t0 = time.perf_counter()
    LinearProblem(a, L, bcs=bcs, petsc_options=opts).solve()
    return (time.perf_counter() - t0) * 1000


def benchmark(mesh, facet_tags, hs, pps, sps, opts):
    times = []
    for i in range(N_WARMUP + N_TRIALS):
        ms = time_solve(mesh, facet_tags, hs, pps, sps, opts)
        if i >= N_WARMUP:
            times.append(ms)
    return np.array(times)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    hs  = heightscape.constructors.rect_interp_htscape((10, 10), (.1, .2, .2, .1))
    ts  = tapescape.constructors.make_rectangle_top_open(-5, -5, 5, 5)
    pps = physical_params()
    sps = solver_params()

    for R in RADII:
        angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        dr_x = R * np.cos(angles)
        dr_y = R * np.sin(angles)

        mesh, facet_tags = build_mesh(dr_x, dr_y, ts, sps)
        n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        n_geom  = mesh.geometry.x.shape[0]

        print(f'\nR={R:.1f}  ({n_cells} cells, {n_geom} geom nodes, '
              f'FEM_ELT_ORDER={sps.FEM_ELT_ORDER})')
        print(f'  {"Solver":<32}  {"mean ms":>8}  {"min ms":>8}  {"std ms":>7}')
        print('  ' + '─' * 60)

        for name, opts in CONFIGS:
            try:
                t = benchmark(mesh, facet_tags, hs, pps, sps, opts)
                print(f'  {name:<32}  {np.mean(t):>8.1f}  {np.min(t):>8.1f}  {np.std(t):>7.1f}')
            except Exception as e:
                print(f'  {name:<32}  FAILED: {e}')

    print()


if __name__ == '__main__':
    main()
