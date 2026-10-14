"""Persistent moving meshes for the canting study's existing elliptic PDEs.

The original solvers remesh for every ODE evaluation and interpolate a
discontinuous element gradient into a continuous space.  Both operations can
make a tiny geometry perturbation produce a finite jump in the velocity RHS.
Here connectivity is fixed, coordinates follow a smooth radial extension of
the contact-line displacement, and gradients are recovered by L2 projection.
The gas/liquid equations, outer boundary conditions and material constants are
the same as in gas_fem.py and liquid_fem.py.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

import gmsh
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx.fem import (
    Constant, Function, dirichletbc, functionspace, locate_dofs_topological,
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import bb_tree, create_midpoint_tree, compute_closest_entity
from dolfinx.io import gmshio
from dolfinx.mesh import entities_to_geometry, locate_entities_boundary
from ufl import SpatialCoordinate, TestFunction, TrialFunction, dx, grad, inner

import gas_fem
import liquid_fem
from tapescape import BC_TYPES


class MovingPhase:
    """One serial finite-element domain with reusable forms and topology."""

    def __init__(self, dr_x, dr_y, universe, *, gas: bool):
        self.gas = gas
        self.universe = universe
        self.initial_boundary = np.column_stack((dr_x, dr_y))
        self.n = len(dr_x)
        self.parameter = np.linspace(0.0, 2.0 * np.pi, self.n + 1)
        self.radius = float(np.mean(np.hypot(dr_x, dr_y)))
        self.last_boundary = self.initial_boundary.copy()
        sps = universe.sol_ps
        if gas:
            gas_fem.make_gmesh_mesh(dr_x, dr_y, universe.tpscp, sps)
        else:
            liquid_fem.make_gmesh_mesh(dr_x, dr_y, sps)
        self.mesh, _, facets = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        self.reference = self.mesh.geometry.x.copy()
        corners = self.reference[self.mesh.geometry.dofmap[:, :3], :2]
        self.reference_determinant = np.linalg.det(corners[:, 1:] - corners[:, :1])
        self.minimum_area_ratio = 1.0
        self.angle = np.mod(np.arctan2(self.reference[:, 1], self.reference[:, 0]), 2 * np.pi)
        radial = np.linalg.norm(self.reference[:, :2], axis=1)

        if gas:
            boundary_facets = facets.find(BC_TYPES.INHOMOGENEOUS_DIRICHLET)
            outer_radius = 5.0 / np.maximum(np.abs(np.cos(self.angle)), np.abs(np.sin(self.angle)))
            self.weight = np.clip((outer_radius - radial) / (outer_radius - self.radius), 0, 1)
        else:
            boundary_facets = locate_entities_boundary(self.mesh, 1, lambda x: np.full(x.shape[1], True))
            self.weight = np.minimum(radial / self.radius, 1.0) ** 2
        geometry_boundary = np.unique(entities_to_geometry(self.mesh, 1, boundary_facets).ravel())
        self.weight[geometry_boundary] = 1.0

        self.V = functionspace(self.mesh, ("CG", sps.FEM_ELT_ORDER))
        u, v = TrialFunction(self.V), TestFunction(self.V)
        x = SpatialCoordinate(self.mesh)
        h = eval(universe.htscp.ufl_str)
        diffusion = h if gas else h**3
        zero = Constant(self.mesh, ScalarType(0))
        if gas:
            outer_dofs = locate_dofs_topological(self.V, 1, facets.find(BC_TYPES.HOMOGENEOUS_DIRICHLET))
            inner_dofs = locate_dofs_topological(self.V, 1, boundary_facets)
            bcs = [
                dirichletbc(ScalarType(-1), outer_dofs, self.V),
                dirichletbc(ScalarType(universe.phys_ps.w_eq), inner_dofs, self.V),
            ]
        else:
            self.boundary_values = Function(self.V)
            dof_coordinates = self.V.tabulate_dof_coordinates()
            self.dof_angle = np.mod(np.arctan2(dof_coordinates[:, 1], dof_coordinates[:, 0]), 2 * np.pi)
            dofs = locate_dofs_topological(self.V, 1, boundary_facets)
            bcs = [dirichletbc(self.boundary_values, dofs)]
        options = {"ksp_type": "preonly", "pc_type": "lu"}
        self.problem = LinearProblem(diffusion * inner(grad(u), grad(v)) * dx,
                                     inner(zero, v) * dx, bcs=bcs, petsc_options=options)
        self.W = functionspace(self.mesh, ("CG", sps.FEM_ELT_ORDER, (2,)))
        gu, gv = TrialFunction(self.W), TestFunction(self.W)
        self.projection = LinearProblem(inner(gu, gv) * dx,
                                        inner(grad(self.problem.u), gv) * dx,
                                        petsc_options=options)

        cell_map = self.mesh.topology.index_map(2)
        cells = np.arange(cell_map.size_local, dtype=np.int32)
        tree = bb_tree(self.mesh, 2)
        midpoint_tree = create_midpoint_tree(self.mesh, 2, cells)
        points = np.column_stack((dr_x, dr_y, np.zeros(self.n)))
        self.sample_cells = compute_closest_entity(tree, midpoint_tree, self.mesh, points)
        self.solves = 0

    def move(self, dr_x, dr_y):
        boundary = np.column_stack((dr_x, dr_y))
        if np.array_equal(boundary, self.last_boundary):
            return
        displacement = boundary - self.initial_boundary
        interpolator = CubicSpline(self.parameter, np.vstack((displacement, displacement[0])), bc_type="periodic")
        extended = interpolator(self.angle)
        if self.gas:
            shifted = self.weight[:, None] * extended
        else:
            center = np.mean(displacement, axis=0)
            shifted = center + self.weight[:, None] * (extended - center)
        self.mesh.geometry.x[:, :2] = self.reference[:, :2] + shifted
        corners = self.mesh.geometry.x[self.mesh.geometry.dofmap[:, :3], :2]
        area_ratio = np.linalg.det(corners[:, 1:] - corners[:, :1]) / self.reference_determinant
        self.minimum_area_ratio = min(self.minimum_area_ratio, float(area_ratio.min()))
        if self.minimum_area_ratio <= 0.15:
            raise FloatingPointError("moving FEM mesh is too distorted; refine/rebuild this run")
        self.last_boundary = boundary.copy()

    def gradients(self, dr_x, dr_y, pressures=None):
        self.move(dr_x, dr_y)
        if not self.gas:
            pressure = CubicSpline(self.parameter, np.r_[pressures, pressures[0]], bc_type="periodic")
            self.boundary_values.x.array[:] = pressure(self.dof_angle)
        self.problem.solve()
        recovered = self.projection.solve()
        points = np.column_stack((dr_x, dr_y, np.zeros(self.n)))
        result = recovered.eval(points, self.sample_cells)
        self.solves += 1
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("non-finite recovered FEM gradient")
        return result


class MovingFEM:
    def __init__(self, initial_x, initial_y, universe):
        # A resumed trajectory must deform the same original reference mesh.
        self.initial_x = np.asarray(initial_x).copy()
        self.initial_y = np.asarray(initial_y).copy()
        self.universe = universe
        self.gas = None
        self.liquid = None

    def concentration(self, dr_x, dr_y, universe):
        if self.gas is None:
            self.gas = MovingPhase(self.initial_x, self.initial_y, self.universe, gas=True)
        return self.gas.gradients(dr_x, dr_y)

    def pressure(self, dr_x, dr_y, pressures, universe):
        if self.liquid is None:
            self.liquid = MovingPhase(self.initial_x, self.initial_y, self.universe, gas=False)
        return self.liquid.gradients(dr_x, dr_y, pressures)
