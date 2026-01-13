import gmsh
import numpy as np
import uuid
from geo_util import gas_lc
from tapescape import EXTERNAL_BC_TYPES, BC_TYPES

from dolfinx.fem import (Constant, functionspace, dirichletbc, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.plot import vtk_mesh
from dolfinx.io import gmshio

from mpi4py import MPI
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, inner, grad, dx)

from petsc4py.PETSc import ScalarType, Options
import pyvista


def make_gmesh_mesh(dr_x, dr_y, ts, sps):
    # Get the mesh fineness dynamically we're gonna use
    lc = gas_lc(dr_x, dr_y, ts, sps)

    # Just initialize a new gmesh
    gmsh.initialize()
    model_id = uuid.uuid4().hex
    gmsh.model.add(model_id)
    factory = gmsh.model.geo

    ### Let's make the external geometry
    outer_n = len(ts.segments)

    # Add external geometry points
    for (ind, seg) in enumerate(ts.segments):
        start_pt = seg.start_pt
        factory.addPoint(start_pt[0], start_pt[1], 0, lc, ind+1)

    # Make the lines for the external geometry
    hd_stuff = []
    for line_ind in range(1, outer_n):
        factory.addLine(line_ind, line_ind+1, line_ind)
        if ts.segments[line_ind-1].bc_type == EXTERNAL_BC_TYPES.HOMOGENEOUS_DIRICHLET:
            hd_stuff.append(line_ind)
    factory.add_line(outer_n, 1, outer_n)

    if ts.segments[outer_n-1].bc_type == EXTERNAL_BC_TYPES.HOMOGENEOUS_DIRICHLET:
        hd_stuff.append(outer_n)

    # Make the outer curve loop
    factory.addCurveLoop(list(range(1, outer_n+1)), 1)

    ## Now, we make the inner droplet arc
    inner_n = len(dr_x)

    # First, add all the points for the inner droplet
    for ind in range(inner_n):
        factory.addPoint(dr_x[ind], dr_y[ind], 0, lc, ind+outer_n+1)
    # Collect the points into a list to make periodic spline through the points
    spline_pts = list(range(outer_n+1, outer_n+inner_n+1)) + [outer_n + 1]
    factory.addSpline(spline_pts, outer_n+1)
    factory.addCurveLoop([outer_n+1], 2)
    id_stuff = [outer_n+1]

    # Make the main physical groups
    factory.addPlaneSurface([1, 2], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [1], name='gas_phase')

    ## Now we handle the boundary tagging
    # Make the physical groups for the boundary tags
    gmsh.model.addPhysicalGroup(1, hd_stuff, BC_TYPES.HOMOGENEOUS_DIRICHLET)
    gmsh.model.addPhysicalGroup(1, id_stuff, BC_TYPES.INHOMOGENEOUS_DIRICHLET)

    ## Finally, we clean up the mesh
        
    # Generate the mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(sps.FEM_SHAPE_ORDER)

    # Write the mesh
    gmsh.write(sps.GAS_PHASE_MESH_FILE)

    # Finalize and return
    gmsh.finalize()
    return 

# viz here is a diagnostic check
def solve_concentration_field(dr_x, dr_y, hs, ts, pps, sps, viz=False):
    # Start by making and reading the mesh
    make_gmesh_mesh(dr_x, dr_y, ts, sps)
    (mesh, cell_tags, facet_tags) = gmshio.read_from_msh(sps.GAS_PHASE_MESH_FILE, MPI.COMM_WORLD, gdim=2)

    ## First, set up the FEM problem

    # set up the weak formulation

    V = functionspace(mesh, ("CG", sps.FEM_ELT_ORDER))
    u, v = TrialFunction(V), TestFunction(V)

    # Get the height field that we need here
    x = SpatialCoordinate(mesh)
    ufl_h = eval(hs.ufl_str)
    a = ufl_h * inner(grad(u), grad(v)) * dx
    ZERO = Constant(mesh, ScalarType(0))
    L= inner(ZERO, v) * dx

    # Homogeneous Dirichlet BC
    dofs_hd = locate_dofs_topological(V, 1, facet_tags.find(BC_TYPES.HOMOGENEOUS_DIRICHLET))
    bc_hd = dirichletbc(value=ScalarType(0), dofs=dofs_hd, V=V)

    # Inhomogeneous Dirichlet BC
    dofs_id = locate_dofs_topological(V, 1, facet_tags.find(BC_TYPES.INHOMOGENEOUS_DIRICHLET))
    bc_id = dirichletbc(value=ScalarType(pps.w_eq), dofs=dofs_hd, V=V)

    # We get homogeneous neumann BC's for free, so all BC's
    bcs = [bc_hd, bc_id]

    # Make the problem and solve it
    problem = LinearProblem(a, L, bcs=bcs, petsc_options={'ksp_type' : 'preonly', 'pc_type' : 'lu'})
    uh = problem.solve() # If you've got a problem yo I'll solve it

    if viz:
        # Visualize the solution
        pyvista_cells, cell_types, geometry = vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geometry)
        grid.point_data["u"] = uh.x.array
        grid.set_active_scalars("u")

        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        plotter.view_xy()
        plotter.show()
        #while True:
        #    continue

        return (uh, V, mesh)



