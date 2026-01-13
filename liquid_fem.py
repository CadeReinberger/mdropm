from geo_util import liquid_lc, make_polygon_projector
import numpy as np
import uuid
import gmsh

from dolfinx.fem import (Constant, Function, functionspace, dirichletbc, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary
from dolfinx.plot import vtk_mesh
from dolfinx.io import gmshio


from mpi4py import MPI
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, inner, grad, dx)

from petsc4py.PETSc import ScalarType, Options
import pyvista


def make_gmesh_mesh(dr_x, dr_y, sps):
    # Get the mesh fineness dynamically we're gona use
    lc = liquid_lc(dr_x, dr_y, sps)

    # Initialize a new gmesh
    gmsh.initialize()
    model_id = uuid.uuid4().hex
    gmsh.model.add(model_id)
    factory = gmsh.model.geo

    ## Make the inner droplet arc
    n = len(dr_x)

    # First, add all the points for the inner droplet
    for ind in range(n):
        factory.addPoint(dr_x[ind], dr_y[ind], 0, lc, ind+1)

    # Collect the points into a list to make a periodic spline through the points
    spline_pts = list(range(1, n+1)) + [1]
    cls = factory.addSpline(spline_pts, n+1)
    cl = factory.addCurveLoop([cls])
    factory.addPlaneSurface([cl])

    # Now do some gmsh bookkeeping
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [1], name='liquid_phase')

    # Generate the mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(sps.FEM_SHAPE_ORDER)

    # Write the mesh
    gmsh.write(sps.LIQUID_PHASE_MESH_FILE)

    # Finalize and return
    gmsh.finalize()
    return


def solve_pressure_field(dr_x, dr_y, ps, hs, pps, sps, viz=False):
    # Make and read the mesh
    make_gmesh_mesh(dr_x, dr_y, sps)
    (mesh, cell_tags, facet_tags) = gmshio.read_from_msh(sps.LIQUID_PHASE_MESH_FILE, MPI.COMM_WORLD, gdim=2)

    ## First, set up the FEM problem 

    # Set up the function space
    V = functionspace(mesh, ("CG", sps.FEM_ELT_ORDER))
    u, v = TrialFunction(V), TestFunction(V)

    # Get the weak form operator
    x = SpatialCoordinate(mesh)
    ufl_h = eval(hs.ufl_str)
    a = ufl_h * ufl_h * ufl_h * inner(grad(u), grad(v)) * dx
    ZERO = Constant(mesh, ScalarType(0))
    L = inner(ZERO, v) * dx

    ## Now we add the dirichlet BC
    
    # Make a function to interpolate to the boundary 
    bdry_funct = make_polygon_projector(dr_x, dr_y, ps)
    def _bdry_interp(x):
        vals = [bdry_funct(x[0][ind], x[1][ind]) for ind in range(len(x[0]))]
        return np.asarray(vals, dtype=ScalarType)

    # Make a Function Object and interpolate to it
    u_D = Function(V)
    u_D.interpolate(_bdry_interp)
    
    # Make the bdry facets type shit
    boundary_facets = locate_entities_boundary(mesh, 1, lambda x: np.full(x.shape[1], True, dtype=bool))
    dofs_bc = locate_dofs_topological(V, 1, boundary_facets)
    bc = dirichletbc(u_D, dofs_bc)

    ## Make and solve the problem
    
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={'ksp_type' : 'preonly', 'pc_type' : 'lu'})
    uh = problem.solve()

    if viz: 
        # Visualize the solution
        pyvista_cells, cell_types, geometry = vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geometry)
        grid.point_data["u"] = uh.x.array
        grid.set_active_scalars("u")

        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=False)
        plotter.view_xy()
        plotter.show()
        #while True:
        #    continue

    return (uh, V, mesh)


def compute_pressure_gradients(dr_x, dr_y, ps, pu):
    # Read the stuff from the problem universe
    hs, pps, sps = pu.htscp, pu.phys_ps, pu.sol_ps

    # First, solve the problem and get the solution
    uh, V, mesh = solve_pressure_field(dr_x, dr_y, ps, hs, pps, sps)

    # Now, interpolate the gradient as we want
    W = functionspace(mesh, ("CG", 2, (2,)))
    guh = Function(W)
    guh_expr = Expression(grad(uh), W.element.interpolation_points())
    guh.interpolate(guh_expr)

    # Make geometry tools to find incident cells
    tree = bb_tree(mesh, 2)

    # Initialize a gradients to return
    grad_vecs = []

    # Iterate over our points and compute derivatives, based on whether or not we project outside mesh
    if not sps.PROJECT_OUTSIDE_MESH:
        # Compute colliding cells and pick one, no projection
        for ind in range(len(dr_x)):
            eval_pt = np.array([dr_x[ind], dr_y[ind], 0])
            cell_candidates = compute_collisions_points(tree, eval_pt)
            cell = compute_colliding_cells(mesh, cell_candidates, eval_pt)
            cell = cell.array[0]
            grad_vec = guh.eval(eval_pt, [cell])
            grad_vecs.append(grad_vec)

    else:
        # Compute the midpoint tree to do the projection shit
        mesh_map = mesh.topology.index_map(2)
        all_faces = np.arange(mesh_map.size_local + mesh_map.num_ghosts, dtype=np.int32)
        face_midpoint_tree = create_midpoint_tree(mesh, 2, all_faces)
        # Now, iterate through and see what we can do
        for ind in range(len(dr_x)):
            eval_pt = np.array([dr_x[ind], dr_y[ind], 0])
            closest_cell = compute_closest_entity(tree, face_midpoint_tree, mesh, eval_pt)[0]
            grad_vec = guh.eval(eval_pt, [closest_cell])
            grad_vecs.append(grad_vec)

    return grad_vecs
