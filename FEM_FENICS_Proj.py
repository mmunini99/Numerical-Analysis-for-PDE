import numpy as np
import mpi4py.MPI
import dolfinx
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
import mpi4py.MPI
import numpy as np
import petsc4py.PETSc
import ufl



def unit_square_poisson_solve(
    mesh: dolfinx.mesh.Mesh, boundaries: dict[int, np.ndarray], order: int, rhs, func_bound, spat_coord
) -> tuple[dolfinx.fem.FunctionSpace, petsc4py.PETSc.Mat, dolfinx.fem.Function]:
    """Solve a Poisson problem on the unit square."""
    # Function space
    Vh = dolfinx.fem.functionspace(mesh, ("Lagrange", order))

    # Weak form
    uh = ufl.TrialFunction(Vh)
    vh = ufl.TestFunction(Vh)
    x = spat_coord
    f = rhs
    dx = ufl.dx
    inner = ufl.inner
    grad = ufl.grad
    a = inner(grad(uh), grad(vh)) * dx
    F = f * vh * dx

    # Boundary conditions
    boundary_value = dolfinx.fem.Function(Vh)
    boundary_value.interpolate(func_bound)
    zero = dolfinx.fem.Constant(mesh, 0.0)
    boundary_dofs = {
        i: dolfinx.fem.locate_dofs_topological(
            Vh, mesh.topology.dim - 1, boundaries.indices[boundaries.values == i]) for i in range(1, 5)}

    bcs = [
        dolfinx.fem.dirichletbc(zero, boundary_dofs[1], Vh),
        dolfinx.fem.dirichletbc(boundary_value, boundary_dofs[2]),
        dolfinx.fem.dirichletbc(zero, boundary_dofs[3], Vh),
        dolfinx.fem.dirichletbc(boundary_value, boundary_dofs[4])]

    # Assemble system
    a_cpp = dolfinx.fem.form(a)
    F_cpp = dolfinx.fem.form(F)
    A = dolfinx.fem.petsc.assemble_matrix(a_cpp, bcs)
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(F_cpp)
    dolfinx.fem.petsc.apply_lifting(b, [a_cpp], [bcs])
    dolfinx.fem.petsc.set_bc(b, bcs)

    # Solve
    solution = dolfinx.fem.Function(Vh)
    ksp = petsc4py.PETSc.KSP()
    ksp.create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setFromOptions()
    ksp.solve(b, solution.x.petsc_vec) # since no more vector attribute in dolfinx v0.9.0, use this the underlying dolfinx.la.Vector the PETSc.Vec wrapper in DOLFINx now lives in dolfinx.fem.Function.x.petsc_vec

    # Return
    return Vh, A, solution


def unit_square_solution_error(
    mesh: dolfinx.mesh.Mesh, solution: dolfinx.fem.Function, space: int, true_sol, spat_coord
) -> float:
    """Compute the error between the FE solution and the exact solution."""
    # Definition of the exact solution
    x = spat_coord
    u_ex = true_sol

    # Computation of the difference between the finite element solution and the exact solution
    diff = solution - u_ex

    # UFL representation of the square of the norm of the error depending on the input argument space
    dx = ufl.dx
    if space == 0:
        eh_squared_ufl = diff * diff * dx
    elif space == 1:
        inner = ufl.inner
        grad = ufl.grad
        eh_squared_ufl = diff * diff * dx + inner(grad(diff), grad(diff)) * dx
    else:
        raise RuntimeError("Invalid space.")

    # Evaluation of the square of the norm of the error by assembling the UFL representation
    eh_squared = dolfinx.fem.assemble_scalar(dolfinx.fem.form(eh_squared_ufl))

    # Compute the square root and return
    return np.sqrt(eh_squared)



def unit_square_structured_mesh(n: int) -> tuple[dolfinx.mesh.Mesh, dict[int, np.ndarray]]:
    """Generate a structured mesh of the unit square, and locate its four boundaries."""
    # Generate structured mesh
    mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, n, n)

    # Locate boundary entities
    boundary_entities = {
        1: dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[0], 0.0)),
        2: dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[1], 0.0)),
        3: dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[0], 1.0)),
        4: dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[1], 1.0))
    }

    # Collect boundary entities in a MeshTags object
    boundary_entities_unsorted = np.hstack(
        [boundary_entities[i] for i in range(1, 5)]
    ).astype(np.int32)
    boundary_values_unsorted = np.hstack(
        [i * np.ones_like(boundary_entities[i]) for i in range(1, 5)]
    ).astype(np.int32)
    boundary_entities_argsort = np.argsort(boundary_entities_unsorted)
    boundary_entities_sorted = boundary_entities_unsorted[boundary_entities_argsort]
    boundary_values_sorted = boundary_values_unsorted[boundary_entities_argsort]
    boundaries = dolfinx.mesh.meshtags(
        mesh, mesh.topology.dim - 1, boundary_entities_sorted, boundary_values_sorted)

    # Return
    return mesh, boundaries