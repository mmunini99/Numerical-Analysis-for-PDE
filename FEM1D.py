import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigsh
import sympy as sym



def quadrature(n_quadrature_points):
    # exploit numpy Gauss quadrature. This is defined in [-1,1]
    q, w = np.polynomial.legendre.leggauss(n_quadrature_points)

    return (q+1)/2, w/2

def unif_mesh(omega,N):
    return np.linspace(omega[0],omega[1],N+1)


# def non_uniform_mesh(omega, N, alpha):
#     # Generate mesh points using exponential spacing
#     x = omega[0] + (omega[1] - omega[0]) * (np.arange(N+1) / (N))**alpha
#     return x

def non_uniform_mesh(omega, N, alpha):

    base = np.linspace(omega[0],omega[1],N+1)

    h = np.diff(base).min()

    l = h/10

    noise = np.random.uniform(low=-1, high=1, size=len(base)-2)

    base[1: -1] = base[1: -1]+ noise*l
    
    return base
#     # Generate mesh points using exponential spacing
#     x = omega[0] + (omega[1] - omega[0]) * (np.arange(N+1) / (N))**alpha
#     return x

def mapping(q, i):
    # check index is within range
    assert i < len(q)-1
    assert i >= 0
    return lambda x: q[i] + (q[i+1]-q[i]) * x

def mapping_J(q,i):
    assert i < len(q)-1
    assert i >= 0
    return q[i+1]-q[i]

# Linear lagrange basis on reference element
def basis1(i):
    assert i < 2
    assert i >= 0
    if i == 0:
        phi = lambda x: 1-x
    else:
        phi = lambda x: x
    return phi

# Linear lagrange basis derivatives on reference element
def basis1_derivative(i):
    assert i < 2
    assert i >= 0
    if i == 0:
        dphi = lambda x: -np.ones(len(x))
    else:
        dphi = lambda x: np.ones(len(x))
    return dphi


def apply_boundary_conditions(N, A, F, lb, ub):
    A[0,0] = 1; A[0,1] = 0; F[0] = lb
    A[N,N] = 1; A[N,N-1] = 0; F[N] = ub

    return A, F

def apply_boundary_conditions_kcase(omega, A, F, lb, ub):
  # Ideally should scale entries as those of A
  N = A.shape[0] - 1
  A[0,A[0].nonzero()] = 0; A[0,0] = 1;  F[0]=lb
  A[N,A[N].nonzero()] = 0; A[N,N] = 1; F[N]=ub

  return A, F

# Workaround, to allow lambdify to work also on constant expressions
def np_lambdify(varname, func):
    lamb = sym.lambdify(varname, func, modules=['numpy'])
    if func.is_constant():
        return lambda t: np.full_like(t, lamb(t))
    else:
        return lambda t: lamb(np.array(t))

def lagrange_basis(q, i):
    assert i < len(q)
    assert i >= 0
    return lambda x: np.prod([(x-q[j])/(q[i]-q[j]) for j in range(len(q)) if i!=j], axis=0)

def lagrange_basis_derivative(q,i,order=1):
    t = sym.var('t')
    return np_lambdify(t, lagrange_basis(q,i)(t).diff(t,order))



class FEM1_1D(object):

    def __init__(self, omega, N, quad_points, rhs, lb, ub, unif):

        self.omega = omega
        self.N = N
        self.quad_points = quad_points
        self.rhs = rhs
        self.lb = lb
        self.ub = ub
        self.unif = unif
        


    def FEM_POISSON(self):

        # grid
        if self.unif:
            vertices = unif_mesh(self.omega, self.N)
        else:
            vertices = non_uniform_mesh(self.omega, self.N, 0.84)

        # quadrature formula on reference element
        q, w = quadrature(self.quad_points)

        # Evaluation of the two local linear Lagrange basis
        phi = np.array([basis1(i)(q) for i in range(2)]).T
        dphi = np.array([basis1_derivative(i)(q) for i in range(2)]).T

        # initialise system
        A = sp.lil_matrix((self.N+1, self.N+1))
        F = np.zeros(self.N+1)

        # Assembly loop
        for i in range(self.N):
            JxW = mapping_J(vertices,i) * w
            A_ele = np.einsum('qi,qj,q',dphi,dphi,JxW) / mapping_J(vertices,i)**2
            F_ele = np.einsum('qi,q,q',phi,self.rhs(mapping(vertices,i)(q)),JxW)
            A[i:i+2,i:i+2] += A_ele
            F[i:i+2] += F_ele

        # boundary condition
        A, F = apply_boundary_conditions(self.N, A, F, self.lb, self.ub)
        # return system matrix and rhs vector
        return A, F
    
    def FEM1_H1(self, Uh, sol, symbol):

        derivative = sym.lambdify(symbol, sol.diff(symbol,1)) # the solution must be sympy function format

        if self.unif:
            vertices = unif_mesh(self.omega, self.N)
        else:
            vertices = non_uniform_mesh(self.omega, self.N, 0.72)

        # quadrature formula on reference element
        q, w = quadrature(self.quad_points)

        # Evaluation of linear Lagrange basis. 
        phi = np.array([basis1(i)(q) for i in range(2)]).T
        dphi = np.array([basis1_derivative(i)(q) for i in range(2)]).T

        # initialise value of norm of error
        H1_err = 0.0

        # Assembly error. 
        for i in range(self.N):
            JxW = mapping_J(vertices, i) * w  # Compute JxW once per element
            
            # Evaluate the term inside the summation
            Uh_dphi = Uh[i] * dphi[:, 0] + Uh[i + 1] * dphi[:, 1]
            sol_values = derivative(mapping(vertices, i)(q))
            
            # Using einsum for summation over quadrature points
            local_err = np.einsum('j,j,j->', (sol_values - Uh_dphi / mapping_J(vertices, i))**2, JxW, np.ones_like(JxW))
            
            H1_err += local_err

        # Return error
        return H1_err
    

    def FEM_GEN(self, a_func, b_func, c_func):

        # grid
        if self.unif:
            vertices = unif_mesh(self.omega, self.N)
        else:
            vertices = non_uniform_mesh(self.omega, self.N, 0.72)

        # quadrature formula on reference element
        q, w = quadrature(self.quad_points)

        # Evaluation of the two local linear Lagrange basis
        phi = np.array([basis1(i)(q) for i in range(2)]).T
        dphi = np.array([basis1_derivative(i)(q) for i in range(2)]).T

        # initialise system
        A = sp.lil_matrix((self.N+1, self.N+1))
        F = np.zeros(self.N+1)

        # Assembly loop
        for i in range(self.N):
            JxW = mapping_J(vertices,i) * w
            A_ele = np.einsum('qi,qj,q,q',dphi,dphi,JxW, np.ones_like(JxW)*a_func(mapping(vertices,i)(q))) / mapping_J(vertices,i)**2
            B_ele = np.einsum('qi,qj,q,q',phi,dphi,JxW, np.ones_like(JxW)*b_func(mapping(vertices,i)(q))) / mapping_J(vertices,i)
            C_ele = np.einsum('qi,qj,q,q',phi,phi,JxW, np.ones_like(JxW)*c_func(mapping(vertices,i)(q)))
            F_ele = np.einsum('qi,q,q',phi,self.rhs(mapping(vertices,i)(q)),JxW)
            A[i:i+2,i:i+2] += (A_ele+B_ele+C_ele)
            F[i:i+2] += F_ele

        # boundary condition
        A, F = apply_boundary_conditions(self.N, A, F, self.lb, self.ub)
        # return system matrix and rhs vector
        return A, F





class FEM_k_1D(object):

    def __init__(self, omega, k, M, quad_points, rhs, lb, ub, unif):

        self.omega = omega
        self.k = k
        self.M = M
        self.quad_points = quad_points
        self.rhs = rhs
        self.lb = lb
        self.ub = ub
        self.unif = unif

    def FEM_POISSON(self):

        # Dimension of the problem
        N = self.M*self.k +1

        # grid
        if self.unif:
            vertices = unif_mesh(self.omega, self.M)
        else:
            vertices = non_uniform_mesh(self.omega, self.M, 0.72)

        # reference element (quadrature and Lagrange points)
        q, w = quadrature(self.quad_points)
        lagrange_points = np.linspace(0, 1, self.k+1)

        # Evaluation of Lagrange basis
        phi = np.array([lagrange_basis(lagrange_points,i)(q) for i in range(self.k+1)]).T
        dphi = np.array([lagrange_basis_derivative(lagrange_points,i)(q) for i in range(self.k+1)]).T

        # initialise system
        A = sp.lil_matrix((N,N))
        F = np.zeros(N)

        # Assembly loop
        for i in range(self.M):
            JxW = mapping_J(vertices, i) * w
            ele_A = np.einsum('qi,qj,q',dphi,dphi,JxW) / mapping_J(vertices,i)**2
            ele_F = np.einsum('qi,q,q',phi,self.rhs(mapping(vertices,i)(q)),JxW)

            # Assembly local-to-global
            j = i * self.k
            m = j + self.k + 1
            A[j:m,j:m] += ele_A
            F[j:m] += ele_F


        A, F = apply_boundary_conditions_kcase(self.omega, A, F, self.lb, self.ub)


        return A, F



    def FEM1_H1(self, Uh, sol, symbol):

        derivative = sym.lambdify(symbol, sol.diff(symbol,1)) # the solution must be sympy function format

        # Dimension of the problem
        N = self.M*self.k +1

        # grid
        if self.unif:
            vertices = unif_mesh(self.omega, self.M)
        else:
            vertices = non_uniform_mesh(self.omega, self.M, 0.72)

        # reference element (quadrature and Lagrange points)
        q, w = quadrature(self.quad_points)
        lagrange_points = np.linspace(0, 1, self.k+1)

        # Evaluation of Lagrange basis
        phi = np.array([lagrange_basis(lagrange_points,i)(q) for i in range(self.k+1)]).T
        dphi = np.array([lagrange_basis_derivative(lagrange_points,i)(q) for i in range(self.k+1)]).T

        # initialise error
        err_H1 = 0.0
        
        # Assembly loop
        for i in range(self.M - 1):
                JxW = mapping_J(vertices, i) * w  # Compute JxW once per element
            

                # Get exact solution values at quadrature points
                sol_values = derivative(mapping(vertices, i)(q))

                # Compute the Uh part for the basis derivatives
                Uh_dphi = Uh[i * self.k:i* self.k + dphi.shape[0] -1] @ dphi.T  # Uh * dphi

                # Compute the error element-wise
                local_err = np.einsum('j,j,j->', (sol_values - Uh_dphi / mapping_J(vertices, i)) ** 2, JxW, np.ones_like(JxW))

                # Add to the total error
                err_H1 += local_err


        # Return error
        return err_H1