import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigsh



def quadrature(n_quadrature_points):
    # exploit numpy Gauss quadrature. This is defined in [-1,1]
    q, w = np.polynomial.legendre.leggauss(n_quadrature_points)

    return (q+1)/2, w/2



def unif_mesh(omega,N):
    return np.linspace(omega[0],omega[1],N+1)


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



class FEM1_1D(object):

    def __init__(self, omega, N, quad_points, rhs, lb, ub):

        self.omega = omega
        self.N = N
        self.quad_points = quad_points
        self.rhs = rhs
        self.lb = lb
        self.ub = ub


    def FEM_POISSON(self):

        # grid
        vertices = unif_mesh(self.omega, self.N)

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