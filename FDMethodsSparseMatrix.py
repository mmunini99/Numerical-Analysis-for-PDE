import scipy.sparse as sp
import numpy as np



class first_order_first_derivative_FiniteDiff_sparse_method(object):

    def __init__(self, n, h):

        self.n = n
        self.h = h

    def FD(self):
        list_diag=[-np.ones(self.n+1), np.ones(self.n)]
        sp_matrix = sp.diags(list_diag,[0,1], format="csr")

        return sp_matrix / self.h


    def CD(self):
        list_diag=[-np.ones(self.n), np.zeros(self.n+1), np.ones(self.n)]
        sp_matrix = sp.diags(list_diag,[-1,0,1], format="csr")

        return sp_matrix / (2*self.h)
    


class second_derivative_FiniteDiff_sparse_method(object):

    def __init__(self, n):

        self.n = n

    def CD(self):
        '''
        This function will return only the matrix A.
        Need to define boundary condition on it and the RHS funtion times h^2
        '''
        list_diag = [np.ones(self.n), -2*np.ones(self.n+1), np.ones(self.n)]
        sp_matrix = sp.diags(list_diag,[-1,0,1], format="csr")

        return sp_matrix
    


class FD_1d_system_setup(object):

    def __init__(self, omega,N,alpha,beta,gamma,rhs):

        self.omega = omega
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rhs = rhs


    def FD1D_Hom(self):
        '''
        Return sparse LHS and RHS matrixes
        '''


        # grid
        h = (self.omega[1]-self.omega[0])/self.N
        x =np.linspace(self.omega[0],self.omega[1],self.N+1)

        # compute coeffs and rhs values
        diff = self.alpha(x)
        conv = self.beta(x)
        reac = self.gamma(x)
        F = self.rhs(x)

        # assemble system matrix
        diff_term = [-diff[1:self.N+1], 2*diff, -diff[0:-1]]
        conv_term = [-conv[1:self.N+1], conv[0:-1]]

        A = (1./h**2) * sp.diags(diff_term,[-1,0,1],format="csr")+(1./(2*h)) * sp.diags(conv_term,[-1,1],format="csr") + sp.diags(reac,0,format="csr")

        # modify system to account for homogeneous Dirichlet boundary conditions
        A[0,0] = 1; A[0,1] = 0; F[0] = 0
        A[self.N,self.N] = 1; A[self.N,self.N-1] = 0; F[self.N] = 0

        return A, F
    

    def FD1D_Non_Hom(self, lb_cond, ub_cond):
        '''
        Return sparse LHS and RHS matrixes
        '''


        # grid
        h = (self.omega[1]-self.omega[0])/self.N
        x =np.linspace(self.omega[0],self.omega[1],self.N+1)

        # compute coeffs and rhs values
        diff = self.alpha(x)
        conv = self.beta(x)
        reac = self.gamma(x)
        F = self.rhs(x)

        # assemble system matrix
        diff_term = [-diff[1:self.N+1], 2*diff, -diff[0:-1]]
        conv_term = [-conv[1:self.N+1], conv[0:-1]]

        A = (1./h**2) * sp.diags(diff_term,[-1,0,1],format="csr")+(1./(2*h)) * sp.diags(conv_term,[-1,1],format="csr") + sp.diags(reac,0,format="csr")

        # modify system to account for homogeneous Dirichlet boundary conditions
        A[0,0] = 1; A[0,1] = 0; F[0] = lb_cond
        A[self.N,self.N] = 1; A[self.N,self.N-1] = 0; F[self.N] = ub_cond

        return A, F




class FD_2D(object):

    def __init__(self, omega, N, rhs):

        self.omega = omega
        self.N = N
        self.rhs = rhs


    def FD2D_Poisson_Homo(self):

        # x and y axis grid
        h = (self.omega[1]-self.omega[0])/self.N
        x = np.linspace(self.omega[0],self.omega[1],self.N+1)
        y = x
        # 2-dim grid
        X, Y = np.meshgrid(x,y)
        X = X.flatten()
        Y = Y.flatten()

        # compute rhs
        F = self.rhs(X,Y)

        # compute system matrix
        coefs = [-1*np.ones((self.N+1)*(self.N)) ,-1*np.ones((self.N+1)*(self.N+1)-1),4*np.ones((self.N+1)*(self.N+1)),-1*np.ones((self.N+1)*(self.N+1)-1),-1*np.ones((self.N+1)*(self.N))]
        A = sp.diags(coefs, [-(self.N+1), -1, 0, 1, self.N+1],format="csr")

        # Implement boundary conditions
        for i in range(self.N+1):
            # y lower bound
            A[i,A[i].nonzero()] = 0; A[i,i] = 1; F[i] = 0
            # y upper bound
            j = (self.N+1) * self.N + i
            A[j,A[j].nonzero()] = 0; A[j,j] = 1; F[j] = 0

        for i in range(1,self.N):
            # x lower bound
            j = i * (self.N+1)
            A[j,A[j].nonzero()] = 0; A[j,j] = 1; F[j] = 0
            # x upper bound
            j = i * (self.N+1) + self.N
            A[j,A[j].nonzero()] = 0; A[j,j] = 1; F[j] = 0

        return (1./h**2) * A, F
    

    def FD2D_GEN_Homo(self,alpha: float, bx : float, by : float ,c: float):

        # x and y axis grid
        h = (self.omega[1]-self.omega[0])/self.N
        x = np.linspace(self.omega[0],self.omega[1],self.N+1)
        y = x
        # 2-dim grid
        X, Y = np.meshgrid(x,y)
        X = X.flatten()
        Y = Y.flatten()

        # compute rhs
        F = self.rhs(X,Y)

        # compute matrix second derivative
        coefs_A = [-1*np.ones((self.N+1)*(self.N)) ,-1*np.ones((self.N+1)*(self.N+1)-1),4*np.ones((self.N+1)*(self.N+1)),-1*np.ones((self.N+1)*(self.N+1)-1),-1*np.ones((self.N+1)*(self.N))]
        A_matrix = sp.diags(coefs_A, [-(self.N+1), -1, 0, 1, self.N+1],format="csr")
        A_matrix *= alpha*(1./h**2)
    

        # no formal method to aggregate the gradient of first order on 2d --> strategy --> decompose in 2 operations separately
        # compute matrix for gradient first derivative --> y component
        coefs_by = [-1*np.ones((self.N+1)*(self.N)),np.ones((self.N+1)*(self.N))]
        matrix_By = (by/(2*h))*sp.diags(coefs_by,[-(self.N+1),self.N+1],format="csr")
        # same thing here --> x component --> same code adjusted only for entries in matrix
        coefs_bx = [-1*np.ones((self.N+1)*(self.N+1)-1),np.ones((self.N+1)*(self.N+1)-1)]
        matrix_Bx = (bx/(2*h))*sp.diags(coefs_bx,[-1,1],format="csr")
        # Assemble matrix first order derivative
        B_matrix = matrix_Bx + matrix_By
        
        # no derivative term
        coefs_C = np.ones((self.N+1)*(self.N+1))
        C_matrix = c*sp.diags(coefs_C,0,format="csr")

        # Assemble LHS matrix
        LHS_matrix = A_matrix + B_matrix + C_matrix

        # Implement boundary conditions
        for i in range(self.N+1):
            # y lower bound
            LHS_matrix[i,LHS_matrix[i].nonzero()] = 0; LHS_matrix[i,i] = 1; F[i] = 0
            # y upper bound
            j = (self.N+1) * self.N + i
            LHS_matrix[j,LHS_matrix[j].nonzero()] = 0; LHS_matrix[j,j] = 1; F[j] = 0

        for i in range(1,self.N):
            # x lower bound
            j = i * (self.N+1)
            LHS_matrix[j,LHS_matrix[j].nonzero()] = 0; LHS_matrix[j,j] = 1; F[j] = 0
            # x upper bound
            j = i * (self.N+1) + self.N
            LHS_matrix[j,LHS_matrix[j].nonzero()] = 0; LHS_matrix[j,j] = 1; F[j] = 0

        return LHS_matrix, F
    


    def FD2D_GEN_Non_Homo(self,alpha: float, bx : float, by : float ,c: float, x_lb, x_ub, y_lb, y_ub):

        # x and y axis grid
        h = (self.omega[1]-self.omega[0])/self.N
        x = np.linspace(self.omega[0],self.omega[1],self.N+1)
        y = x
        # 2-dim grid
        X, Y = np.meshgrid(x,y)
        X = X.flatten()
        Y = Y.flatten()

        # compute rhs
        F = self.rhs(X,Y)

        # compute matrix second derivative
        coefs_A = [-1*np.ones((self.N+1)*(self.N)) ,-1*np.ones((self.N+1)*(self.N+1)-1),4*np.ones((self.N+1)*(self.N+1)),-1*np.ones((self.N+1)*(self.N+1)-1),-1*np.ones((self.N+1)*(self.N))]
        A_matrix = sp.diags(coefs_A, [-(self.N+1), -1, 0, 1, self.N+1],format="csr")
        A_matrix *= alpha*(1./h**2)
    

        # no formal method to aggregate the gradient of first order on 2d --> strategy --> decompose in 2 operations separately
        # compute matrix for gradient first derivative --> y component
        coefs_by = [-1*np.ones((self.N+1)*(self.N)),np.ones((self.N+1)*(self.N))]
        matrix_By = (by/(2*h))*sp.diags(coefs_by,[-(self.N+1),self.N+1],format="csr")
        # same thing here --> x component --> same code adjusted only for entries in matrix
        coefs_bx = [-1*np.ones((self.N+1)*(self.N+1)-1),np.ones((self.N+1)*(self.N+1)-1)]
        matrix_Bx = (bx/(2*h))*sp.diags(coefs_bx,[-1,1],format="csr")
        # Assemble matrix first order derivative
        B_matrix = matrix_Bx + matrix_By
        
        # no derivative term
        coefs_C = np.ones((self.N+1)*(self.N+1))
        C_matrix = c*sp.diags(coefs_C,0,format="csr")

        # Assemble LHS matrix
        LHS_matrix = A_matrix + B_matrix + C_matrix

        # Implement boundary conditions
        for i in range(self.N+1):
            # y lower bound
            LHS_matrix[i,LHS_matrix[i].nonzero()] = 0; LHS_matrix[i,i] = 1; F[i] = y_lb
            # y upper bound
            j = (self.N+1) * self.N + i
            LHS_matrix[j,LHS_matrix[j].nonzero()] = 0; LHS_matrix[j,j] = 1; F[j] = y_ub

        for i in range(1,self.N):
            # x lower bound
            j = i * (self.N+1)
            LHS_matrix[j,LHS_matrix[j].nonzero()] = 0; LHS_matrix[j,j] = 1; F[j] = x_lb
            # x upper bound
            j = i * (self.N+1) + self.N
            LHS_matrix[j,LHS_matrix[j].nonzero()] = 0; LHS_matrix[j,j] = 1; F[j] = x_ub

        return LHS_matrix, F