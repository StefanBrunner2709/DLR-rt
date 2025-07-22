"""
Contains classes and functions to set up low rank structure.
"""

import numpy as np

from DLR_rt.src.util import computeD_cendiff_2x1d


class LR:
    """
    Low rank class.

    Generate low rank structure using matrices U, S and V.
    """
    def __init__(self, U, S, V):
        self.U = U
        self.S = S
        self.V = V

def computeF_b(t: float, f, grid, f_left = None, f_right= None):
    """
    Generate discretization of inflow/outflow boundary.
    
    Generate the discretization of the full boundary, depending on position of the current subdomain.
    If only one domain is present, we don't need to declare f_left or f_right.
    If the subdomain is the leftmost domain, declare f_right (Values for domain on the right).
    If the subdomain is the rightmost domain, declare f_left (Values for domain on the left).
    If the subdomain is between two other subdomains, declare both.

    Parameters
    ----------
    t : float
        Current time.
    f
        Values of subdomain, given as matrix.
    grid
        Grid class of subdomain.
    f_left
        Values of subdomain on left side, given as matrix.
    f_right
        Values of subdomain on right side, given as matrix.
    """
    
    F_b = np.zeros((2, len(grid.MU)))

    if f_left is not None and f_right is not None:
        for i in range(len(grid.MU)):           # Middle domain
            if grid.MU[i] > 0:
                F_b[0, i] = f_left[-1, i]       # Here we use inflow from domain on the left
                F_b[1, i] = f[grid.Nx-1,i] + (f[grid.Nx-1,i]-f[grid.Nx-2,i])/grid.dx * (1-grid.X[grid.Nx-1])
            elif grid.MU[i] < 0:
                F_b[1, i] = f_right[0, i]                   # Here we use inflow from domain on the right
                F_b[0, i] = f[0,i] - (f[1,i]-f[0,i])/grid.dx * grid.X[0]
    elif f_left is not None:                        # Left sided domain
        for i in range(len(grid.MU)):
            if grid.MU[i] > 0:
                F_b[0, i] = f_left[-1, i]       # Here we use inflow from domain on the left
                F_b[1, i] = f[grid.Nx-1,i] + (f[grid.Nx-1,i]-f[grid.Nx-2,i])/grid.dx * (1-grid.X[grid.Nx-1])
            elif grid.MU[i] < 0:
                F_b[1, i] = np.tanh(t)
                F_b[0, i] = f[0,i] - (f[1,i]-f[0,i])/grid.dx * grid.X[0]
    elif f_right is not None:                       # Right sided domain
        for i in range(len(grid.MU)):
            if grid.MU[i] > 0:
                F_b[0, i] = np.tanh(t)
                F_b[1, i] = f[grid.Nx-1,i] + (f[grid.Nx-1,i]-f[grid.Nx-2,i])/grid.dx * (1-grid.X[grid.Nx-1])
            elif grid.MU[i] < 0:
                F_b[1, i] = f_right[0, i]                   # Here we use inflow from domain on the right
                F_b[0, i] = f[0,i] - (f[1,i]-f[0,i])/grid.dx * grid.X[0]
    else:                                       # Only one domain
        for i in range(len(grid.MU)):
            if grid.MU[i] > 0:
                F_b[0, i] = np.tanh(t)
                F_b[1, i] = f[grid.Nx-1,i] + (f[grid.Nx-1,i]-f[grid.Nx-2,i])/grid.dx * (1-grid.X[grid.Nx-1])
            elif grid.MU[i] < 0:
                F_b[1, i] = np.tanh(t)
                F_b[0, i] = f[0,i] - (f[1,i]-f[0,i])/grid.dx * grid.X[0]

    return F_b

def computeK_bdry(lr, grid, F_b):
    """
    Compute boundary values for K.

    Transforms the boundary information given by F_b (discretization of inflow/outflow function) into a boundary information in K.
    """

    e_vec_left = np.zeros([len(grid.MU)])
    e_vec_right = np.zeros([len(grid.MU)])

    #Values from boundary condition:
    for i in range(len(grid.MU)):       # compute e-vector
        if grid.MU[i] > 0:
            e_vec_left[i] = F_b[0, i]
        elif grid.MU[i] < 0:
            e_vec_right[i] = F_b[1, i]
    
    int_exp_left = (e_vec_left @ lr.V) * grid.dmu   # compute integral from inflow, contains information from inflow from every K_j
    int_exp_right = (e_vec_right @ lr.V) * grid.dmu

    K = lr.U @ lr.S
    K_extrapol_left = np.zeros([grid.r])
    K_extrapol_right = np.zeros([grid.r])
    for i in range(grid.r):     # calculate extrapolated values
        K_extrapol_left[i] = K[0,i] - (K[1,i]-K[0,i])/grid.dx * grid.X[0]
        K_extrapol_right[i] = K[grid.Nx-1,i] + (K[grid.Nx-1,i]-K[grid.Nx-2,i])/grid.dx * (1-grid.X[grid.Nx-1])

    V_indicator_left = np.copy(lr.V)     # generate V*indicator, Note: Only works for Nx even
    V_indicator_left[int(grid.Nmu/2):,:] = 0
    V_indicator_right = np.copy(lr.V)
    V_indicator_right[:int(grid.Nmu/2),:] = 0

    int_V_left = (V_indicator_left.T @ lr.V) * grid.dmu        # compute integrals over V
    int_V_right = (V_indicator_right.T @ lr.V) * grid.dmu 

    sum_vector_left = K_extrapol_left @ int_V_left              # compute vector of size r with all the sums inside
    sum_vector_right = K_extrapol_right @ int_V_right

    K_bdry_left = int_exp_left + sum_vector_left            # add all together to get boundary info (vector with info for 1<=j<=r)
    K_bdry_right = int_exp_right + sum_vector_right    

    return K_bdry_left, K_bdry_right

def computedxK(lr, K_bdry_left, K_bdry_right, grid):
    """
    Compute the derivative of K.

    Compute the first derivative of K in x using a centered difference stencil.
    """
    K = lr.U @ lr.S

    Dx = np.zeros((grid.Nx, grid.Nx), dtype=int)
    np.fill_diagonal(Dx[1:], -1)
    np.fill_diagonal(Dx[:, 1:], 1)

    dxK = Dx @ K / (2*grid.dx)

    dxK[0,:] -= K_bdry_left / (2*grid.dx)
    dxK[-1,:] += K_bdry_right / (2*grid.dx)

    return dxK

def computeC(lr, grid, dimensions = "1x1d"):
    """
    Compute C coefficient.

    For higher dimensional simulations set i.e. dimensions = "2x1d"
    """
    if dimensions == "1x1d":

        C1 = (lr.V.T @ np.diag(grid.MU) @ lr.V) * grid.dmu

        C2 = (lr.V.T @ np.ones((grid.Nmu,grid.Nmu))).T * grid.dmu

        ### Alternative option, faster but harder to understand
        # muV = grid.MU[:, None] * lr.V
        # C1 = lr.V.T @ muV * grid.dmu
        # C2 = lr.V * grid.dmu

    elif dimensions == "2x1d":

        C1_1 = (lr.V.T @ np.diag(np.cos(grid.PHI)) @ lr.V) * grid.dphi
        C1_2 = (lr.V.T @ np.diag(np.sin(grid.PHI)) @ lr.V) * grid.dphi
        C1 = [C1_1, C1_2]

        C2 = (lr.V.T @ np.ones((grid.Nphi,grid.Nphi))).T * grid.dphi

    return C1, C2

def computeB(L, grid, dimensions = "1x1d"):
    """
    Compute B coeffiecient.

    For higher dimensional simulations set i.e. dimensions = "2x1d"
    """
    if dimensions == "1x1d":

        B1 = (L.T @ np.ones((grid.Nmu,grid.Nmu))).T * grid.dmu

    elif dimensions == "2x1d":

        B1 = (L.T @ np.ones((grid.Nphi,grid.Nphi))).T * grid.dphi

    return B1

def computeD(lr, grid, F_b = None, dimensions = "1x1d"):
    """
    Compute D coeffiecient.

    To compute D coefficient for periodic simulations, leave the standard value F_b = None.
    To compute D coefficient for inflow simulations, set F_b.
    For higher dimensional simulations set i.e. dimensions = "2x1d"
    """
    if dimensions == "1x1d":

        if F_b is not None:
            K_bdry_left, K_bdry_right = computeK_bdry(lr, grid, F_b)
            dxK = computedxK(lr, K_bdry_left, K_bdry_right, grid)
            D1 = lr.U.T @ dxK * grid.dx

        else:
            dxU = 0.5 * (np.roll(lr.U, -1, axis=0) - np.roll(lr.U, 1, axis=0)) / grid.dx
            D1 = lr.U.T @ dxU * grid.dx

    elif dimensions == "2x1d":
        
        DX, DY = computeD_cendiff_2x1d(grid)
        D1X = lr.U.T @ DX @ lr.U * grid.dx
        D1Y = lr.U.T @ DY @ lr.U * grid.dy
        D1 = [D1X, D1Y]
        
    return D1

def Kstep(K, C1, C2, grid, lr = None, F_b = None, inflow = False, dimensions = "1x1d"):
    """
    K step of 1-d radiative transfer equation.

    For K step for periodic simulations, leave the standard values inflow = False.
    For K step for inflow simulations, set inflow = True.
    For higher dimensional simulations set i.e. dimensions = "2x1d"
    """
    if dimensions == "1x1d":

        if inflow == True:
            K_bdry_left, K_bdry_right = computeK_bdry(lr, grid, F_b)
            dxK = computedxK(lr, K_bdry_left, K_bdry_right, grid)    
            rhs = - (grid.coeff) * dxK @ C1 + 0.5 * (grid.coeff)**2 * K @ C2.T @ C2 - (grid.coeff)**2 * K

        else:
            dxK = 0.5 * (np.roll(K, -1, axis=0) - np.roll(K, 1, axis=0)) / grid.dx    
            rhs = - (grid.coeff) * dxK @ C1 + 0.5 * (grid.coeff)**2 * K @ C2.T @ C2 - (grid.coeff)**2 * K

    elif dimensions == "2x1d":

        DX, DY = computeD_cendiff_2x1d(grid)
        rhs = - (grid.coeff) * DX @ K @ C1[0] - (grid.coeff) * DY @ K @ C1[1] + 0.5 / (np.pi) * (grid.coeff)**2 * K @ C2.T @ C2 - (grid.coeff)**2 * K
    
    return rhs

def Sstep(S, C1, C2, D1, grid, inflow = False, dimensions = "1x1d"):
    """
    S step of 1-d radiative transfer equation.

    For S step for periodic simulations, leave the standard values inflow = False.
    For S step for inflow simulations, set inflow = True.
    For higher dimensional simulations set i.e. dimensions = "2x1d"
    """
    if dimensions == "1x1d":

        if inflow == False:
            rhs = (grid.coeff) * D1 @ S @ C1 - 0.5 * (grid.coeff)**2 * S @ C2.T @ C2 + (grid.coeff)**2 * S

        elif inflow == True:
            rhs = (grid.coeff) * D1 @ C1 - 0.5 * (grid.coeff)**2 * S @ C2.T @ C2 + (grid.coeff)**2 * S

    elif dimensions == "2x1d":

        rhs = (grid.coeff) * D1[0] @ S @ C1[0] + (grid.coeff) * D1[1] @ S @ C1[1] - 0.5 / (np.pi) * (grid.coeff)**2 * S @ C2.T @ C2 + (grid.coeff)**2 * S

    return rhs

def Lstep(L, D1, B1, grid, lr = None, inflow = False, dimensions = "1x1d"):
    """
    L step of 1-d radiative transfer equation.

    For L step for periodic simulations, leave the standard values inflow = False.
    For L step for inflow simulations, set inflow = True.
    For higher dimensional simulations set i.e. dimensions = "2x1d"
    """
    if dimensions == "1x1d":

        if inflow == True:
            rhs = - (grid.coeff) * np.diag(grid.MU) @ lr.V @ D1.T + 0.5 * (grid.coeff)**2 * B1 - (grid.coeff)**2 * L
        else:
            rhs = - (grid.coeff) * np.diag(grid.MU) @ L @ D1.T + 0.5 * (grid.coeff)**2 * B1 - (grid.coeff)**2 * L

    elif dimensions == "2x1d":

        rhs = - (grid.coeff) * np.diag(np.cos(grid.PHI)) @ L @ D1[0].T - (grid.coeff) * np.diag(np.sin(grid.PHI)) @ L @ D1[1].T + 0.5 / (np.pi) * (grid.coeff)**2 * B1 - (grid.coeff)**2 * L
    
    return rhs

def add_basis_functions(lr, grid, F_b, tol_sing_val):
    """
    Add basis functions.

    Add basis functions according to the inflow condition of current subdomain and a tolarance for singular values.
    """
    # Compute SVD and drop singular values
    X, sing_val, QT = np.linalg.svd(F_b)
    r_b = np.sum(sing_val > tol_sing_val)
    Sigma = np.zeros((F_b.shape[0], r_b))
    np.fill_diagonal(Sigma, sing_val[:r_b])
    Q = QT.T[:,:r_b]

    # Concatenate
    X_h = np.random.rand(grid.Nx, r_b)
    lr.U = np.concatenate((lr.U, X_h), axis=1)
    lr.V = np.concatenate((lr.V, Q), axis=1)
    S_extended = np.zeros((grid.r + r_b, grid.r + r_b))
    S_extended[:grid.r, :grid.r] = lr.S
    lr.S = S_extended

    # QR-decomp
    lr.U, R_U = np.linalg.qr(lr.U, mode="reduced")
    lr.U /= np.sqrt(grid.dx)
    R_U *= np.sqrt(grid.dx)
    lr.V, R_V = np.linalg.qr(lr.V, mode="reduced")
    lr.V /= np.sqrt(grid.dmu)
    R_V *= np.sqrt(grid.dmu)
    lr.S = R_U @ lr.S @ R_V.T

    grid.r += r_b

    return lr, grid

def drop_basis_functions(lr, grid, drop_tol):
    """
    Drop basis functions.

    Drop basis functions according to some drop tolerance, such that the rank does not grow drastically.
    """
    U, sing_val, QT = np.linalg.svd(lr.S)
    r_prime = np.sum(sing_val > drop_tol)
    if r_prime < 5:
        r_prime = 5
    lr.S = np.zeros((r_prime, r_prime))
    np.fill_diagonal(lr.S, sing_val[:r_prime])
    U = U[:, :r_prime]
    Q = QT.T[:, :r_prime]
    lr.U = lr.U @ U
    lr.V = lr.V @ Q
    grid.r = r_prime

    return lr, grid