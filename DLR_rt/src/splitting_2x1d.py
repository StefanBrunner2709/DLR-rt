"""
All functions for simulation of 2x1d DD simulations with splitting
ToDo: Incorporate functions into rest of the code (here we have code duplication)
"""

import numpy as np


class LR:
    """
    Low rank class.

    Generate low rank structure using matrices U, S and V.
    """
    def __init__(self, U, S, V):
        self.U = U
        self.S = S
        self.V = V


class Grid_2x1d:
    """
    Generate 2x1 dimensional grid.

    Helps to generate an equidistant grid. For calculations with DD and splitting approach.
    Angle domain is set from [0, 2*pi]. Spacial domain is [0,1]x[0,1].

    Parameters
    ----------
    _Nx : int
        Number of gridpoints in x.
    _Ny : int
        Number of gridpoints in y.
    _Nphi : int
        Number of gridpoints in phi.
    _r : int
        Initial rank of the simulation.
    _coeff
        1/epsilon for radiative transfer equation on this domain.
    """
    def __init__(self, _Nx: int, _Ny: int, _Nphi: int, _r: int = 5, _X = None, _coeff : float = 1.0):
        self.Nx = _Nx
        self.Ny = _Ny
        self.Nphi = _Nphi
        self.r = _r
        self.coeff = _coeff

        
        if _X is None:
            self.X = np.linspace(1/(2*self.Nx), 1 - 1/(2*self.Nx), self.Nx, endpoint=True) # Point 0 and 1 are not on the grid, spacing to first gridpoint: delta_x/2
        else:
            self.X = _X

        self.Y = np.linspace(1/(2*self.Ny), 1 - 1/(2*self.Ny), self.Ny, endpoint=True) # Point 0 and 1 are not on the grid, spacing to first gridpoint: delta_x/2

        self.PHI = np.linspace(0.0, 2*np.pi, self.Nphi, endpoint=False)        # 2*pi is the same angle as 0
        
        self.dx = self.X[1] - self.X[0]
        self.dy = self.Y[1] - self.Y[0]
        self.dphi = self.PHI[1] - self.PHI[0]

    def split_x(self, _coeff_left = None, _coeff_right = None):
        """
        Split domain into 2 subdomains in x dimension.

        Split the domain into 2 subdomains by dividing the domain in half in the middle of the X grid.

        Parameters
        ----------
        _coeff_left
            1/epsilon for radiative transfer equation on left subdomain. If None, value from whole domain is taken.
        _coeff_right
            1/epsilon for radiative transfer equation on right subdomain. If None, value from whole domain is taken.
        """
        if _coeff_left is None:
            _coeff_left = self.coeff
        if _coeff_right is None:
            _coeff_right = self.coeff

        # Split grid
        X_left = self.X[:int(self.Nx/2)]
        X_right = self.X[int(self.Nx/2):]

        # Create new Grid instances for left and right
        left_grid = Grid_2x1d(int(self.Nx/2), self.Ny, self.Nphi, self.r, _X=X_left, _coeff = _coeff_left)
        right_grid = Grid_2x1d(int(self.Nx/2), self.Ny, self.Nphi, self.r, _X=X_right, _coeff = _coeff_right)

        return left_grid, right_grid
    

def setInitialCondition_2x1d_lr(grid: Grid_2x1d):
    """
    Set initial condition.

    Set initial condition for 2x1d low rank grid with domain decomposition.
    """
    S = np.zeros((grid.r, grid.r))
    U = np.zeros((grid.Nx * grid.Ny, grid.r))
    V = np.zeros((grid.Nphi, grid.r))
    for i in range(grid.Ny):
        U[i*grid.Nx:(i+1)*grid.Nx, 0] = 1/(2 * np.pi) * np.exp(-((grid.X-0.5)**2)/0.07) * np.exp(-((grid.Y[i]-0.5)**2)/0.07)
        # U[i*grid.Nx:(i+1)*grid.Nx, 0] = np.sin(2*np.pi*grid.X)*np.sin(2*np.pi*grid.Y[i])
    V[0, 0] = 1.0
    S[0, 0] = 1.0

    U_ortho, R_U = np.linalg.qr(U, mode="reduced")
    V_ortho, R_V = np.linalg.qr(V, mode="reduced")
    S_ortho = R_U @ S @ R_V.T

    lr = LR(U_ortho, S_ortho, V_ortho)
    return lr
    

def computeD_cendiff_2x1d(grid):
    """
    Compute centered difference matrices.

    Compute centered difference matrices for 2x1d with domain decomposition. Output is DX and DY.
    Because we only split domain in x, we can assume perioidic boundary conditions in y.
    """
    ### Compute DX
    # Step 1: Set up cen difference matrix
    Dx = np.zeros((grid.Nx, grid.Nx), dtype=int)
    np.fill_diagonal(Dx[1:], -1)
    np.fill_diagonal(Dx[:, 1:], 1)

    # We have no periodic boundary condition in x. We need to add information afterwards with inflow/outflow

    I = np.eye(grid.Ny, grid.Ny)

    # Step 2: Use np.kron
    DX = np.kron(I, Dx)

    ### Compute DY
    # Step 1: Set up cen difference matrix
    Dy = np.zeros((grid.Ny, grid.Ny), dtype=int)
    np.fill_diagonal(Dy[1:], -1)
    np.fill_diagonal(Dy[:, 1:], 1)
    Dy[0, grid.Ny-1] = -1           # We have periodic boundary conditions in y, because we do not split the domain in y
    Dy[grid.Ny-1, 0] = 1

    I = np.eye(grid.Nx, grid.Nx)

    # Step 2: Use np.kron
    DY = np.kron(Dy, I)

    ### Scale matrices
    DX = 0.5 * DX / grid.dx
    DY = 0.5 * DY / grid.dy

    return DX, DY


def computeF_b_2x1d(f, grid, f_left = None, f_right = None, f_periodic = None):
    """
    Generate discretization of 2x1d full boundary.
    
    Generate the discretization of the full boundary, depending on position of the current subdomain.
    If only one domain is present, we don't need to declare f_left, f_right or f_periodic
    If the subdomain is the leftmost domain, declare f_right (Values for domain on the right) and f_periodic (values for rightmost domain).
    If the subdomain is the rightmost domain, declare f_left (Values for domain on the left)  and f_periodic (values for leftmost domain).
    If the subdomain is between two other subdomains, declare f_left and f_right.

    Parameters
    ----------
    f
        Values of subdomain, given as matrix.
    grid
        Grid class of subdomain.
    f_left
        Values of subdomain on left side, given as matrix.
    f_right
        Values of subdomain on right side, given as matrix.
    f_periodic
        Values of subdomain on other side of periodic boundary, given as matrix.
    """

    F_b = np.zeros((2*len(grid.Y), len(grid.PHI)))

    # ToDo: make it work for all Phi. Right now only Phi=0

    if f_periodic is not None:  # left or right sided domain
        if f_right is not None:  # left sided domain

            for i in range(len(grid.PHI)):

                if grid.PHI[i] < np.pi/2 or grid.PHI[i] > 3/2*np.pi:

                    indices_left = list(range(grid.Nx-1, grid.Nx*(grid.Ny+1)-1, grid.Nx))       # pick every Nxth row
                    F_b[:len(grid.Y),i] = f_periodic[indices_left, i]                 # This is inflow from left

                    indices_outflow_1 = list(range(grid.Nx-1, grid.Nx*(grid.Ny+1)-1, grid.Nx))
                    indices_outflow_2 = list(range(grid.Nx-2, grid.Nx*(grid.Ny+1)-2, grid.Nx))
                    F_b[len(grid.Y):,i] = f[indices_outflow_1, i] + (f[indices_outflow_1, i] - f[indices_outflow_2, i]) # outflow to right side

                else:

                    indices_right = list(range(0, grid.Nx*(grid.Ny), grid.Nx))
                    F_b[len(grid.Y):,i] = f_right[indices_right, i]                     # This is inflow from right

                    indices_outflow_0 = list(range(0, grid.Nx*(grid.Ny), grid.Nx))
                    indices_outflow_1 = list(range(1, grid.Nx*(grid.Ny)+1, grid.Nx))
                    F_b[:len(grid.Y),i] = f[indices_outflow_0, i] - (f[indices_outflow_1, i]-f[indices_outflow_0, i]) # outflow to left side

        elif f_left is not None:   # right sided domain

            for i in range(len(grid.PHI)):

                if grid.PHI[i] < np.pi/2 or grid.PHI[i] > 3/2*np.pi:

                    indices_left = list(range(grid.Nx-1, grid.Nx*(grid.Ny+1)-1, grid.Nx))       # pick every Nxth row
                    F_b[:len(grid.Y),i] = f_left[indices_left, i]                 # This is inflow from left

                    indices_outflow_1 = list(range(grid.Nx-1, grid.Nx*(grid.Ny+1)-1, grid.Nx))
                    indices_outflow_2 = list(range(grid.Nx-2, grid.Nx*(grid.Ny+1)-2, grid.Nx))
                    F_b[len(grid.Y):,i] = f[indices_outflow_1, i] + (f[indices_outflow_1, i] - f[indices_outflow_2, i]) # outflow to right side

                else:

                    indices_right = list(range(0, grid.Nx*(grid.Ny), grid.Nx))
                    F_b[len(grid.Y):,i] = f_periodic[indices_right, i]                     # This is inflow from right

                    indices_outflow_0 = list(range(0, grid.Nx*(grid.Ny), grid.Nx))
                    indices_outflow_1 = list(range(1, grid.Nx*(grid.Ny)+1, grid.Nx))
                    F_b[:len(grid.Y),i] = f[indices_outflow_0, i] - (f[indices_outflow_1, i]-f[indices_outflow_0, i]) # outflow to left side


    elif f_left is not None and f_right is not None:    # middle domain

        ToDo=0

    else:       # only one domain

        ToDo=0

    return F_b

def computeK_bdry_2x1d(lr, grid, F_b):
    """
    Compute boundary values for K in 2x1d.

    Transforms the boundary information given by F_b into a boundary information in K.
    """
    e_mat_left = np.zeros((len(grid.Y), len(grid.PHI)))
    e_mat_right = np.zeros((len(grid.Y), len(grid.PHI)))

    #Values from boundary condition:
    for i in range(len(grid.PHI)):       # compute e-vector
        if grid.PHI[i] < np.pi/2 or grid.PHI[i] > 3*np.pi/2:
            e_mat_left[:,i] = F_b[:len(grid.Y),i]
        else:
            e_mat_right[:,i] = F_b[len(grid.Y):,i]
    
    int_exp_left = (e_mat_left @ lr.V) * grid.dphi   # compute integral from inflow, contains information from inflow from every K_j
    int_exp_right = (e_mat_right @ lr.V) * grid.dphi     # now matrix of dimension Ny x r

    K = lr.U @ lr.S
    K_extrapol_left = np.zeros([len(grid.Y), grid.r])
    K_extrapol_right = np.zeros([len(grid.Y), grid.r])

    for i in range(grid.r):     # calculate extrapolated values

        indices_outflow_left_0 = list(range(0, grid.Nx*(grid.Ny), grid.Nx))
        indices_outflow_left_1 = list(range(1, grid.Nx*(grid.Ny)+1, grid.Nx))
        K_extrapol_left[:,i] = K[indices_outflow_left_0,i] - (K[indices_outflow_left_1,i]-K[indices_outflow_left_0,i])

        indices_outflow_right_1 = list(range(grid.Nx-1, grid.Nx*(grid.Ny+1)-1, grid.Nx))
        indices_outflow_right_2 = list(range(grid.Nx-2, grid.Nx*(grid.Ny+1)-2, grid.Nx))
        K_extrapol_right[:,i] = K[indices_outflow_right_1,i] + (K[indices_outflow_right_1,i]-K[indices_outflow_right_2,i])

    V_indicator_left = np.copy(lr.V)     # generate V*indicator, Note: Only works for Nx even
    V_indicator_left[:int(grid.Nphi/4),:] = 0
    V_indicator_left[int(3*grid.Nphi/4):,:] = 0
    V_indicator_right = np.copy(lr.V)
    V_indicator_right[int(grid.Nphi/4):int(3*grid.Nphi/4),:] = 0

    int_V_left = (V_indicator_left.T @ lr.V) * grid.dphi        # compute integrals over V
    int_V_right = (V_indicator_right.T @ lr.V) * grid.dphi 

    sum_vector_left = K_extrapol_left @ int_V_left              # compute matrix of size Ny x r with all the sums inside
    sum_vector_right = K_extrapol_right @ int_V_right

    K_bdry_left = int_exp_left + sum_vector_left            # add all together to get boundary info (matrix with info for 1<=j<=r)
    K_bdry_right = int_exp_right + sum_vector_right    

    return K_bdry_left, K_bdry_right

def computedxK_2x1d(lr, K_bdry_left, K_bdry_right, grid):
    """
    Compute the x derivative of K.

    Compute the first derivative of K in x using a centered difference stencil.
    """
    K = lr.U @ lr.S
    DX, DY = computeD_cendiff_2x1d(grid)

    DXK = DX @ K

    # Still need to add boundary information
    indices_1 = list(range(0, grid.Nx*(grid.Ny), grid.Nx))
    DXK[indices_1,:] = DXK[indices_1,:] - K_bdry_left / (2*grid.dx)
    indices_2 = list(range(grid.Nx-1, grid.Nx*(grid.Ny+1)-1, grid.Nx))
    DXK[indices_2,:] = DXK[indices_2,:] + K_bdry_right / (2*grid.dx)

    return DXK


def computeC(lr, grid):
    """
    Compute C coefficient.

    For simulation with dimension 2x1d.
    """

    C1_1 = (lr.V.T @ np.diag(np.cos(grid.PHI)) @ lr.V) * grid.dphi
    C1_2 = (lr.V.T @ np.diag(np.sin(grid.PHI)) @ lr.V) * grid.dphi
    C1 = [C1_1, C1_2]

    C2 = (lr.V.T @ np.ones((grid.Nphi,grid.Nphi))).T * grid.dphi

    return C1, C2

def computeB(L, grid):
    """
    Compute B coeffiecient.

    For simulation with dimension 2x1d.
    """

    B1 = (L.T @ np.ones((grid.Nphi,grid.Nphi))).T * grid.dphi

    return B1

def computeD(lr, grid, F_b):
    """
    Compute D coeffiecient.

    For simulation with dimension 2x1d and domain decomposition.
    """

    K_bdry_left, K_bdry_right = computeK_bdry_2x1d(lr, grid, F_b)
    DXK = computedxK_2x1d(lr, K_bdry_left, K_bdry_right, grid)
    D1X = lr.U.T @ DXK * grid.dx

    DX, DY = computeD_cendiff_2x1d(grid)
    D1Y = lr.U.T @ DY @ lr.U * grid.dy      # For D1Y we can stay with the more simple approach, because we do not do split the boundary in y

    D1 = [D1X, D1Y]

    return D1


def add_basis_functions(lr, grid, F_b, tol_sing_val):
    """
    Add basis functions.

    Add basis functions according to the inflow condition of current subdomain and a tolarance for singular values.
    For 2x1d DD simulations.
    """
    # Compute SVD and drop singular values
    X, sing_val, QT = np.linalg.svd(F_b)
    r_b = np.sum(sing_val > tol_sing_val)
    if (grid.r + r_b) > grid.Nphi:      # because rank cannot be bigger than our amount of gridpoints
        r_b = grid.Nphi - grid.r
    Sigma = np.zeros((F_b.shape[0], r_b))
    np.fill_diagonal(Sigma, sing_val[:r_b])
    Q = QT.T[:,:r_b]

    # Concatenate
    
    X_h = np.random.rand(grid.Nx*grid.Ny, r_b)
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
    
    lr.V /= np.sqrt(grid.dphi)
    R_V *= np.sqrt(grid.dphi)
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


def Kstep1(C1, grid, lr, F_b):

    K_bdry_left, K_bdry_right = computeK_bdry_2x1d(lr, grid, F_b)
    DXK = computedxK_2x1d(lr, K_bdry_left, K_bdry_right, grid)
    rhs = - (grid.coeff) * DXK @ C1[0]

    return rhs

def Kstep2(K, C1, grid):

    DX, DY = computeD_cendiff_2x1d(grid)
    rhs = - (grid.coeff) * DY @ K @ C1[1]

    return rhs

def Kstep3(K, C2, grid):

    rhs = 0.5 / (np.pi) * (grid.coeff)**2 * K @ C2.T @ C2 - (grid.coeff)**2 * K

    return rhs

def Sstep1(C1, D1, grid):

    rhs = (grid.coeff) * D1[0] @ C1[0]

    return rhs

def Sstep2(S, C1, D1, grid):

    rhs = (grid.coeff) * D1[1] @ S @ C1[1]

    return rhs

def Sstep3(S, C2, grid):

    rhs = - 0.5 / (np.pi) * (grid.coeff)**2 * S @ C2.T @ C2 + (grid.coeff)**2 * S

    return rhs

def Lstep1(lr, D1, grid):

    rhs = - (grid.coeff) * np.diag(np.cos(grid.PHI)) @ lr.V @ D1[0].T

    return rhs

def Lstep2(L, D1, grid):

    rhs = - (grid.coeff) * np.diag(np.sin(grid.PHI)) @ L @ D1[1].T

    return rhs

def Lstep3(L, B1, grid):

    rhs = 0.5 / (np.pi) * (grid.coeff)**2 * B1 - (grid.coeff)**2 * L

    return rhs


def RK4(f, rhs, dt):
    """
    Runge Kutta 4.

    Time integration method.
    """
    b_coeff = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
    
    k_coeff0 = rhs(f)
    k_coeff1 = rhs(f + dt * 0.5 * k_coeff0)
    k_coeff2 = rhs(f + dt * 0.5 * k_coeff1)
    k_coeff3 = rhs(f + dt * k_coeff2)

    return b_coeff[0] * k_coeff0 + b_coeff[1] * k_coeff1 + b_coeff[2] * k_coeff2 + b_coeff[3] * k_coeff3

def PSI_splitting(lr, grid, dt, F_b, lr_periodic = None, option = "lie", location = "left"):
    """
    Projector splitting integrator with splitting.

    For simulations in 2x1d and DD, together with a splitting aproach.

    Parameters
    ----------
    lr
        LR class of subdomain.
    grid
        Grid class of subdomain.
    dt
        Time step size.
    F_b
        Boundary condition matrix for inflow conditions.
    """
    
    # Step 1: advection in x

    if option == "lie":
        # K step
        C1, C2 = computeC(lr, grid)
        K = lr.U @ lr.S
        K += dt * RK4(K, lambda K: Kstep1(C1, grid, lr, F_b), dt)
        lr.U, lr.S = np.linalg.qr(K, mode="reduced")
        lr.U /= np.sqrt(grid.dx)
        lr.S *= np.sqrt(grid.dx)

        # S step
        D1 = computeD(lr, grid, F_b)
        lr.S += dt * RK4(lr.S, lambda S: Sstep1(C1, D1, grid), dt)

        # L step
        L = lr.V @ lr.S.T
        B1 = computeB(L, grid)
        L += dt * RK4(L, lambda L: Lstep1(lr, D1, grid), dt)
        lr.V, St = np.linalg.qr(L, mode="reduced")
        lr.S = St.T
        lr.V /= np.sqrt(grid.dphi)
        lr.S *= np.sqrt(grid.dphi)

    
    elif option == "strang":
        # 1/2 K step
        C1, C2 = computeC(lr, grid)
        K = lr.U @ lr.S
        K += 0.5 * dt * RK4(K, lambda K: Kstep1(C1, grid, lr, F_b), 0.5 * dt)
        lr.U, lr.S = np.linalg.qr(K, mode="reduced")
        lr.U /= np.sqrt(grid.dx)
        lr.S *= np.sqrt(grid.dx)

        # 1/2 S step
        D1 = computeD(lr, grid, F_b)
        lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep1(C1, D1, grid), 0.5 * dt)

        # L step
        L = lr.V @ lr.S.T
        B1 = computeB(L, grid)
        L += dt * RK4(L, lambda L: Lstep1(lr, D1, grid), dt)
        lr.V, St = np.linalg.qr(L, mode="reduced")
        lr.S = St.T
        lr.V /= np.sqrt(grid.dphi)
        lr.S *= np.sqrt(grid.dphi)

        # Compute F_b
        if location == "left":
            F_b = computeF_b_2x1d(lr.U @ lr.S @ lr.V.T, grid, f_right = lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T, f_periodic = lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T)      # recalculate F_b at time t + 0.5 dt
        elif location == "right":
            F_b = computeF_b_2x1d(lr.U @ lr.S @ lr.V.T, grid, f_left = lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T, f_periodic = lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T)      # recalculate F_b at time t + 0.5 dt
        
        D1 = computeD(lr, grid, F_b)                                    # recalculate D1 because we recalculated F_b

        # 1/2 S step
        C1, C2 = computeC(lr, grid)     # need to recalculate C1 and C2 because we changed V in L step     
        lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep1(C1, D1, grid), 0.5 * dt)

        # 1/2 K step
        K = lr.U @ lr.S
        K += 0.5 * dt * RK4(K, lambda K: Kstep1(C1, grid, lr, F_b), 0.5 * dt)
        lr.U, lr.S = np.linalg.qr(K, mode="reduced")
        lr.U /= np.sqrt(grid.dx)
        lr.S *= np.sqrt(grid.dx)


    # Step 2: advection in y

    if option == "lie":
        # K step
        C1, C2 = computeC(lr, grid)
        K = lr.U @ lr.S
        K += dt * RK4(K, lambda K: Kstep2(K, C1, grid), dt)
        lr.U, lr.S = np.linalg.qr(K, mode="reduced")
        lr.U /= np.sqrt(grid.dx)
        lr.S *= np.sqrt(grid.dx)

        # S step
        D1 = computeD(lr, grid, F_b)
        lr.S += dt * RK4(lr.S, lambda S: Sstep2(S, C1, D1, grid), dt)

        # L step
        L = lr.V @ lr.S.T
        L += dt * RK4(L, lambda L: Lstep2(L, D1, grid), dt)
        lr.V, St = np.linalg.qr(L, mode="reduced")
        lr.S = St.T
        lr.V /= np.sqrt(grid.dphi)
        lr.S *= np.sqrt(grid.dphi)

    elif option == "strang":
        # 1/2 K step
        C1, C2 = computeC(lr, grid)
        K = lr.U @ lr.S
        K += 0.5 * dt * RK4(K, lambda K: Kstep2(K, C1, grid), 0.5 * dt)
        lr.U, lr.S = np.linalg.qr(K, mode="reduced")
        lr.U /= np.sqrt(grid.dx)
        lr.S *= np.sqrt(grid.dx)

        # 1/2 S step
        D1 = computeD(lr, grid, F_b)
        lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep2(S, C1, D1, grid), 0.5 * dt)

        # L step
        L = lr.V @ lr.S.T
        L += dt * RK4(L, lambda L: Lstep2(L, D1, grid), dt)
        lr.V, St = np.linalg.qr(L, mode="reduced")
        lr.S = St.T
        lr.V /= np.sqrt(grid.dphi)
        lr.S *= np.sqrt(grid.dphi)

        # Compute F_b
        if location == "left":
            F_b = computeF_b_2x1d(lr.U @ lr.S @ lr.V.T, grid, f_right = lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T, f_periodic = lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T)      # recalculate F_b at time t + 0.5 dt
        elif location == "right":
            F_b = computeF_b_2x1d(lr.U @ lr.S @ lr.V.T, grid, f_left = lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T, f_periodic = lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T)      # recalculate F_b at time t + 0.5 dt
        
        D1 = computeD(lr, grid, F_b)                                    # maybe we dont even have to recompute here

        # 1/2 S step
        C1, C2 = computeC(lr, grid)     # need to recalculate C1 and C2 because we changed V in L step     
        lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep2(S, C1, D1, grid), 0.5 * dt)

        # 1/2 K step
        K = lr.U @ lr.S
        K += 0.5 * dt * RK4(K, lambda K: Kstep2(K, C1, grid), 0.5 * dt)
        lr.U, lr.S = np.linalg.qr(K, mode="reduced")
        lr.U /= np.sqrt(grid.dx)
        lr.S *= np.sqrt(grid.dx)


    # Step 3: collisions

    if option == "lie":
        # K step
        C1, C2 = computeC(lr, grid)
        K = lr.U @ lr.S
        K += dt * RK4(K, lambda K: Kstep3(K, C2, grid), dt)
        lr.U, lr.S = np.linalg.qr(K, mode="reduced")
        lr.U /= np.sqrt(grid.dx)
        lr.S *= np.sqrt(grid.dx)

        # S step
        lr.S += dt * RK4(lr.S, lambda S: Sstep3(S, C2, grid), dt)

        # L step
        L = lr.V @ lr.S.T
        B1 = computeB(L, grid)
        L += dt * RK4(L, lambda L: Lstep3(L, B1, grid), dt)
        lr.V, St = np.linalg.qr(L, mode="reduced")
        lr.S = St.T
        lr.V /= np.sqrt(grid.dphi)
        lr.S *= np.sqrt(grid.dphi)

    elif option == "strang":
        # 1/2 K step
        C1, C2 = computeC(lr, grid)
        K = lr.U @ lr.S
        K += 0.5 * dt * RK4(K, lambda K: Kstep3(K, C2, grid), 0.5 * dt)
        lr.U, lr.S = np.linalg.qr(K, mode="reduced")
        lr.U /= np.sqrt(grid.dx)
        lr.S *= np.sqrt(grid.dx)

        # 1/2 S step
        lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep3(S, C2, grid), 0.5 * dt)

        # L step
        L = lr.V @ lr.S.T
        B1 = computeB(L, grid)
        L += dt * RK4(L, lambda L: Lstep3(L, B1, grid), dt)
        lr.V, St = np.linalg.qr(L, mode="reduced")
        lr.S = St.T
        lr.V /= np.sqrt(grid.dphi)
        lr.S *= np.sqrt(grid.dphi)

        # 1/2 S step
        C1, C2 = computeC(lr, grid)     # need to recalculate C1 and C2 because we changed V in L step     
        lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep3(S, C2, grid), 0.5 * dt)

        # 1/2 K step
        K = lr.U @ lr.S
        K += 0.5 * dt * RK4(K, lambda K: Kstep3(K, C2, grid), 0.5 * dt)
        lr.U, lr.S = np.linalg.qr(K, mode="reduced")
        lr.U /= np.sqrt(grid.dx)
        lr.S *= np.sqrt(grid.dx)

    return lr, grid
