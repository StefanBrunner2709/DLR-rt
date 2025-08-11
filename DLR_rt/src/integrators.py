"""
Contains different time integrators.
"""

import numpy as np

from DLR_rt.src.lr import computeC, computeB, computeD, Kstep, Sstep, Lstep, computeF_b, add_basis_functions, drop_basis_functions, computeF_b_2x1d_X
from DLR_rt.src.lr import Kstep1, Kstep2, Kstep3, Sstep1, Sstep2, Sstep3, Lstep1, Lstep2, Lstep3


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

def PSI_lie(lr, grid, dt, F_b = None, DX = None, DY = None, dimensions = "1x1d"):
    """
    Projector splitting integrator with lie splitting.

    To run periodic simulations, leave the standard value F_b = None.
    To run inflow simulations, set F_b.
    For higher dimensional simulations set i.e. dimensions = "2x1d"

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
    dimensions
        Number of dimensions, given as a string.
    """
    if F_b is not None:
        inflow = True
    else:
        inflow = False

    # K step
    C1, C2 = computeC(lr, grid, dimensions = dimensions)
    K = lr.U @ lr.S
    K += dt * RK4(K, lambda K: Kstep(K, C1, C2, grid, lr, F_b, DX = DX, DY = DY, inflow = inflow, dimensions = dimensions), dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= np.sqrt(grid.dx)
    lr.S *= np.sqrt(grid.dx)

    # S step
    D1 = computeD(lr, grid, F_b, DX = DX, DY = DY, dimensions = dimensions)
    lr.S += dt * RK4(lr.S, lambda S: Sstep(S, C1, C2, D1, grid, inflow, dimensions = dimensions), dt)

    # L step
    L = lr.V @ lr.S.T
    B1 = computeB(L, grid, dimensions = dimensions)
    L += dt * RK4(L, lambda L: Lstep(L, D1, B1, grid, lr, inflow, dimensions = dimensions), dt)
    lr.V, St = np.linalg.qr(L, mode="reduced")
    lr.S = St.T
    if dimensions == "1x1d":
        lr.V /= np.sqrt(grid.dmu)
        lr.S *= np.sqrt(grid.dmu)
    elif dimensions == "2x1d":
        lr.V /= np.sqrt(grid.dphi)
        lr.S *= np.sqrt(grid.dphi)

    return lr, grid

def PSI_strang(lr, grid, dt, t, F_b = None, DX = None, DY = None):
    """
    Projector splitting integrator with strang splitting.

    To run periodic simulations, leave the standard value F_b = None.
    To run inflow simulations, set F_b.

    Parameters
    ----------
    lr
        LR class of subdomain.
    grid
        Grid class of subdomain.
    dt
        Time step size.
    t
        Current time.
    F_b
        Boundary condition matrix for inflow conditions.
    """
    if F_b is not None:
        inflow = True
    else:
        inflow = False

    # 1/2 K step
    C1, C2 = computeC(lr, grid)
    K = lr.U @ lr.S
    K += 0.5 * dt * RK4(K, lambda K: Kstep(K, C1, C2, grid, lr, F_b, inflow = inflow), 0.5 * dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= np.sqrt(grid.dx)
    lr.S *= np.sqrt(grid.dx)

    # 1/2 S step
    D1 = computeD(lr, grid, F_b, DX = DX, DY = DY)
    lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep(S, C1, C2, D1, grid, inflow), 0.5 * dt)

    # L step
    L = lr.V @ lr.S.T
    B1 = computeB(L, grid)
    L += dt * RK4(L, lambda L: Lstep(L, D1, B1, grid, lr, inflow), dt)
    lr.V, St = np.linalg.qr(L, mode="reduced")
    lr.S = St.T
    lr.V /= np.sqrt(grid.dmu)
    lr.S *= np.sqrt(grid.dmu)

    if inflow == True:
        # Compute F_b
        F_b = computeF_b(t + 0.5 * dt, lr.U @ lr.S @ lr.V.T, grid)      # recalculate F_b at time t + 0.5 dt
        D1 = computeD(lr, grid, F_b, DX = DX, DY = DY)                                    # recalculate D1 because we recalculated F_b

    # 1/2 S step
    C1, C2 = computeC(lr, grid)     # need to recalculate C1 and C2 because we changed V in L step     
    lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep(S, C1, C2, D1, grid, inflow), 0.5 * dt)

    # 1/2 K step
    K = lr.U @ lr.S
    K += 0.5 * dt * RK4(K, lambda K: Kstep(K, C1, C2, grid, lr, F_b, inflow = inflow), 0.5 * dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= np.sqrt(grid.dx)
    lr.S *= np.sqrt(grid.dx)

    return lr, grid

def PSI_splitting_lie(lr, grid, dt, F_b, F_b_top_bottom, DX = None, DY = None, lr_periodic = None, location = "left", tol_sing_val = 1e-6, drop_tol = 1e-6, rank_adapted = None, rank_dropped = None):
    """
    Projector splitting integrator with equation splitting and lie splitting.

    For simulations in 2x1d and DD, together with a splitting approach.

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
    F_b_top_bottom
        Boundary condition matrix for inflow conditions.
    """
    
    # Step 1: advection in x

    ### Add basis for adaptive rank strategy:
    lr, grid = add_basis_functions(lr, grid, F_b, tol_sing_val, dimensions = "2x1d", option = "x_advection")

    # K step
    C1, C2 = computeC(lr, grid, dimensions = "2x1d")
    K = lr.U @ lr.S
    K += dt * RK4(K, lambda K: Kstep1(C1, grid, lr, F_b, F_b_top_bottom, DX, DY), dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= np.sqrt(grid.dx)
    lr.S *= np.sqrt(grid.dx)

    # S step
    D1 = computeD(lr, grid, F_b, F_b_top_bottom, DX = DX, DY = DY, dimensions = "2x1d", option_dd = "dd")
    lr.S += dt * RK4(lr.S, lambda S: Sstep1(C1, D1, grid), dt)

    # L step
    L = lr.V @ lr.S.T
    B1 = computeB(L, grid, dimensions = "2x1d")
    L += dt * RK4(L, lambda L: Lstep1(lr, D1, grid), dt)
    lr.V, St = np.linalg.qr(L, mode="reduced")
    lr.S = St.T
    lr.V /= np.sqrt(grid.dphi)
    lr.S *= np.sqrt(grid.dphi)

    ### Drop basis for adaptive rank strategy:
    lr, grid = drop_basis_functions(lr, grid, drop_tol)

    # Step 2: advection in y

    ### Add basis for adaptive rank strategy:
    lr, grid = add_basis_functions(lr, grid, F_b_top_bottom, tol_sing_val, dimensions = "2x1d", option = "y_advection")
    if rank_adapted is not None:
        rank_adapted.append(grid.r)

    # K step
    C1, C2 = computeC(lr, grid, dimensions = "2x1d")
    K = lr.U @ lr.S
    K += dt * RK4(K, lambda K: Kstep2(C1, grid, lr, F_b, F_b_top_bottom, DX, DY), dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= np.sqrt(grid.dx)
    lr.S *= np.sqrt(grid.dx)

    # S step
    D1 = computeD(lr, grid, F_b, F_b_top_bottom, DX = DX, DY = DY, dimensions = "2x1d", option_dd = "dd")
    lr.S += dt * RK4(lr.S, lambda S: Sstep2(C1, D1, grid), dt)

    # L step
    L = lr.V @ lr.S.T
    L += dt * RK4(L, lambda L: Lstep2(lr, D1, grid), dt)
    lr.V, St = np.linalg.qr(L, mode="reduced")
    lr.S = St.T
    lr.V /= np.sqrt(grid.dphi)
    lr.S *= np.sqrt(grid.dphi)

    ### Drop basis for adaptive rank strategy:
    lr, grid = drop_basis_functions(lr, grid, drop_tol)
    if rank_dropped is not None:
        rank_dropped.append(grid.r)

    # Step 3: collisions

    # K step
    C1, C2 = computeC(lr, grid, dimensions = "2x1d")
    K = lr.U @ lr.S
    K += dt * RK4(K, lambda K: Kstep3(K, C2, grid), dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= np.sqrt(grid.dx)
    lr.S *= np.sqrt(grid.dx)

    # S step
    lr.S += dt * RK4(lr.S, lambda S: Sstep3(S, C2, grid), dt)

    # L step
    L = lr.V @ lr.S.T
    B1 = computeB(L, grid, dimensions = "2x1d")
    L += dt * RK4(L, lambda L: Lstep3(L, B1, grid), dt)
    lr.V, St = np.linalg.qr(L, mode="reduced")
    lr.S = St.T
    lr.V /= np.sqrt(grid.dphi)
    lr.S *= np.sqrt(grid.dphi)

    return lr, grid, rank_adapted, rank_dropped

def PSI_splitting_strang(lr, grid, dt, F_b, F_b_top_bottom, DX = None, DY = None, lr_periodic = None, location = "left", tol_sing_val = 1e-6, drop_tol = 1e-6, rank_adapted = None, rank_dropped = None):
    """
    Projector splitting integrator with equation splitting and strang splitting.

    For simulations in 2x1d and DD, together with a splitting approach.

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
    F_b_top_bottom
        Boundary condition matrix for inflow conditions.
    """

    # ToDo: Still need to add domain decomposition in Y to strang splitting, for now it does not work at all.
    
    # Step 1: advection in x

    ### Add basis for adaptive rank strategy:
    lr, grid = add_basis_functions(lr, grid, F_b, tol_sing_val, dimensions = "2x1d", option = "x_advection")

    # 1/2 K step
    C1, C2 = computeC(lr, grid, dimensions = "2x1d")
    K = lr.U @ lr.S
    K += 0.5 * dt * RK4(K, lambda K: Kstep1(C1, grid, lr, F_b, DX, DY), 0.5 * dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= np.sqrt(grid.dx)
    lr.S *= np.sqrt(grid.dx)

    # 1/2 S step
    D1 = computeD(lr, grid, F_b, F_b_top_bottom, DX = DX, DY = DY, dimensions = "2x1d", option_dd = "dd")
    lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep1(C1, D1, grid), 0.5 * dt)

    # L step
    L = lr.V @ lr.S.T
    B1 = computeB(L, grid, dimensions = "2x1d")
    L += dt * RK4(L, lambda L: Lstep1(lr, D1, grid), dt)
    lr.V, St = np.linalg.qr(L, mode="reduced")
    lr.S = St.T
    lr.V /= np.sqrt(grid.dphi)
    lr.S *= np.sqrt(grid.dphi)

    # Compute F_b
    if location == "left":
        F_b = computeF_b_2x1d_X(lr.U @ lr.S @ lr.V.T, grid, f_right = lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T, f_periodic = lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T)      # recalculate F_b at time t + 0.5 dt
    elif location == "right":
        F_b = computeF_b_2x1d_X(lr.U @ lr.S @ lr.V.T, grid, f_left = lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T, f_periodic = lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T)      # recalculate F_b at time t + 0.5 dt
    
    D1 = computeD(lr, grid, F_b, F_b_top_bottom, DX = DX, DY = DY, dimensions = "2x1d", option_dd = "dd")                                    # recalculate D1 because we recalculated F_b

    # 1/2 S step
    C1, C2 = computeC(lr, grid, dimensions = "2x1d")     # need to recalculate C1 and C2 because we changed V in L step     
    lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep1(C1, D1, grid), 0.5 * dt)

    # 1/2 K step
    K = lr.U @ lr.S
    K += 0.5 * dt * RK4(K, lambda K: Kstep1(C1, grid, lr, F_b, DX, DY), 0.5 * dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= np.sqrt(grid.dx)
    lr.S *= np.sqrt(grid.dx)

    ### Drop basis for adaptive rank strategy:
    lr, grid = drop_basis_functions(lr, grid, drop_tol)

    # Step 2: advection in y

    ### Add basis for adaptive rank strategy:
    lr, grid = add_basis_functions(lr, grid, F_b_top_bottom, tol_sing_val, dimensions = "2x1d", option = "y_advection")
    if rank_adapted is not None:
        rank_adapted.append(grid.r)

    # 1/2 K step
    C1, C2 = computeC(lr, grid, dimensions = "2x1d")
    K = lr.U @ lr.S
    K += 0.5 * dt * RK4(K, lambda K: Kstep2(K, C1, grid, DX, DY), 0.5 * dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= np.sqrt(grid.dx)
    lr.S *= np.sqrt(grid.dx)

    # 1/2 S step
    D1 = computeD(lr, grid, F_b, F_b_top_bottom, DX = DX, DY = DY, dimensions = "2x1d", option_dd = "dd")
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
        F_b = computeF_b_2x1d_X(lr.U @ lr.S @ lr.V.T, grid, f_right = lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T, f_periodic = lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T)      # recalculate F_b at time t + 0.5 dt
    elif location == "right":
        F_b = computeF_b_2x1d_X(lr.U @ lr.S @ lr.V.T, grid, f_left = lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T, f_periodic = lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T)      # recalculate F_b at time t + 0.5 dt
    
    D1 = computeD(lr, grid, F_b, F_b_top_bottom, DX = DX, DY = DY, dimensions = "2x1d", option_dd = "dd")                                    # maybe we dont even have to recompute here

    # 1/2 S step
    C1, C2 = computeC(lr, grid, dimensions = "2x1d")     # need to recalculate C1 and C2 because we changed V in L step     
    lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep2(S, C1, D1, grid), 0.5 * dt)

    # 1/2 K step
    K = lr.U @ lr.S
    K += 0.5 * dt * RK4(K, lambda K: Kstep2(K, C1, grid, DX, DY), 0.5 * dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= np.sqrt(grid.dx)
    lr.S *= np.sqrt(grid.dx)

    ### Drop basis for adaptive rank strategy:
    lr, grid = drop_basis_functions(lr, grid, drop_tol)
    if rank_dropped is not None:
        rank_dropped.append(grid.r)

    # Step 3: collisions

    # 1/2 K step
    C1, C2 = computeC(lr, grid, dimensions = "2x1d")
    K = lr.U @ lr.S
    K += 0.5 * dt * RK4(K, lambda K: Kstep3(K, C2, grid), 0.5 * dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= np.sqrt(grid.dx)
    lr.S *= np.sqrt(grid.dx)

    # 1/2 S step
    lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep3(S, C2, grid), 0.5 * dt)

    # L step
    L = lr.V @ lr.S.T
    B1 = computeB(L, grid, dimensions = "2x1d")
    L += dt * RK4(L, lambda L: Lstep3(L, B1, grid), dt)
    lr.V, St = np.linalg.qr(L, mode="reduced")
    lr.S = St.T
    lr.V /= np.sqrt(grid.dphi)
    lr.S *= np.sqrt(grid.dphi)

    # 1/2 S step
    C1, C2 = computeC(lr, grid, dimensions = "2x1d")     # need to recalculate C1 and C2 because we changed V in L step     
    lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep3(S, C2, grid), 0.5 * dt)

    # 1/2 K step
    K = lr.U @ lr.S
    K += 0.5 * dt * RK4(K, lambda K: Kstep3(K, C2, grid), 0.5 * dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= np.sqrt(grid.dx)
    lr.S *= np.sqrt(grid.dx)

    return lr, grid, rank_adapted, rank_dropped