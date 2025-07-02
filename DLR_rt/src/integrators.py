"""
Contains different time integrators.
"""

import numpy as np

from DLR_rt.src.lr import computeC, computeB, computeD, Kstep, Sstep, Lstep, computeF_b


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

def PSI_lie(lr, grid, dt, F_b = None):
    """
    Projector splitting integrator with lie splitting.

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
    F_b
        Boundary condition matrix for inflow conditions.
    """
    if F_b is not None:
        inflow = True
    else:
        inflow = False

    # K step
    C1, C2 = computeC(lr, grid)
    K = lr.U @ lr.S
    K += dt * RK4(K, lambda K: Kstep(K, C1, C2, grid, lr, F_b, inflow), dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= np.sqrt(grid.dx)
    lr.S *= np.sqrt(grid.dx)

    # S step
    D1 = computeD(lr, grid, F_b)
    lr.S += dt * RK4(lr.S, lambda S: Sstep(S, C1, C2, D1, inflow), dt)

    # L step
    L = lr.V @ lr.S.T
    B1 = computeB(L, grid)
    L += dt * RK4(L, lambda L: Lstep(L, D1, B1, grid, lr, inflow), dt)
    lr.V, St = np.linalg.qr(L, mode="reduced")
    lr.S = St.T
    lr.V /= np.sqrt(grid.dmu)
    lr.S *= np.sqrt(grid.dmu)

    return lr, grid

def PSI_strang(lr, grid, dt, t, F_b = None):
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
    K += 0.5 * dt * RK4(K, lambda K: Kstep(K, C1, C2, grid, lr, F_b, inflow), 0.5 * dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= np.sqrt(grid.dx)
    lr.S *= np.sqrt(grid.dx)

    # 1/2 S step
    D1 = computeD(lr, grid, F_b)
    lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep(S, C1, C2, D1, inflow), 0.5 * dt)

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
        D1 = computeD(lr, grid, F_b)                                    # recalculate D1 because we recalculated F_b

    # 1/2 S step
    C1, C2 = computeC(lr, grid)     # need to recalculate C1 and C2 because we changed V in L step     
    lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep(S, C1, C2, D1, inflow), 0.5 * dt)

    # 1/2 K step
    K = lr.U @ lr.S
    K += 0.5 * dt * RK4(K, lambda K: Kstep(K, C1, C2, grid, lr, F_b, inflow), 0.5 * dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= np.sqrt(grid.dx)
    lr.S *= np.sqrt(grid.dx)

    return lr, grid