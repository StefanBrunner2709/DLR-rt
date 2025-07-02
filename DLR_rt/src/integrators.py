"""
Contains different time integrators.
"""

import numpy as np

from DLR_rt.src.lr import computeC, computeB, computeD, Kstep, Sstep, Lstep


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