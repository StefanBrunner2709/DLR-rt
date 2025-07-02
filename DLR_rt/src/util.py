"""
Contains functions like mass computation.
"""

import numpy as np


def compute_mass(lr, grid):
    K = lr.U @ lr.S
    rho = K @ np.trapezoid(lr.V, dx=grid.dmu, axis=0).T
    M = np.trapezoid(rho, dx=grid.dx, axis=0)
    return M
