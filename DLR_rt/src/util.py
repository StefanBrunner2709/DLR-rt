"""
Contains functions like mass computation.
"""

import numpy as np
from scipy import sparse

from DLR_rt.src.grid import Grid_2x1d


def compute_mass(lr, grid):
    K = lr.U @ lr.S
    rho = K @ np.trapezoid(lr.V, dx=grid.dmu, axis=0).T
    M = np.trapezoid(rho, dx=grid.dx, axis=0)
    return M


def computeD_cendiff_2x1d(grid: Grid_2x1d, option_dd: str = "no_dd"):
    """
    Compute centered difference matrices.

    Compute centered difference matrices for 2x1d with or without domain decmoposition.
    Output is DX and DY.

    Parameters
    ----------
    grid
        Grid class of subdomain
    option_dd : str
        Can be chosen either "dd" or "no_dd"
    """
    ### Compute DX
    # Step 1: Set up cen difference matrix
    Dx = np.zeros((grid.Nx, grid.Nx), dtype=int)
    np.fill_diagonal(Dx[1:], -1)
    np.fill_diagonal(Dx[:, 1:], 1)

    if option_dd == "no_dd":
        Dx[0, grid.Nx - 1] = -1
        Dx[grid.Nx - 1, 0] = 1

    # If option = "dd", we need to add information afterwards with inflow/outflow 
    # and cannot just impose periodic b.c.

    identity = np.eye(grid.Ny, grid.Ny)

    # Step 2: Use np.kron
    DX = np.kron(identity, Dx)

    ### Compute DY
    # Step 1: Set up cen difference matrix
    Dy = np.zeros((grid.Ny, grid.Ny), dtype=int)
    np.fill_diagonal(Dy[1:], -1)
    np.fill_diagonal(Dy[:, 1:], 1)

    if option_dd == "no_dd":
        Dy[0, grid.Ny - 1] = -1
        Dy[grid.Ny - 1, 0] = 1

    # If option = "dd", we need to add information afterwards with inflow/outflow 
    # and cannot just impose periodic b.c.

    identity = np.eye(grid.Nx, grid.Nx)

    # Step 2: Use np.kron
    DY = np.kron(Dy, identity)

    ### Scale matrices
    DX = 0.5 * DX / grid.dx
    DY = 0.5 * DY / grid.dy

    ### Make matrix sparse
    DX = sparse.csr_matrix(DX)
    DY = sparse.csr_matrix(DY)

    return DX, DY
