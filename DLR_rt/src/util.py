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
    Dx = sparse.lil_matrix((grid.Nx, grid.Nx), dtype=float)
    Dx.setdiag(-1, k=-1)
    Dx.setdiag(1, k=1)

    if option_dd == "no_dd":
        Dx[0, grid.Nx - 1] = -1
        Dx[grid.Nx - 1, 0] = 1

    # If option = "dd", we need to add information afterwards with inflow/outflow 
    # and cannot just impose periodic b.c.

    Ix = sparse.identity(grid.Ny, format="csr", dtype=float)

    # Step 2: Use np.kron
    DX = sparse.kron(Ix, Dx, format="csr")  # Ny × Ny blocks, each block is Dx

    ### Compute DY
    # Step 1: Set up cen difference matrix
    Dy = sparse.lil_matrix((grid.Ny, grid.Ny), dtype=float)
    Dy.setdiag(-1, k=-1)
    Dy.setdiag(1, k=1)

    if option_dd == "no_dd":
        Dy[0, grid.Ny - 1] = -1
        Dy[grid.Ny - 1, 0] = 1

    # If option = "dd", we need to add information afterwards with inflow/outflow 
    # and cannot just impose periodic b.c.

    Iy = sparse.identity(grid.Nx, format="csr", dtype=float)

    # Step 2: Use np.kron
    DY = sparse.kron(Dy, Iy, format="csr")  # Nx × Nx blocks, each block is Dy

    ### Scale matrices
    DX *= 0.5 / grid.dx
    DY *= 0.5 / grid.dy

    return DX.tocsr(), DY.tocsr()
