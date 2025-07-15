"""
Contains functions like mass computation.
"""

import numpy as np


def compute_mass(lr, grid):
    K = lr.U @ lr.S
    rho = K @ np.trapezoid(lr.V, dx=grid.dmu, axis=0).T
    M = np.trapezoid(rho, dx=grid.dx, axis=0)
    return M

def computeD_cendiff_2x1d(grid, option = "periodic"):
    """
    Compute centered difference matrices.

    Compute centered difference matrices for 2x1d. Output is DX and DY.
    """
    ### Compute DX
    # Step 1: Set up cen difference matrix
    Dx = np.zeros((grid.Nx, grid.Nx), dtype=int)
    np.fill_diagonal(Dx[1:], -1)
    np.fill_diagonal(Dx[:, 1:], 1)
    if option == "periodic":
        Dx[0, grid.Nx-1] = -1
        Dx[grid.Nx-1, 0] = 1

    I = np.eye(grid.Ny, grid.Ny)

    # Step 2: Use np.kron
    DX = np.kron(I, Dx)

    ### Compute DY
    # Step 1: Set up cen difference matrix
    Dy = np.zeros((grid.Ny, grid.Ny), dtype=int)
    np.fill_diagonal(Dy[1:], -1)
    np.fill_diagonal(Dy[:, 1:], 1)
    if option == "periodic":
        Dy[0, grid.Ny-1] = -1
        Dy[grid.Ny-1, 0] = 1

    I = np.eye(grid.Nx, grid.Nx)

    # Step 2: Use np.kron
    DY = np.kron(Dy, I)

    ### Scale matrices
    DX = 0.5 * DX / grid.dx
    DY = 0.5 * DY / grid.dy

    return DX, DY