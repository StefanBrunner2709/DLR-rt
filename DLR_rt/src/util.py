"""
Contains functions like mass computation.
"""

import numpy as np


def compute_mass(lr, grid):
    K = lr.U @ lr.S
    rho = K @ np.trapezoid(lr.V, dx=grid.dmu, axis=0).T
    M = np.trapezoid(rho, dx=grid.dx, axis=0)
    return M

def computeD_cendiff_2x1d(grid):
    """
    Compute centered difference matrices.

    Compute centered difference matrices for 2x1d. Output is DX and DY.
    """
    ### Compute DX
    # Step 1: Set up cen difference matrix
    Dx = np.zeros((grid.Nx, grid.Nx), dtype=int)
    np.fill_diagonal(Dx[1:], -1)
    np.fill_diagonal(Dx[:, 1:], 1)
    Dx[0, grid.Nx-1] = -1
    Dx[grid.Nx-1, 0] = 1

    # Step 2: Initialize the big zero matrix
    DX = np.zeros((grid.Nx**2, grid.Nx**2))

    # Step 3: Insert Dx along the block diagonal
    for i in range(grid.Nx):
        start = i * grid.Nx
        DX[start:start + grid.Nx, start:start + grid.Nx] = Dx

    ### Compute DY
    # Parameters
    block_size = grid.Ny            # size of each identity block
    num_blocks = grid.Ny            # number of blocks
    matrix_size = block_size**2     # full matrix size

    # Create identity block
    I = np.eye(block_size)

    # Initialize full matrix
    DY = np.zeros((matrix_size, matrix_size))

    # Fill superdiagonal (i = j - 1) with +I
    # Fill subdiagonal (i = j + 1) with -I
    for i in range(num_blocks):
        for j in range(num_blocks):
            if i == j - 1:
                DY[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = I
            elif i == j + 1:
                DY[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = -I

    # Add blocks because of bc
    DY[:block_size, (num_blocks-1)*block_size:num_blocks*block_size] = -I
    DY[(num_blocks-1)*block_size:num_blocks*block_size, :block_size] = I

    ### Scale matrices
    DX = 0.5 * DX / grid.dx
    DY = 0.5 * DY / grid.dy

    return DX, DY