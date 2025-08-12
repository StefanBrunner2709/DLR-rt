"""
Contains functions to set initial condition.
"""

import numpy as np

from DLR_rt.src.grid import Grid_1x1d, Grid_2x1d
from DLR_rt.src.lr import LR


def setInitialCondition_1x1d_full(grid: Grid_1x1d, sigma: float = 1.0) -> np.ndarray:
    """
    Set initial condition.

    Set initial condition for full grid with periodic boundary conditions.
    """
    f0 = np.zeros((grid.Nx, grid.Nmu))
    xx = 1 / (2 * np.pi * sigma**2) * np.exp(-((grid.X - 0.5) ** 2) / (2 * sigma**2))
    vv = np.exp(-(np.abs(grid.MU) ** 2) / (16 * sigma**2))
    f0 = np.outer(xx, vv)
    return f0


def setInitialCondition_1x1d_lr(grid: Grid_1x1d, sigma: float = 1.0):
    """
    Set initial condition.

    Set initial condition for low rank grid with periodic or inflow boundary conditions.
    Boundary conditions are determined according to the boundary conditions in grid.
    """
    S = np.zeros((grid.r, grid.r))

    if grid.option_bc == "inflow":
        U = np.random.rand(grid.Nx, grid.r)
        V = np.random.rand(grid.Nmu, grid.r)
    elif grid.option_bc == "periodic":
        U = np.zeros((grid.Nx, grid.r))
        V = np.zeros((grid.Nmu, grid.r))
        U[:, 0] = (
            1 / (2 * np.pi * sigma**2) * np.exp(-((grid.X - 0.5) ** 2) / (2 * sigma**2))
        )
        V[:, 0] = np.exp(-(np.abs(grid.MU) ** 2) / (16 * sigma**2))
        S[0, 0] = 1.0

    U_ortho, R_U = np.linalg.qr(U, mode="reduced")
    V_ortho, R_V = np.linalg.qr(V, mode="reduced")
    S_ortho = R_U @ S @ R_V.T

    lr = LR(U_ortho, S_ortho, V_ortho)
    return lr


def setInitialCondition_2x1d_lr(grid: Grid_2x1d):
    """
    Set initial condition.

    Set initial condition for 2x1d low rank grid with or without domain decomposition 
    and periodic boundary conditions.
    """
    S = np.zeros((grid.r, grid.r))
    U = np.zeros((grid.Nx * grid.Ny, grid.r))
    V = np.zeros((grid.Nphi, grid.r))
    for i in range(grid.Ny):
        U[i * grid.Nx : (i + 1) * grid.Nx, 0] = (
            1
            / (2 * np.pi)
            * np.exp(-((grid.X - 0.5) ** 2) / 0.07)
            * np.exp(-((grid.Y[i] - 0.5) ** 2) / 0.07)
        )
        # U[i*grid.Nx:(i+1)*grid.Nx, 0] = (
        #     np.sin(2*np.pi*grid.X)*np.sin(2*np.pi*grid.Y[i])
        # )
    V[4, 0] = 1.0
    S[0, 0] = 1.0

    U_ortho, R_U = np.linalg.qr(U, mode="reduced")
    V_ortho, R_V = np.linalg.qr(V, mode="reduced")
    S_ortho = R_U @ S @ R_V.T

    lr = LR(U_ortho, S_ortho, V_ortho)
    return lr
