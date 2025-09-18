"""
Contains classes and functions to set up low rank structure.
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


def computeF_b(t: float, f, grid, f_left=None, f_right=None):
    """
    Generate discretization of inflow/outflow boundary.

    Generate the discretization of the full boundary, 
    depending on position of the current subdomain.
    If only one domain is present, we don't need to declare f_left or f_right.
    If the subdomain is the leftmost domain, declare f_right 
    (Values for domain on the right).
    If the subdomain is the rightmost domain, declare f_left 
    (Values for domain on the left).
    If the subdomain is between two other subdomains, declare both.

    Parameters
    ----------
    t : float
        Current time.
    f
        Values of subdomain, given as matrix.
    grid
        Grid class of subdomain.
    f_left
        Values of subdomain on left side, given as matrix.
    f_right
        Values of subdomain on right side, given as matrix.
    """

    F_b = np.zeros((2, len(grid.MU)))

    if f_left is not None and f_right is not None:
        for i in range(len(grid.MU)):  # Middle domain
            if grid.MU[i] > 0:
                F_b[0, i] = f_left[-1, i]  # Here we use inflow from domain on the left
                F_b[1, i] = f[grid.Nx - 1, i] + (f[grid.Nx - 1, i] - f[grid.Nx - 2, i])
            elif grid.MU[i] < 0:
                F_b[1, i] = f_right[0, i]  # Here we use inflow from domain on the right
                F_b[0, i] = f[0, i] - (f[1, i] - f[0, i])
    elif f_left is not None:  # Left sided domain
        for i in range(len(grid.MU)):
            if grid.MU[i] > 0:
                F_b[0, i] = f_left[-1, i]  # Here we use inflow from domain on the left
                F_b[1, i] = f[grid.Nx - 1, i] + (f[grid.Nx - 1, i] - f[grid.Nx - 2, i])
            elif grid.MU[i] < 0:
                F_b[1, i] = np.tanh(t)
                F_b[0, i] = f[0, i] - (f[1, i] - f[0, i])
    elif f_right is not None:  # Right sided domain
        for i in range(len(grid.MU)):
            if grid.MU[i] > 0:
                F_b[0, i] = np.tanh(t)
                F_b[1, i] = f[grid.Nx - 1, i] + (f[grid.Nx - 1, i] - f[grid.Nx - 2, i])
            elif grid.MU[i] < 0:
                F_b[1, i] = f_right[0, i]  # Here we use inflow from domain on the right
                F_b[0, i] = f[0, i] - (f[1, i] - f[0, i])
    else:  # Only one domain
        for i in range(len(grid.MU)):
            if grid.MU[i] > 0:
                F_b[0, i] = np.tanh(t)
                F_b[1, i] = f[grid.Nx - 1, i] + (f[grid.Nx - 1, i] - f[grid.Nx - 2, i])
            elif grid.MU[i] < 0:
                F_b[1, i] = np.tanh(t)
                F_b[0, i] = f[0, i] - (f[1, i] - f[0, i])

    return F_b


def computeF_b_2x1d_X(f, grid, f_left=None, f_right=None, f_periodic=None, 
                      grid_left=None, grid_right=None, option_bc="standard"):
    """
    Generate discretization of 2x1d full X boundary.

    Generate the discretization of the full boundary for the X grid, 
    depending on position of the current subdomain.
    If only one domain is present, 
    we don't need to declare f_left, f_right or f_periodic.
    If the subdomain is the leftmost domain, declare f_right 
    (Values for domain on the right) and f_periodic (values for rightmost domain).
    If the subdomain is the rightmost domain, declare f_left 
    (Values for domain on the left)  and f_periodic (values for leftmost domain).
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

    if grid_left is None:
        grid_left = grid

    if grid_right is None:
        grid_right = grid

    F_b_X = np.zeros((2 * len(grid.Y), len(grid.PHI)))

    if f_periodic is not None:  # left or right sided domain
        if f_right is not None:  # left sided domain
            for i in range(len(grid.PHI)):
                if grid.PHI[i] < np.pi / 2 or grid.PHI[i] > 3 / 2 * np.pi:
                    indices_left = list(
                        range(grid.Nx - 1, grid.Nx * (grid.Ny + 1) - 1, grid.Nx)
                    )  # pick every Nxth row
                    F_b_X[: len(grid.Y), i] = f_periodic[
                        indices_left, i
                    ]  # This is inflow from left

                    indices_outflow_1 = list(
                        range(grid.Nx - 1, grid.Nx * (grid.Ny + 1) - 1, grid.Nx)
                    )
                    indices_outflow_2 = list(
                        range(grid.Nx - 2, grid.Nx * (grid.Ny + 1) - 2, grid.Nx)
                    )
                    F_b_X[len(grid.Y) :, i] = f[indices_outflow_1, i] + (
                        f[indices_outflow_1, i] - f[indices_outflow_2, i]
                    )  # outflow to right side

                else:
                    indices_right = list(range(0, grid.Nx * (grid.Ny), grid.Nx))
                    F_b_X[len(grid.Y) :, i] = f_right[
                        indices_right, i
                    ]  # This is inflow from right

                    indices_outflow_0 = list(range(0, grid.Nx * (grid.Ny), grid.Nx))
                    indices_outflow_1 = list(range(1, grid.Nx * (grid.Ny) + 1, grid.Nx))
                    F_b_X[: len(grid.Y), i] = f[indices_outflow_0, i] - (
                        f[indices_outflow_1, i] - f[indices_outflow_0, i]
                    )  # outflow to left side

        elif f_left is not None:  # right sided domain
            for i in range(len(grid.PHI)):
                if grid.PHI[i] < np.pi / 2 or grid.PHI[i] > 3 / 2 * np.pi:
                    indices_left = list(
                        range(grid.Nx - 1, grid.Nx * (grid.Ny + 1) - 1, grid.Nx)
                    )  # pick every Nxth row
                    F_b_X[: len(grid.Y), i] = f_left[
                        indices_left, i
                    ]  # This is inflow from left

                    indices_outflow_1 = list(
                        range(grid.Nx - 1, grid.Nx * (grid.Ny + 1) - 1, grid.Nx)
                    )
                    indices_outflow_2 = list(
                        range(grid.Nx - 2, grid.Nx * (grid.Ny + 1) - 2, grid.Nx)
                    )
                    F_b_X[len(grid.Y) :, i] = f[indices_outflow_1, i] + (
                        f[indices_outflow_1, i] - f[indices_outflow_2, i]
                    )  # outflow to right side

                else:
                    indices_right = list(range(0, grid.Nx * (grid.Ny), grid.Nx))
                    F_b_X[len(grid.Y) :, i] = f_periodic[
                        indices_right, i
                    ]  # This is inflow from right

                    indices_outflow_0 = list(range(0, grid.Nx * (grid.Ny), grid.Nx))
                    indices_outflow_1 = list(range(1, grid.Nx * (grid.Ny) + 1, grid.Nx))
                    F_b_X[: len(grid.Y), i] = f[indices_outflow_0, i] - (
                        f[indices_outflow_1, i] - f[indices_outflow_0, i]
                    )  # outflow to left side

    elif f_left is not None and f_right is not None:  # middle domain
        for i in range(len(grid.PHI)):
                if grid.PHI[i] < np.pi / 2 or grid.PHI[i] > 3 / 2 * np.pi:
                    indices_left = list(
                        range(grid_left.Nx - 1, grid_left.Nx * (grid_left.Ny + 1) - 1, 
                              grid_left.Nx)
                    )  # pick every Nxth row
                    F_b_X[: len(grid.Y), i] = f_left[
                        indices_left, i
                    ]  # This is inflow from left

                    indices_outflow_1 = list(
                        range(grid.Nx - 1, grid.Nx * (grid.Ny + 1) - 1, grid.Nx)
                    )
                    indices_outflow_2 = list(
                        range(grid.Nx - 2, grid.Nx * (grid.Ny + 1) - 2, grid.Nx)
                    )
                    F_b_X[len(grid.Y) :, i] = f[indices_outflow_1, i] + (
                        f[indices_outflow_1, i] - f[indices_outflow_2, i]
                    )  # outflow to right side

                else:
                    indices_right = list(range(0, grid_right.Nx * (grid_right.Ny), 
                                               grid_right.Nx))
                    F_b_X[len(grid.Y) :, i] = f_right[
                        indices_right, i
                    ]  # This is inflow from right

                    indices_outflow_0 = list(range(0, grid.Nx * (grid.Ny), grid.Nx))
                    indices_outflow_1 = list(range(1, grid.Nx * (grid.Ny) + 1, grid.Nx))
                    F_b_X[: len(grid.Y), i] = f[indices_outflow_0, i] - (
                        f[indices_outflow_1, i] - f[indices_outflow_0, i]
                    )  # outflow to left side

    else:  # only one domain
        for i in range(len(grid.PHI)):
            if grid.PHI[i] < np.pi / 2 or grid.PHI[i] > 3 / 2 * np.pi:

                if option_bc == "standard":
                    indices_left = list(
                        range(grid.Nx - 1, grid.Nx * (grid.Ny + 1) - 1, grid.Nx)
                    )  # pick every Nxth row
                    F_b_X[: len(grid.Y), i] = f[
                        indices_left, i
                    ]  # This is inflow from left

                elif option_bc == "lattice":
                    F_b_X[: len(grid.Y), i] = 0  # This is inflow from left

                elif option_bc == "hohlraum":
                    F_b_X[: len(grid.Y), i] = 1  # inflow = 1


                indices_outflow_1 = list(
                    range(grid.Nx - 1, grid.Nx * (grid.Ny + 1) - 1, grid.Nx)
                )
                indices_outflow_2 = list(
                    range(grid.Nx - 2, grid.Nx * (grid.Ny + 1) - 2, grid.Nx)
                )
                F_b_X[len(grid.Y) :, i] = f[indices_outflow_1, i] + (
                    f[indices_outflow_1, i] - f[indices_outflow_2, i]
                )  # outflow to right side

            else:

                if option_bc == "standard":
                    indices_right = list(range(0, grid.Nx * (grid.Ny), grid.Nx))
                    F_b_X[len(grid.Y) :, i] = f[
                        indices_right, i
                    ]  # This is inflow from right

                elif option_bc == "lattice" or option_bc == "hohlraum":
                    F_b_X[len(grid.Y) :, i] = 0  # This is inflow from right


                indices_outflow_0 = list(range(0, grid.Nx * (grid.Ny), grid.Nx))
                indices_outflow_1 = list(range(1, grid.Nx * (grid.Ny) + 1, grid.Nx))
                F_b_X[: len(grid.Y), i] = f[indices_outflow_0, i] - (
                    f[indices_outflow_1, i] - f[indices_outflow_0, i]
                )  # outflow to left side

    return F_b_X


def computeF_b_2x1d_Y(f, grid, f_bottom=None, f_top=None, f_periodic=None, 
                      grid_bottom=None, grid_top=None, option_bc="standard"):
    """
    Generate discretization of 2x1d full Y boundary.

    Generate the discretization of the full boundary for the Y grid, 
    depending on position of the current subdomain.
    If only one domain is present, 
    we don't need to declare f_bottom, f_top or f_periodic
    If the subdomain is the lowest domain, declare f_top 
    (Values for domain on the top) and f_periodic (values for highest domain).
    If the subdomain is the highest domain, declare f_bottom 
    (Values for domain on the bottom)  and f_periodic (values for lowest domain).
    If the subdomain is between two other subdomains, declare f_bottom and f_top.

    Parameters
    ----------
    f
        Values of subdomain, given as matrix.
    grid
        Grid class of subdomain.
    f_bottom
        Values of subdomain on bottom, given as matrix.
    f_top
        Values of subdomain on top, given as matrix.
    f_periodic
        Values of subdomain on other side of periodic boundary, given as matrix.
    """

    if grid_bottom is None:
        grid_bottom = grid

    if grid_top is None:
        grid_top = grid

    F_b_Y = np.zeros((2 * len(grid.X), len(grid.PHI)))

    if f_periodic is not None:  # lowest or highest domain
        if f_top is not None:  # lowest domain
            for i in range(len(grid.PHI)):
                if grid.PHI[i] < np.pi:
                    F_b_Y[: len(grid.X), i] = f_periodic[
                        grid.Nx * (grid.Ny - 1) : grid.Nx * grid.Ny, i
                    ]  # This is inflow from bottom

                    F_b_Y[len(grid.X) :, i] = f[
                        grid.Nx * (grid.Ny - 1) : grid.Nx * grid.Ny, i
                    ] + (
                        f[grid.Nx * (grid.Ny - 1) : grid.Nx * grid.Ny, i]
                        - f[grid.Nx * (grid.Ny - 2) : grid.Nx * (grid.Ny - 1), i]
                    )  # outflow to top
                else:
                    F_b_Y[len(grid.X) :, i] = f_top[
                        : grid.Nx, i
                    ]  # This is inflow from top

                    F_b_Y[: len(grid.X), i] = f[: grid.Nx, i] - (
                        f[grid.Nx : grid.Nx * 2, i] - f[: grid.Nx, i]
                    )  # outflow to bottom

        elif f_bottom is not None:  # highest domain
            for i in range(len(grid.PHI)):
                if grid.PHI[i] < np.pi:
                    F_b_Y[: len(grid.X), i] = f_bottom[
                        grid.Nx * (grid.Ny - 1) : grid.Nx * grid.Ny, i
                    ]  # This is inflow from bottom

                    F_b_Y[len(grid.X) :, i] = f[
                        grid.Nx * (grid.Ny - 1) : grid.Nx * grid.Ny, i
                    ] + (
                        f[grid.Nx * (grid.Ny - 1) : grid.Nx * grid.Ny, i]
                        - f[grid.Nx * (grid.Ny - 2) : grid.Nx * (grid.Ny - 1), i]
                    )  # outflow to top

                else:
                    F_b_Y[len(grid.X) :, i] = f_periodic[
                        : grid.Nx, i
                    ]  # This is inflow from top

                    F_b_Y[: len(grid.X), i] = f[: grid.Nx, i] - (
                        f[grid.Nx : grid.Nx * 2, i] - f[: grid.Nx, i]
                    )  # outflow to bottom

    elif f_bottom is not None and f_top is not None:  # middle domain
        for i in range(len(grid.PHI)):
                if grid.PHI[i] < np.pi:
                    F_b_Y[: len(grid.X), i] = f_bottom[
                        grid_bottom.Nx * (grid_bottom.Ny - 1) : 
                        grid_bottom.Nx * grid_bottom.Ny, i
                    ]  # This is inflow from bottom

                    F_b_Y[len(grid.X) :, i] = f[
                        grid.Nx * (grid.Ny - 1) : grid.Nx * grid.Ny, i
                    ] + (
                        f[grid.Nx * (grid.Ny - 1) : grid.Nx * grid.Ny, i]
                        - f[grid.Nx * (grid.Ny - 2) : grid.Nx * (grid.Ny - 1), i]
                    )  # outflow to top

                else:
                    F_b_Y[len(grid.X) :, i] = f_top[
                        : grid_top.Nx, i
                    ]  # This is inflow from top

                    F_b_Y[: len(grid.X), i] = f[: grid.Nx, i] - (
                        f[grid.Nx : grid.Nx * 2, i] - f[: grid.Nx, i]
                    )  # outflow to bottom

    else:  # only one domain
        for i in range(len(grid.PHI)):
            if grid.PHI[i] < np.pi:

                if option_bc == "standard":
                    F_b_Y[: len(grid.X), i] = f[
                        grid.Nx * (grid.Ny - 1) : grid.Nx * grid.Ny, i
                    ]  # This is inflow from bottom

                elif option_bc == "lattice" or option_bc == "hohlraum":
                    F_b_Y[: len(grid.X), i] = 0  # This is inflow from bottom

                F_b_Y[len(grid.X) :, i] = f[
                    grid.Nx * (grid.Ny - 1) : grid.Nx * grid.Ny, i
                ] + (
                    f[grid.Nx * (grid.Ny - 1) : grid.Nx * grid.Ny, i]
                    - f[grid.Nx * (grid.Ny - 2) : grid.Nx * (grid.Ny - 1), i]
                )  # outflow to top
            else:

                if option_bc == "standard":
                    F_b_Y[len(grid.X) :, i] = f[: grid.Nx, i]  # This is inflow from top

                elif option_bc == "lattice" or option_bc == "hohlraum":
                    F_b_Y[len(grid.X) :, i] = 0  # This is inflow from top

                F_b_Y[: len(grid.X), i] = f[: grid.Nx, i] - (
                    f[grid.Nx : grid.Nx * 2, i] - f[: grid.Nx, i]
                )  # outflow to bottom

    return F_b_Y


def computeK_bdry(lr, grid, F_b):
    """
    Compute boundary values for K.

    Transforms the boundary information given by F_b 
    (discretization of inflow/outflow function) into a boundary information in K.
    """

    e_vec_left = np.zeros([len(grid.MU)])
    e_vec_right = np.zeros([len(grid.MU)])

    # Values from boundary condition:
    for i in range(len(grid.MU)):  # compute e-vector
        if grid.MU[i] > 0:
            e_vec_left[i] = F_b[0, i]
        elif grid.MU[i] < 0:
            e_vec_right[i] = F_b[1, i]

    int_exp_left = (
        (e_vec_left @ lr.V) * grid.dmu
    )  # compute integral from inflow, contains information from inflow from every K_j
    int_exp_right = (e_vec_right @ lr.V) * grid.dmu

    K = lr.U @ lr.S
    K_extrapol_left = np.zeros([grid.r])
    K_extrapol_right = np.zeros([grid.r])
    for i in range(grid.r):  # calculate extrapolated values
        K_extrapol_left[i] = K[0, i] - (K[1, i] - K[0, i])
        K_extrapol_right[i] = K[grid.Nx - 1, i] + (
            K[grid.Nx - 1, i] - K[grid.Nx - 2, i]
        )

    V_indicator_left = np.copy(
        lr.V
    )  # generate V*indicator, Note: Only works for Nx even
    V_indicator_left[int(grid.Nmu / 2) :, :] = 0
    V_indicator_right = np.copy(lr.V)
    V_indicator_right[: int(grid.Nmu / 2), :] = 0

    int_V_left = (V_indicator_left.T @ lr.V) * grid.dmu  # compute integrals over V
    int_V_right = (V_indicator_right.T @ lr.V) * grid.dmu

    sum_vector_left = (
        K_extrapol_left @ int_V_left
    )  # compute vector of size r with all the sums inside
    sum_vector_right = K_extrapol_right @ int_V_right

    K_bdry_left = (
        int_exp_left + sum_vector_left
    )  # add all together to get boundary info (vector with info for 1<=j<=r)
    K_bdry_right = int_exp_right + sum_vector_right

    return K_bdry_left, K_bdry_right


def computeK_bdry_2x1d_X(lr, grid, F_b_X):
    """
    Compute X grid boundary values for K in 2x1d.

    Transforms the boundary information given by F_b_X into a boundary information in K.
    """
    # --- Precompute masks for PHI direction ---
    mask_left = (np.pi / 2 > grid.PHI) | (3 * np.pi / 2 < grid.PHI)
    mask_right = ~mask_left  # opposite of left

    # --- Fill e_mats using masks (vectorized) ---
    e_mat_left = np.zeros((grid.Ny, grid.Nphi))
    e_mat_right = np.zeros((grid.Ny, grid.Nphi))

    e_mat_left[:, mask_left] = F_b_X[:grid.Ny, mask_left]
    e_mat_right[:, mask_right] = F_b_X[grid.Ny:, mask_right]

    int_exp_left = (
        (e_mat_left @ lr.V) * grid.dphi
    )  # compute integral from inflow, contains information from inflow from every K_j
    int_exp_right = (e_mat_right @ lr.V) * grid.dphi  # now matrix of dimension Ny x r

    K = lr.U @ lr.S

    # --- Precompute indices once ---
    idx_outflow_left_0 = np.arange(0, grid.Nx * grid.Ny, grid.Nx)
    idx_outflow_left_1 = idx_outflow_left_0 + 1
    idx_outflow_right_1 = np.arange(grid.Nx - 1, grid.Nx * (grid.Ny + 1) - 1, grid.Nx)
    idx_outflow_right_2 = idx_outflow_right_1 - 1

    # --- Vectorized extrapolation for extrapolated values ---
    K_extrapol_left = 2 * K[idx_outflow_left_0] - K[idx_outflow_left_1]
    K_extrapol_right = 2 * K[idx_outflow_right_1] - K[idx_outflow_right_2]

    # compute V_int
    V_indicator_left = np.copy(
        lr.V
    )  # generate V*indicator, Note: Only works for Nx even
    V_indicator_left[: int(grid.Nphi / 4), :] = 0
    V_indicator_left[int(3 * grid.Nphi / 4) :, :] = 0
    V_indicator_right = np.copy(lr.V)
    V_indicator_right[int(grid.Nphi / 4) : int(3 * grid.Nphi / 4), :] = 0

    int_V_left = (V_indicator_left.T @ lr.V) * grid.dphi  # compute integrals over V
    int_V_right = (V_indicator_right.T @ lr.V) * grid.dphi

    sum_vector_left = (
        K_extrapol_left @ int_V_left
    )  # compute matrix of size Ny x r with all the sums inside
    sum_vector_right = K_extrapol_right @ int_V_right

    K_bdry_left = (
        int_exp_left + sum_vector_left
    )  # add all together to get boundary info (matrix with info for 1<=j<=r)
    K_bdry_right = int_exp_right + sum_vector_right

    return K_bdry_left, K_bdry_right


def computeK_bdry_2x1d_Y(lr, grid, F_b_Y):
    """
    Compute Y grid boundary values for K in 2x1d.

    Transforms the boundary information given by F_b_Y into a boundary information in K.
    """
    # --- Precompute masks for PHI direction ---
    mask_bottom = (np.pi > grid.PHI)
    mask_top = ~mask_bottom  # opposite of bpttom

    # --- Fill e_mats using masks (vectorized) ---
    e_mat_bottom = np.zeros((grid.Nx, grid.Nphi))
    e_mat_top = np.zeros((grid.Nx, grid.Nphi))

    e_mat_bottom[:, mask_bottom] = F_b_Y[:grid.Nx, mask_bottom]
    e_mat_top[:, mask_top] = F_b_Y[grid.Nx:, mask_top]

    int_exp_bottom = (
        (e_mat_bottom @ lr.V) * grid.dphi
    )  # compute integral from inflow, contains information from inflow from every K_j
    int_exp_top = (e_mat_top @ lr.V) * grid.dphi  # now matrix of dimension Nx x r

    K = lr.U @ lr.S

    # --- Vectorized extrapolation ---
    K_extrapol_bottom = 2 * K[:grid.Nx, :] - K[grid.Nx : 2*grid.Nx, :]
    K_extrapol_top = (2 * K[grid.Nx*(grid.Ny-1) : grid.Nx*grid.Ny, :] - 
                      K[grid.Nx*(grid.Ny-2) : grid.Nx*(grid.Ny-1), :])

    # compute V_int
    V_indicator_bottom = np.copy(
        lr.V
    )  # generate V*indicator, Note: Only works for Ny even
    V_indicator_bottom[: int(grid.Nphi / 2), :] = 0
    V_indicator_top = np.copy(lr.V)
    V_indicator_top[int(grid.Nphi / 2) :, :] = 0

    int_V_bottom = (V_indicator_bottom.T @ lr.V) * grid.dphi  # compute integrals over V
    int_V_top = (V_indicator_top.T @ lr.V) * grid.dphi

    sum_vector_bottom = (
        K_extrapol_bottom @ int_V_bottom
    )  # compute matrix of size Nx x r with all the sums inside
    sum_vector_top = K_extrapol_top @ int_V_top

    K_bdry_bottom = (
        int_exp_bottom + sum_vector_bottom
    )  # add all together to get boundary info (matrix with info for 1<=j<=r)
    K_bdry_top = int_exp_top + sum_vector_top

    return K_bdry_bottom, K_bdry_top


def computedxK(lr, K_bdry_left, K_bdry_right, grid):
    """
    Compute the derivative of K.

    Compute the first derivative of K in x using a centered difference stencil.
    """
    K = lr.U @ lr.S

    Dx = np.zeros((grid.Nx, grid.Nx), dtype=int)
    np.fill_diagonal(Dx[1:], -1)
    np.fill_diagonal(Dx[:, 1:], 1)

    dxK = Dx @ K / (2 * grid.dx)

    dxK[0, :] -= K_bdry_left / (2 * grid.dx)
    dxK[-1, :] += K_bdry_right / (2 * grid.dx)

    return dxK


def computedxK_2x1d(
    lr, K_bdry_left, K_bdry_right, K_bdry_bottom, K_bdry_top, grid, DX, DY
):
    """
    Compute the gradient of K.

    Compute the first derivative of K in x and y using a centered difference stencil.
    """
    K = lr.U @ lr.S

    DXK = DX @ K

    # Still need to add boundary information
    indices_1 = list(range(0, grid.Nx * (grid.Ny), grid.Nx))
    DXK[indices_1, :] = DXK[indices_1, :] - K_bdry_left / (2 * grid.dx)
    indices_2 = list(range(grid.Nx - 1, grid.Nx * (grid.Ny + 1) - 1, grid.Nx))
    DXK[indices_2, :] = DXK[indices_2, :] + K_bdry_right / (2 * grid.dx)

    DYK = DY @ K

    # Still need to add boundary information
    DYK[: grid.Nx, :] = DYK[: grid.Nx, :] - K_bdry_bottom / (2 * grid.dy)
    DYK[grid.Nx * (grid.Ny - 1) : grid.Nx * grid.Ny, :] = DYK[
        grid.Nx * (grid.Ny - 1) : grid.Nx * grid.Ny, :
    ] + K_bdry_top / (2 * grid.dy)

    return DXK, DYK

def computedxK_2x1d_upwind(
    lr, K_bdry_left, K_bdry_right, K_bdry_bottom, K_bdry_top, grid, 
    DX_0, DX_1, DY_0, DY_1, eigvals_0, P_0, eigvals_1, P_1
):
    """
    Compute the gradient of K.

    Compute the first derivative of K in x and y using an upwind stencil.
    """
    K = lr.U @ lr.S

    ### Compute DX

    # Project into eigenbasis
    KP_0 = K @ P_0

    # Set up upwind matrices without boundary information
    DXK_0 = DX_0 @ KP_0
    DXK_1 = DX_1 @ KP_0

    # Add boundary information to those matrices
    indices_1 = list(range(0, grid.Nx * (grid.Ny), grid.Nx))
    DXK_0[indices_1, :] = DXK_0[indices_1, :] - (K_bdry_left @ P_0) / (grid.dx)
    indices_2 = list(range(grid.Nx - 1, grid.Nx * (grid.Ny + 1) - 1, grid.Nx))
    DXK_1[indices_2, :] = DXK_1[indices_2, :] + (K_bdry_right @ P_0) / (grid.dx)

    # calculate DXK
    DXK = np.zeros((grid.Nx*grid.Ny,grid.r))
    for i in range(grid.r):
        if eigvals_0[i] > 0:
            DXK[:,i] = DXK_0[:,i]
        else:
            DXK[:,i] = DXK_1[:,i]

    ### Compute DYK

    # Project into eigenbasis
    KP_1 = K @ P_1

    # Set up upwind matrices without boundary information
    DYK_0 = DY_0 @ KP_1
    DYK_1 = DY_1 @ KP_1

    # Add boundary information to those matrices
    DYK_0[: grid.Nx, :] = DYK_0[: grid.Nx, :] - (K_bdry_bottom @ P_1) / (grid.dy)
    DYK_1[grid.Nx * (grid.Ny - 1) : grid.Nx * grid.Ny, :] = DYK_1[
        grid.Nx * (grid.Ny - 1) : grid.Nx * grid.Ny, :
    ] + (K_bdry_top @ P_1) / (grid.dy)    

    # calculate DYK
    DYK = np.zeros((grid.Nx*grid.Ny,grid.r))
    for i in range(grid.r):
        if eigvals_1[i] > 0:
            DYK[:,i] = DYK_0[:,i]
        else:
            DYK[:,i] = DYK_1[:,i]

    return DXK, DYK


def computeC(lr, grid, dimensions="1x1d"):
    """
    Compute C coefficient.

    For higher dimensional simulations set i.e. dimensions = "2x1d"
    """
    if dimensions == "1x1d":
        C1 = (lr.V.T @ np.diag(grid.MU) @ lr.V) * grid.dmu

        C2 = (lr.V.T @ np.ones((grid.Nmu, 1))).T * grid.dmu

        ### Alternative option, faster but harder to understand
        # muV = grid.MU[:, None] * lr.V
        # C1 = lr.V.T @ muV * grid.dmu
        # C2 = lr.V * grid.dmu

    elif dimensions == "2x1d":
        C1_1 = (lr.V.T @ np.diag(np.cos(grid.PHI)) @ lr.V) * grid.dphi
        C1_2 = (lr.V.T @ np.diag(np.sin(grid.PHI)) @ lr.V) * grid.dphi
        C1 = [C1_1, C1_2]

        C2 = (lr.V.T @ np.ones((grid.Nphi, 1))).T * grid.dphi
    return C1, C2


def computeB(L, grid, dimensions="1x1d"):
    """
    Compute B coeffiecient.

    For higher dimensional simulations set i.e. dimensions = "2x1d"
    """
    if dimensions == "1x1d":
        B1 = (L.T @ np.ones((grid.Nmu, 1))).T * grid.dmu

    elif dimensions == "2x1d":
        B1 = (L.T @ np.ones((grid.Nphi, 1))).T * grid.dphi

    return B1


def computeD(
    lr,
    grid,
    F_b=None,
    F_b_Y=None,
    DX=None,
    DY=None,
    dimensions="1x1d",
    option_dd="no_dd",
    option_coeff="constant"
):
    """
    Compute D coeffiecient.

    For 1x1d simulations, to compute D coefficient for periodic simulations, 
    leave the standard value F_b = None.
    For 1x1d simulations, to compute D coefficient for inflow simulations, set F_b.
    For higher dimensional simulations set i.e. dimensions = "2x1d".
    For higher dimensional simulation with domain decomp set option_dd = "dd" and 
    set F_b(F_b_X) and F_b_Y.
    """
    if dimensions == "1x1d":
        if F_b is not None:
            K_bdry_left, K_bdry_right = computeK_bdry(lr, grid, F_b)
            dxK = computedxK(lr, K_bdry_left, K_bdry_right, grid)
            D1 = lr.U.T @ dxK * grid.dx

        else:
            dxU = 0.5 * (np.roll(lr.U, -1, axis=0) - np.roll(lr.U, 1, axis=0)) / grid.dx
            D1 = lr.U.T @ dxU * grid.dx

    elif dimensions == "2x1d":
        if option_dd == "no_dd":
            if option_coeff == "constant":
                D1X = lr.U.T @ DX @ lr.U * grid.dx * grid.dy
                D1Y = lr.U.T @ DY @ lr.U * grid.dy * grid.dx
                D1 = [D1X, D1Y]
            
            elif option_coeff == "space_dep":
                D1X = lr.U.T @ grid.coeff[0] @ DX @ lr.U * grid.dx * grid.dy
                D1Y = lr.U.T @ grid.coeff[0] @ DY @ lr.U * grid.dy * grid.dx
                D1 = [D1X, D1Y]

        elif option_dd == "dd":
            K_bdry_left, K_bdry_right = computeK_bdry_2x1d_X(lr, grid, F_b)
            K_bdry_bottom, K_bdry_top = computeK_bdry_2x1d_Y(lr, grid, F_b_Y)
            DXK, DYK = computedxK_2x1d(
                lr, K_bdry_left, K_bdry_right, K_bdry_bottom, K_bdry_top, grid, DX, DY
            )
            if option_coeff == "constant":
                D1X = lr.U.T @ DXK * grid.dx * grid.dy

                D1Y = lr.U.T @ DYK * grid.dy * grid.dx

                D1 = [D1X, D1Y]
            
            elif option_coeff == "space_dep":
                D1X = lr.U.T @ grid.coeff[0] @ DXK * grid.dx * grid.dy

                D1Y = lr.U.T @ grid.coeff[0] @ DYK * grid.dy * grid.dx

                D1 = [D1X, D1Y]

    return D1

def computeE(lr, grid):

    E1_1 = lr.U.T @ grid.coeff[1] @ lr.U * grid.dx * grid.dy
    E1_2 = lr.U.T @ grid.coeff[2] @ lr.U * grid.dx * grid.dy
    E1 = [E1_1, E1_2]

    return E1


def Kstep(
    K,
    C1,
    C2,
    grid,
    lr=None,
    F_b=None,
    DX=None,
    DY=None,
    inflow=False,
    dimensions="1x1d",
    option_coeff="constant",
    source=None,
    option_scheme="cendiff",
    DX_0=None,
    DX_1=None,
    DY_0=None,
    DY_1=None
):
    """
    K step of radiative transfer equation.

    For K step for periodic simulations, leave the standard values inflow = False.
    For K step for inflow simulations, set inflow = True.
    For higher dimensional simulations set i.e. dimensions = "2x1d"
    """
    if dimensions == "1x1d":
        if inflow:
            K_bdry_left, K_bdry_right = computeK_bdry(lr, grid, F_b)
            dxK = computedxK(lr, K_bdry_left, K_bdry_right, grid)
            rhs = (
                -(grid.coeff) * dxK @ C1
                + 0.5 * (grid.coeff) ** 2 * K @ C2.T @ C2
                - (grid.coeff) ** 2 * K
            )

        else:
            dxK = 0.5 * (np.roll(K, -1, axis=0) - np.roll(K, 1, axis=0)) / grid.dx
            rhs = (
                -(grid.coeff) * dxK @ C1
                + 0.5 * (grid.coeff) ** 2 * K @ C2.T @ C2
                - (grid.coeff) ** 2 * K
            )

    elif dimensions == "2x1d":
        if option_scheme=="cendiff":

            if option_coeff == "constant":
                rhs = (
                    -(grid.coeff[0]) * DX @ K @ C1[0]
                    - (grid.coeff[0]) * DY @ K @ C1[1]
                    + 0.5 / (np.pi) * (grid.coeff[1]) * K @ C2.T @ C2
                    - (grid.coeff[2]) * K
                )

            elif option_coeff == "space_dep":

                rhs = (
                    -grid.coeff[0] @ DX @ K @ C1[0]
                    - grid.coeff[0] @ DY @ K @ C1[1]
                    + 0.5 / (np.pi) * grid.coeff[1] @ K @ C2.T @ C2
                    - grid.coeff[2] @ K
                    + 0.5 / (np.pi) * source @ C2
                )
        
        elif option_scheme=="upwind":
            ### Diagonalize matrix C
            # Eigen-decomposition
            eigvals_0, P_0 = np.linalg.eigh(C1[0])  # matrix C1 is symmetric
            eigvals_1, P_1 = np.linalg.eigh(C1[1])  

            # Construct diagonal matrix of eigenvalues
            T1_0 = np.diag(eigvals_0)
            T1_1 = np.diag(eigvals_1)

            ### Obtain matrices from upwind
            # Project into eigenbasis
            KP_0 = K @ P_0
            KP_1 = K @ P_1

            # Set up upwind matrices
            DXK_0 = DX_0 @ KP_0
            DXK_1 = DX_1 @ KP_0
            DYK_0 = DY_0 @ KP_1
            DYK_1 = DY_1 @ KP_1

            # calculate DXK and DYK
            DXK = np.zeros((grid.Nx*grid.Ny,grid.r))
            DYK = np.zeros((grid.Nx*grid.Ny,grid.r))
            for i in range(grid.r):

                if eigvals_0[i] > 0:
                    DXK[:,i] = DXK_0[:,i]
                else:
                    DXK[:,i] = DXK_1[:,i]

                if eigvals_1[i] > 0:
                    DYK[:,i] = DYK_0[:,i]
                else:
                    DYK[:,i] = DYK_1[:,i]

            if option_coeff == "constant":
                rhs = (
                    -(grid.coeff[0]) * DXK @ T1_0 @ P_0.T   # P is orthogonal
                    - (grid.coeff[0]) * DYK @ T1_1 @ P_1.T
                    + 0.5 / (np.pi) * (grid.coeff[1]) * K @ C2.T @ C2
                    - (grid.coeff[2]) * K
                )

            elif option_coeff == "space_dep":

                rhs = (
                    -grid.coeff[0] @ DXK @ T1_0 @ P_0.T
                    - grid.coeff[0] @ DYK @ T1_1 @ P_1.T
                    + 0.5 / (np.pi) * grid.coeff[1] @ K @ C2.T @ C2
                    - grid.coeff[2] @ K
                    + 0.5 / (np.pi) * source @ C2
                )

    return rhs


def Sstep(S, C1, C2, D1, grid, inflow=False, 
          dimensions="1x1d", option_coeff="constant", E1=None, source=None, lr=None):
    """
    S step of radiative transfer equation.

    For S step for periodic simulations, leave the standard values inflow = False.
    For S step for inflow simulations, set inflow = True.
    For higher dimensional simulations set i.e. dimensions = "2x1d"
    """
    if dimensions == "1x1d":
        if not inflow:
            rhs = (
                (grid.coeff) * D1 @ S @ C1
                - 0.5 * (grid.coeff) ** 2 * S @ C2.T @ C2
                + (grid.coeff) ** 2 * S
            )

        elif inflow:
            rhs = (
                (grid.coeff) * D1 @ C1
                - 0.5 * (grid.coeff) ** 2 * S @ C2.T @ C2
                + (grid.coeff) ** 2 * S
            )

    elif dimensions == "2x1d":
        if option_coeff == "constant":
            rhs = (
                (grid.coeff[0]) * D1[0] @ S @ C1[0]
                + (grid.coeff[0]) * D1[1] @ S @ C1[1]
                - 0.5 / (np.pi) * (grid.coeff[1]) * S @ C2.T @ C2
                + (grid.coeff[2]) * S
            )

        elif option_coeff == "space_dep":

            rhs = (
                D1[0] @ S @ C1[0]
                + D1[1] @ S @ C1[1]
                - 0.5 / (np.pi) * E1[0] @ S @ C2.T @ C2
                + E1[1] @ S
                - 0.5 / (np.pi) * lr.U.T @ source @ C2 * grid.dx * grid.dy
            )

    return rhs


def Lstep(L, D1, B1, grid, lr=None, inflow=False, 
          dimensions="1x1d", option_coeff="constant", E1=None, source=None):
    """
    L step of radiative transfer equation.

    For L step for periodic simulations, leave the standard values inflow = False.
    For L step for inflow simulations, set inflow = True.
    For higher dimensional simulations set i.e. dimensions = "2x1d".
    """
    if dimensions == "1x1d":
        if inflow:
            rhs = (
                -(grid.coeff) * np.diag(grid.MU) @ lr.V @ D1.T
                + 0.5 * (grid.coeff) ** 2 * np.ones((grid.Nmu, 1)) @ B1
                - (grid.coeff) ** 2 * L
            )
        else:
            rhs = (
                -(grid.coeff) * np.diag(grid.MU) @ L @ D1.T
                + 0.5 * (grid.coeff) ** 2 * np.ones((grid.Nmu, 1)) @ B1
                - (grid.coeff) ** 2 * L
            )

    elif dimensions == "2x1d":
        if option_coeff == "constant":
            rhs = (
                -(grid.coeff[0]) * np.diag(np.cos(grid.PHI)) @ L @ D1[0].T
                - (grid.coeff[0]) * np.diag(np.sin(grid.PHI)) @ L @ D1[1].T
                + 0.5 / (np.pi) * (grid.coeff[1]) * np.ones((grid.Nphi, 1)) @ B1
                - (grid.coeff[2]) * L
            )
        
        elif option_coeff == "space_dep":

            M = np.ones((1, grid.Nphi))

            rhs = (
                -np.diag(np.cos(grid.PHI)) @ L @ D1[0].T
                - np.diag(np.sin(grid.PHI)) @ L @ D1[1].T
                + 0.5 / (np.pi) * np.ones((grid.Nphi, 1)) @ B1 @ E1[0]
                - L @ E1[1]
                + 0.5 / (np.pi) * ((lr.U.T @ source) @ M).T * grid.dx * grid.dy
            )

    return rhs


def Kstep1(C1, grid, lr, F_b_X, F_b_Y, DX, DY, option_scheme="cendiff", 
           DX_0=None, DX_1=None, DY_0=None, DY_1=None):
    """
    K step of radiative transfer equation.

    Part of the K step associated to the x advection 
    after splitting the full equation in 2x1d.
    """

    K_bdry_left, K_bdry_right = computeK_bdry_2x1d_X(lr, grid, F_b_X)
    K_bdry_bottom, K_bdry_top = computeK_bdry_2x1d_Y(lr, grid, F_b_Y)

    if option_scheme=="cendiff":
        DXK = computedxK_2x1d(
            lr, K_bdry_left, K_bdry_right, K_bdry_bottom, K_bdry_top, grid, DX, DY
        )[0]
        rhs = -(grid.coeff[0]) * DXK @ C1[0]

    elif option_scheme=="upwind":

        ### Diagonalize matrix C
        # Eigen-decomposition
        eigvals_0, P_0 = np.linalg.eigh(C1[0])  # matrix C1 is symmetric
        eigvals_1, P_1 = np.linalg.eigh(C1[1])  

        # Construct diagonal matrix of eigenvalues
        T1_0 = np.diag(eigvals_0)

        ### Obtain matrix from upwind (multiplied with P_0 already)
        DXK = computedxK_2x1d_upwind(
            lr, K_bdry_left, K_bdry_right, K_bdry_bottom, K_bdry_top, grid, 
            DX_0, DX_1, DY_0, DY_1, eigvals_0, P_0, eigvals_1, P_1
        )[0]

        ### Calculate rhs
        rhs = -(grid.coeff[0]) * DXK @ T1_0 @ P_0.T   # P is orthogonal

    return rhs


def Kstep2(C1, grid, lr, F_b_X, F_b_Y, DX, DY, option_scheme="cendiff", 
           DX_0=None, DX_1=None, DY_0=None, DY_1=None):
    """
    K step of radiative transfer equation.

    Part of the K step associated to the y advection 
    after splitting the full equation in 2x1d.
    """

    K_bdry_left, K_bdry_right = computeK_bdry_2x1d_X(lr, grid, F_b_X)
    K_bdry_bottom, K_bdry_top = computeK_bdry_2x1d_Y(lr, grid, F_b_Y)

    if option_scheme=="cendiff":
        DYK = computedxK_2x1d(
            lr, K_bdry_left, K_bdry_right, K_bdry_bottom, K_bdry_top, grid, DX, DY
        )[1]
        rhs = -(grid.coeff[0]) * DYK @ C1[1]
    
    elif option_scheme=="upwind":

        ### Diagonalize matrix C
        # Eigen-decomposition
        eigvals_0, P_0 = np.linalg.eigh(C1[0])  # matrix C1 is symmetric
        eigvals_1, P_1 = np.linalg.eigh(C1[1])  

        # Construct diagonal matrix of eigenvalues
        T1_1 = np.diag(eigvals_1)

        ### Obtain matrix from upwind (multiplied with P_1 already)
        DYK = computedxK_2x1d_upwind(
            lr, K_bdry_left, K_bdry_right, K_bdry_bottom, K_bdry_top, grid, 
            DX_0, DX_1, DY_0, DY_1, eigvals_0, P_0, eigvals_1, P_1
        )[1]

        ### Calculate rhs
        rhs = -(grid.coeff[0]) * DYK @ T1_1 @ P_1.T   # P is orthogonal

    return rhs


def Kstep3(K, C2, grid, lr, source=None):
    """
    K step of radiative transfer equation.

    Part of the K step associated to the collision term 
    after splitting the full equation in 2x1d.
    """

    if source is None:
        rhs = 0.5 / (np.pi) * (grid.coeff[1]) * K @ C2.T @ C2 - (grid.coeff[2]) * K
    else:
        rhs = (0.5 / (np.pi) * (grid.coeff[1]) * K @ C2.T @ C2 - (grid.coeff[2]) * K 
               + 0.5 / (np.pi) * source @ C2)

    return rhs


def Sstep1(C1, D1, grid):
    """
    S step of radiative transfer equation.

    Part of the S step associated to the x advection 
    after splitting the full equation in 2x1d.
    """

    rhs = (grid.coeff[0]) * D1[0] @ C1[0]

    return rhs


def Sstep2(C1, D1, grid):
    """
    S step of radiative transfer equation.

    Part of the S step associated to the y advection 
    after splitting the full equation in 2x1d.
    """
    rhs = (grid.coeff[0]) * D1[1] @ C1[1]

    return rhs


def Sstep3(S, C2, grid, lr, source=None):
    """
    S step of radiative transfer equation.

    Part of the S step associated to the collision term 
    after splitting the full equation in 2x1d.
    """

    if source is None:
        rhs = -0.5 / (np.pi) * (grid.coeff[1]) * S @ C2.T @ C2 + (grid.coeff[2]) * S
    else:
        rhs = (-0.5 / (np.pi) * (grid.coeff[1]) * S @ C2.T @ C2 + (grid.coeff[2]) * S 
               - 0.5 / (np.pi) * lr.U.T @ source @ C2 * grid.dx * grid.dy)

    return rhs


def Lstep1(lr, D1, grid):
    """
    L step of radiative transfer equation.

    Part of the L step associated to the x advection 
    after splitting the full equation in 2x1d.
    """

    rhs = -(grid.coeff[0]) * np.diag(np.cos(grid.PHI)) @ lr.V @ D1[0].T

    return rhs


def Lstep2(lr, D1, grid):
    """
    L step of radiative transfer equation.

    Part of the L step associated to the y advection 
    after splitting the full equation in 2x1d.
    """

    rhs = -(grid.coeff[0]) * np.diag(np.sin(grid.PHI)) @ lr.V @ D1[1].T

    return rhs


def Lstep3(L, B1, grid, lr, source=None):
    """
    L step of radiative transfer equation.

    Part of the L step associated to the collision term 
    after splitting the full equation in 2x1d.
    """

    if source is None:
        rhs = (0.5 / (np.pi) * (grid.coeff[1]) * np.ones((grid.Nphi, 1)) @ B1 
               - (grid.coeff[2]) * L)
    else:
        M = np.ones((1, grid.Nphi))
        rhs = (0.5 / (np.pi) * (grid.coeff[1]) * np.ones((grid.Nphi, 1)) @ B1 
               - (grid.coeff[2]) * L 
               + 0.5 / (np.pi) * ((lr.U.T @ source) @ M).T * grid.dx * grid.dy)

    return rhs


def add_basis_functions(
    lr, grid, F_b, tol_sing_val, dimensions="1x1d"
):
    """
    Add basis functions.

    Add basis functions according to the inflow condition of current subdomain 
    and a tolarance for singular values.
    For higher dimensional simulations set i.e. dimensions = "2x1d".
    """
    # Compute SVD and drop singular values
    X, sing_val, QT = np.linalg.svd(F_b)
    r_b = np.sum(sing_val > tol_sing_val)
    if dimensions == "1x1d" and (
        grid.r + r_b
    ) > grid.Nmu:  # because rank cannot be bigger than our amount of gridpoints
        r_b = grid.Nmu - grid.r
    elif dimensions == "2x1d" and (
        grid.r + r_b
    ) > grid.Nphi:  # because rank cannot be bigger than our amount of gridpoints
        r_b = grid.Nphi - grid.r
    Sigma = np.zeros((F_b.shape[0], r_b))
    np.fill_diagonal(Sigma, sing_val[:r_b])
    Q = QT.T[:, :r_b]

    # Concatenate
    if dimensions == "1x1d":
        X_h = np.random.rand(grid.Nx, r_b)
    elif dimensions == "2x1d":
        X_h = np.random.rand(grid.Nx * grid.Ny, r_b)
    lr.U = np.concatenate((lr.U, X_h), axis=1)
    lr.V = np.concatenate((lr.V, Q), axis=1)
    S_extended = np.zeros((grid.r + r_b, grid.r + r_b))
    S_extended[: grid.r, : grid.r] = lr.S
    lr.S = S_extended

    # QR-decomp
    lr.U, R_U = np.linalg.qr(lr.U, mode="reduced")
    if dimensions == "1x1d":
        lr.U /= np.sqrt(grid.dx)
        R_U *= np.sqrt(grid.dx)
    elif dimensions == "2x1d":
        lr.U /= (np.sqrt(grid.dx) * np.sqrt(grid.dy))
        R_U *= (np.sqrt(grid.dx) * np.sqrt(grid.dy))
    lr.V, R_V = np.linalg.qr(lr.V, mode="reduced")
    if dimensions == "1x1d":
        lr.V /= np.sqrt(grid.dmu)
        R_V *= np.sqrt(grid.dmu)
    elif dimensions == "2x1d":
        lr.V /= np.sqrt(grid.dphi)
        R_V *= np.sqrt(grid.dphi)
    lr.S = R_U @ lr.S @ R_V.T

    grid.r += r_b

    return lr, grid


def drop_basis_functions(lr, grid, drop_tol):
    """
    Drop basis functions.

    Drop basis functions according to some drop tolerance, 
    such that the rank does not grow drastically.
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
