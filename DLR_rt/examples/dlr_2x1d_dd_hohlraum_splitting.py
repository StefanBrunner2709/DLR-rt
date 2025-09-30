import numpy as np
from tqdm import tqdm

from DLR_rt.src.grid import Grid_2x1d
from DLR_rt.src.initial_condition import setInitialCondition_2x1d_lr_subgrids
from DLR_rt.src.integrators import PSI_splitting_lie
from DLR_rt.src.lr import LR, computeF_b_2x1d_X, computeF_b_2x1d_Y
from DLR_rt.src.util import (
    computeD_cendiff_2x1d,
    computeD_upwind_2x1d,
    plot_ranks_subgrids,
    plot_rho_subgrids,
)


def integrate(lr0_on_subgrids: LR, subgrids: Grid_2x1d, t_f: float, dt: float,
               tol_sing_val: float = 1e-3, drop_tol: float = 1e-7, method="lie", 
               option_scheme: str = "cendiff", option_problem : str = "hohlraum"):
    
    lr_on_subgrids = lr0_on_subgrids
    t = 0
    time = []
    time.append(t)

    n_split_x = subgrids[0][0].n_split_x
    n_split_y = subgrids[0][0].n_split_y

    D_on_subgrids = []

    for j in range(n_split_y):
        row = []
        for i in range(n_split_x):

            DX, DY = computeD_cendiff_2x1d(subgrids[j][i], "dd")
            # all grids have same size, thus enough to compute once

            if option_scheme == "upwind":
                DX_0, DX_1, DY_0, DY_1 = computeD_upwind_2x1d(subgrids[j][i], "dd")
            else:
                DX_0 = None
                DX_1 = None
                DY_0 = None
                DY_1 = None

            D = [DX, DY, DX_0, DX_1, DY_0, DY_1]

            row.append(D)
        D_on_subgrids.append(row)

    rank_on_subgrids_adapted = []
    rank_on_subgrids_dropped = []

    for j in range(n_split_y):
        row_adapted = []
        row_dropped = []
        for i in range(n_split_x):

            rank_adapted = [subgrids[j][i].r]
            rank_dropped = [subgrids[j][i].r]

            row_adapted.append(rank_adapted)
            row_dropped.append(rank_dropped)
        rank_on_subgrids_adapted.append(row_adapted)
        rank_on_subgrids_dropped.append(row_dropped)

    with tqdm(total=t_f / dt, desc="Running Simulation") as pbar:
        while t < t_f:
            pbar.update(1)

            if t + dt > t_f:
                dt = t_f - t

            ### Calculate f
            f_on_subgrids = []

            for j in range(n_split_y):
                row = []
                for i in range(n_split_x):
                    
                    f = (lr_on_subgrids[j][i].U @ lr_on_subgrids[j][i].S @ 
                        lr_on_subgrids[j][i].V.T)
                    
                    row.append(f)
                f_on_subgrids.append(row)

            ### Calculate F_b
            F_b_X_on_subgrids = []
            F_b_Y_on_subgrids = []
            
            for j in range(n_split_y):
                row_F_b_X = []
                row_F_b_Y = []
                for i in range(n_split_x):

                    if i==0:
                        F_b_X = computeF_b_2x1d_X(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_left=f_on_subgrids[j][n_split_x-1],
                                                  f_right=f_on_subgrids[j][i+1],
                                                  grid_left=subgrids[j][n_split_x-1],
                                                  grid_right=subgrids[j][i+1])     
                        if option_problem == "hohlraum":
                            for k in range(len(subgrids[j][i].PHI)):  # inflow left is 1
                                if (subgrids[j][i].PHI[k] < np.pi / 2 
                                    or subgrids[j][i].PHI[k] > 3 / 2 * np.pi):
                                    F_b_X[: len(subgrids[j][i].Y), k] = 1
                        elif option_problem == "pointsource":
                            for k in range(len(subgrids[j][i].PHI)):
                                if (subgrids[j][i].PHI[k] < np.pi / 2 
                                    or subgrids[j][i].PHI[k] > 3 / 2 * np.pi):
                                    F_b_X[: len(subgrids[j][i].Y), k] = 0
                                    if j == n_split_y-2:
                                        F_b_X[int(len(subgrids[j][i].Y)*3/4), k] = 1
                    elif i==n_split_x-1:
                        F_b_X = computeF_b_2x1d_X(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_left=f_on_subgrids[j][i-1],
                                                  f_right=f_on_subgrids[j][0],
                                                  grid_left=subgrids[j][i-1],
                                                  grid_right=subgrids[j][0])
                        for k in range(len(subgrids[j][i].PHI)):  # set outflow boundary
                            if (subgrids[j][i].PHI[k] >= np.pi / 2 
                                and subgrids[j][i].PHI[k] <= 3 / 2 * np.pi):
                                F_b_X[len(subgrids[j][i].Y) :, k] = 0
                    else:
                        F_b_X = computeF_b_2x1d_X(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_left=f_on_subgrids[j][i-1],
                                                  f_right=f_on_subgrids[j][i+1],
                                                  grid_left=subgrids[j][i-1],
                                                  grid_right=subgrids[j][i+1])
                        
                    if j==0:
                        F_b_Y = computeF_b_2x1d_Y(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_bottom=f_on_subgrids[n_split_y-1][i],
                                                  f_top=f_on_subgrids[j+1][i],
                                                  grid_bottom=subgrids[n_split_y-1][i],
                                                  grid_top=subgrids[j+1][i])
                        for k in range(len(subgrids[j][i].PHI)):  # set outflow boundary
                            if subgrids[j][i].PHI[k] < np.pi:
                                F_b_Y[: len(subgrids[j][i].X), k] = 0
                    elif j==n_split_y-1:
                        F_b_Y = computeF_b_2x1d_Y(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_bottom=f_on_subgrids[j-1][i],
                                                  f_top=f_on_subgrids[0][i],
                                                  grid_bottom=subgrids[j-1][i],
                                                  grid_top=subgrids[0][i])
                        for k in range(len(subgrids[j][i].PHI)):  # set outflow boundary
                            if subgrids[j][i].PHI[k] >= np.pi:
                                F_b_Y[len(subgrids[j][i].X) :, k] = 0
                    else:
                        F_b_Y = computeF_b_2x1d_Y(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_bottom=f_on_subgrids[j-1][i],
                                                  f_top=f_on_subgrids[j+1][i],
                                                  grid_bottom=subgrids[j-1][i],
                                                  grid_top=subgrids[j+1][i])
                        
                    row_F_b_X.append(F_b_X)
                    row_F_b_Y.append(F_b_Y)
                F_b_X_on_subgrids.append(row_F_b_X)
                F_b_Y_on_subgrids.append(row_F_b_Y)

            ### Update lr by PSI with adaptive rank strategy
            ### Run PSI with adaptive rank strategy
            for j in range(n_split_y):
                for i in range(n_split_x):

                    # if (j==1 or j==3) and i==0:
                    #     source= np.ones((subgrids[j][i].Nx, subgrids[j][i].Ny))
                    #     source = source.flatten()[:, None]
                    # else:
                    source = None

                    (lr_on_subgrids[j][i], 
                     subgrids[j][i], 
                     rank_on_subgrids_adapted[j][i], 
                     rank_on_subgrids_dropped[j][i]) = PSI_splitting_lie(
                        lr_on_subgrids[j][i],
                        subgrids[j][i],
                        dt,
                        F_b_X_on_subgrids[j][i],
                        F_b_Y_on_subgrids[j][i],
                        DX=D_on_subgrids[j][i][0],
                        DY=D_on_subgrids[j][i][1],
                        tol_sing_val=tol_sing_val,
                        drop_tol=drop_tol,
                        rank_adapted=rank_on_subgrids_adapted[j][i],
                        rank_dropped=rank_on_subgrids_dropped[j][i],
                        source=source,
                        option_scheme=option_scheme, 
                        DX_0=D_on_subgrids[j][i][2], 
                        DX_1=D_on_subgrids[j][i][3], 
                        DY_0=D_on_subgrids[j][i][4], 
                        DY_1=D_on_subgrids[j][i][5]
                    )

            ### Update time
            t += dt
            time.append(t)

    return lr_on_subgrids, time, rank_on_subgrids_adapted, rank_on_subgrids_dropped


### Plotting

Nx = 200
Ny = 200
Nphi = 200
dt = 0.5 / Nx
r = 5
t_f = 1.5
fs = 16
savepath = "plots/"
method = "lie"
option_scheme = "upwind"
option_problem = "hohlraum"


### Initial configuration
grid = Grid_2x1d(Nx, Ny, Nphi, r, _option_dd="dd")
subgrids = grid.split_grid_into_subgrids(option_split="hohlraum")

# # Print subgrids as a test:
# for j in range(5):
#     for i in range(5):
#         print("At subgrid position ", (j,i), " we have: ", 
#               subgrids[j][i].X, subgrids[j][i].Y, " with coefficients: ", 
#               subgrids[j][i].coeff)


lr0_on_subgrids = setInitialCondition_2x1d_lr_subgrids(subgrids, option_cond="lattice")

plot_rho_subgrids(subgrids, lr0_on_subgrids)

### Final configuration
lr_on_subgrids, time, rank_on_subgrids_adapted, rank_on_subgrids_dropped = integrate(
    lr0_on_subgrids, subgrids, t_f, dt, option_scheme=option_scheme, 
    tol_sing_val=1e-3, drop_tol=1e-5, option_problem=option_problem
    )

# plot_rho_subgrids(subgrids, lr_on_subgrids, t=t_f)

plot_rho_subgrids(subgrids, lr_on_subgrids, t=t_f, plot_option="log")

plot_ranks_subgrids(subgrids, time, rank_on_subgrids_adapted, rank_on_subgrids_dropped,
                    option="hohlraum")
