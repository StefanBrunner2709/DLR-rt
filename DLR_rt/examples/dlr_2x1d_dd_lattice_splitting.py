import numpy as np
from tqdm import tqdm

from DLR_rt.src.grid import Grid_2x1d
from DLR_rt.src.initial_condition import setInitialCondition_2x1d_lr_subgrids
from DLR_rt.src.integrators import PSI_splitting_lie
from DLR_rt.src.lr import LR, computeF_b_2x1d_X, computeF_b_2x1d_Y
from DLR_rt.src.util import (
    computeD_cendiff_2x1d,
    computeD_upwind_2x1d,
    plot_rho_subgrids,
)


def integrate(lr0_on_subgrids: LR, subgrids: Grid_2x1d, t_f: float, dt: float,
               tol_sing_val: float = 1e-3, drop_tol: float = 1e-7, method="lie", 
               option_scheme: str = "cendiff"):
    
    lr_on_subgrids = lr0_on_subgrids
    t = 0
    time = []
    time.append(t)

    DX, DY = computeD_cendiff_2x1d(subgrids[0][0], "dd")
    # all grids have same size, thus enough to compute once

    if option_scheme == "upwind":
        DX_0, DX_1, DY_0, DY_1 = computeD_upwind_2x1d(subgrids[0][0], "dd")
    else:
        DX_0 = None
        DX_1 = None
        DY_0 = None
        DY_1 = None

    n_split_x = subgrids[0][0].n_split_x
    n_split_y = subgrids[0][0].n_split_y

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
                                                  f_right=f_on_subgrids[j][i+1])       
                    elif i==n_split_x-1:
                        F_b_X = computeF_b_2x1d_X(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_left=f_on_subgrids[j][i-1],
                                                  f_right=f_on_subgrids[j][0])
                    else:
                        F_b_X = computeF_b_2x1d_X(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_left=f_on_subgrids[j][i-1],
                                                  f_right=f_on_subgrids[j][i+1])
                        
                    if j==0:
                        F_b_Y = computeF_b_2x1d_Y(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_bottom=f_on_subgrids[n_split_y-1][i],
                                                  f_top=f_on_subgrids[j+1][i])
                    elif j==n_split_y-1:
                        F_b_Y = computeF_b_2x1d_Y(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_bottom=f_on_subgrids[j-1][i],
                                                  f_top=f_on_subgrids[0][i])
                    else:
                        F_b_Y = computeF_b_2x1d_Y(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_bottom=f_on_subgrids[j-1][i],
                                                  f_top=f_on_subgrids[j+1][i])
                        
                    row_F_b_X.append(F_b_X)
                    row_F_b_Y.append(F_b_Y)
                F_b_X_on_subgrids.append(row_F_b_X)
                F_b_Y_on_subgrids.append(row_F_b_Y)

            ### Update lr by PSI with adaptive rank strategy
            ### Run PSI with adaptive rank strategy
            for j in range(n_split_y):
                for i in range(n_split_x):

                    if i==3 and j==3:
                        source= np.ones((subgrids[j][i].Nx, subgrids[j][i].Ny))
                        source = source.flatten()[:, None]
                    else:
                        source = None

                    (lr_on_subgrids[j][i], 
                     subgrids[j][i], 
                     rank_adapted, 
                     rank_dropped) = PSI_splitting_lie(
                        lr_on_subgrids[j][i],
                        subgrids[j][i],
                        dt,
                        F_b_X_on_subgrids[j][i],
                        F_b_Y_on_subgrids[j][i],
                        DX=DX,
                        DY=DY,
                        tol_sing_val=tol_sing_val,
                        drop_tol=drop_tol,
                        source=source,
                        option_scheme=option_scheme, 
                        DX_0=DX_0, DX_1=DX_1, DY_0=DY_0, DY_1=DY_1
                    )

            ### Update time
            t += dt
            time.append(t)

    return lr_on_subgrids, time


### Plotting

Nx = 252
Ny = 252
Nphi = 252
dt = 0.95 / Nx
r = 5
t_f = 0.4
fs = 16
savepath = "plots/"
method = "lie"
option_scheme = "upwind"


### Initial configuration
grid = Grid_2x1d(Nx, Ny, Nphi, r, _option_dd="dd", _coeff=[1.0, 1.0, 1.0])
subgrids = grid.split_grid_into_subgrids(option_coeff="lattice", 
                                         n_split_y=7, n_split_x=7)

# # Print subgrids as a test:
# for j in range(8):
#     for i in range(8):
#         print("At subgrid position ", (j,i), " we have: ", 
#               subgrids[j][i].X, subgrids[j][i].Y, " with coefficients: ", 
#               subgrids[j][i].coeff)
        
lr0_on_subgrids = setInitialCondition_2x1d_lr_subgrids(subgrids, option_cond="lattice")

plot_rho_subgrids(subgrids, lr0_on_subgrids, plot_option="log")

### Final configuration
lr_on_subgrids, time = integrate(lr0_on_subgrids, subgrids, t_f, dt, 
                                 option_scheme=option_scheme)

plot_rho_subgrids(subgrids, lr_on_subgrids, t=t_f, plot_option="log")
