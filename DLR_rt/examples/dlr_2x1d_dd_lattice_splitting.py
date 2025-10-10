import numpy as np

from DLR_rt.src.grid import Grid_2x1d
from DLR_rt.src.initial_condition import setInitialCondition_2x1d_lr_subgrids
from DLR_rt.src.run_functions import integrate_dd_lattice
from DLR_rt.src.util import plot_ranks_subgrids

### Plotting

Nx = 252
Ny = 252
Nphi = 252
dt = 0.5 / Nx
r = 5
t_f = 0.1
snapshots = 2
fs = 16
savepath = "plots/"
method = "lie"
option_scheme = "upwind"
option_timescheme = "RK4"

option_error_estimate = True

tol_sing_val = 1e-8
drop_tol = 2e-12


### Initial configuration
grid = Grid_2x1d(Nx, Ny, Nphi, r, _option_dd="dd", _coeff=[1.0, 1.0, 1.0])
subgrids = grid.split_grid_into_subgrids(option_coeff="lattice", 
                                         n_split_y=7, n_split_x=7)


lr0_on_subgrids = setInitialCondition_2x1d_lr_subgrids(subgrids, option_cond="lattice")

### Final configuration
(lr_on_subgrids, time, 
 rank_on_subgrids_adapted, rank_on_subgrids_dropped) = integrate_dd_lattice(
    lr0_on_subgrids, subgrids, t_f, dt, 
    option_scheme=option_scheme, option_timescheme=option_timescheme,
    tol_sing_val=tol_sing_val, drop_tol=drop_tol, snapshots=snapshots
    )

plot_ranks_subgrids(subgrids, time, rank_on_subgrids_adapted, rank_on_subgrids_dropped)


### Compare to higher rank solution
if option_error_estimate:

    ### Initial configuration
    grid_2 = Grid_2x1d(Nx, Ny, Nphi, r, _option_dd="dd", _coeff=[1.0, 1.0, 1.0])
    subgrids_2 = grid_2.split_grid_into_subgrids(option_coeff="lattice", 
                                                n_split_y=7, n_split_x=7)


    lr0_on_subgrids_2 = setInitialCondition_2x1d_lr_subgrids(subgrids_2, 
                                                            option_cond="lattice")

    ### Final configuration
    (lr_on_subgrids_2, time_2, rank_on_subgrids_adapted_2, 
    rank_on_subgrids_dropped_2) = integrate_dd_lattice(
        lr0_on_subgrids_2, subgrids_2, t_f, dt, option_scheme=option_scheme, 
        option_timescheme=option_timescheme, 
        tol_sing_val=tol_sing_val*0.001, drop_tol=drop_tol*0.001, 
        snapshots=snapshots, plot_name_add="high_rank_"
        )

    plot_ranks_subgrids(subgrids_2, time_2, 
                        rank_on_subgrids_adapted_2, rank_on_subgrids_dropped_2,
                        plot_name_add="high_rank_")

    Frob = 0

    n_split_x = subgrids[0][0].n_split_x
    n_split_y = subgrids[0][0].n_split_y

    for j in range(n_split_y):
        for i in range(n_split_x):

            f = (lr_on_subgrids[j][i].U @ lr_on_subgrids[j][i].S @ 
                    lr_on_subgrids[j][i].V.T)
            
            f_2 = (lr_on_subgrids_2[j][i].U @ lr_on_subgrids_2[j][i].S @ 
                    lr_on_subgrids_2[j][i].V.T)
            
            Frob += (np.linalg.norm(f - f_2, ord='fro'))**2

    Frob = np.sqrt(Frob)

    print("Frobenius: ", Frob)
