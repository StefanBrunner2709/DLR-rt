"""
Contains integration functions for different example scripts
"""

import numpy as np
from tqdm import tqdm

from DLR_rt.src.grid import Grid_2x1d
from DLR_rt.src.integrators import PSI_lie, PSI_splitting_lie
from DLR_rt.src.lr import LR, computeF_b_2x1d_X, computeF_b_2x1d_Y
from DLR_rt.src.util import (
    computeD_cendiff_2x1d,
    computeD_upwind_2x1d,
    plot_rho_onedomain,
    plot_rho_subgrids,
)


def integrate_dd_hohlraum(lr0_on_subgrids: LR, subgrids: Grid_2x1d, 
                          t_f: float, dt: float,
                          tol_sing_val: float = 1e-3, drop_tol: float = 1e-7, 
                          option_scheme: str = "cendiff", 
                          option_problem : str = "hohlraum", 
                          snapshots: int = 2, plot_name_add = ""):
    
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

    # --- SNAPSHOT setup ---
    if snapshots < 2:
        snapshots = 2  # At least initial and final
    snapshot_times = [i * t_f / (snapshots - 1) for i in range(snapshots)]
    next_snapshot_idx = 1  # first snapshot after t=0

    # --- Initial snapshot ---
    print(f"📸 Snapshot 1/{snapshots} at t = {t:.4f}")
    plot_rho_subgrids(subgrids, lr_on_subgrids, t=t, plot_option="log", 
                      plot_name_add=plot_name_add)

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
                                        F_b_X[: len(subgrids[j][i].Y), k] = (
                                            1
                                            / (2 * np.pi)
                                            * np.exp(-((subgrids[j][i].Y - 0.85
                                                        -subgrids[j][i].dy/2) ** 2) 
                                                     / (1e-5))
                                        )
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

            # --- Check for snapshot condition ---
            if next_snapshot_idx < snapshots and t >= snapshot_times[next_snapshot_idx]:
                print(f"📸 Snapshot {next_snapshot_idx+1}/{snapshots} at t = {t:.4f}")
                plot_rho_subgrids(subgrids, lr_on_subgrids, t=t, plot_option="log", 
                                  plot_name_add=plot_name_add)
                next_snapshot_idx += 1

    return lr_on_subgrids, time, rank_on_subgrids_adapted, rank_on_subgrids_dropped


def integrate_dd_lattice(lr0_on_subgrids: LR, subgrids: Grid_2x1d, 
                         t_f: float, dt: float,
                         tol_sing_val: float = 1e-3, drop_tol: float = 1e-7,
                         option_scheme: str = "cendiff", 
                         option_timescheme : str = "RK4", 
                         snapshots: int = 2, plot_name_add = ""):
    
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

    # --- SNAPSHOT setup ---
    if snapshots < 2:
        snapshots = 2  # At least initial and final
    snapshot_times = [i * t_f / (snapshots - 1) for i in range(snapshots)]
    next_snapshot_idx = 1  # first snapshot after t=0

    # --- Initial snapshot ---
    print(f"📸 Snapshot 1/{snapshots} at t = {t:.4f}")
    plot_rho_subgrids(subgrids, lr_on_subgrids, t=t, plot_option="log", 
                      plot_name_add=plot_name_add)
        
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
                        for k in range(len(subgrids[j][i].PHI)):  # set outflow boundary
                            if (subgrids[j][i].PHI[k] < np.pi / 2 
                                or subgrids[j][i].PHI[k] > 3 / 2 * np.pi):
                                F_b_X[: len(subgrids[j][i].Y), k] = 0
                    elif i==n_split_x-1:
                        F_b_X = computeF_b_2x1d_X(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_left=f_on_subgrids[j][i-1],
                                                  f_right=f_on_subgrids[j][0])
                        for k in range(len(subgrids[j][i].PHI)):  # set outflow boundary
                            if (subgrids[j][i].PHI[k] >= np.pi / 2 
                                and subgrids[j][i].PHI[k] <= 3 / 2 * np.pi):
                                F_b_X[len(subgrids[j][i].Y) :, k] = 0
                    else:
                        F_b_X = computeF_b_2x1d_X(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_left=f_on_subgrids[j][i-1],
                                                  f_right=f_on_subgrids[j][i+1])
                        
                    if j==0:
                        F_b_Y = computeF_b_2x1d_Y(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_bottom=f_on_subgrids[n_split_y-1][i],
                                                  f_top=f_on_subgrids[j+1][i])
                        for k in range(len(subgrids[j][i].PHI)):  # set outflow boundary
                            if subgrids[j][i].PHI[k] < np.pi:
                                F_b_Y[: len(subgrids[j][i].X), k] = 0
                    elif j==n_split_y-1:
                        F_b_Y = computeF_b_2x1d_Y(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_bottom=f_on_subgrids[j-1][i],
                                                  f_top=f_on_subgrids[0][i])
                        for k in range(len(subgrids[j][i].PHI)):  # set outflow boundary
                            if subgrids[j][i].PHI[k] >= np.pi:
                                F_b_Y[len(subgrids[j][i].X) :, k] = 0
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
                     rank_on_subgrids_adapted[j][i], 
                     rank_on_subgrids_dropped[j][i]) = PSI_splitting_lie(
                        lr_on_subgrids[j][i],
                        subgrids[j][i],
                        dt,
                        F_b_X_on_subgrids[j][i],
                        F_b_Y_on_subgrids[j][i],
                        DX=DX,
                        DY=DY,
                        tol_sing_val=tol_sing_val,
                        drop_tol=drop_tol,
                        rank_adapted=rank_on_subgrids_adapted[j][i],
                        rank_dropped=rank_on_subgrids_dropped[j][i],
                        source=source,
                        option_scheme=option_scheme, 
                        DX_0=DX_0, DX_1=DX_1, DY_0=DY_0, DY_1=DY_1,
                        option_timescheme=option_timescheme
                    )

            ### Update time
            t += dt
            time.append(t)

            # --- Check for snapshot condition ---
            if next_snapshot_idx < snapshots and t >= snapshot_times[next_snapshot_idx]:
                print(f"📸 Snapshot {next_snapshot_idx+1}/{snapshots} at t = {t:.4f}")
                plot_rho_subgrids(subgrids, lr_on_subgrids, t=t, plot_option="log", 
                                  plot_name_add=plot_name_add)
                next_snapshot_idx += 1

    return lr_on_subgrids, time, rank_on_subgrids_adapted, rank_on_subgrids_dropped

def integrate_1domain(lr0: LR, grid: Grid_2x1d, t_f: float, dt: float, 
              option: str = "lie", source = None, 
              option_scheme : str = "cendiff", option_timescheme : str = "RK4",
              option_bc : str = "standard", tol_sing_val = 1e-2, drop_tol = 1e-3, 
              tol_lattice = 1e-5, snapshots: int = 2, plot_name_add = ""):
    
    min_rank = grid.r

    if option_bc == "lattice" or option_bc == "hohlraum" or option_bc == "pointsource":
        rank_adapted = [grid.r]
        rank_dropped = [grid.r]
    else:
        rank_adapted = None
        rank_dropped = None

    lr = lr0
    t = 0
    time = []
    time.append(t)

    DX, DY = computeD_cendiff_2x1d(grid, "dd")

    if option_scheme == "upwind":
        DX_0, DX_1, DY_0, DY_1 = computeD_upwind_2x1d(grid, "dd")
    else:
        DX_0 = None
        DX_1 = None
        DY_0 = None
        DY_1 = None

    # --- SNAPSHOT setup ---
    if snapshots < 2:
        snapshots = 2  # At least initial and final
    snapshot_times = [i * t_f / (snapshots - 1) for i in range(snapshots)]
    next_snapshot_idx = 1  # first snapshot after t=0

    # --- Initial snapshot ---
    print(f"📸 Snapshot 1/{snapshots} at t = {t:.4f}")
    plot_rho_onedomain(grid, lr, t=t, plot_name_add=plot_name_add)

    with tqdm(total=t_f / dt, desc="Running Simulation") as pbar:
        while t < t_f:
            pbar.update(1)

            if t + dt > t_f:
                dt = t_f - t

            if (option_bc == "lattice" or option_bc == "hohlraum" 
                or option_bc == "pointsource"):

                f = lr.U @ lr.S @ lr.V.T

                F_b_X = computeF_b_2x1d_X(f, grid, option_bc = option_bc)
                F_b_Y = computeF_b_2x1d_Y(f, grid, option_bc = option_bc)

            else:
                F_b_X = None
                F_b_Y = None

            if option == "lie":
                (lr, grid, 
                 rank_adapted, rank_dropped) = PSI_lie(lr, grid, dt, DX=DX, DY=DY, 
                                   dimensions="2x1d", option_coeff="space_dep", 
                                   source=source, option_scheme=option_scheme,
                                   DX_0=DX_0, DX_1=DX_1, DY_0=DY_0, DY_1=DY_1,
                                   option_timescheme=option_timescheme,
                                   option_bc = option_bc, F_b_X = F_b_X, F_b_Y = F_b_Y,
                                   tol_sing_val=tol_sing_val, drop_tol=drop_tol, 
                                   min_rank=min_rank, 
                                   rank_adapted=rank_adapted, rank_dropped=rank_dropped,
                                   tol_lattice=tol_lattice)

            t += dt
            time.append(t)

            # --- Check for snapshot condition ---
            if next_snapshot_idx < snapshots and t >= snapshot_times[next_snapshot_idx]:
                print(f"📸 Snapshot {next_snapshot_idx+1}/{snapshots} at t = {t:.4f}")
                plot_rho_onedomain(grid, lr, t=t, plot_name_add=plot_name_add)
                next_snapshot_idx += 1

    return lr, time, rank_adapted, rank_dropped
