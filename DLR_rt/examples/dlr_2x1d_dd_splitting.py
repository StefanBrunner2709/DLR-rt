import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from DLR_rt.src.splitting_2x1d import LR, Grid_2x1d, setInitialCondition_2x1d_lr, computeF_b_2x1d, add_basis_functions, drop_basis_functions, PSI_lie_splitting


def integrate(lr0_left: LR, lr0_right: LR, grid_left: Grid_2x1d, grid_right: Grid_2x1d, t_f: float, dt: float, tol_sing_val: float = 1e-6, drop_tol: float = 1e-6):
    lr_left = lr0_left
    lr_right = lr0_right
    t = 0
    time = []
    time.append(t)

    with tqdm(total=t_f/dt, desc="Running Simulation") as pbar:

        while t < t_f:

            pbar.update(1)

            if (t + dt > t_f):
                dt = t_f - t

            ### Compute F_b
            F_b_left = computeF_b_2x1d(lr_left.U @ lr_left.S @ lr_left.V.T, grid_left, f_right = lr_right.U @ lr_right.S @ lr_right.V.T, f_periodic = lr_right.U @ lr_right.S @ lr_right.V.T)
            F_b_right = computeF_b_2x1d(lr_right.U @ lr_right.S @ lr_right.V.T, grid_right, f_left = lr_left.U @ lr_left.S @ lr_left.V.T, f_periodic = lr_left.U @ lr_left.S @ lr_left.V.T)

            ### Update left side

            ### Add basis for adaptive rank strategy:
            lr_left, grid_left = add_basis_functions(lr_left, grid_left, F_b_left, tol_sing_val)

            ### Run PSI
            lr_left, grid_left = PSI_lie_splitting(lr_left, grid_left, dt, F_b_left)

            ### Drop basis for adaptive rank strategy:
            lr_left, grid_left = drop_basis_functions(lr_left, grid_left, drop_tol)

            ### Update right side

            ### Add basis for adaptive rank strategy:
            lr_right, grid_right = add_basis_functions(lr_right, grid_right, F_b_right, tol_sing_val)

            ### Run PSI
            lr_right, grid_right = PSI_lie_splitting(lr_right, grid_right, dt, F_b_right)

            ### Drop basis for adaptive rank strategy:
            lr_right, grid_right = drop_basis_functions(lr_right, grid_right, drop_tol)

            ### Update time
            t += dt
            time.append(t)

    return lr_left, lr_right, time



### Plotting

Nx = 32
Ny = 32
Nphi = 32
dt = 1e-4
r = 8
t_f = 0.5
fs = 16
savepath = "plots/"
method = "lie"


### Initial configuration

grid = Grid_2x1d(Nx, Ny, Nphi, r)
grid_left, grid_right = grid.split_x()

lr0_left = setInitialCondition_2x1d_lr(grid_left)
lr0_right = setInitialCondition_2x1d_lr(grid_right)

f0_left = lr0_left.U @ lr0_left.S @ lr0_left.V.T
f0_right = lr0_right.U @ lr0_right.S @ lr0_right.V.T

rho0_left = (f0_left @ np.ones(grid_left.Nphi)) * grid_left.dphi    # This is now a vector, only depends on x and y
rho0_right = (f0_right @ np.ones(grid_right.Nphi)) * grid_right.dphi    # This is now a vector, only depends on x and y

rho0_matrix_left = rho0_left.reshape((grid_left.Nx, grid_left.Ny), order='F')
rho0_matrix_right = rho0_right.reshape((grid_right.Nx, grid_right.Ny), order='F')

rho0_matrix = np.concatenate((rho0_matrix_left, rho0_matrix_right), axis=0)

extent = [grid.X[0], grid.X[-1], grid.Y[0], grid.Y[-1]]

fig, axes = plt.subplots(1, 1, figsize=(10, 8))

im = axes.imshow(rho0_matrix.T, extent=extent, origin='lower')
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel("y", fontsize=fs, labelpad=-5)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([0, 0.5, 1])
axes.tick_params(axis='both', labelsize=fs, pad=10)

cbar_fixed = fig.colorbar(im, ax=axes)
cbar_fixed.set_ticks([np.min(rho0_matrix), np.max(rho0_matrix)])
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "dd_splitting_2x1d_rho_initial.pdf")


### Final configuration

lr_left, lr_right, time = integrate(lr0_left, lr0_right, grid_left, grid_right, t_f, dt)

f_left = lr_left.U @ lr_left.S @ lr_left.V.T
f_right = lr_right.U @ lr_right.S @ lr_right.V.T

rho_left = (f_left @ np.ones(grid_left.Nphi)) * grid_left.dphi    # This is now a vector, only depends on x and y
rho_right = (f_right @ np.ones(grid_right.Nphi)) * grid_right.dphi    # This is now a vector, only depends on x and y

rho_matrix_left = rho_left.reshape((grid_left.Nx, grid_left.Ny), order='F')
rho_matrix_right = rho_right.reshape((grid_right.Nx, grid_right.Ny), order='F')

rho_matrix = np.concatenate((rho_matrix_left, rho_matrix_right), axis=0)

extent = [grid.X[0], grid.X[-1], grid.Y[0], grid.Y[-1]]

fig, axes = plt.subplots(1, 1, figsize=(10, 8))

im = axes.imshow(rho_matrix.T, extent=extent, origin='lower')
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel("y", fontsize=fs, labelpad=-5)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([0, 0.5, 1])
axes.tick_params(axis='both', labelsize=fs, pad=10)

cbar_fixed = fig.colorbar(im, ax=axes)
cbar_fixed.set_ticks([np.min(rho_matrix), np.max(rho_matrix)])
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "dd_splitting_2x1d_rho_final.pdf")

