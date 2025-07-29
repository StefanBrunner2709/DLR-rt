import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from DLR_rt.src.splitting_2x1d import LR, Grid_2x1d, setInitialCondition_2x1d_lr, computeF_b_2x1d, add_basis_functions, drop_basis_functions, PSI_splitting


def integrate(lr0_left: LR, lr0_right: LR, grid_left: Grid_2x1d, grid_right: Grid_2x1d, t_f: float, dt: float, tol_sing_val: float = 1e-6, drop_tol: float = 1e-6, method = "lie"):
    lr_left = lr0_left
    lr_right = lr0_right
    t = 0
    time = []
    time.append(t)
    rank_left_adapted = [grid_left.r]
    rank_left_dropped = [grid_left.r]
    rank_right_adapted = [grid_right.r]
    rank_right_dropped = [grid_right.r]

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
            lr_left_new, grid_left_new = add_basis_functions(lr_left, grid_left, F_b_left, tol_sing_val)
            rank_left_adapted.append(grid_left.r)

            ### Run PSI
            lr_left_new, grid_left_new = PSI_splitting(lr_left_new, grid_left_new, dt, F_b_left, lr_right, method, "left")

            ### Drop basis for adaptive rank strategy:
            lr_left_new, grid_left_new = drop_basis_functions(lr_left_new, grid_left_new, drop_tol)
            rank_left_dropped.append(grid_left.r)

            ### Update right side

            ### Add basis for adaptive rank strategy:
            lr_right, grid_right = add_basis_functions(lr_right, grid_right, F_b_right, tol_sing_val)
            rank_right_adapted.append(grid_right.r)

            ### Run PSI
            lr_right, grid_right = PSI_splitting(lr_right, grid_right, dt, F_b_right, lr_left, method, "right")

            ### Drop basis for adaptive rank strategy:
            lr_right, grid_right = drop_basis_functions(lr_right, grid_right, drop_tol)
            rank_right_dropped.append(grid_right.r)

            ### Assign lr_left, grid_left
            lr_left, grid_left = lr_left_new, grid_left_new

            ### Update time
            t += dt
            time.append(t)

    return lr_left, lr_right, time, rank_left_adapted, rank_left_dropped, rank_right_adapted, rank_right_dropped



### Plotting

Nx = 32
Ny = 32
Nphi = 32
dt = 1e-3
r = 5
t_f = 1.0
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
axes.set_ylabel("$y$", fontsize=fs)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([0, 0.5, 1])
axes.tick_params(axis='both', labelsize=fs, pad=10)

cbar_fixed = fig.colorbar(im, ax=axes)
cbar_fixed.set_ticks([np.min(rho0_matrix), np.max(rho0_matrix)])
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "dd_splitting_2x1d_rho_initial.pdf")


### Final configuration

lr_left, lr_right, time, rank_left_adapted, rank_left_dropped, rank_right_adapted, rank_right_dropped = integrate(lr0_left, lr0_right, grid_left, grid_right, t_f, dt, method = method)

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
axes.set_ylabel("$y$", fontsize=fs)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([0, 0.5, 1])
axes.tick_params(axis='both', labelsize=fs, pad=10)

cbar_fixed = fig.colorbar(im, ax=axes)
cbar_fixed.set_ticks([np.min(rho_matrix), np.max(rho_matrix)])
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "dd_splitting_2x1d_rho_final.pdf")


### Slice plot

rho_vector_x = rho_matrix[:,int(grid_left.Ny/2)]

fig, axes = plt.subplots(1, 1, figsize=(10, 8))
plt.plot(grid.X, rho_vector_x)
plt.title("$y$ fixed")
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel("$f(x)$", fontsize=fs)
plt.savefig(savepath + "dd_splitting_2x1d_rho_x_final.pdf")


### Rank plots

fig, axes = plt.subplots(1, 1, figsize=(10, 8))
plt.plot(time,  rank_left_adapted)
plt.title("rank left adapted")
axes.set_xlabel("$t$", fontsize=fs)
axes.set_ylabel("$r(t)$", fontsize=fs)
plt.savefig(savepath + "dd_splitting_2x1d_rank_left_adapted.pdf")

fig, axes = plt.subplots(1, 1, figsize=(10, 8))
plt.plot(time,  rank_left_dropped)
plt.title("rank left dropped")
axes.set_xlabel("$t$", fontsize=fs)
axes.set_ylabel("$r(t)$", fontsize=fs)
plt.savefig(savepath + "dd_splitting_2x1d_rank_left_dropped.pdf")

fig, axes = plt.subplots(1, 1, figsize=(10, 8))
plt.plot(time,  rank_right_adapted)
plt.title("rank right adapted")
axes.set_xlabel("$t$", fontsize=fs)
axes.set_ylabel("$r(t)$", fontsize=fs)
plt.savefig(savepath + "dd_splitting_2x1d_rank_right_adapted.pdf")

fig, axes = plt.subplots(1, 1, figsize=(10, 8))
plt.plot(time,  rank_right_dropped)
plt.title("rank right dropped")
axes.set_xlabel("$t$", fontsize=fs)
axes.set_ylabel("$r(t)$", fontsize=fs)
plt.savefig(savepath + "dd_splitting_2x1d_rank_right_dropped.pdf")


### Print frobenius error
print(np.linalg.norm(rho0_matrix-rho_matrix, 'fro'))