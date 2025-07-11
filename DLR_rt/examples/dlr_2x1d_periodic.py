import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from DLR_rt.src.grid import Grid_2x1d
from DLR_rt.src.integrators import PSI_lie
from DLR_rt.src.initial_condition import setInitialCondition_2x1d_lr
from DLR_rt.src.lr import LR


def integrate(lr0: LR, grid: Grid_2x1d, t_f: float, dt: float, option: str = "lie"):
    lr = lr0
    t = 0
    time = []
    time.append(t)

    with tqdm(total=t_f/dt, desc="Running Simulation") as pbar:

        while t < t_f:

            pbar.update(1)

            if (t + dt > t_f):
                dt = t_f - t

            if option=="lie":
                lr, grid = PSI_lie(lr, grid, dt, dimensions = "2x1d")

            t += dt
            time.append(t)

    return lr, time


### Plotting

Nx = 32
Ny = 32
Nphi = 32
dt = 1e-3
r = 8
t_f = 1.0
fs = 16
savepath = "plots/"
method = "lie"

grid = Grid_2x1d(Nx, Ny, Nphi, r)
lr0 = setInitialCondition_2x1d_lr(grid)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr, time = integrate(lr0, grid, t_f, dt)
f = lr.U @ lr.S @ lr.V.T

rho0 = (f0 @ np.ones(grid.Nphi)) * grid.dphi    # This is now a vector, only depends on x and y
rho = (f @ np.ones(grid.Nphi)) * grid.dphi    # This is now a vector, only depends on x and y

rho0_matrix = rho0.reshape((grid.Nx, grid.Ny), order='F')
rho_matrix = rho.reshape((grid.Nx, grid.Ny), order='F')

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
plt.savefig(savepath + "2x1d_rho_initial.pdf")

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
plt.savefig(savepath + "2x1d_rho_final.pdf")


### Calculate Frobenius norm

fro_norm = np.linalg.norm(f0-f, 'fro')
print("Frobenius norm f(t=0) - f(t=1): ", fro_norm)


### Plots integrating over x:

# Reshape into (n_y, n_x, n_phi)
f0_reshaped = f0.reshape(Ny, Nx, Nphi)

# Apply trapezoidal integration along the x-axis (axis=1)
f0_integrated = np.trapezoid(f0_reshaped, grid.X, axis=1)  # Result shape: (n_y, n_phi)

fig, axes = plt.subplots(1, 1, figsize=(10, 8))

extent = [grid.PHI[0], grid.PHI[-1], grid.Y[0], grid.Y[-1]]

im = axes.imshow(f0_integrated, extent=extent, origin='lower', aspect = "auto")
axes.set_xlabel(r"$\phi$", fontsize=fs)
axes.set_ylabel("y", fontsize=fs, labelpad=-5)
axes.set_yticks([0, 0.5, 1])
axes.tick_params(axis='both', labelsize=fs, pad=10)

cbar_fixed = fig.colorbar(im, ax=axes)
cbar_fixed.set_ticks([np.min(f0_integrated), np.max(f0_integrated)])
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "2x1d_collisions_yphi_initial_piquarter_t10.pdf")


# Reshape into (n_y, n_x, n_phi)
f_reshaped = f.reshape(Ny, Nx, Nphi)

# Apply trapezoidal integration along the x-axis (axis=1)
f_integrated = np.trapezoid(f_reshaped, grid.X, axis=1)  # Result shape: (n_y, n_phi)

fig, axes = plt.subplots(1, 1, figsize=(10, 8))

extent = [grid.PHI[0], grid.PHI[-1], grid.Y[0], grid.Y[-1]]

im = axes.imshow(f_integrated, extent=extent, origin='lower', aspect = "auto")
axes.set_xlabel(r"$\phi$", fontsize=fs)
axes.set_ylabel("y", fontsize=fs, labelpad=-5)
axes.set_yticks([0, 0.5, 1])
axes.tick_params(axis='both', labelsize=fs, pad=10)

cbar_fixed = fig.colorbar(im, ax=axes)
cbar_fixed.set_ticks([np.min(f_integrated), np.max(f_integrated)])
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "2x1d_collisions_yphi_final_piquarter_t10.pdf")