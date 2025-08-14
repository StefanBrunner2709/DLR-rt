import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from tqdm import tqdm

from DLR_rt.src.grid import Grid_2x1d
from DLR_rt.src.initial_condition import setInitialCondition_2x1d_lr
from DLR_rt.src.integrators import PSI_lie
from DLR_rt.src.lr import LR
from DLR_rt.src.util import computeD_cendiff_2x1d


def integrate(lr0: LR, grid: Grid_2x1d, t_f: float, dt: float, option: str = "lie"):
    lr = lr0
    t = 0
    time = []
    time.append(t)

    DX, DY = computeD_cendiff_2x1d(grid, "no_dd")

    with tqdm(total=t_f / dt, desc="Running Simulation") as pbar:
        while t < t_f:
            pbar.update(1)

            if t + dt > t_f:
                dt = t_f - t

            if option == "lie":
                lr, grid = PSI_lie(lr, grid, dt, DX=DX, DY=DY, 
                                   dimensions="2x1d", option_coeff="space_dep")

            t += dt
            time.append(t)

    return lr, time


### Plotting

Nx = 32
Ny = 32
Nphi = 32
dt = 1e-3
r = 5
t_f = 0.1
fs = 16
savepath = "plots/"
method = "lie"
option_grid = "dd"      # Just changes how gridpoints are chosen

### To compare with constant coefficient results
c_adv = diags(np.ones(Nx*Ny))
c_s = diags(np.ones(Nx*Ny))
c_t = diags(np.ones(Nx*Ny))

# ### Simplified lattice test setup
# c_adv = np.ones(Nx*Ny)

# # Block and grid sizes
# num_blocks = 8
# block_size = int(Nx/num_blocks)

# # The pattern of numbers in each row of blocks
# pattern_s = [1, 0, 1, 0, 0, 1, 0, 1]
# pattern_t = [1, 10, 1, 10, 10, 1, 10, 1]

# # Create the 8x8 block matrix with pattern repeated in each row
# c_s_block_matrix = np.array([pattern_s] * num_blocks)
# c_t_block_matrix = np.array([pattern_t] * num_blocks)

# # Expand each block to be block_size x block_size
# c_s_matrix = np.kron(c_s_block_matrix, np.ones((block_size, block_size), dtype=int))
# c_t_matrix = np.kron(c_t_block_matrix, np.ones((block_size, block_size), dtype=int))

# # Change to vector
# c_s = c_s_matrix.flatten()
# c_t = c_t_matrix.flatten()

### Do the plotting
grid = Grid_2x1d(Nx, Ny, Nphi, r, _option_dd=option_grid, _coeff=[c_adv, c_s, c_t])
lr0 = setInitialCondition_2x1d_lr(grid)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr, time = integrate(lr0, grid, t_f, dt)
f = lr.U @ lr.S @ lr.V.T

rho0 = (
    f0 @ np.ones(grid.Nphi)
) * grid.dphi  # This is now a vector, only depends on x and y
rho = (
    f @ np.ones(grid.Nphi)
) * grid.dphi  # This is now a vector, only depends on x and y

rho0_matrix = rho0.reshape((grid.Nx, grid.Ny), order="F")
rho_matrix = rho.reshape((grid.Nx, grid.Ny), order="F")

extent = [grid.X[0], grid.X[-1], grid.Y[0], grid.Y[-1]]

fig, axes = plt.subplots(1, 1, figsize=(10, 8))

im = axes.imshow(rho0_matrix.T, extent=extent, origin="lower")
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel("y", fontsize=fs, labelpad=-5)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([0, 0.5, 1])
axes.tick_params(axis="both", labelsize=fs, pad=10)

cbar_fixed = fig.colorbar(im, ax=axes)
cbar_fixed.set_ticks([np.min(rho0_matrix), np.max(rho0_matrix)])
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "2x1d_rho_initial_spacedepcoeff.pdf")

fig, axes = plt.subplots(1, 1, figsize=(10, 8))

im = axes.imshow(rho_matrix.T, extent=extent, origin="lower")
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel("y", fontsize=fs, labelpad=-5)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([0, 0.5, 1])
axes.tick_params(axis="both", labelsize=fs, pad=10)

cbar_fixed = fig.colorbar(im, ax=axes)
cbar_fixed.set_ticks([np.min(rho_matrix), np.max(rho_matrix)])
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "2x1d_rho_final_spacedepcoeff.pdf")
