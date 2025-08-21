import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from tqdm import tqdm

from DLR_rt.src.grid import Grid_2x1d
from DLR_rt.src.initial_condition import setInitialCondition_2x1d_lr
from DLR_rt.src.integrators import PSI_lie
from DLR_rt.src.lr import LR
from DLR_rt.src.util import computeD_cendiff_2x1d


def integrate(lr0: LR, grid: Grid_2x1d, t_f: float, dt: float, 
              option: str = "lie", source = None):
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
                                   dimensions="2x1d", option_coeff="space_dep", 
                                   source=source)

            t += dt
            time.append(t)

    return lr, time


### Plotting

Nx = 128
Ny = 128
Nphi = 128
dt = 1e-3
r = 128
t_f = 0.1
fs = 16
savepath = "plots/"
method = "lie"
option_grid = "dd"      # Just changes how gridpoints are chosen

# ### To compare with constant coefficient results
# c_adv = diags(np.ones(Nx*Ny))
# c_s = diags(np.ones(Nx*Ny))
# c_t = diags(np.ones(Nx*Ny))

# ### For question 1
# s = np.ones(Nx*Ny)
# s[int(Nx*Ny/8*7):]=0
# t = np.ones(Nx*Ny)
# t[int(Nx*Ny/8*7):]=10
# c_s = diags(s)
# c_t = diags(t)


# ### Simplified lattice test setup
# c_adv_vec = np.ones(Nx*Ny)
# c_adv = diags(c_adv_vec)

# # Block and grid sizes
# num_blocks = 8
# block_size = int(Nx/num_blocks)

# # The pattern of numbers in each row of blocks
# pattern_s = [1, 0, 1, 0, 1, 0, 1, 1]
# pattern_t = [1, 10, 1, 10, 1, 10, 1, 1]

# # Create the 8x8 block matrix with pattern repeated in each row
# c_s_block_matrix = np.array([pattern_s] * num_blocks)
# c_t_block_matrix = np.array([pattern_t] * num_blocks)

# # Expand each block to be block_size x block_size
# c_s_matrix = np.kron(c_s_block_matrix, np.ones((block_size, block_size), dtype=int))
# c_t_matrix = np.kron(c_t_block_matrix, np.ones((block_size, block_size), dtype=int))

# # Change to vector
# c_s_vec = c_s_matrix.flatten()
# c_t_vec = c_t_matrix.flatten()

# # Change to diag matrix
# c_s = diags(c_s_vec)
# c_t = diags(c_t_vec)

### Full lattice setup

c_adv_vec = np.ones(Nx*Ny)
c_adv = diags(c_adv_vec)

# Parameters
num_blocks = 8        # number of blocks in each row/col
block_size = int(Nx/num_blocks)        # size of each block

# Pattern of blocks
block_pattern_s = np.array([[1,1,1,1,1,1,1,1],
                            [1,0,1,0,1,0,1,1],
                            [1,1,0,1,0,1,1,1],
                            [1,0,1,0,1,0,1,1],
                            [1,1,0,1,0,1,1,1],
                            [1,0,1,1,1,0,1,1],
                            [1,1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1]])
block_pattern_t = np.array([[1,1,1,1,1,1,1,1],
                            [1,10,1,10,1,10,1,1],
                            [1,1,10,1,10,1,1,1],
                            [1,10,1,10,1,10,1,1],
                            [1,1,10,1,10,1,1,1],
                            [1,10,1,1,1,10,1,1],
                            [1,1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1]])

# Expand each block into block_size x block_size
c_s_matrix = np.kron(block_pattern_s, np.ones((block_size, block_size), dtype=int))
c_t_matrix = np.kron(block_pattern_t, np.ones((block_size, block_size), dtype=int))

# Change to vector
c_s_vec = c_s_matrix.flatten()
c_t_vec = c_t_matrix.flatten()

# Change to diag matrix
c_s = diags(c_s_vec)
c_t = diags(c_t_vec)

# c_adv = diags(np.zeros(Nx*Ny))
# c_s = diags(np.zeros(Nx*Ny))
# c_t = diags(np.zeros(Nx*Ny))

# ### To test
# c_adv = diags(np.zeros(Nx*Ny))
# c_s = diags(np.zeros(Nx*Ny))
# c_t = diags(np.zeros(Nx*Ny))

# num_blocks = 8
# block_size = int(Nx/num_blocks)  

### Calculate source

# Start with all zeros
block_matrix = np.zeros((num_blocks, num_blocks))

# Set block (4,4) to 1
block_row = 3
block_col = 3
block_matrix[block_row, block_col] = 1

# Expand to full matrix
matrix = np.kron(block_matrix, np.ones((block_size, block_size)))

source = matrix.flatten()[:, None]

### Do the plotting
grid = Grid_2x1d(Nx, Ny, Nphi, r, _option_dd=option_grid, _coeff=[c_adv, c_s, c_t])
lr0 = setInitialCondition_2x1d_lr(grid, option_cond="lattice")
f0 = lr0.U @ lr0.S @ lr0.V.T
lr, time = integrate(lr0, grid, t_f, dt, source=source)
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
