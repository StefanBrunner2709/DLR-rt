import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from tqdm import tqdm

from DLR_rt.src.grid import Grid_2x1d
from DLR_rt.src.initial_condition import setInitialCondition_2x1d_lr
from DLR_rt.src.integrators import PSI_lie
from DLR_rt.src.lr import LR
from DLR_rt.src.util import computeD_cendiff_2x1d, computeD_upwind_2x1d


def integrate(lr0: LR, grid: Grid_2x1d, t_f: float, dt: float, 
              option: str = "lie", source = None, 
              option_scheme : str = "cendiff", option_timescheme : str = "RK4"):
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

    with tqdm(total=t_f / dt, desc="Running Simulation") as pbar:
        while t < t_f:
            pbar.update(1)

            if t + dt > t_f:
                dt = t_f - t

            if option == "lie":
                lr, grid = PSI_lie(lr, grid, dt, DX=DX, DY=DY, 
                                   dimensions="2x1d", option_coeff="space_dep", 
                                   source=source, option_scheme=option_scheme,
                                   DX_0=DX_0, DX_1=DX_1, DY_0=DY_0, DY_1=DY_1,
                                   option_timescheme=option_timescheme)

            t += dt
            time.append(t)

    return lr, time


### Plotting

Nx = 252
Ny = 252
Nphi = 252
dt = 0.02
r = 50
t_f = 0.7
fs = 16
savepath = "plots/"
method = "lie"
option_grid = "dd"      # Just changes how gridpoints are chosen
option_scheme = "upwind"
option_timescheme = "impl_Euler"


# ### To compare with constant coefficient results
# c_adv = diags(np.ones(Nx*Ny))
# c_s = diags(np.zeros(Nx*Ny))
# c_t = diags(np.zeros(Nx*Ny))


### Full lattice setup
c_adv_vec = np.ones(Nx*Ny)
c_adv = diags(c_adv_vec)

# Parameters
num_blocks = 7        # number of blocks in each row/col
block_size = int(Nx/num_blocks)        # size of each block

# Pattern of blocks
block_pattern_s = np.array([[1,1,1,1,1,1,1],
                            [1,0,1,0,1,0,1],
                            [1,1,0,1,0,1,1],
                            [1,0,1,1,1,0,1],
                            [1,1,0,1,0,1,1],
                            [1,0,1,1,1,0,1],
                            [1,1,1,1,1,1,1]])
block_pattern_t = np.array([[1,1,1,1,1,1,1],
                            [1,10,1,10,1,10,1],
                            [1,1,10,1,10,1,1],
                            [1,10,1,1,1,10,1],
                            [1,1,10,1,10,1,1],
                            [1,10,1,1,1,10,1],
                            [1,1,1,1,1,1,1]])
# block_pattern_s = np.array([[1,0,1,1,1,0,1],
#                             [1,0,1,1,1,0,1],
#                             [1,0,1,1,1,0,1],
#                             [1,0,1,1,1,0,1],
#                             [1,0,1,1,1,0,1],
#                             [1,0,1,1,1,0,1],
#                             [1,0,1,1,1,0,1]])
# block_pattern_t = np.array([[1,10,1,1,1,10,1],
#                             [1,10,1,1,1,10,1],
#                             [1,10,1,1,1,10,1],
#                             [1,10,1,1,1,10,1],
#                             [1,10,1,1,1,10,1],
#                             [1,10,1,1,1,10,1],
#                             [1,10,1,1,1,10,1]])

# Expand each block into block_size x block_size
c_s_matrix = np.kron(block_pattern_s, np.ones((block_size, block_size), dtype=int))
c_t_matrix = np.kron(block_pattern_t, np.ones((block_size, block_size), dtype=int))

# Change to vector
c_s_vec = c_s_matrix.flatten()
c_t_vec = c_t_matrix.flatten()

# Change to diag matrix
c_s = diags(c_s_vec)
c_t = diags(c_t_vec)


### Setup grid and initial condition
grid = Grid_2x1d(Nx, Ny, Nphi, r, _option_dd=option_grid, _coeff=[c_adv, c_s, c_t])
lr0 = setInitialCondition_2x1d_lr(grid, option_cond="lattice")
f0 = lr0.U @ lr0.S @ lr0.V.T


### Plot lattice
extent = [grid.X[0], grid.X[-1], grid.Y[0], grid.Y[-1]]
fig, axes = plt.subplots(1, 1, figsize=(10, 8))

im = axes.imshow(c_s_matrix, extent=extent, origin="lower", cmap="jet")
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel("y", fontsize=fs, labelpad=-5)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([0, 0.5, 1])
axes.tick_params(axis="both", labelsize=fs, pad=10)

cbar_fixed = fig.colorbar(im, ax=axes)
cbar_fixed.set_ticks([np.min(c_s_matrix), np.max(c_s_matrix)])
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "lattice1.pdf")


### Do normal 1 source
# Start with all zeros
block_matrix = np.zeros((num_blocks, num_blocks))

# Set block (4,4) to 1
block_row = 3
block_col = 3
block_matrix[block_row, block_col] = 1

# Expand to full matrix
source = np.kron(block_matrix, np.ones((block_size, block_size)))


# ### Do Gaussian source
# source = np.zeros((Nx,Ny))
# for i in range(grid.Nx):
#     for j in range(grid.Ny):
#         source[i,j] = (
#                         1
#                         / (2 * np.pi)
#                         * np.exp(-((grid.X[i] - 0.5) ** 2) / 0.02)
#                         * np.exp(-((grid.Y[j] - 0.5) ** 2) / 0.02)
#                     )


# ### Do sinus source
# source = np.zeros((Nx,Ny))
# for i in range(3*block_size, 4*block_size):
#     for j in range(3*block_size, 4*block_size):
#         source[i,j] = (
#                         np.sin((grid.X[i] - 3*block_size/Nx) * Nx/block_size * np.pi)
#                         * np.sin((grid.Y[j] - 3*block_size/Ny) 
#                                  * Ny/block_size * np.pi)
#                     )
        

### Plot source
extent = [grid.X[0], grid.X[-1], grid.Y[0], grid.Y[-1]]
fig, axes = plt.subplots(1, 1, figsize=(10, 8))

im = axes.imshow(source.T, extent=extent, origin="lower", cmap="jet")
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel("y", fontsize=fs, labelpad=-5)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([0, 0.5, 1])
axes.tick_params(axis="both", labelsize=fs, pad=10)

cbar_fixed = fig.colorbar(im, ax=axes)
cbar_fixed.set_ticks([np.min(source), np.max(source)])
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "source.pdf")

# Prepare source for code
source = source.flatten()[:, None]


### Run code and do the plotting
lr, time = integrate(lr0, grid, t_f, dt, source=source, 
                     option_scheme=option_scheme, option_timescheme=option_timescheme)
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

im = axes.imshow(np.log(rho0_matrix.T), extent=extent, origin="lower", 
                 vmin=np.log(1e-3), vmax=np.log(np.max(rho0_matrix)), 
                 cmap="jet")
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel("y", fontsize=fs, labelpad=-5)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([0, 0.5, 1])
axes.tick_params(axis="both", labelsize=fs, pad=10)

cbar_fixed = fig.colorbar(im, ax=axes)
cbar_fixed.set_ticks([np.log(1e-3), np.log(np.max(rho0_matrix))])
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "2x1d_rho_initial_spacedepcoeff.pdf")

fig, axes = plt.subplots(1, 1, figsize=(10, 8))

# im = axes.imshow(rho_matrix.T, extent=extent, origin="lower")
im = axes.imshow(np.log(rho_matrix.T), extent=extent, origin="lower", 
                 vmin=np.log(1e-3), vmax=np.log(np.max(rho_matrix)), 
                 cmap="jet")
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel("y", fontsize=fs, labelpad=-5)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([0, 0.5, 1])
axes.tick_params(axis="both", labelsize=fs, pad=10)

cbar_fixed = fig.colorbar(im, ax=axes)
# cbar_fixed.set_ticks([np.min(rho_matrix), np.max(rho_matrix)])
cbar_fixed.set_ticks([np.log(1e-3), np.log(np.max(rho_matrix))])
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "2x1d_rho_final_spacedepcoeff.pdf")
