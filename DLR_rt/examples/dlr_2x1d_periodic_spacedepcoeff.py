import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.sparse import diags
from tqdm import tqdm

from DLR_rt.src.grid import Grid_2x1d
from DLR_rt.src.initial_condition import setInitialCondition_2x1d_lr
from DLR_rt.src.integrators import PSI_lie
from DLR_rt.src.lr import LR, computeF_b_2x1d_X, computeF_b_2x1d_Y
from DLR_rt.src.util import computeD_cendiff_2x1d, computeD_upwind_2x1d


def integrate(lr0: LR, grid: Grid_2x1d, t_f: float, dt: float, 
              option: str = "lie", source = None, 
              option_scheme : str = "cendiff", option_timescheme : str = "RK4",
              option_bc : str = "standard", tol_sing_val = 1e-2, drop_tol = 1e-3, 
              tol_lattice = 1e-5):
    
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

    return lr, time, rank_adapted, rank_dropped


### Plotting

Nx = 252
Ny = 252
Nphi = 252
dt = 0.5 / Nx
r = 5
t_f = 0.1
fs = 16
savepath = "plots/"
method = "lie"
option_grid = "dd"      # Just changes how gridpoints are chosen
option_scheme = "upwind"
option_timescheme = "RK4"
option_bc = "lattice"
tol_sing_val = 1e-3 
drop_tol = 1e-4
tol_lattice = 1e-5


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


# ### Full hohlraum setup

# c_adv_vec = np.ones(Nx*Ny)
# c_adv = diags(c_adv_vec)

# c_s_matrix = np.zeros((Nx,Ny))
# c_t_matrix = np.zeros((Nx,Ny))

# # Set c_t for absorbing parts
# for i in range(Nx):
#     for j in range(Ny):

#         if j <= 0.05*Ny or j >= 0.95*Ny:    # upper and lower blocks
#             c_t_matrix[j,i] = 100

#         else:
#             if (i >= 0.95*Nx or i<=0.05*Nx and (0.25*Ny <= j <= 0.75*Ny)
#                 or (0.25*Nx <= i <= 0.75*Nx) and (0.25*Ny <= j <= 0.75*Ny)):
#                 c_t_matrix[j,i] = 100

# # Change to vector
# c_s_vec = c_s_matrix.flatten()
# c_t_vec = c_t_matrix.flatten()

# # Change to diag matrix
# c_s = diags(c_s_vec)
# c_t = diags(c_t_vec)


### Setup grid and initial condition
grid = Grid_2x1d(Nx, Ny, Nphi, r, _option_dd=option_grid, _coeff=[c_adv, c_s, c_t])
lr0 = setInitialCondition_2x1d_lr(grid, option_cond="lattice")
f0 = lr0.U @ lr0.S @ lr0.V.T


### Plot lattice
extent = [grid.X[0], grid.X[-1], grid.Y[0], grid.Y[-1]]
fig, axes = plt.subplots(1, 1, figsize=(10, 8))

im = axes.imshow(c_t_matrix, extent=extent, origin="lower", cmap="jet", 
                 interpolation="none")
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel("$y$", fontsize=fs)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([0, 0.5, 1])
axes.tick_params(axis="both", labelsize=fs, pad=10)

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


# ### Hohlraum already has inflow source
# source = np.zeros((Nx,Ny))

### Plot source
extent = [grid.X[0], grid.X[-1], grid.Y[0], grid.Y[-1]]
fig, axes = plt.subplots(1, 1, figsize=(10, 8))

im = axes.imshow(source, extent=extent, origin="lower", cmap="viridis", 
                 interpolation="none")
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel("$y$", fontsize=fs)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([0, 0.5, 1])
axes.tick_params(axis="both", labelsize=fs, pad=10)

plt.tight_layout()
plt.savefig(savepath + "source.pdf")

# Prepare source for code
source = source.flatten()[:, None]


### Run code and do the plotting
lr, time, rank_adapted, rank_dropped = integrate(lr0, grid, t_f, dt, source=source, 
                     option_scheme=option_scheme, option_timescheme=option_timescheme,
                     option_bc=option_bc, tol_sing_val=tol_sing_val, drop_tol=drop_tol, 
                     tol_lattice=tol_lattice)
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

im = axes.imshow(rho0_matrix.T, extent=extent, origin="lower", 
                 norm=LogNorm(vmin=np.exp(-7), vmax=np.max(rho0_matrix)), 
                 cmap="jet")
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel("$y$", fontsize=fs)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([0, 0.5, 1])
axes.tick_params(axis="both", labelsize=fs, pad=10)

cbar_fixed = fig.colorbar(im, ax=axes)
ticks = [np.exp(-7), np.max(rho0_matrix)]
cbar_fixed.set_ticks(ticks)
cbar_fixed.ax.set_yticklabels([f"{np.log(t):.2f}" for t in ticks])
cbar_fixed.ax.minorticks_off()
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "2x1d_rho_initial_spacedepcoeff.pdf")

fig, axes = plt.subplots(1, 1, figsize=(10, 8))

# im = axes.imshow(rho_matrix.T, extent=extent, origin="lower")
im = axes.imshow(rho_matrix.T, extent=extent, origin="lower", 
                 norm=LogNorm(vmin=np.exp(-7), vmax=np.max(rho_matrix)), 
                 cmap="jet")
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel("$y$", fontsize=fs)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([0, 0.5, 1])
axes.tick_params(axis="both", labelsize=fs, pad=10)

cbar_fixed = fig.colorbar(im, ax=axes)
ticks = [np.exp(-7), np.max(rho_matrix)]
cbar_fixed.set_ticks(ticks)
cbar_fixed.ax.set_yticklabels([f"{np.log(t):.2f}" for t in ticks])
cbar_fixed.ax.minorticks_off()
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "2x1d_rho_final_spacedepcoeff.pdf")


### Plot for rank over time

fig, axes = plt.subplots(1, 1, figsize=(10, 8))
plt.plot(time, rank_adapted)
plt.title("adapted rank " + option_bc + " simulation")
axes.set_xlabel("$t$", fontsize=fs)
axes.set_ylabel("$r(t)$", fontsize=fs)
axes.set_xlim(time[0], time[-1]) # Remove extra padding: set x-limits to data range  
plt.savefig(savepath + "1domainsim_rank_adapted.pdf")

fig, axes = plt.subplots(1, 1, figsize=(10, 8))
plt.plot(time, rank_dropped)
plt.title("dropped rank " + option_bc + " simulation")
axes.set_xlabel("$t$", fontsize=fs)
axes.set_ylabel("$r(t)$", fontsize=fs)
axes.set_xlim(time[0], time[-1]) # Remove extra padding: set x-limits to data range  
plt.savefig(savepath + "1domainsim_rank_dropped.pdf")
