import matplotlib.pyplot as plt
import numpy as np

from DLR_rt.src.grid import Grid_2x1d
from DLR_rt.src.initial_condition import setInitialCondition_2x1d_lr
from DLR_rt.src.run_functions import integrate_1domain
from DLR_rt.src.util import setup_coeff_source_1domain

### Plotting

option_bc = "hohlraum"
r = 5
t_f = 0.1
snapshots = 2
tol_sing_val = 1e-6
drop_tol = 5e-10
tol_lattice = 2e-11

method = "lie"
option_grid = "dd"      # Just changes how gridpoints are chosen
option_scheme = "upwind"
option_timescheme = "RK4"

option_error_estimate = True

fs = 16
savepath = "plots/"


# Set amount of gridpoints according to problem
if option_bc == "lattice":
    Nx = 252
    Ny = 252
    Nphi = 252
elif option_bc == "hohlraum":
    Nx = 200
    Ny = 200
    Nphi = 200
elif option_bc == "pointsource":
    Nx = 600
    Ny = 600
    Nphi = 600

# Timestepsize
dt = 0.5 / Nx


### Setup coefficients and source
c_adv, c_s, c_t, source, c_s_matrix, c_t_matrix = setup_coeff_source_1domain(Nx, Ny, 
                                                                             option_bc)


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
lr, time, rank_adapted, rank_dropped = integrate_1domain(lr0, grid, 
                                                         t_f, dt, source=source, 
                     option_scheme=option_scheme, option_timescheme=option_timescheme,
                     option_bc=option_bc, tol_sing_val=tol_sing_val, drop_tol=drop_tol, 
                     tol_lattice=tol_lattice, snapshots=snapshots)


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



### Compare to higher rank solution
if option_error_estimate:

    ### Setup grid and initial condition
    grid_2 = Grid_2x1d(Nx, Ny, Nphi, r, _option_dd=option_grid, 
                       _coeff=[c_adv, c_s, c_t])
    lr0_2 = setInitialCondition_2x1d_lr(grid_2, option_cond="lattice")
    f0_2 = lr0_2.U @ lr0_2.S @ lr0_2.V.T

    ### Run code and do the plotting
    lr_2, time_2, rank_adapted_2, rank_dropped_2 = integrate_1domain(
                        lr0_2, grid_2, t_f, dt, source=source, 
                        option_scheme=option_scheme, 
                        option_timescheme=option_timescheme,
                        option_bc=option_bc, tol_sing_val=tol_sing_val*0.001, 
                        drop_tol=drop_tol*0.001, 
                        tol_lattice=tol_lattice*0.001, snapshots=snapshots,
                        plot_name_add = "high_rank_")


    f = lr.U @ lr.S @ lr.V.T

    f_2 = lr_2.U @ lr_2.S @ lr_2.V.T


    Frob = np.linalg.norm(f - f_2, ord='fro')

    print("Frobenius: ", Frob)

    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    plt.plot(time_2, rank_adapted_2)
    plt.title("adapted rank " + option_bc + " simulation")
    axes.set_xlabel("$t$", fontsize=fs)
    axes.set_ylabel("$r(t)$", fontsize=fs)
    axes.set_xlim(time[0], time[-1]) # Remove extra padding: set x-limits to data range
    plt.savefig(savepath + "high_rank_1domainsim_rank_adapted_2.pdf")

    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    plt.plot(time_2, rank_dropped_2)
    plt.title("dropped rank " + option_bc + " simulation")
    axes.set_xlabel("$t$", fontsize=fs)
    axes.set_ylabel("$r(t)$", fontsize=fs)
    axes.set_xlim(time[0], time[-1]) # Remove extra padding: set x-limits to data range
    plt.savefig(savepath + "high_rank_1domainsim_rank_dropped_2.pdf")
