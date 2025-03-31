import numpy as np
import matplotlib.pyplot as plt

''' Programming steps

1) Generate grid for function f with variabels x [0,1] and mu [-1,1] 

2) Choose initial condition

3) Make quick, brute force implementation to update solution for time steps

'''

### Generate grid

class Grid:
    def __init__(self, Nx: int, Nmu: int):
        self.Nx = Nx
        self.Nmu = Nmu
        self.X = np.linspace(0.0, 1.0, Nx, endpoint=False)
        self.MU = np.linspace(-1.0, 1.0, Nmu, endpoint=False)
        self.dx = self.X[1] - self.X[0]
        self.dmu = self.MU[1] - self.MU[0]

### Set initial condition

def setInitialCondition(grid: Grid, option: str) -> np.ndarray:
    f0 = np.zeros((grid.Nx, grid.Nmu))
    sigma = 1
    if option == "no_mu":
        xx = 1/(np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(((grid.X-0.5)**2)/(2*sigma))**2)
        f0[:] = xx
    elif option == "with_mu":
        xx = 1/(2 * np.pi * sigma**2) * np.exp(-((grid.X-0.5)**2)/(2*sigma**2))
        vv = np.exp(-(np.abs(grid.MU)**2)/(16*sigma**2))
        f0 = np.outer(xx, vv)
    return f0

### Implementation of solver

def integrate(f0: np.ndarray, grid: Grid, t_f: float, dt: float, epsilon: float, option: str, tol):
    f = np.copy(f0)
    t = 0
    time = [0]
    rank = []
    rank.append(np.linalg.matrix_rank(f, tol))
    while t < t_f:
        if (t + dt > t_f):
            dt = t_f - t

        f = f + dt * rhs(f, grid, epsilon, option)
        t += dt

        time.append(t)
        rank.append(np.linalg.matrix_rank(f, tol))

        # Clock for progress
        if np.round(t*1000) % 100 == 0:
            print(f"Timestep: {t}")

        # Write values to file
        if np.round(t*1000) % 1000 == 0:
            time_string = str(np.round(t*1000))
            np.save("C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_after_correction/Plots_classical/_" + option + "_" + time_string + "_.npy", f)

    return f, time, rank

def rhs(f: np.ndarray, grid: Grid, epsilon: float, option: str):
    # integrate over mu to get rho
    rho = np.zeros((grid.Nx, grid.Nmu))
    rho[:] = (1/np.sqrt(2)) * np.trapz(f, grid.MU, axis=1)

    # do cen diff and rest
    res = np.zeros((grid.Nx, grid.Nmu))
    if option == "cen_diff":
        for k in range(0, grid.Nmu):
            for l in range(1, grid.Nx-1):
                res[l, k] = -(1/epsilon) * grid.MU[k] * (f[l+1, k] - f[l-1, k]) / (2 * grid.dx) + (1/epsilon**2) * ((1/np.sqrt(2)) * rho[l, k] - f[l, k])

            res[0, k] = -(1/epsilon) * grid.MU[k] * (f[1, k] - f[grid.Nx-1, k]) / (2 * grid.dx) + (1/epsilon**2) * ((1/np.sqrt(2)) * rho[0, k] - f[0, k])
            res[grid.Nx-1, k] = -(1/epsilon) * grid.MU[k] * (f[0, k] - f[grid.Nx-2, k]) / (2 * grid.dx) + (1/epsilon**2) * ((1/np.sqrt(2)) * rho[grid.Nx-1, k] - f[grid.Nx-1, k])
    elif option == "upwind":
        for k in range(0, grid.Nmu):
            if grid.MU[k] >= 0:
                for l in range(1, grid.Nx):
                    res[l, k] = -(1/epsilon) * grid.MU[k] * (f[l, k] - f[l-1, k]) / (grid.dx) + (1/epsilon**2) * ((1/np.sqrt(2)) * rho[l, k] - f[l, k])
                    res[0, k] = -(1/epsilon) * grid.MU[k] * (f[0, k] - f[grid.Nx-1, k]) / (grid.dx) + (1/epsilon**2) * ((1/np.sqrt(2)) * rho[0, k] - f[0, k])
            elif grid.MU[k] < 0:
                for l in range(0, grid.Nx-1):
                    res[l, k] = -(1/epsilon) * grid.MU[k] * (f[l+1, k] - f[l, k]) / (grid.dx) + (1/epsilon**2) * ((1/np.sqrt(2)) * rho[l, k] - f[l, k])
                    res[grid.Nx-1, k] = -(1/epsilon) * grid.MU[k] * (f[0, k] - f[grid.Nx-1, k]) / (grid.dx) + (1/epsilon**2) * ((1/np.sqrt(2)) * rho[grid.Nx-1, k] - f[grid.Nx-1, k])
    return(res)      

'''
# First simulation
### Check initial condition

grid = Grid(64, 64)
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]
f0 = setInitialCondition(grid, "with_mu")
plt.subplot(1, 3, 1)
plt.imshow(f0.T, extent=extent, origin='lower')
plt.colorbar(orientation='horizontal', pad=0.08, fraction=0.035)
plt.xlabel("$x$")
plt.ylabel("mu")
plt.title("inital values")

### Do simulations

f1 = integrate(f0, grid, 1, 1e-3, 1, "upwind", 1e-2)[0]
plt.subplot(1, 3, 2)
plt.imshow(f1.T, extent=extent, origin='lower')
plt.colorbar(orientation='horizontal', pad=0.08, fraction=0.035)
plt.xlabel("$x$")
plt.ylabel("mu")
plt.title("upwind")

f2 = integrate(f0, grid, 1, 1e-3, 1, "cen_diff", 1e-2)[0]
plt.subplot(1, 3, 3)
plt.imshow(f2.T, extent=extent, origin='lower')
plt.colorbar(orientation='horizontal', pad=0.08, fraction=0.035)
plt.xlabel("$x$")
plt.ylabel("mu")
plt.title("cen_diff")
plt.show()
'''
'''
# Print singular values over time
grid = Grid(64, 64)
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]
f0 = setInitialCondition(grid, "with_mu")
tfinal = np.linspace(0,1,11)
for t in tfinal:
    f = integrate(f0, grid, t, 1e-3, 1, "cen_diff", 1e-2)[0]
    print(np.linalg.svd(f)[1])

# Plot rank of solution over time
grid = Grid(64, 64)
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]
f0 = setInitialCondition(grid, "with_mu")
tfinal = np.linspace(0,10,11)
f_rank = [np.linalg.matrix_rank(integrate(f0, grid, t, 1e-3, 1, "cen_diff", 1e-2)[0], tol=1e-4) for t in tfinal]
fig, ax = plt.subplots()
ax.plot(tfinal, f_rank)
ax.set_xlabel("$t$")
ax.set_ylabel("rank $r(t)$")
plt.show()
'''
'''
# Plot rank of solution over time
grid = Grid(64, 64)
f0 = setInitialCondition(grid, "with_mu")
res = integrate(f0, grid, 10, 1e-3, 1, "cen_diff", 1e-2)
time_array = res[1]
rank_array = res[2]
fig, ax = plt.subplots()
ax.plot(time_array, rank_array)
ax.set_xlabel("$t$")
ax.set_ylabel("rank $r(t)$")
plt.show()
'''

# Plotting after correction

fs = 16
n = 256
t_final = 0.5
t_string = "t005"


# Inital condition plot
'''
grid = Grid(n, n)
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]
f0 = setInitialCondition(grid, "with_mu")
plt.imshow(f0.T, extent=extent, origin='lower', aspect=0.5, vmin=0.13, vmax=0.16)
cbar = plt.colorbar()
cbar.set_ticks([0.13, 0.145, 0.16])
cbar.ax.tick_params(labelsize=fs)
plt.xlabel("$x$", fontsize=fs)
plt.ylabel("$\mu$", fontsize=fs)
plt.xticks([0, 0.5, 1], fontsize=fs)
plt.yticks([-1, 0, 1], fontsize=fs)
plt.tick_params(axis='x', pad=10)  # Moves x-axis labels farther
plt.tick_params(axis='y', pad=10)  # Moves y-axis labels farther
plt.title("$f(t=0,x,\mu)$", fontsize=fs)
plt.tight_layout()
plt.savefig("C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_after_correction/Plots_classical/init_cond_sigma1.pdf")
'''


# Distribution function f for different times using centered differences

grid = Grid(n, n)
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]
f0 = setInitialCondition(grid, "with_mu")
f1_all = integrate(f0, grid, t_final, 1e-4, 1, "cen_diff", 1e-4)
f1 = f1_all[0]

# Save resulting matrix
np.save("C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_after_correction/Plots_classical/f1_" + t_string + "_.npy", f1)

plt.figure()
plt.imshow(f1.T, extent=extent, origin='lower', aspect=0.5, vmin=0.13, vmax=0.16)
cbar_fixed = plt.colorbar()
cbar_fixed.set_ticks([0.13, 0.145, 0.16])
cbar_fixed.ax.tick_params(labelsize=fs)
plt.xlabel("$x$", fontsize=fs)
plt.ylabel("$\mu$", fontsize=fs)
plt.xticks([0, 0.5, 1], fontsize=fs)
plt.yticks([-1, 0, 1], fontsize=fs)
plt.tick_params(axis='x', pad=10)
plt.tick_params(axis='y', pad=10)
plt.title("RK4, centered differences", fontsize=fs)
plt.tight_layout()
plt.savefig("C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_after_correction/Plots_classical/distr_funct_" + t_string + "_cendiff_sigma1_fixedaxis.pdf")

plt.figure()
plt.imshow(f1.T, extent=extent, origin='lower', aspect=0.5)
cbar_f1 = plt.colorbar()
cbar_f1.set_ticks([np.ceil(np.min(f1)*10000)/10000, np.floor(np.max(f1)*10000)/10000])
cbar_f1.ax.tick_params(labelsize=fs)
plt.xlabel("$x$", fontsize=fs)
plt.ylabel("$\mu$", fontsize=fs)
plt.xticks([0, 0.5, 1], fontsize=fs)
plt.yticks([-1, 0, 1], fontsize=fs)
plt.tick_params(axis='x', pad=10)
plt.tick_params(axis='y', pad=10)
plt.title("RK4, centered differences", fontsize=fs)
plt.tight_layout()
plt.savefig("C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_after_correction/Plots_classical/distr_funct_" + t_string + "_cendiff_sigma1.pdf")

# Distribution function f for different times using upwind

f2_all = integrate(f0, grid, t_final, 1e-3, 1, "upwind", 1e-2)
f2 = f2_all[0]

# Save resulting matrix
np.save("C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_after_correction/Plots_classical/f2_" + t_string + "_.npy", f2)

plt.figure()
plt.imshow(f2.T, extent=extent, origin='lower', aspect=0.5, vmin=0.13, vmax=0.16)
cbar_fixed = plt.colorbar()
cbar_fixed.set_ticks([0.13, 0.145, 0.16])
cbar_fixed.ax.tick_params(labelsize=fs)
plt.xlabel("$x$", fontsize=fs)
plt.ylabel("$\mu$", fontsize=fs)
plt.xticks([0, 0.5, 1], fontsize=fs)
plt.yticks([-1, 0, 1], fontsize=fs)
plt.tick_params(axis='x', pad=10)
plt.tick_params(axis='y', pad=10)
plt.title("Explicit Euler, upwind", fontsize=fs)
plt.tight_layout()
plt.savefig("C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_after_correction/Plots_classical/distr_funct_" + t_string + "_upwind_sigma1_fixedaxis.pdf")

plt.figure()
plt.imshow(f2.T, extent=extent, origin='lower', aspect=0.5)
cbar_f1 = plt.colorbar()
cbar_f1.set_ticks([np.ceil(np.min(f2)*10000)/10000, np.floor(np.max(f2)*1000)/1000])
cbar_f1.ax.tick_params(labelsize=fs)
plt.xlabel("$x$", fontsize=fs)
plt.ylabel("$\mu$", fontsize=fs)
plt.xticks([0, 0.5, 1], fontsize=fs)
plt.yticks([-1, 0, 1], fontsize=fs)
plt.tick_params(axis='x', pad=10)
plt.tick_params(axis='y', pad=10)
plt.title("Explicit Euler, upwind", fontsize=fs)
plt.tight_layout()
plt.savefig("C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_after_correction/Plots_classical/distr_funct_" + t_string + "_upwind_sigma1.pdf")

# Rank plot for multiple times
'''
fig, ax = plt.subplots()
ax.plot(f1_all[1], f1_all[2])
ax.set_xlabel("$t$", fontsize=fs)
ax.set_ylabel("rank $r(t)$", fontsize=fs)
ax.tick_params(axis='both', labelsize=fs)
plt.tight_layout()
plt.savefig("C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_after_correction/Plots_classical/rank_over_time_cendiff_sigma1_tol1e-4.pdf")


fig, ax = plt.subplots()
ax.plot(f2_all[1], f2_all[2])
ax.set_xlabel("$t$", fontsize=fs)
ax.set_ylabel("rank $r(t)$", fontsize=fs)
ax.tick_params(axis='both', labelsize=fs)
plt.tight_layout()
plt.savefig("C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_after_correction/Plots_classical/rank_over_time_upwind_sigma1_tol1e-2.pdf")
'''