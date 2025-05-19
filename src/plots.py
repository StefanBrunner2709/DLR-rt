import numpy as np
import matplotlib.pyplot as plt

class Grid:
    def __init__(self, Nx: int, Nmu: int):
        self.Nx = Nx
        self.Nmu = Nmu
        self.X = np.linspace(0.0, 1.0, Nx, endpoint=False)
        self.MU = np.linspace(-1.0, 1.0, Nmu, endpoint=False)
        self.dx = self.X[1] - self.X[0]
        self.dmu = self.MU[1] - self.MU[0]

### Plotting

fs = 16
n = 256
t_string = "t03"
t_string_load = "3000"
sigma = 8e-2
sigma_str = "8e-2"
min_col = 1.2
max_col = 2.8
cbar_ticks = [1.2, 2, 2.8]
savepath = "C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_latex_250430/periodic_cendiff_upwind/sigma8e-2/"

# Load matrix
f1 = np.load("C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_250418/Plots_classical/solution_matrices_sigma8e-2/cen_diff_" + t_string_load + ".0.npy")
f2 = np.load("C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_250418/Plots_classical/solution_matrices_sigma8e-2/upwind_" + t_string_load + ".0.npy")


# Distribution function f for different times using centered differences

grid = Grid(n, n)
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

plt.figure()
plt.imshow(f1.T, extent=extent, origin='lower', aspect=0.5, vmin=min_col, vmax=max_col)
cbar_fixed = plt.colorbar()
cbar_fixed.set_ticks(cbar_ticks)
cbar_fixed.ax.tick_params(labelsize=fs)
plt.xlabel("$x$", fontsize=fs)
plt.ylabel("$\mu$", fontsize=fs)
plt.xticks([0, 0.5, 1], fontsize=fs)
plt.yticks([-1, 0, 1], fontsize=fs)
plt.tick_params(axis='x', pad=10)
plt.tick_params(axis='y', pad=10)
plt.title("RK4, centered differences", fontsize=fs)
plt.tight_layout()
plt.savefig(savepath + "distr_funct_" + t_string + "_cendiff_sigma" + sigma_str + "_fixedaxis.pdf")

plt.figure()
plt.imshow(f1.T, extent=extent, origin='lower', aspect=0.5)
cbar_f1 = plt.colorbar()
cbar_f1.set_ticks([np.min(f1), np.max(f1)])
cbar_f1.ax.tick_params(labelsize=fs)
plt.xlabel("$x$", fontsize=fs)
plt.ylabel("$\mu$", fontsize=fs)
plt.xticks([0, 0.5, 1], fontsize=fs)
plt.yticks([-1, 0, 1], fontsize=fs)
plt.tick_params(axis='x', pad=10)
plt.tick_params(axis='y', pad=10)
plt.title("RK4, centered differences", fontsize=fs)
plt.tight_layout()
plt.savefig(savepath + "distr_funct_" + t_string + "_cendiff_sigma" + sigma_str + ".pdf")


# Distribution function f for different times using upwind

plt.figure()
plt.imshow(f2.T, extent=extent, origin='lower', aspect=0.5, vmin=min_col, vmax=max_col)
cbar_fixed = plt.colorbar()
cbar_fixed.set_ticks(cbar_ticks)
cbar_fixed.ax.tick_params(labelsize=fs)
plt.xlabel("$x$", fontsize=fs)
plt.ylabel("$\mu$", fontsize=fs)
plt.xticks([0, 0.5, 1], fontsize=fs)
plt.yticks([-1, 0, 1], fontsize=fs)
plt.tick_params(axis='x', pad=10)
plt.tick_params(axis='y', pad=10)
plt.title("Explicit Euler, upwind", fontsize=fs)
plt.tight_layout()
plt.savefig(savepath + "distr_funct_" + t_string + "_upwind_sigma" + sigma_str + "_fixedaxis.pdf")

plt.figure()
plt.imshow(f2.T, extent=extent, origin='lower', aspect=0.5)
cbar_f1 = plt.colorbar()
cbar_f1.set_ticks([np.min(f2), np.max(f2)])
cbar_f1.ax.tick_params(labelsize=fs)
plt.xlabel("$x$", fontsize=fs)
plt.ylabel("$\mu$", fontsize=fs)
plt.xticks([0, 0.5, 1], fontsize=fs)
plt.yticks([-1, 0, 1], fontsize=fs)
plt.tick_params(axis='x', pad=10)
plt.tick_params(axis='y', pad=10)
plt.title("Explicit Euler, upwind", fontsize=fs)
plt.tight_layout()
plt.savefig(savepath + "distr_funct_" + t_string + "_upwind_sigma" + sigma_str + ".pdf")