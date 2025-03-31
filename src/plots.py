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

# Load matrix
f1 = np.load("C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_after_correction/Plots_classical/solution_matrices/_cen_diff_2000.0_.npy")
f2 = np.load("C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_after_correction/Plots_classical/solution_matrices/_upwind_2000.0_.npy")


# Distribution function f for different times using centered differences

fs = 16
n = 256
t_final = 0.05
t_string = "t02"
grid = Grid(n, n)
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]


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
