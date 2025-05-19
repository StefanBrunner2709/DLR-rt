import numpy as np
import matplotlib.pyplot as plt

### Generate grid

class Grid:
    def __init__(self, Nx: int, Nmu: int):
        self.Nx = Nx
        self.Nmu = Nmu
        self.X = np.linspace(0.0, 1.0, Nx, endpoint=False)
        self.MU = np.linspace(-1.0, 1.0, Nmu, endpoint=True)
        self.dx = self.X[1] - self.X[0]
        self.dmu = self.MU[1] - self.MU[0]

### Set initial condition

def setInitialCondition(grid: Grid, option: str, sigma: float) -> np.ndarray:
    f0 = np.zeros((grid.Nx, grid.Nmu))
    if option == "no_mu":
        xx = 1/(np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(((grid.X-0.5)**2)/(2*sigma))**2)
        f0[:] = xx
    elif option == "with_mu":
        xx = 1/(2 * np.pi * sigma**2) * np.exp(-((grid.X-0.5)**2)/(2*sigma**2))
        vv = np.exp(-(np.abs(grid.MU)**2)/(16*sigma**2))
        f0 = np.outer(xx, vv)
    return f0

### Implementation of RK4

def RK4(f, rhs, dt):
    b_coeff = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
    
    k_coeff0 = rhs(f)
    k_coeff1 = rhs(f + dt * 0.5 * k_coeff0)
    k_coeff2 = rhs(f + dt * 0.5 * k_coeff1)
    k_coeff3 = rhs(f + dt * k_coeff2)

    return b_coeff[0] * k_coeff0 + b_coeff[1] * k_coeff1 + b_coeff[2] * k_coeff2 + b_coeff[3] * k_coeff3

### Implementation of solver

def integrate(f0: np.ndarray, grid: Grid, t_f: float, dt: float, epsilon: float, option: str, tol: float = 1e-2, tol2: float = 1e-4):
    f = np.copy(f0)
    t = 0
    time = [0]
    rank = []
    rank2 = []
    rank.append(np.linalg.matrix_rank(f, tol))
    rank2.append(np.linalg.matrix_rank(f, tol2))
    while t < t_f:
        if (t + dt > t_f):
            dt = t_f - t

        if option == "upwind":  # For upwind use Euler
            f = f + dt * rhs(f, grid, epsilon, option)
            t += dt
        
        elif option == "cen_diff":  # For cen_diff use RK4
            f += dt * RK4(f, lambda f: rhs(f, grid, epsilon, option), dt)
            t += dt

        time.append(t)
        rank.append(np.linalg.matrix_rank(f, tol))
        rank2.append(np.linalg.matrix_rank(f, tol2))

        ### Clock for progress
        if np.round(t*1000) % 100 == 0:
            print(f"Timestep: {t}")

        ''' ### Write values to file
        if np.round(t*1000) % 50 == 0:
            time_string = str(np.round(t*1000))
            np.save("C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_250418/Plots_classical/solution_matrices/" + option + "_" + time_string + ".npy", f)
        '''
    return f, time, rank, rank2

### Implementation of cen_diff and upwind

def rhs(f: np.ndarray, grid: Grid, epsilon: float, option: str):
    # integrate over mu to get rho
    rho = np.zeros((grid.Nx, grid.Nmu))
    rho[:] = (1/np.sqrt(2)) * np.trapezoid(f, grid.MU, axis=1)

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


### Plotting after correction

fs = 16
n = 256
t_final = 3.0
t_string = "t03"
sigma = 1.0
savepath = "C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_latex_250430/"


''' ### Inital condition plot
grid = Grid(n, n)
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]
f0 = setInitialCondition(grid, "with_mu", sigma)
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
plt.savefig(savepath + "init_cond_sigma" + str(sigma) + ".pdf")


### Distribution function f for different times using centered differences

grid = Grid(n, n)
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]
f0 = setInitialCondition(grid, "with_mu", sigma)
f1_all = integrate(f0, grid, t_final, 1e-3, 1, "cen_diff")
f1 = f1_all[0]

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
plt.savefig(savepath + "distr_funct_" + t_string + "_cendiff_sigma" + str(sigma) + "_fixedaxis.pdf")

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
plt.savefig(savepath + "distr_funct_" + t_string + "_cendiff_sigma" + str(sigma) + ".pdf")


### Distribution function f for different times using upwind

f2_all = integrate(f0, grid, t_final, 1e-3, 1, "upwind")
f2 = f2_all[0]

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
plt.savefig(savepath + "distr_funct_" + t_string + "_upwind_sigma" + str(sigma) + "_fixedaxis.pdf")

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
plt.savefig(savepath + "distr_funct_" + t_string + "_upwind_sigma" + str(sigma) + ".pdf")


### Rank plot for multiple times

fig, ax = plt.subplots()
ax.plot(f1_all[1], f1_all[2])
ax.set_xlabel("$t$", fontsize=fs)
ax.set_ylabel("rank $r(t)$", fontsize=fs)
ax.tick_params(axis='both', labelsize=fs)
plt.tight_layout()
plt.savefig(savepath + "rank_over_time_cendiff_sigma" + str(sigma) + "_tol1e-2.pdf")

fig, ax = plt.subplots()
ax.plot(f1_all[1], f1_all[3])
ax.set_xlabel("$t$", fontsize=fs)
ax.set_ylabel("rank $r(t)$", fontsize=fs)
ax.tick_params(axis='both', labelsize=fs)
plt.tight_layout()
plt.savefig(savepath + "rank_over_time_cendiff_sigma" + str(sigma) + "_tol1e-4.pdf")

fig, ax = plt.subplots()
ax.plot(f2_all[1], f2_all[2])
ax.set_xlabel("$t$", fontsize=fs)
ax.set_ylabel("rank $r(t)$", fontsize=fs)
ax.tick_params(axis='both', labelsize=fs)
plt.tight_layout()
plt.savefig(savepath + "rank_over_time_upwind_sigma" + str(sigma) + "_tol1e-2.pdf")

fig, ax = plt.subplots()
ax.plot(f2_all[1], f2_all[3])
ax.set_xlabel("$t$", fontsize=fs)
ax.set_ylabel("rank $r(t)$", fontsize=fs)
ax.tick_params(axis='both', labelsize=fs)
plt.tight_layout()
plt.savefig(savepath + "rank_over_time_upwind_sigma" + str(sigma) + "_tol1e-4.pdf")
'''