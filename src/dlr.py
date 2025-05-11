import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class LR:
    def __init__(self, U, S, V):
        self.U = U
        self.S = S
        self.V = V

class Grid:
    def __init__(self, Nx, Nmu, r):
        self.Nx = Nx
        self.Nmu = Nmu
        self.r = r
        self.X = np.linspace(0.0, 1.0, Nx, endpoint=False)
        self.MU = np.linspace(-1.0, 1.0, Nmu, endpoint=True)
        self.dx = self.X[1] - self.X[0]
        self.dmu = self.MU[1] - self.MU[0]

def setInitialCondition(grid, sigma):
    U = np.zeros((grid.Nx, grid.r))
    V = np.zeros((grid.Nmu, grid.r))
    S = np.zeros((grid.r, grid.r))

    U[:, 0] = 1/(2 * np.pi * sigma**2) * np.exp(-((grid.X-0.5)**2)/(2*sigma**2))
    V[:, 0] = np.exp(-(np.abs(grid.MU)**2)/(16*sigma**2))
    S[0, 0] = 1.0

    U_ortho, R_U = np.linalg.qr(U, mode="reduced")
    V_ortho, R_V = np.linalg.qr(V, mode="reduced")
    S_ortho = R_U @ S @ R_V.T

    lr = LR(U_ortho, S_ortho, V_ortho)
    return lr

def RK4(f, rhs, dt):
    b_coeff = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
    
    k_coeff0 = rhs(f)
    k_coeff1 = rhs(f + dt * 0.5 * k_coeff0)
    k_coeff2 = rhs(f + dt * 0.5 * k_coeff1)
    k_coeff3 = rhs(f + dt * k_coeff2)

    return b_coeff[0] * k_coeff0 + b_coeff[1] * k_coeff1 + b_coeff[2] * k_coeff2 + b_coeff[3] * k_coeff3


def computeC(lr, grid):

    C1 = (lr.V.T @ np.diag(grid.MU) @ lr.V) * grid.dmu

    C2 = (lr.V.T @ np.ones((grid.Nmu,grid.Nmu))).T * grid.dmu

    ### Alternative option, faster but harder to understand
    # muV = grid.MU[:, None] * lr.V
    # C1 = lr.V.T @ muV * grid.dmu
    # C2 = lr.V * grid.dmu

    return C1, C2

def computeB(L, grid):

    B1 = (L.T @ np.ones((grid.Nmu,grid.Nmu))).T * grid.dmu

    return B1

def computeD(lr, grid):

    dxU = 0.5 * (np.roll(lr.U, -1, axis=0) - np.roll(lr.U, 1, axis=0)) / grid.dx
    D1 = lr.U.T @ dxU * grid.dx

    return D1

def Kstep(K, C1, C2, grid):
    dxK = 0.5 * (np.roll(K, -1, axis=0) - np.roll(K, 1, axis=0)) / grid.dx    
    rhs = - dxK @ C1 + 0.5 * K @ C2.T @ C2 - K
    return rhs

def Sstep(S, C1, C2, D1):
    rhs = D1 @ S @ C1 - 0.5 * S @ C2.T @ C2 + S
    return rhs

def Lstep(L, D1, B1, grid):
    rhs = - np.diag(grid.MU) @ L @ D1.T + 0.5 * B1 - L
    return rhs

def compute_mass(lr, grid):
    K = lr.U @ lr.S
    rho = K @ np.trapz(lr.V, dx=grid.dmu, axis=0).T
    M = np.trapz(rho, dx=grid.dx, axis=0)
    return M

def integrate(lr0: LR, grid: Grid, t_f: float, dt: float, option: str = "lie"):
    lr = lr0
    t = 0
    time = []
    time.append(t)
    # lr_array = []     # only needed for error plots
    # lr_array.append(lr.U @ lr.S @ lr.V.T)
    # mass_array = []     # only needed for mass plots
    # mass_array.append(compute_mass(lr, grid))

    with tqdm(total=t_f/dt, desc="Running Simulation") as pbar:

        while t < t_f:

            pbar.update(1)

            if (t + dt > t_f):
                dt = t_f - t

            t += dt
            time.append(t)

            if option=="lie":
                # K step
                C1, C2 = computeC(lr, grid)
                K = lr.U @ lr.S
                K += dt * RK4(K, lambda K: Kstep(K, C1, C2, grid), dt)
                lr.U, lr.S = np.linalg.qr(K, mode="reduced")
                lr.U /= np.sqrt(grid.dx)
                lr.S *= np.sqrt(grid.dx)

                # S step
                D1 = computeD(lr, grid)
                lr.S += dt * RK4(lr.S, lambda S: Sstep(S, C1, C2, D1), dt)

                # L step
                L = lr.V @ lr.S.T
                B1 = computeB(L, grid)
                L += dt * RK4(L, lambda L: Lstep(L, D1, B1, grid), dt)
                lr.V, St = np.linalg.qr(L, mode="reduced")
                lr.S = St.T
                lr.V /= np.sqrt(grid.dmu)
                lr.S *= np.sqrt(grid.dmu)
            
            if option=="strang":
                # 1/2 K step
                C1, C2 = computeC(lr, grid)
                K = lr.U @ lr.S
                K += 0.5 * dt * RK4(K, lambda K: Kstep(K, C1, C2, grid), 0.5 * dt)
                lr.U, lr.S = np.linalg.qr(K, mode="reduced")
                lr.U /= np.sqrt(grid.dx)
                lr.S *= np.sqrt(grid.dx)

                # 1/2 S step
                D1 = computeD(lr, grid)
                lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep(S, C1, C2, D1), 0.5 * dt)

                # L step
                L = lr.V @ lr.S.T
                B1 = computeB(L, grid)
                L += dt * RK4(L, lambda L: Lstep(L, D1, B1, grid), dt)
                lr.V, St = np.linalg.qr(L, mode="reduced")
                lr.S = St.T
                lr.V /= np.sqrt(grid.dmu)
                lr.S *= np.sqrt(grid.dmu)

                # 1/2 S step
                C1, C2 = computeC(lr, grid)     # need to recalculate C1 and C2 because we changed V in L step     
                lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep(S, C1, C2, D1), 0.5 * dt)

                # 1/2 K step
                K = lr.U @ lr.S
                K += 0.5 * dt * RK4(K, lambda K: Kstep(K, C1, C2, grid), 0.5 * dt)
                lr.U, lr.S = np.linalg.qr(K, mode="reduced")
                lr.U /= np.sqrt(grid.dx)
                lr.S *= np.sqrt(grid.dx)

            # lr_array.append(lr.U @ lr.S @ lr.V.T)     # again only for error plots

            # mass_array.append(compute_mass(lr, grid))       # again only for mass plots

    return lr, time


''' ### Just one plot for certain rank and certain time

Nx = 256
Nmu = 256
dt = 1e-3
r = 16
t_f = 3.0
t_string = "03"
sigma = 1
fs = 16
savepath = "C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_latex_250430/periodic_dlr/sigma1/"
method = "strang"

fig, axes = plt.subplots(1, 1, figsize=(10, 8))

grid = Grid(Nx, Nmu, r)
lr0 = setInitialCondition(grid, sigma)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr, time = integrate(lr0, grid, t_f, dt, method)
f = lr.U @ lr.S @ lr.V.T
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

im = axes.imshow(f.T, extent=extent, origin='lower', aspect=0.5)
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel("$\mu$", fontsize=fs, labelpad=-5)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([-1, 0, 1])
axes.tick_params(axis='both', labelsize=fs, pad=10)

cbar_fixed = fig.colorbar(im, ax=axes)
cbar_fixed.set_ticks([np.min(f), np.max(f)])
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "distr_funct_t" + t_string + "_" + method + "_1e-3_r16.pdf")
'''

''' ### 4 plots, same time, different ranks

Nx = 256
Nmu = 256
dt = 1e-3
r_array = [4, 8, 16, 32]
t_f = 1.0
t_string = "01"
sigma = 1
fs = 16
savepath = "C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_latex_250430/periodic_dlr/sigma1/"
method = "strang"

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

grid = Grid(Nx, Nmu, r_array[0])
lr0 = setInitialCondition(grid, sigma)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr, time = integrate(lr0, grid, t_f, dt, method)
f = lr.U @ lr.S @ lr.V.T
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

im1 = axes[0, 0].imshow(f.T, extent=extent, origin='lower', aspect=0.5, vmin=0.143, vmax=0.155)
axes[0, 0].set_title("$r=$" + str(r_array[0]), fontsize=fs)
axes[0, 0].set_xlabel("$x$", fontsize=fs)
axes[0, 0].set_ylabel("$\mu$", fontsize=fs, labelpad=-5)
axes[0, 0].set_xticks([0, 0.5, 1])
axes[0, 0].set_yticks([-1, 0, 1])
axes[0, 0].tick_params(axis='both', labelsize=fs, pad=10)

grid = Grid(Nx, Nmu, r_array[1])
lr0 = setInitialCondition(grid, sigma)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr, time = integrate(lr0, grid, t_f, dt, method)
f = lr.U @ lr.S @ lr.V.T
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

im2 = axes[0, 1].imshow(f.T, extent=extent, origin='lower', aspect=0.5, vmin=0.143, vmax=0.155)
axes[0, 1].set_title("$r=$" + str(r_array[1]), fontsize=fs)
axes[0, 1].set_xlabel("$x$", fontsize=fs)
axes[0, 1].set_ylabel("$\mu$", fontsize=fs, labelpad=-5)
axes[0, 1].set_xticks([0, 0.5, 1])
axes[0, 1].set_yticks([-1, 0, 1])
axes[0, 1].tick_params(axis='both', labelsize=fs, pad=10)

grid = Grid(Nx, Nmu, r_array[2])
lr0 = setInitialCondition(grid, sigma)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr, time = integrate(lr0, grid, t_f, dt, method)
f = lr.U @ lr.S @ lr.V.T
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

im3 = axes[1, 0].imshow(f.T, extent=extent, origin='lower', aspect=0.5, vmin=0.143, vmax=0.155)
axes[1, 0].set_title("$r=$" + str(r_array[2]), fontsize=fs)
axes[1, 0].set_xlabel("$x$", fontsize=fs)
axes[1, 0].set_ylabel("$\mu$", fontsize=fs, labelpad=-5)
axes[1, 0].set_xticks([0, 0.5, 1])
axes[1, 0].set_yticks([-1, 0, 1])
axes[1, 0].tick_params(axis='both', labelsize=fs, pad=10)

grid = Grid(Nx, Nmu, r_array[3])
lr0 = setInitialCondition(grid, sigma)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr, time = integrate(lr0, grid, t_f, dt, method)
f = lr.U @ lr.S @ lr.V.T
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

im4 = axes[1, 1].imshow(f.T, extent=extent, origin='lower', aspect=0.5, vmin=0.143, vmax=0.155)
axes[1, 1].set_title("$r=$" + str(r_array[3]), fontsize=fs)
axes[1, 1].set_xlabel("$x$", fontsize=fs)
axes[1, 1].set_ylabel("$\mu$", fontsize=fs, labelpad=-5)
axes[1, 1].set_xticks([0, 0.5, 1])
axes[1, 1].set_yticks([-1, 0, 1])
axes[1, 1].tick_params(axis='both', labelsize=fs, pad=10)

cbar_fixed1 = fig.colorbar(im1, ax=axes[0, 0])
cbar_fixed1.set_ticks([0.144, 0.149, 0.154])
cbar_fixed1.ax.tick_params(labelsize=fs)
cbar_fixed2 = fig.colorbar(im2, ax=axes[0, 1])
cbar_fixed2.set_ticks([0.144, 0.149, 0.154])
cbar_fixed2.ax.tick_params(labelsize=fs)
cbar_fixed3 = fig.colorbar(im3, ax=axes[1, 0])
cbar_fixed3.set_ticks([0.144, 0.149, 0.154])
cbar_fixed3.ax.tick_params(labelsize=fs)
cbar_fixed4 = fig.colorbar(im4, ax=axes[1, 1])
cbar_fixed4.set_ticks([0.144, 0.149, 0.154])
cbar_fixed4.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.6)
plt.savefig(savepath + "distr_funct_different_ranks_t" + t_string + "_" + method + "_1e-3.pdf")
'''

''' ### 4 plots, same rank, different times

Nx = 256
Nmu = 256
dt = 1e-3
r = 16
t_f_array = [0.5, 1.0, 2.0, 3.0]
sigma = 1
fs = 16
savepath = "C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_latex_250430/periodic_dlr/sigma1/"
method = "strang"

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

grid = Grid(Nx, Nmu, r)
lr0 = setInitialCondition(grid, sigma)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr, time = integrate(lr0, grid, t_f_array[0], dt, method)
f = lr.U @ lr.S @ lr.V.T
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

im1 = axes[0, 0].imshow(f.T, extent=extent, origin='lower', aspect=0.5, vmin=0.138, vmax=0.158)
axes[0, 0].set_title("$t=$" + str(t_f_array[0]), fontsize=fs)
axes[0, 0].set_xlabel("$x$", fontsize=fs)
axes[0, 0].set_ylabel("$\mu$", fontsize=fs, labelpad=-5)
axes[0, 0].set_xticks([0, 0.5, 1])
axes[0, 0].set_yticks([-1, 0, 1])
axes[0, 0].tick_params(axis='both', labelsize=fs, pad=10)

grid = Grid(Nx, Nmu, r)
lr0 = setInitialCondition(grid, sigma)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr, time = integrate(lr0, grid, t_f_array[1], dt, method)
f = lr.U @ lr.S @ lr.V.T
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

im2 = axes[0, 1].imshow(f.T, extent=extent, origin='lower', aspect=0.5, vmin=0.143, vmax=0.155)
axes[0, 1].set_title("$t=$" + str(t_f_array[1]), fontsize=fs)
axes[0, 1].set_xlabel("$x$", fontsize=fs)
axes[0, 1].set_ylabel("$\mu$", fontsize=fs, labelpad=-5)
axes[0, 1].set_xticks([0, 0.5, 1])
axes[0, 1].set_yticks([-1, 0, 1])
axes[0, 1].tick_params(axis='both', labelsize=fs, pad=10)

grid = Grid(Nx, Nmu, r)
lr0 = setInitialCondition(grid, sigma)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr, time = integrate(lr0, grid, t_f_array[2], dt, method)
f = lr.U @ lr.S @ lr.V.T
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

im3 = axes[1, 0].imshow(f.T, extent=extent, origin='lower', aspect=0.5, vmin=0.1484, vmax=0.1526)
axes[1, 0].set_title("$t=$" + str(t_f_array[2]), fontsize=fs)
axes[1, 0].set_xlabel("$x$", fontsize=fs)
axes[1, 0].set_ylabel("$\mu$", fontsize=fs, labelpad=-5)
axes[1, 0].set_xticks([0, 0.5, 1])
axes[1, 0].set_yticks([-1, 0, 1])
axes[1, 0].tick_params(axis='both', labelsize=fs, pad=10)

grid = Grid(Nx, Nmu, r)
lr0 = setInitialCondition(grid, sigma)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr, time = integrate(lr0, grid, t_f_array[3], dt, method)
f = lr.U @ lr.S @ lr.V.T
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

im4 = axes[1, 1].imshow(f.T, extent=extent, origin='lower', aspect=0.5, vmin=0.1505, vmax=0.1521)
axes[1, 1].set_title("$t=$" + str(t_f_array[3]), fontsize=fs)
axes[1, 1].set_xlabel("$x$", fontsize=fs)
axes[1, 1].set_ylabel("$\mu$", fontsize=fs, labelpad=-5)
axes[1, 1].set_xticks([0, 0.5, 1])
axes[1, 1].set_yticks([-1, 0, 1])
axes[1, 1].tick_params(axis='both', labelsize=fs, pad=10)

cbar_fixed1 = fig.colorbar(im1, ax=axes[0, 0])
cbar_fixed1.set_ticks([0.139, 0.148, 0.157])
cbar_fixed1.ax.tick_params(labelsize=fs)
cbar_fixed2 = fig.colorbar(im2, ax=axes[0, 1])
cbar_fixed2.set_ticks([0.144, 0.149, 0.154])
cbar_fixed2.ax.tick_params(labelsize=fs)
cbar_fixed3 = fig.colorbar(im3, ax=axes[1, 0])
cbar_fixed3.set_ticks([0.149, 0.152])
cbar_fixed3.ax.tick_params(labelsize=fs)
cbar_fixed4 = fig.colorbar(im4, ax=axes[1, 1])
cbar_fixed4.set_ticks([0.151, 0.152])
cbar_fixed4.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.6)
plt.savefig(savepath + "distr_funct_different_times_r" + str(r) + "_" + method + "_1e-3.pdf")
'''

''' ### Values for colorbar sigma 1

im1 = axes[0, 0].imshow(f.T, extent=extent, origin='lower', aspect=0.5, vmin=0.138, vmax=0.158)
im2 = axes[0, 1].imshow(f.T, extent=extent, origin='lower', aspect=0.5, vmin=0.143, vmax=0.155)
im3 = axes[1, 0].imshow(f.T, extent=extent, origin='lower', aspect=0.5, vmin=0.1484, vmax=0.1526)
im4 = axes[1, 1].imshow(f.T, extent=extent, origin='lower', aspect=0.5, vmin=0.1505, vmax=0.1521)

cbar_fixed1 = fig.colorbar(im1, ax=axes[0, 0])
cbar_fixed1.set_ticks([0.139, 0.148, 0.157])
cbar_fixed1.ax.tick_params(labelsize=fs)
cbar_fixed2 = fig.colorbar(im2, ax=axes[0, 1])
cbar_fixed2.set_ticks([0.144, 0.149, 0.154])
cbar_fixed2.ax.tick_params(labelsize=fs)
cbar_fixed3 = fig.colorbar(im3, ax=axes[1, 0])
cbar_fixed3.set_ticks([0.149, 0.152])
cbar_fixed3.ax.tick_params(labelsize=fs)
cbar_fixed4 = fig.colorbar(im4, ax=axes[1, 1])
cbar_fixed4.set_ticks([0.151, 0.152])
cbar_fixed4.ax.tick_params(labelsize=fs)
'''

''' ### Plot mass over time

Nx = 256
Nmu = 256
dt = 1e-3
r = 16
t_f = 3.0
t_string = "03"
sigma = 1
fs = 16
savepath = "C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_latex_250430/periodic_dlr/sigma1/"
method = "strang"

fig, axes = plt.subplots(1, 1, figsize=(10, 8))

grid = Grid(Nx, Nmu, r)
lr0 = setInitialCondition(grid, sigma)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr, time, mass = integrate(lr0, grid, t_f, dt, method)
f = lr.U @ lr.S @ lr.V.T
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

plt.plot(time, mass)
plt.xlabel("$t$")
plt.ylabel("mass")

plt.savefig(savepath + "mass_over_time_" + method + "_1e-3_r" + str(r) + "_sigma" + str(sigma) + ".pdf")
'''