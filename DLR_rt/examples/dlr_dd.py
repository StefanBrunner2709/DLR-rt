import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from DLR_rt.src.grid import Grid_1x1d
from DLR_rt.src.integrators import RK4
from DLR_rt.src.initial_condition import setInitialCondition_1x1d_lr
from DLR_rt.src.lr import LR, computeC, computeB, computeD, Kstep, Sstep, Lstep, computeF_b, add_basis_functions, drop_basis_functions


def integrate(lr0_left: LR, lr0_right: LR, grid_left, grid_right, t_f: float, dt: float, option: str = "lie", tol: float = 1e-2, tol_sing_val: float = 1e-6, drop_tol: float = 1e-6):
    lr_left = lr0_left
    lr_right = lr0_right
    t = 0
    time = []
    time.append(t)
    #adapt_rank = []
    #adapt_rank.append(grid.r)

    with tqdm(total=t_f/dt, desc="Running Simulation") as pbar:

        while t < t_f:

            pbar.update(1)

            if (t + dt > t_f):
                dt = t_f - t

            ### Compute F_b
            F_b_left = computeF_b(t, lr_left.U @ lr_left.S @ lr_left.V.T, grid_left, f_right = lr_right.U @ lr_right.S @ lr_right.V.T)
            F_b_right = computeF_b(t, lr_right.U @ lr_right.S @ lr_right.V.T, grid_right, f_left = lr_left.U @ lr_left.S @ lr_left.V.T)

            ### Update left side

            ### Add basis for adaptive rank strategy:
            lr_left, grid_left = add_basis_functions(lr_left, grid_left, F_b_left, tol_sing_val)

            ### Run PSI
            if option=="lie":

                # K step
                C1, C2 = computeC(lr_left, grid_left)
                K = lr_left.U @ lr_left.S
                K += dt * RK4(K, lambda K: Kstep(K, C1, C2, grid_left, lr_left, F_b_left), dt)
                lr_left.U, lr_left.S = np.linalg.qr(K, mode="reduced")
                lr_left.U /= np.sqrt(grid_left.dx)
                lr_left.S *= np.sqrt(grid_left.dx)

                # S step
                D1 = computeD(lr_left, grid_left, F_b_left)
                lr_left.S += dt * RK4(lr_left.S, lambda S: Sstep(S, C1, C2, D1, inflow = True), dt)

                # L step
                L = lr_left.V @ lr_left.S.T
                B1 = computeB(L, grid_left)
                L += dt * RK4(L, lambda L: Lstep(L, D1, B1, grid_left, lr_left), dt)
                lr_left.V, St = np.linalg.qr(L, mode="reduced")
                lr_left.S = St.T
                lr_left.V /= np.sqrt(grid_left.dmu)
                lr_left.S *= np.sqrt(grid_left.dmu)

            ### Drop basis for adaptive rank strategy:
            lr_left, grid_left = drop_basis_functions(lr_left, grid_left, drop_tol)

            ### Update right side

            ### Add basis for adaptive rank strategy:
            lr_right, grid_right = add_basis_functions(lr_right, grid_right, F_b_right, tol_sing_val)

            if option=="lie":

                # K step
                C1, C2 = computeC(lr_right, grid_right)
                K = lr_right.U @ lr_right.S
                K += dt * RK4(K, lambda K: Kstep(K, C1, C2, grid_right, lr_right, F_b_right), dt)
                lr_right.U, lr_right.S = np.linalg.qr(K, mode="reduced")
                lr_right.U /= np.sqrt(grid_right.dx)
                lr_right.S *= np.sqrt(grid_right.dx)

                # S step
                D1 = computeD(lr_right, grid_right, F_b_right)
                lr_right.S += dt * RK4(lr_right.S, lambda S: Sstep(S, C1, C2, D1, inflow = True), dt)

                # L step
                L = lr_right.V @ lr_right.S.T
                B1 = computeB(L, grid_right)
                L += dt * RK4(L, lambda L: Lstep(L, D1, B1, grid_right, lr_right), dt)
                lr_right.V, St = np.linalg.qr(L, mode="reduced")
                lr_right.S = St.T
                lr_right.V /= np.sqrt(grid_right.dmu)
                lr_right.S *= np.sqrt(grid_right.dmu)

            ### Drop basis for adaptive rank strategy:
            lr_right, grid_right = drop_basis_functions(lr_right, grid_right, drop_tol)


            # Update time
            t += dt
            time.append(t)

            #adapt_rank.append(grid.r)

    return lr_left, lr_right, time



### Just one plot for certain rank and certain time

Nx = 64
Nmu = 64
dt = 1e-4
r = 5
t_f = 2.0
fs = 30
method = "lie"
savepath = "plots/"

fig, axes = plt.subplots(1, 1, figsize=(10, 8))

grid = Grid_1x1d(Nx, Nmu, r)
grid_left, grid_right = grid.split()
lr0_left = setInitialCondition_1x1d_lr(grid_left)
lr0_right = setInitialCondition_1x1d_lr(grid_right)
extent = [grid_left.X[0], grid_right.X[-1], grid_left.MU[0], grid_left.MU[-1]]

lr_left, lr_right, time = integrate(lr0_left, lr0_right, grid_left, grid_right, t_f, dt, option=method, tol_sing_val=1e-5, drop_tol=1e-5)
f_left = lr_left.U @ lr_left.S @ lr_left.V.T
f_right = lr_right.U @ lr_right.S @ lr_right.V.T
# Concatenate left and right domain
f = np.concatenate((f_left, f_right), axis=0)

im = axes.imshow(f.T, extent=extent, origin='lower', aspect=0.5, vmin=0.0, vmax=1.0)
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel(r"$\mu$", fontsize=fs, labelpad=-5)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([-1, 0, 1])
axes.tick_params(axis='both', labelsize=fs, pad=20)
axes.set_title("$t=$" + str(t_f), fontsize=fs)

cbar_fixed = fig.colorbar(im, ax=axes)
cbar_fixed.set_ticks([0, 0.5, 1])
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "dd_distr_funct_initalltanh_fixedcol_t" + str(t_f) + "_" + method + "_" + str(dt) + "_adaptrank_" + str(Nx) + "x" + str(Nmu) + ".pdf")
