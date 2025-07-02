import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from DLR_rt.src.grid import Grid_1x1d
from DLR_rt.src.integrators import RK4
from DLR_rt.src.initial_condition import setInitialCondition_1x1d_lr
from DLR_rt.src.lr import LR, computeC, computeB, computeD, Kstep, Sstep, Lstep, computeF_b, add_basis_functions, drop_basis_functions


def integrate(lr0: LR, grid: Grid_1x1d, t_f: float, dt: float, option: str = "lie", tol: float = 1e-2, tol_sing_val: float = 1e-6, drop_tol: float = 1e-6):
    lr = lr0
    t = 0
    time = []
    time.append(t)
    adapt_rank = []
    f = lr.U @ lr.S @ lr.V.T
    adapt_rank.append(grid.r)
    vid_frame = 0

    with tqdm(total=t_f/dt, desc="Running Simulation") as pbar:

        while t < t_f:

            pbar.update(1)

            if (t + dt > t_f):
                dt = t_f - t

            ### Compute F_b
            F_b = computeF_b(t, lr.U @ lr.S @ lr.V.T, grid)
            
            ### Add basis for adaptive rank strategy:
            lr, grid = add_basis_functions(lr, grid, F_b, tol_sing_val)

            ### Run PSI
            if option=="lie":

                # K step
                C1, C2 = computeC(lr, grid)
                K = lr.U @ lr.S
                K += dt * RK4(K, lambda K: Kstep(K, C1, C2, grid, lr, F_b), dt)
                lr.U, lr.S = np.linalg.qr(K, mode="reduced")
                lr.U /= np.sqrt(grid.dx)
                lr.S *= np.sqrt(grid.dx)

                # S step
                D1 = computeD(lr, grid, F_b)
                lr.S += dt * RK4(lr.S, lambda S: Sstep(S, C1, C2, D1, inflow = True), dt)

                # L step
                L = lr.V @ lr.S.T
                B1 = computeB(L, grid)
                L += dt * RK4(L, lambda L: Lstep(L, D1, B1, grid, lr), dt)
                lr.V, St = np.linalg.qr(L, mode="reduced")
                lr.S = St.T
                lr.V /= np.sqrt(grid.dmu)
                lr.S *= np.sqrt(grid.dmu)

            if option=="strang":
                # 1/2 K step
                C1, C2 = computeC(lr, grid)
                K = lr.U @ lr.S
                K += 0.5 * dt * RK4(K, lambda K: Kstep(K, C1, C2, grid, lr, F_b), 0.5 * dt)
                lr.U, lr.S = np.linalg.qr(K, mode="reduced")
                lr.U /= np.sqrt(grid.dx)
                lr.S *= np.sqrt(grid.dx)

                # 1/2 S step
                D1 = computeD(lr, grid, F_b)
                lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep(S, C1, C2, D1, inflow = True), 0.5 * dt)

                # L step
                L = lr.V @ lr.S.T
                B1 = computeB(L, grid)
                L += dt * RK4(L, lambda L: Lstep(L, D1, B1, grid, lr), dt)
                lr.V, St = np.linalg.qr(L, mode="reduced")
                lr.S = St.T
                lr.V /= np.sqrt(grid.dmu)
                lr.S *= np.sqrt(grid.dmu)

                # Compute F_b
                F_b = computeF_b(t + 0.5 * dt, lr.U @ lr.S @ lr.V.T, grid)      # recalculate F_b at time t + 0.5 dt

                D1 = computeD(lr, grid, F_b)    # recalculate D1 because we recalculated F_b

                # 1/2 S step
                C1, C2 = computeC(lr, grid)     # need to recalculate C1 and C2 because we changed V in L step     
                lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep(S, C1, C2, D1, inflow = True), 0.5 * dt)

                # 1/2 K step
                K = lr.U @ lr.S
                K += 0.5 * dt * RK4(K, lambda K: Kstep(K, C1, C2, grid, lr, F_b), 0.5 * dt)
                lr.U, lr.S = np.linalg.qr(K, mode="reduced")
                lr.U /= np.sqrt(grid.dx)
                lr.S *= np.sqrt(grid.dx)

            ### Drop basis for adaptive rank strategy:
            lr, grid = drop_basis_functions(lr, grid, drop_tol)

            t += dt
            time.append(t)

            f = lr.U @ lr.S @ lr.V.T
            adapt_rank.append(grid.r)



            r''' ### Do the plotting for video
            
            if np.round(t/dt) % 50 == 0:
                fs = 22
                savepath = "plots/"
                extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]
                f = lr.U @ lr.S @ lr.V.T

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                #im = ax1.imshow(f.T, extent=extent, origin='lower', aspect=0.5)
                im = ax1.imshow(f.T, extent=extent, origin='lower', aspect=0.5, vmin=0.0, vmax=1.0)
                ax1.set_xlabel("$x$", fontsize=fs)
                ax1.set_ylabel(r"$\mu$", fontsize=fs, labelpad=-5)
                ax1.set_xticks([0, 0.5, 1])
                ax1.set_yticks([-1, 0, 1])
                ax1.tick_params(axis='both', labelsize=fs, pad=20)
                ax1.set_title(r"$f(t,x,\mu)$", fontsize=fs)

                cbar_fixed = fig.colorbar(im, ax=ax1, shrink=1)
                #cbar_fixed.set_ticks([np.ceil(np.min(f)*10000)/10000, np.floor(np.max(f)*10000)/10000])
                cbar_fixed.set_ticks([0, 0.5, 1])
                cbar_fixed.ax.tick_params(labelsize=fs)

                ax2.plot(time, adapt_rank)
                ax2.set_xlabel("$t$", fontsize=22)
                ax2.set_ylabel("rank $r(t)$", fontsize=22)
                ax2.tick_params(axis='both', labelsize=22)
                ax2.set_yticks([5,6,7,8,9,10])
                ax2.set_xticks([0,1,2])
                ax2.margins(x=0)
                ax2.set_xlim(0, 2)
                ax2.set_ylim(4.8, 10.2)

                vid_frame += 1

                plt.tight_layout()
                plt.savefig(savepath + f"frame_{vid_frame:04d}.png")
            '''
            


    return lr, time, adapt_rank



### Just one plot for certain rank and certain time

Nx = 64
Nmu = 64
dt = 1e-4
r = 5
t_f = 2.0
fs = 22
method = "lie"
savepath = "plots/"

grid = Grid_1x1d(Nx, Nmu, r)
lr0 = setInitialCondition_1x1d_lr(grid)
f0 = lr0.U @ lr0.S @ lr0.V.T
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

lr, time, rank = integrate(lr0, grid, t_f, dt, option=method, tol_sing_val=1e-5, drop_tol=1e-5)
f = lr.U @ lr.S @ lr.V.T

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

#im = ax1.imshow(f.T, extent=extent, origin='lower', aspect=0.5)
im = ax1.imshow(f.T, extent=extent, origin='lower', aspect=0.5, vmin=0.0, vmax=1.0)
ax1.set_xlabel("$x$", fontsize=fs)
ax1.set_ylabel(r"$\mu$", fontsize=fs, labelpad=-5)
ax1.set_xticks([0, 0.5, 1])
ax1.set_yticks([-1, 0, 1])
ax1.tick_params(axis='both', labelsize=fs, pad=20)
ax1.set_title(r"$f(t,x,\mu)$", fontsize=fs)

cbar_fixed = fig.colorbar(im, ax=ax1, shrink=1)
#cbar_fixed.set_ticks([np.ceil(np.min(f)*10000)/10000, np.floor(np.max(f)*10000)/10000])
cbar_fixed.set_ticks([0, 0.5, 1])
cbar_fixed.ax.tick_params(labelsize=fs)

ax2.plot(time, rank)
ax2.set_xlabel("$t$", fontsize=22)
ax2.set_ylabel("rank $r(t)$", fontsize=22)
ax2.tick_params(axis='both', labelsize=22)
ax2.set_yticks([5,6,7,8,9,10])
ax2.set_xticks([0,1,2])
ax2.margins(x=0)
ax2.set_xlim(0, 2)

plt.tight_layout()
plt.savefig(savepath + "distr_funct_adapt_rank_t" + str(t_f) + ".pdf")
