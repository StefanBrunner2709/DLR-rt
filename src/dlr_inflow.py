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
        self.X = np.linspace(0.0, 1.0, Nx+1, endpoint=False)[1:]     # We don't want starting point because of our boundary conditions now
        self.MU = np.linspace(-1.0, 1.0, Nmu, endpoint=True)       # For mu we don't have boundary conditions
        self.dx = self.X[1] - self.X[0]
        self.dmu = self.MU[1] - self.MU[0]

def setInitialCondition(grid):
    S = np.zeros((grid.r, grid.r))
    U = np.random.rand(grid.Nx, grid.r)
    V = np.random.rand(grid.Nmu, grid.r)

    U_ortho, R_U = np.linalg.qr(U, mode="reduced")
    V_ortho, R_V = np.linalg.qr(V, mode="reduced")
    S_ortho = R_U @ S @R_V.T

    lr = LR(U_ortho, S_ortho, V_ortho)
    return lr

def RK4(f, rhs, dt):
    b_coeff = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
    
    k_coeff0 = rhs(f)
    k_coeff1 = rhs(f + dt * 0.5 * k_coeff0)
    k_coeff2 = rhs(f + dt * 0.5 * k_coeff1)
    k_coeff3 = rhs(f + dt * k_coeff2)

    return b_coeff[0] * k_coeff0 + b_coeff[1] * k_coeff1 + b_coeff[2] * k_coeff2 + b_coeff[3] * k_coeff3

def computeF_b(f, grid, t):
    
    F_b = np.zeros((2, len(grid.MU)))
    
    # values from inflow:
    if grid.MU[191] > 0:     # leftmost entries are for negative mu, rightmost por positive mu
        F_b[0, 191] = np.tanh(t)
    elif grid.MU[191] < 0:
        F_b[1, 191] = np.tanh(t)
    """
    for i in range(len(grid.MU)):
        if grid.MU[i] > 0:
            F_b[0, i] = np.tanh(t)
        elif grid.MU[i] < 0:
            F_b[1, i] = np.tanh(t)
    """
    #Values from extrapolation from f:          # not completely sure about indices
    for i in range(int(grid.Nx/2)):
        F_b[0, i] = f[0,i] - (f[1,i]-f[0,i])/grid.dx * grid.X[0]
    for i in range(int(grid.Nx/2), grid.Nx):
        F_b[1, i] = f[grid.Nx-1,i] + (f[grid.Nx-1,i]-f[grid.Nx-2,i])/grid.dx * (1-grid.X[grid.Nx-1])
    
    return F_b

def computeK_bdry(lr, grid, t):

    e_vec_left = np.zeros([len(grid.MU)])
    e_vec_right = np.zeros([len(grid.MU)])
    
    if grid.MU[191] > 0:
        e_vec_left[191] = np.tanh(t)
    elif grid.MU[191] < 0:
        e_vec_right[191] = np.tanh(t)
    """
    for i in range(len(grid.MU)):       # compute e-vector
        if grid.MU[i] > 0:
            # e_vec_left[i] = np.exp(-(grid.MU[i])**2/2)
            e_vec_left[i] = np.tanh(t)
        elif grid.MU[i] < 0:
            # e_vec_right[i] = np.exp(-(grid.MU[i])**2/2)
            e_vec_right[i] = np.tanh(t)
    """
    int_exp_left = (e_vec_left @ lr.V) * grid.dmu   # compute integral from inflow, contains information from inflow from every K_j
    int_exp_right = (e_vec_right @ lr.V) * grid.dmu

    K = lr.U @ lr.S
    K_extrapol_left = np.zeros([grid.r])
    K_extrapol_right = np.zeros([grid.r])
    for i in range(grid.r):     # calculate extrapolated values
        K_extrapol_left[i] = K[0,i] - (K[1,i]-K[0,i])/grid.dx * grid.X[0]
        K_extrapol_right[i] = K[grid.Nx-1,i] + (K[grid.Nx-1,i]-K[grid.Nx-2,i])/grid.dx * (1-grid.X[grid.Nx-1])

    V_indicator_left = np.copy(lr.V)     # generate V*indicator, Note: Only works for Nx even
    V_indicator_left[int(grid.Nmu/2):,:] = 0
    V_indicator_right = np.copy(lr.V)
    V_indicator_right[:int(grid.Nmu/2),:] = 0

    int_V_left = (V_indicator_left.T @ lr.V) * grid.dmu        # compute integrals over V
    int_V_right = (V_indicator_right.T @ lr.V) * grid.dmu 

    sum_vector_left = K_extrapol_left @ int_V_left              # compute vector of size r with all the sums inside
    sum_vector_right = K_extrapol_right @ int_V_right

    K_bdry_left = int_exp_left + sum_vector_left            # add all together to get boundary info (vector with info for 1<=j<=r)
    K_bdry_right = int_exp_right + sum_vector_right    

    return K_bdry_left, K_bdry_right

def computedxK(lr, K_bdry_left, K_bdry_right, grid):

    K = lr.U @ lr.S

    Dx = np.zeros((grid.Nx, grid.Nx), dtype=int)
    np.fill_diagonal(Dx[1:], -1)
    np.fill_diagonal(Dx[:, 1:], 1)

    dxK = Dx @ K / (2*grid.dx)

    dxK[0,:] -= K_bdry_left / (2*grid.dx)
    dxK[-1,:] += K_bdry_right / (2*grid.dx)

    return dxK

def computeC(lr, grid):

    C1 = (lr.V.T @ np.diag(grid.MU) @ lr.V) * grid.dmu

    C2 = (lr.V.T @ np.ones((grid.Nmu,grid.Nmu))).T * grid.dmu

    return C1, C2

def computeB(L, grid):

    B1 = (L.T @ np.ones((grid.Nmu,grid.Nmu))).T * grid.dmu

    return B1

def computeD(lr, grid, t):

    K_bdry_left, K_bdry_right = computeK_bdry(lr, grid, t)
    dxK = computedxK(lr, K_bdry_left, K_bdry_right, grid)
    D1 = lr.U.T @ dxK * grid.dx

    return D1

def Kstep(K, C1, C2, grid, lr, t):
    K_bdry_left, K_bdry_right = computeK_bdry(lr, grid, t)
    dxK = computedxK(lr, K_bdry_left, K_bdry_right, grid)    
    rhs = - dxK @ C1 + 0.5 * K @ C2.T @ C2 - K
    return rhs

def Sstep(S, C1, C2, D1):
    rhs = D1 @ C1 - 0.5 * S @ C2.T @ C2 + S
    return rhs

def Lstep(L, D1, B1, grid, lr):
    rhs = - np.diag(grid.MU) @ lr.V @ D1.T + 0.5 * B1 - L
    return rhs

def integrate(lr0: LR, grid: Grid, t_f: float, dt: float, option: str = "lie", tol: float = 1e-2, tol_sing_val: float = 1e-6, drop_tol: float = 1e-6):
    lr = lr0
    t = 0
    time = []
    time.append(t)
    #rank = []
    adapt_rank = []
    f = lr.U @ lr.S @ lr.V.T
    #rank.append(np.linalg.matrix_rank(f, tol))
    adapt_rank.append(grid.r)

    with tqdm(total=t_f/dt, desc="Running Simulation") as pbar:

        while t < t_f:

            pbar.update(1)

            if (t + dt > t_f):
                dt = t_f - t
            
            ### Add basis for adaptive rank strategy:

            # Compute F_b
            F_b = computeF_b(lr.U @ lr.S @ lr.V.T, grid, t)

            # Compute SVD and drop singular values
            X, sing_val, QT = np.linalg.svd(F_b)
            r_b = np.sum(sing_val > tol_sing_val)
            Sigma = np.zeros((F_b.shape[0], r_b))
            np.fill_diagonal(Sigma, sing_val[:r_b])
            Q = QT.T[:,:r_b]

            # Concatenate
            X_h = np.random.rand(grid.Nx, r_b)
            lr.U = np.concatenate((lr.U, X_h), axis=1)
            lr.V = np.concatenate((lr.V, Q), axis=1)
            S_extended = np.zeros((grid.r + r_b, grid.r + r_b))
            S_extended[:grid.r, :grid.r] = lr.S
            lr.S = S_extended

            # QR-decomp
            lr.U, R_U = np.linalg.qr(lr.U, mode="reduced")
            lr.U /= np.sqrt(grid.dx)
            R_U *= np.sqrt(grid.dx)
            lr.V, R_V = np.linalg.qr(lr.V, mode="reduced")
            lr.V /= np.sqrt(grid.dmu)
            R_V *= np.sqrt(grid.dmu)
            lr.S = R_U @ lr.S @ R_V.T

            grid.r += r_b

            if option=="lie":

                # K step
                C1, C2 = computeC(lr, grid)
                K = lr.U @ lr.S
                K += dt * RK4(K, lambda K: Kstep(K, C1, C2, grid, lr, t), dt)
                lr.U, lr.S = np.linalg.qr(K, mode="reduced")
                lr.U /= np.sqrt(grid.dx)
                lr.S *= np.sqrt(grid.dx)

                # S step
                D1 = computeD(lr, grid, t)
                lr.S += dt * RK4(lr.S, lambda S: Sstep(S, C1, C2, D1), dt)

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
                K += 0.5 * dt * RK4(K, lambda K: Kstep(K, C1, C2, grid, lr, t), 0.5 * dt)
                lr.U, lr.S = np.linalg.qr(K, mode="reduced")
                lr.U /= np.sqrt(grid.dx)
                lr.S *= np.sqrt(grid.dx)

                # 1/2 S step
                D1 = computeD(lr, grid, t)
                lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep(S, C1, C2, D1), 0.5 * dt)

                # L step
                L = lr.V @ lr.S.T
                B1 = computeB(L, grid)
                L += dt * RK4(L, lambda L: Lstep(L, D1, B1, grid, lr), dt)
                lr.V, St = np.linalg.qr(L, mode="reduced")
                lr.S = St.T
                lr.V /= np.sqrt(grid.dmu)
                lr.S *= np.sqrt(grid.dmu)

                # 1/2 S step
                C1, C2 = computeC(lr, grid)     # need to recalculate C1 and C2 because we changed V in L step     
                lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep(S, C1, C2, D1), 0.5 * dt)

                # 1/2 K step
                K = lr.U @ lr.S
                K += 0.5 * dt * RK4(K, lambda K: Kstep(K, C1, C2, grid, lr, t), 0.5 * dt)      # ToDo: Do i need t + 0.5*dt for K-step?
                lr.U, lr.S = np.linalg.qr(K, mode="reduced")
                lr.U /= np.sqrt(grid.dx)
                lr.S *= np.sqrt(grid.dx)

            ### Drop basis for adaptive rank strategy:
            U, sing_val, QT = np.linalg.svd(lr.S)
            r_prime = np.sum(sing_val > drop_tol)
            if r_prime < 5:
                r_prime = 5
            lr.S = np.zeros((r_prime, r_prime))
            np.fill_diagonal(lr.S, sing_val[:r_prime])
            U = U[:, :r_prime]
            Q = QT.T[:, :r_prime]
            lr.U = lr.U @ U
            lr.V = lr.V @ Q
            grid.r = r_prime

            t += dt
            time.append(t)

            f = lr.U @ lr.S @ lr.V.T
            #rank.append(np.linalg.matrix_rank(f, tol))
            adapt_rank.append(grid.r)

    return lr, time, adapt_rank



### Just one plot for certain rank and certain time

Nx = 256
Nmu = 256
dt = 1e-4
r = 5
t_f = 2.0
fs = 30
method = "lie"
savepath = "C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Plots_latex_250430/inflow/"

fig, axes = plt.subplots(1, 1, figsize=(10, 8))

grid = Grid(Nx, Nmu, r)
lr0 = setInitialCondition(grid)
f0 = lr0.U @ lr0.S @ lr0.V.T
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

lr, time, rank = integrate(lr0, grid, t_f, dt, option=method, tol_sing_val=1e-5, drop_tol=1e-5)
f = lr.U @ lr.S @ lr.V.T
'''
#im = axes.imshow(f.T, extent=extent, origin='lower', aspect=0.5)
im = axes.imshow(f.T, extent=extent, origin='lower', aspect=0.5, vmin=0.0, vmax=1.0)
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel("$\mu$", fontsize=fs, labelpad=-5)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([-1, 0, 1])
axes.tick_params(axis='both', labelsize=fs, pad=20)
axes.set_title("$t=$" + str(t_f), fontsize=fs)

cbar_fixed = fig.colorbar(im, ax=axes)
#cbar_fixed.set_ticks([np.ceil(np.min(f)*10000)/10000, np.floor(np.max(f)*10000)/10000])
cbar_fixed.set_ticks([0, 0.5, 1])
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "distr_funct_initalltanh_fixedcol_t" + str(t_f) + "_" + method + "_" + str(dt) + "_adaptrank_" + str(Nx) + "x" + str(Nmu) + ".pdf")
'''
fig, ax = plt.subplots()
ax.plot(time, rank)
ax.set_xlabel("$t$", fontsize=22)
ax.set_ylabel("rank $r(t)$", fontsize=22)
ax.tick_params(axis='both', labelsize=22)
ax.set_yticks([5])
ax.margins(x=0)
plt.tight_layout()
plt.savefig(savepath + "adaptive_rank_initsingletanh_t" + str(t_f) + "_" + method + "_" + str(dt) + "_adaptrank_" + str(Nx) + "x" + str(Nmu) + ".pdf")
