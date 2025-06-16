import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class LR:
    def __init__(self, U, S, V):
        self.U = U
        self.S = S
        self.V = V

class Grid_left:
    def __init__(self, Nx, Nmu, r):
        self.Nx = int(Nx/2)
        self.Nmu = Nmu
        self.r = r
        self.X = np.linspace(0.0, 1.0, Nx + 1, endpoint=False)[1:int(Nx/2)+1]
        self.MU = np.linspace(-1.0, 1.0, Nmu, endpoint=True)
        self.dx = self.X[1] - self.X[0]
        self.dmu = self.MU[1] - self.MU[0]

class Grid_right:
    def __init__(self, Nx, Nmu, r):
        self.Nx = int(Nx/2)
        self.Nmu = Nmu
        self.r = r
        self.X = np.linspace(0.0, 1.0, Nx + 1, endpoint=False)[int(Nx/2)+1:]
        self.MU = np.linspace(-1.0, 1.0, Nmu, endpoint=True)
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

def computeF_b(f_left, f_right, grid_left, grid_right, t):
    
    F_b_left = np.zeros((2, len(grid_left.MU)))
    F_b_right = np.zeros((2, len(grid_right.MU)))
    
    #Values from extrapolation from f left domain:
    for i in range(len(grid_left.MU)):
        if grid_left.MU[i] > 0:
            F_b_left[1, i] = f_left[grid_left.Nx-1,i] + (f_left[grid_left.Nx-1,i]-f_left[grid_left.Nx-2,i])/grid_left.dx * (1-grid_left.X[grid_left.Nx-1])
        elif grid_left.MU[i] < 0:
            F_b_left[0, i] = f_left[0,i] - (f_left[1,i]-f_left[0,i])/grid_left.dx * grid_left.X[0]
        
    #Values from extrapolation from f right domain:
    for i in range(len(grid_right.MU)):
        if grid_right.MU[i] > 0:
            F_b_right[1, i] = f_right[grid_right.Nx-1,i] + (f_right[grid_right.Nx-1,i]-f_right[grid_right.Nx-2,i])/grid_right.dx * (1-grid_right.X[grid_right.Nx-1])
        elif grid_right.MU[i] < 0:
            F_b_right[0, i] = f_right[0,i] - (f_right[1,i]-f_right[0,i])/grid_right.dx * grid_right.X[0]
            
    #Values from boundary condition left domain:
    for i in range(len(grid_left.MU)):
        if grid_left.MU[i] > 0:
            F_b_left[0, i] = np.tanh(t)
        elif grid_left.MU[i] < 0:
            #F_b_left[1, i] = f_right[0, i]      # is this ok for inflow boundary from right domain or do I need to extrapolate?
            F_b_left[1, i] = F_b_right[0, i]
    
    #Values from boundary condition right domain:
    for i in range(len(grid_right.MU)):
        if grid_right.MU[i] > 0:
            #F_b_right[0, i] = f_left[grid_left.Nx-1, i]      # is this ok for inflow boundary from left domain or do I need to extrapolate?
            F_b_right[0, i] = F_b_left[1, i]
        elif grid_right.MU[i] < 0:
            F_b_right[1, i] = np.tanh(t)

    return F_b_left, F_b_right

def computeK_bdry(lr, grid, t, left_right, F_b_left, F_b_right):

    e_vec_left = np.zeros([len(grid.MU)])
    e_vec_right = np.zeros([len(grid.MU)])

    if left_right == "left":

        #Values from boundary condition:
        for i in range(len(grid.MU)):       # compute e-vector
            if grid.MU[i] > 0:
                e_vec_left[i] = np.tanh(t)
            elif grid.MU[i] < 0:
                e_vec_right[i] = F_b_left[1, i]

    if left_right == "right":

        #Values from boundary condition:
        for i in range(len(grid.MU)):       # compute e-vector
            if grid.MU[i] > 0:
                e_vec_left[i] = F_b_right[0, i]
            elif grid.MU[i] < 0:
                e_vec_right[i] = np.tanh(t)
    
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

def computeD(lr, grid, t, left_right, F_b_left, F_b_right):

    K_bdry_left, K_bdry_right = computeK_bdry(lr, grid, t, left_right, F_b_left, F_b_right)
    dxK = computedxK(lr, K_bdry_left, K_bdry_right, grid)
    D1 = lr.U.T @ dxK * grid.dx

    return D1

def Kstep(K, C1, C2, grid, lr, t, left_right, F_b_left, F_b_right):
    K_bdry_left, K_bdry_right = computeK_bdry(lr, grid, t, left_right, F_b_left, F_b_right)
    dxK = computedxK(lr, K_bdry_left, K_bdry_right, grid)    
    rhs = - dxK @ C1 + 0.5 * K @ C2.T @ C2 - K
    return rhs

def Sstep(S, C1, C2, D1):
    rhs = D1 @ C1 - 0.5 * S @ C2.T @ C2 + S
    return rhs

def Lstep(L, D1, B1, grid, lr):
    rhs = - np.diag(grid.MU) @ lr.V @ D1.T + 0.5 * B1 - L
    return rhs

def integrate(lr0_left: LR, lr0_right: LR, grid_left, grid_right, t_f: float, dt: float, option: str = "lie", tol: float = 1e-2, tol_sing_val: float = 1e-6, drop_tol: float = 1e-6):
    lr_left = lr0_left
    lr_right = lr0_right
    t = 0
    time = []
    time.append(t)
    #rank = []
    #adapt_rank = []
    #f = lr.U @ lr.S @ lr.V.T
    #rank.append(np.linalg.matrix_rank(f, tol))
    #adapt_rank.append(grid.r)

    with tqdm(total=t_f/dt, desc="Running Simulation") as pbar:

        while t < t_f:

            pbar.update(1)

            if (t + dt > t_f):
                dt = t_f - t
            
            ### Add basis for adaptive rank strategy:

            #print(np.shape(lr_left.U), np.shape(lr_left.S), np.shape(lr_left.V.T))
            #print(np.shape(lr_right.U), np.shape(lr_right.S), np.shape(lr_right.V.T))

            # Compute F_b
            F_b_left, F_b_right = computeF_b(lr_left.U @ lr_left.S @ lr_left.V.T, lr_right.U @ lr_right.S @ lr_right.V.T, grid_left, grid_right, t)

            #print("F_b_left:", F_b_left)
            #print("F_b_right:", F_b_right)


            # Update left side

            left_right = "left"

            # Compute SVD and drop singular values
            X, sing_val, QT = np.linalg.svd(F_b_left)
            r_b = np.sum(sing_val > tol_sing_val)
            Sigma = np.zeros((F_b_left.shape[0], r_b))
            np.fill_diagonal(Sigma, sing_val[:r_b])
            Q = QT.T[:,:r_b]

            # Concatenate
            X_h = np.random.rand(grid_left.Nx, r_b)
            lr_left.U = np.concatenate((lr_left.U, X_h), axis=1)
            lr_left.V = np.concatenate((lr_left.V, Q), axis=1)
            S_extended = np.zeros((grid_left.r + r_b, grid_left.r + r_b))
            S_extended[:grid_left.r, :grid_left.r] = lr_left.S
            lr_left.S = S_extended

            # QR-decomp
            lr_left.U, R_U = np.linalg.qr(lr_left.U, mode="reduced")
            lr_left.U /= np.sqrt(grid_left.dx)
            R_U *= np.sqrt(grid_left.dx)
            lr_left.V, R_V = np.linalg.qr(lr_left.V, mode="reduced")
            lr_left.V /= np.sqrt(grid_left.dmu)
            R_V *= np.sqrt(grid_left.dmu)
            lr_left.S = R_U @ lr_left.S @ R_V.T

            grid_left.r += r_b

            if option=="lie":

                # K step
                C1, C2 = computeC(lr_left, grid_left)
                K = lr_left.U @ lr_left.S
                K += dt * RK4(K, lambda K: Kstep(K, C1, C2, grid_left, lr_left, t, left_right, F_b_left, F_b_right), dt)
                lr_left.U, lr_left.S = np.linalg.qr(K, mode="reduced")
                lr_left.U /= np.sqrt(grid_left.dx)
                lr_left.S *= np.sqrt(grid_left.dx)

                # S step
                D1 = computeD(lr_left, grid_left, t, left_right, F_b_left, F_b_right)
                lr_left.S += dt * RK4(lr_left.S, lambda S: Sstep(S, C1, C2, D1), dt)

                # L step
                L = lr_left.V @ lr_left.S.T
                B1 = computeB(L, grid_left)
                L += dt * RK4(L, lambda L: Lstep(L, D1, B1, grid_left, lr_left), dt)
                lr_left.V, St = np.linalg.qr(L, mode="reduced")
                lr_left.S = St.T
                lr_left.V /= np.sqrt(grid_left.dmu)
                lr_left.S *= np.sqrt(grid_left.dmu)

            ### Drop basis for adaptive rank strategy:
            U, sing_val, QT = np.linalg.svd(lr_left.S)
            r_prime = np.sum(sing_val > drop_tol)
            if r_prime < 5:
                r_prime = 5
            lr_left.S = np.zeros((r_prime, r_prime))
            np.fill_diagonal(lr_left.S, sing_val[:r_prime])
            U = U[:, :r_prime]
            Q = QT.T[:, :r_prime]
            lr_left.U = lr_left.U @ U
            lr_left.V = lr_left.V @ Q
            grid_left.r = r_prime


            # Update right side

            left_right = "right"

            # Compute SVD and drop singular values
            X, sing_val, QT = np.linalg.svd(F_b_right)
            r_b = np.sum(sing_val > tol_sing_val)
            Sigma = np.zeros((F_b_right.shape[0], r_b))
            np.fill_diagonal(Sigma, sing_val[:r_b])
            Q = QT.T[:,:r_b]

            # Concatenate
            X_h = np.random.rand(grid_right.Nx, r_b)
            lr_right.U = np.concatenate((lr_right.U, X_h), axis=1)
            lr_right.V = np.concatenate((lr_right.V, Q), axis=1)
            S_extended = np.zeros((grid_right.r + r_b, grid_right.r + r_b))
            S_extended[:grid_right.r, :grid_right.r] = lr_right.S
            lr_right.S = S_extended

            # QR-decomp
            lr_right.U, R_U = np.linalg.qr(lr_right.U, mode="reduced")
            lr_right.U /= np.sqrt(grid_right.dx)
            R_U *= np.sqrt(grid_right.dx)
            lr_right.V, R_V = np.linalg.qr(lr_right.V, mode="reduced")
            lr_right.V /= np.sqrt(grid_right.dmu)
            R_V *= np.sqrt(grid_right.dmu)
            lr_right.S = R_U @ lr_right.S @ R_V.T

            grid_right.r += r_b

            if option=="lie":

                # K step
                C1, C2 = computeC(lr_right, grid_right)
                K = lr_right.U @ lr_right.S
                K += dt * RK4(K, lambda K: Kstep(K, C1, C2, grid_right, lr_right, t, left_right, F_b_left, F_b_right), dt)
                lr_right.U, lr_right.S = np.linalg.qr(K, mode="reduced")
                lr_right.U /= np.sqrt(grid_right.dx)
                lr_right.S *= np.sqrt(grid_right.dx)

                # S step
                D1 = computeD(lr_right, grid_right, t, left_right, F_b_left, F_b_right)
                lr_right.S += dt * RK4(lr_right.S, lambda S: Sstep(S, C1, C2, D1), dt)

                # L step
                L = lr_right.V @ lr_right.S.T
                B1 = computeB(L, grid_right)
                L += dt * RK4(L, lambda L: Lstep(L, D1, B1, grid_right, lr_right), dt)
                lr_right.V, St = np.linalg.qr(L, mode="reduced")
                lr_right.S = St.T
                lr_right.V /= np.sqrt(grid_right.dmu)
                lr_right.S *= np.sqrt(grid_right.dmu)

            ### Drop basis for adaptive rank strategy:
            U, sing_val, QT = np.linalg.svd(lr_right.S)
            r_prime = np.sum(sing_val > drop_tol)
            if r_prime < 5:
                r_prime = 5
            lr_right.S = np.zeros((r_prime, r_prime))
            np.fill_diagonal(lr_right.S, sing_val[:r_prime])
            U = U[:, :r_prime]
            Q = QT.T[:, :r_prime]
            lr_right.U = lr_right.U @ U
            lr_right.V = lr_right.V @ Q
            grid_right.r = r_prime


            # Update time
            t += dt
            time.append(t)

            #f = lr.U @ lr.S @ lr.V.T
            #rank.append(np.linalg.matrix_rank(f, tol))
            #adapt_rank.append(grid.r)

    return lr_left, lr_right, time



### Just one plot for certain rank and certain time

Nx = 256
Nmu = 256
dt = 1e-4
r = 5
t_f = 2.0
fs = 30
method = "lie"
savepath = "/home/stefan/DLR-rt/"

fig, axes = plt.subplots(1, 1, figsize=(10, 8))

grid_left = Grid_left(Nx, Nmu, r)
grid_right = Grid_right(Nx, Nmu, r)
lr0_left = setInitialCondition(grid_left)
lr0_right = setInitialCondition(grid_right)
extent = [grid_left.X[0], grid_right.X[-1], grid_left.MU[0], grid_left.MU[-1]]

lr_left, lr_right, time = integrate(lr0_left, lr0_right, grid_left, grid_right, t_f, dt, option=method, tol_sing_val=1e-5, drop_tol=1e-5)
f_left = lr_left.U @ lr_left.S @ lr_left.V.T
f_right = lr_right.U @ lr_right.S @ lr_right.V.T
# Concatenate left and right domain
f = np.concatenate((f_left, f_right), axis=0)

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
plt.savefig(savepath + "dd_distr_funct_initalltanh_fixedcol_t" + str(t_f) + "_" + method + "_" + str(dt) + "_adaptrank_" + str(Nx) + "x" + str(Nmu) + ".pdf")
