import numpy as np
import matplotlib.pyplot as plt

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
        self.MU = np.linspace(-1.0, 1.0, Nmu, endpoint=False)       # For mu we don't have boundary conditions
        self.dx = self.X[1] - self.X[0]
        self.dmu = self.MU[1] - self.MU[0]

def setInitialCondition(grid):
    U = np.zeros((grid.Nx, grid.r))
    V = np.zeros((grid.Nmu, grid.r))
    S = np.zeros((grid.r, grid.r))

    sigma = 1
    # U[:, 0] = 1/(2 * np.pi * sigma**2) * np.exp(-((grid.X-0.5)**2)/(2*sigma**2))
    # V[:, 0] = np.exp(-(np.abs(grid.MU)**2)/(16*sigma**2))
    # S[0, 0] = 1.0

    U[:, 0] = 0
    V[:, 0] = 0
    S[0, 0] = 0

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


def computeK_bdry(lr, grid, t):

    e_vec_left = np.zeros([len(grid.MU)])
    e_vec_right = np.zeros([len(grid.MU)])

    for i in range(len(grid.MU)):       # compute e-vector
        if grid.MU[i] > 0:
            # e_vec_left[i] = np.exp(-(grid.MU[i])**2/2)
            e_vec_left[i] = np.tanh(t)
        elif grid.MU[i] < 0:
            # e_vec_right[i] = np.exp(-(grid.MU[i])**2/2)
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
    V_indicator_left[int(grid.Nx/2):,:] = 0
    V_indicator_right = np.copy(lr.V)
    V_indicator_right[:int(grid.Nx/2),:] = 0

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

def integrate(lr0, grid, t_f, dt):
    lr = lr0
    t = 0
    time = []
    time.append(t)
    while t < t_f:
        if (t + dt > t_f):
            dt = t_f - t

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

        t += dt
        time.append(t)

    return lr, time



# Plots for different times
Nx = 64
Nmu = 64
dt = 1e-3
r = 64
t_f = 1.0

grid = Grid(Nx, Nmu, r)
lr0 = setInitialCondition(grid)
f0 = lr0.U @ lr0.S @ lr0.V.T
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

lr1, time1 = integrate(lr0, grid, t_f, dt)
f1 = lr1.U @ lr1.S @ lr1.V.T

plt.subplot(1, 2, 1)
plt.imshow(f0.T, extent=extent, origin='lower')
plt.colorbar(orientation='horizontal', pad=0.08, fraction=0.035)
plt.xlabel("$x$")
plt.ylabel("mu")
plt.title("t=0")

plt.subplot(1, 2, 2)
plt.imshow(f1.T, extent=extent, origin='lower')
plt.colorbar(orientation='horizontal', pad=0.08, fraction=0.035)
plt.xlabel("$x$")
plt.ylabel("mu")
plt.title("t=1")

plt.show()
