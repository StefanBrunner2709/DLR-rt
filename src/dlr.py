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
        self.X = np.linspace(0.0, 1.0, Nx, endpoint=False)
        self.MU = np.linspace(-1.0, 1.0, Nmu, endpoint=False)
        self.dx = self.X[1] - self.X[0]
        self.dmu = self.MU[1] - self.MU[0]

def setInitialCondition(grid):
    U = np.zeros((grid.Nx, grid.r))
    V = np.zeros((grid.Nmu, grid.r))
    S = np.zeros((grid.r, grid.r))

    sigma = 1
    U[:, 0] = 1/(2 * np.pi * sigma**2) * np.exp(-((grid.X-0.5)**2)/(2*sigma**2))
    V[:, 0] = np.exp(-(np.abs(grid.MU)**2)/(16*sigma**2))
    S[0, 0] = 1.0

    sigma_x = 1e-1
    sigma_mu = 1
    #U[:, 0] = 1/(2 * np.pi * sigma_x**2) * np.exp(-((grid.X-0.5)**2)/(2*sigma_x**2))
    #V[:, 0] = np.exp(-(np.abs(grid.MU)**2)/(sigma_mu**2))
    #S[0, 0] = 1.0

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

    # Alternatively: (slower but easier to understand)
    # inner1 = lr.V.T @ np.diag(grid.MU) @ lr.V
    # C1 = inner1 * grid.dmu

    # muV = grid.MU[:, None] * lr.V
    # C1 = lr.V.T @ muV * grid.dmu         #still should implement trapezoidal instead of just multiplying by dmu everywhere

    # C2 = lr.V * grid.dmu

    C1 = (lr.V.T @ np.diag(grid.MU) @ lr.V) * grid.dmu

    # C2 = lr.V * grid.dmu

    C2 = (lr.V.T @ np.ones((grid.Nmu,grid.Nmu))).T * grid.dmu

    return C1, C2

def computeB(L, grid):

    # B1 = L * grid.dmu

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
    # rhs = - D1 @ L @ np.diag(grid.MU) + 0.5 * B1 - L
    rhs = - np.diag(grid.MU) @ L @ D1.T + 0.5 * B1 - L
    return rhs

def integrate(lr0, grid, t_f, dt):
    lr = lr0
    t = 0
    time = []
    while t < t_f:
        if (t + dt > t_f):
            dt = t_f - t

        t += dt
        time.append(t)

        # K step
        C1, C2 = computeC(lr, grid)
        K = lr.U @ lr.S
        K += dt * RK4(K, lambda K: Kstep(K, C1, C2, grid), dt)
        lr.U, lr.S = np.linalg.qr(K, mode="reduced")
        lr.U /= np.sqrt(grid.dx)
        lr.S *= np.sqrt(grid.dx)

        # S step
        D1 = computeD(lr, grid)
        lr.S += dt * RK4(lr.S, lambda S: Sstep(S, C1, C2, D1), dt) # Projector-splitting integrator + or - here?

        # L step
        # L = lr.S @ lr.V.T
        L = lr.V @ lr.S.T
        B1 = computeB(L, grid)
        L += dt * RK4(L, lambda L: Lstep(L, D1, B1, grid), dt)
        # lr.S, Vt = np.linalg.qr(L, mode="reduced")
        # lr.V = Vt.T
        lr.V, St = np.linalg.qr(L, mode="reduced")
        lr.S = St.T
        lr.V /= np.sqrt(grid.dmu)
        lr.S *= np.sqrt(grid.dmu)

    return lr, time



r = 64
Nx = 64
Nmu = 64
dt = 1e-3
t_f = 1.0

t_f_array = np.linspace(0.0, t_f, 11)
grid = Grid(Nx, Nmu, r)
lr0 = setInitialCondition(grid)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr, time = integrate(lr0, grid, t_f, dt)
f = lr.U @ lr.S @ lr.V.T

extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

plt.subplot(1, 2, 1)
plt.imshow(f0.T, extent=extent, origin='lower')
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("mu")
plt.title("t=0")

plt.subplot(1, 2, 2)
plt.imshow(f.T, extent=extent, origin='lower')
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("mu")
plt.title("t=1")
plt.show()