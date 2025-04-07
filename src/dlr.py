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
    time.append(t)
    # lr_array = []     # only needed for error plots
    # lr_array.append(lr.U @ lr.S @ lr.V.T)
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

        # lr_array.append(lr.U @ lr.S @ lr.V.T)

    return lr, time #, lr_array


''' # Original plotting
r = 32
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
'''

''' # New plots with 4 plots in one figure
Nx = 64
Nmu = 64
dt = 1e-3
t_f = 1.0

fig, axes = plt.subplots(2, 2, figsize=(8, 8))

grid = Grid(Nx, Nmu, 4)
lr0 = setInitialCondition(grid)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr, time = integrate(lr0, grid, t_f, dt)
f = lr.U @ lr.S @ lr.V.T
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

im1 = axes[0, 0].imshow(f.T, extent=extent, origin='lower')
axes[0, 0].set_title("$r=4$")
axes[0, 0].set_xlabel("$x$")
axes[0, 0].set_ylabel("mu")

grid = Grid(Nx, Nmu, 8)
lr0 = setInitialCondition(grid)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr, time = integrate(lr0, grid, t_f, dt)
f = lr.U @ lr.S @ lr.V.T
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

im2 = axes[0, 1].imshow(f.T, extent=extent, origin='lower')
axes[0, 1].set_title("$r=8$")
axes[0, 1].set_xlabel("$x$")
axes[0, 1].set_ylabel("mu")

grid = Grid(Nx, Nmu, 16)
lr0 = setInitialCondition(grid)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr, time = integrate(lr0, grid, t_f, dt)
f = lr.U @ lr.S @ lr.V.T
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

im3 = axes[1, 0].imshow(f.T, extent=extent, origin='lower')
axes[1, 0].set_title("$r=16$")
axes[1, 0].set_xlabel("$x$")
axes[1, 0].set_ylabel("mu")

grid = Grid(Nx, Nmu, 32)
lr0 = setInitialCondition(grid)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr, time = integrate(lr0, grid, t_f, dt)
f = lr.U @ lr.S @ lr.V.T
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

im4 = axes[1, 1].imshow(f.T, extent=extent, origin='lower')
axes[1, 1].set_title("$r=32$")
axes[1, 1].set_xlabel("$x$")
axes[1, 1].set_ylabel("mu")

fig.colorbar(im1, ax=axes[0, 0])
fig.colorbar(im2, ax=axes[0, 1])
fig.colorbar(im3, ax=axes[1, 0])
fig.colorbar(im4, ax=axes[1, 1])

plt.tight_layout()
plt.savefig("C:/Users/brunn/OneDrive/Dokumente/00_Uni/Masterarbeit/PHD_project_master_thesis/Second_results/Plots_low_rank/test.pdf")
'''

''' # Error plots
Nx = 64
Nmu = 64
dt = 1e-3
t_f = 5.0

grid = Grid(Nx, Nmu, 64)
lr0 = setInitialCondition(grid)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr64, time64, lr_array64 = integrate(lr0, grid, t_f, dt)

grid = Grid(Nx, Nmu, 4)
lr0 = setInitialCondition(grid)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr4, time4, lr_array4 = integrate(lr0, grid, t_f, dt)

grid = Grid(Nx, Nmu, 8)
lr0 = setInitialCondition(grid)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr8, time8, lr_array8 = integrate(lr0, grid, t_f, dt)

grid = Grid(Nx, Nmu, 16)
lr0 = setInitialCondition(grid)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr16, time16, lr_array16 = integrate(lr0, grid, t_f, dt)

grid = Grid(Nx, Nmu, 32)
lr0 = setInitialCondition(grid)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr32, time32, lr_array32 = integrate(lr0, grid, t_f, dt)



error_array4=[]
error_array8=[]
error_array16=[]
error_array32=[]
for i in range(len(lr_array64)):
    error_array4.append(np.linalg.norm(lr_array64[i]-lr_array4[i]))
    error_array8.append(np.linalg.norm(lr_array64[i]-lr_array8[i]))
    error_array16.append(np.linalg.norm(lr_array64[i]-lr_array16[i]))
    error_array32.append(np.linalg.norm(lr_array64[i]-lr_array32[i]))

plt.semilogy(time64, error_array4, label='$r=4$')
plt.semilogy(time64, error_array8, label='$r=8$')
plt.semilogy(time64, error_array16, label='$r=16$')
plt.semilogy(time64, error_array32, label='$r=32$')
plt.xlabel("t")
plt.ylabel("error")
plt.legend()
plt.show()
'''

# Plots for different times
Nx = 256
Nmu = 256
dt = 1e-3
r = 16

grid = Grid(Nx, Nmu, r)
lr0 = setInitialCondition(grid)
f0 = lr0.U @ lr0.S @ lr0.V.T
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

lr1, time1 = integrate(lr0, grid, 1.0, dt)
f1 = lr1.U @ lr1.S @ lr1.V.T

plt.subplot(1, 3, 1)
plt.imshow(f1.T, extent=extent, origin='lower')
plt.colorbar(orientation='horizontal', pad=0.08, fraction=0.035)
plt.xlabel("$x$")
plt.ylabel("mu")
plt.title("t=1")

lr3, time1 = integrate(lr0, grid, 3.0, dt)
f3 = lr3.U @ lr3.S @ lr3.V.T

plt.subplot(1, 3, 2)
plt.imshow(f3.T, extent=extent, origin='lower')
plt.colorbar(orientation='horizontal', pad=0.08, fraction=0.035)
plt.xlabel("$x$")
plt.ylabel("mu")
plt.title("t=3")

lr5, time1 = integrate(lr0, grid, 5.0, dt)
f5 = lr5.U @ lr5.S @ lr5.V.T

plt.subplot(1, 3, 3)
plt.imshow(f5.T, extent=extent, origin='lower')
plt.colorbar(orientation='horizontal', pad=0.08, fraction=0.035)
plt.xlabel("$x$")
plt.ylabel("mu")
plt.title("t=5")
plt.show()
