import numpy as np
import matplotlib.pyplot as plt

''' Programming steps

1) Generate grid for function f with variabels x [0,1] and mu [-1,1] 

2) Choose initial condition

3) Make quick, brute force implementation to update solution for time steps

'''

### Generate grid

class Grid:
    def __init__(self, Nx: int, Nmu: int):
        self.Nx = Nx
        self.Nmu = Nmu
        self.X = np.linspace(0.0, 1.0, Nx, endpoint=False)
        self.MU = np.linspace(-1.0, 1.0, Nmu, endpoint=False)
        self.dx = self.X[1] - self.X[0]
        self.dmu = self.MU[1] - self.MU[0]

### Set initial condition

def setInitialCondition(grid: Grid, option: str) -> np.ndarray:
    f0 = np.zeros((grid.Nx, grid.Nmu))
    sigma = 8e-2
    if option == "no_mu":
        xx = 1/(np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(((grid.X-0.5)**2)/(2*sigma))**2)
        f0[:] = xx
    elif option == "with_mu":
        xx = 1/(2 * np.pi * sigma**2) * np.exp(-((grid.X-0.5)**2)/(2*sigma**2))
        vv = np.exp(-(np.abs(grid.MU)**2)/(16*sigma**2))
        f0 = np.outer(xx, vv)
    return f0

### Implementation of solver

def integrate(f0: np.ndarray, grid: Grid, t_f: float, dt: float, epsilon: float, option: str):
    f = np.copy(f0)
    t = 0
    time = [0]
    rank = []
    rank.append(np.linalg.matrix_rank(f))
    while t < t_f:
        if (t + dt > t_f):
            dt = t_f - t

        f = f + dt * rhs(f, grid, epsilon, option)
        t += dt

        time.append(t)
        rank.append(np.linalg.matrix_rank(f))

    return f, time, rank

def rhs(f: np.ndarray, grid: Grid, epsilon: float, option: str):
    # integrate over mu to get rho
    rho = np.zeros((grid.Nx, grid.Nmu))
    rho[:] = (1/np.sqrt(2)) * np.trapz(f, grid.MU, axis=1)

    # do cen diff and rest
    res = np.zeros((grid.Nx, grid.Nmu))
    if option == "cen_diff":
        for k in range(0, grid.Nmu):
            for l in range(1, grid.Nx-1):
                res[k, l] = -(1/epsilon) * grid.MU[grid.Nmu-1-k] * (f[k, l+1] - f[k, l-1]) / (2 * grid.dx) + (1/epsilon**2) * ((1/np.sqrt(2)) * rho[k, l] - f[k, l])

            res[k, 0] = -(1/epsilon) * grid.MU[grid.Nmu-1-k] * (f[k, 1] - f[k, grid.Nx-1]) / (2 * grid.dx) + (1/epsilon**2) * ((1/np.sqrt(2)) * rho[k, 0] - f[k, 0])
            res[k, grid.Nx-1] = -(1/epsilon) * grid.MU[grid.Nmu-1-k] * (f[k, 0] - f[k, grid.Nx-2]) / (2 * grid.dx) + (1/epsilon**2) * ((1/np.sqrt(2)) * rho[k, grid.Nx-1] - f[k, grid.Nx-1])
    elif option == "upwind":
        for k in range(0, grid.Nmu):
            if grid.MU[grid.Nmu-1-k] >= 0:
                for l in range(1, grid.Nx):
                    res[k, l] = -(1/epsilon) * grid.MU[grid.Nmu-1-k] * (f[k, l] - f[k, l-1]) / (grid.dx) + (1/epsilon**2) * ((1/np.sqrt(2)) * rho[k, l] - f[k, l])
                    res[k, 0] = -(1/epsilon) * grid.MU[grid.Nmu-1-k] * (f[k, 0] - f[k, grid.Nx-1]) / (grid.dx) + (1/epsilon**2) * ((1/np.sqrt(2)) * rho[k, 0] - f[k, 0])
            elif grid.MU[grid.Nmu-1-k] < 0:
                for l in range(0, grid.Nx-1):
                    res[k, l] = -(1/epsilon) * grid.MU[grid.Nmu-1-k] * (f[k, l+1] - f[k, l]) / (grid.dx) + (1/epsilon**2) * ((1/np.sqrt(2)) * rho[k, l] - f[k, l])
                    res[k, grid.Nx-1] = -(1/epsilon) * grid.MU[grid.Nmu-1-k] * (f[k, 0] - f[k, grid.Nx-1]) / (grid.dx) + (1/epsilon**2) * ((1/np.sqrt(2)) * rho[k, grid.Nx-1] - f[k, grid.Nx-1])
    # I want low k to be at bottom of graph, high k to be at the top (thus restructure res)
    return(res)      

# First simulation
### Check initial condition

grid = Grid(64, 64)
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]
f0 = setInitialCondition(grid, "with_mu")
plt.subplot(1, 3, 1)
plt.imshow(f0, extent=extent)
plt.colorbar(orientation='horizontal', pad=0.08, fraction=0.035)
plt.xlabel("$x$")
plt.ylabel("mu")
plt.title("inital values")

### Do simulations

f1 = integrate(f0, grid, 1, 1e-3, 1, "upwind")[0]
plt.subplot(1, 3, 2)
plt.imshow(f1, extent=extent)
plt.colorbar(orientation='horizontal', pad=0.08, fraction=0.035)
plt.xlabel("$x$")
plt.ylabel("mu")
plt.title("upwind")

f2 = integrate(f0, grid, 1, 1e-3, 1, "cen_diff")[0]
plt.subplot(1, 3, 3)
plt.imshow(f2, extent=extent)
plt.colorbar(orientation='horizontal', pad=0.08, fraction=0.035)
plt.xlabel("$x$")
plt.ylabel("mu")
plt.title("cen_diff")
plt.show()

# Print singular values over time
grid = Grid(64, 64)
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]
f0 = setInitialCondition(grid, "with_mu")
tfinal = np.linspace(0,1,11)
for t in tfinal:
    f = integrate(f0, grid, t, 1e-3, 1, "cen_diff")[0]
    print(np.linalg.svd(f)[1])

# Plot rank of solution over time
grid = Grid(64, 64)
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]
f0 = setInitialCondition(grid, "with_mu")
tfinal = np.linspace(0,10,11)
f_rank = [np.linalg.matrix_rank(integrate(f0, grid, t, 1e-3, 1, "cen_diff")[0], tol=1e-2) for t in tfinal]
fig, ax = plt.subplots()
ax.plot(tfinal, f_rank)
ax.set_xlabel("$t$")
ax.set_ylabel("rank $r(t)$")
plt.show()
