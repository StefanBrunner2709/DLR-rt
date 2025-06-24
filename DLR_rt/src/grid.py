"""
Contains classes to set up grid.
"""

import numpy as np


class Grid_1x1d:
    """
    Generate 1x1 dimensional grid.

    Helps to generate an equidistant grid. Angular domain is set from [-1,1]. 
    Space can be set differently, but standard value is given by [0,1].

    Parameters
    ----------
    _Nx : int
        Number of gridpoints in x.
    _Nmu : int
        Number of gridpoints in mu.
    _r : int
        Initial rank of the simulation.
    _option_bc : str
        Can be chosen either "inflow" or "periodic".
    _X
        Optional X grid, given as np.array. Standard value is interval [0,1].
    """
    def __init__(self, _Nx: int, _Nmu: int, _r: int = 5, _option_bc: str = "inflow", _X = None):
        self.Nx = _Nx
        self.Nmu = _Nmu
        self.r = _r
        self.option_bc = _option_bc

        if _option_bc == "inflow":
            if _X is None:
                self.X = np.linspace(0.0, 1.0, self.Nx+1, endpoint=False)[1:]    # Point 0 and 1 are not on our grid
            else:
                self.X = _X                                              # If somebody wants to directly give an X domain
            self.MU = np.linspace(-1.0, 1.0, self.Nmu, endpoint=True)        # For mu we don't have boundary conditions
        elif _option_bc == "periodic":
            self.X = np.linspace(0.0, 1.0, self.Nx, endpoint=False)          # Point 0 is on the grid, point 1 is not on the grid
            self.MU = np.linspace(-1.0, 1.0, self.Nmu, endpoint=True)        # For mu we don't have boundary conditions
        
        self.dx = self.X[1] - self.X[0]
        self.dmu = self.MU[1] - self.MU[0]

    def split(self):
        """
        Split domain into 2 subdomains.

        Split the domain into 2 subdomains by dividing the domain in half in the middle of the X grid.
        """
        # Split grid
        X_left = self.X[:int(self.Nx/2)]
        X_right = self.X[int(self.Nx/2):]

        # Create new Grid instances for left and right
        left_grid = Grid_1x1d(int(self.Nx/2), self.Nmu, self.r, _X=X_left)
        right_grid = Grid_1x1d(int(self.Nx/2), self.Nmu, self.r, _X=X_right)

        return left_grid, right_grid
