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
    _coeff
        1/epsilon for radiative transfer equation on this domain.
    """
    def __init__(self, _Nx: int, _Nmu: int, _r: int = 5, _option_bc: str = "inflow", _X = None, _coeff : float = 1.0):
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
        self.coeff = _coeff

    def split(self, _coeff_left = None, _coeff_right = None):
        """
        Split domain into 2 subdomains.

        Split the domain into 2 subdomains by dividing the domain in half in the middle of the X grid.

        Parameters
        ----------
        _coeff_left
            1/epsilon for radiative transfer equation on left subdomain. If None, value from whole domain is taken.
        _coeff_right
            1/epsilon for radiative transfer equation on right subdomain. If None, value from whole domain is taken.
        """
        if _coeff_left is None:
            _coeff_left = self.coeff
        if _coeff_right is None:
            _coeff_right = self.coeff

        # Split grid
        X_left = self.X[:int(self.Nx/2)]
        X_right = self.X[int(self.Nx/2):]

        # Create new Grid instances for left and right
        left_grid = Grid_1x1d(int(self.Nx/2), self.Nmu, self.r, _X=X_left, _coeff = _coeff_left)
        right_grid = Grid_1x1d(int(self.Nx/2), self.Nmu, self.r, _X=X_right, _coeff = _coeff_right)

        return left_grid, right_grid
    

class Grid_2x1d:
    """
    Generate 2x1 dimensional grid.

    Helps to generate an equidistant grid. For calculations with or without domain decomposition and periodic boundary conditions.
    Angle domain is set from [0, 2*pi]. Spacial domain is [0,1]x[0,1].
    For simulations with domain decomposition set _option_dd to dd

    Parameters
    ----------
    _Nx : int
        Number of gridpoints in x.
    _Ny : int
        Number of gridpoints in y.
    _Nphi : int
        Number of gridpoints in phi.
    _r : int
        Initial rank of the simulation.
    _option_dd : str
        Can be chosen either "dd" or "no_dd"
    _X
        Optional X grid, given as np.array. Standard value is interval [0,1].
    _Y
        Optional Y grid, given as np.array. Standard value is interval [0,1].
    _coeff
        1/epsilon for radiative transfer equation on this domain.
    """
    def __init__(self, _Nx: int, _Ny: int, _Nphi: int, _r: int = 5, _option_dd : str = "no_dd", _X = None, _Y = None, _coeff : float = 1.0):
        self.Nx = _Nx
        self.Ny = _Ny
        self.Nphi = _Nphi
        self.r = _r
        self.coeff = _coeff

        if _option_dd == "no_dd":
            self.X = np.linspace(0.0, 1.0, self.Nx, endpoint=False)          # Point 0 is on the grid, point 1 is not on the grid
            self.Y = np.linspace(0.0, 1.0, self.Ny, endpoint=False)          # Point 0 is on the grid, point 1 is not on the grid
        elif _option_dd == "dd":
            if _X is None:
                self.X = np.linspace(1/(2*self.Nx), 1 - 1/(2*self.Nx), self.Nx, endpoint=True) # Point 0 and 1 are not on the grid, spacing to first gridpoint: delta_x/2
            else:
                self.X = _X

            if _Y is None:
                self.Y = np.linspace(1/(2*self.Ny), 1 - 1/(2*self.Ny), self.Ny, endpoint=True) # Point 0 and 1 are not on the grid, spacing to first gridpoint: delta_y/2
            else:
                self.Y = _Y
        self.PHI = np.linspace(0.0, 2*np.pi, self.Nphi, endpoint=False)        # 2*pi is the same angle as 0
        
        self.dx = self.X[1] - self.X[0]
        self.dy = self.Y[1] - self.Y[0]
        self.dphi = self.PHI[1] - self.PHI[0]

    def split_x(self, _coeff_left = None, _coeff_right = None):
        """
        Split domain into 2 subdomains in x dimension.

        Split the domain into 2 subdomains by dividing the domain in half in the middle of the X grid.

        Parameters
        ----------
        _coeff_left
            1/epsilon for radiative transfer equation on left subdomain. If None, value from whole domain is taken.
        _coeff_right
            1/epsilon for radiative transfer equation on right subdomain. If None, value from whole domain is taken.
        """
        if _coeff_left is None:
            _coeff_left = self.coeff
        if _coeff_right is None:
            _coeff_right = self.coeff

        # Split grid
        X_left = self.X[:int(self.Nx/2)]
        X_right = self.X[int(self.Nx/2):]

        # Create new Grid instances for left and right
        left_grid = Grid_2x1d(int(self.Nx/2), self.Ny, self.Nphi, self.r, _option_dd = "dd", _X=X_left, _Y=self.Y, _coeff = _coeff_left)
        right_grid = Grid_2x1d(int(self.Nx/2), self.Ny, self.Nphi, self.r, _option_dd = "dd", _X=X_right, _Y=self.Y, _coeff = _coeff_right)

        return left_grid, right_grid
    
    def split_y(self, _coeff_bottom = None, _coeff_top = None):
        """
        Split domain into 2 subdomains in y dimension.

        Split the domain into 2 subdomains by dividing the domain in half in the middle of the Y grid.

        Parameters
        ----------
        _coeff_bottom
            1/epsilon for radiative transfer equation on bottom subdomain. If None, value from whole domain is taken.
        _coeff_top
            1/epsilon for radiative transfer equation on top subdomain. If None, value from whole domain is taken.
        """
        if _coeff_bottom is None:
            _coeff_bottom = self.coeff
        if _coeff_top is None:
            _coeff_top = self.coeff

        # Split grid
        Y_bottom = self.Y[:int(self.Ny/2)]
        Y_top = self.Y[int(self.Ny/2):]

        # Create new Grid instances for left and right
        bottom_grid = Grid_2x1d(self.Nx, int(self.Ny/2), self.Nphi, self.r, _option_dd = "dd", _X=self.X, _Y=Y_bottom, _coeff = _coeff_bottom)
        top_grid = Grid_2x1d(self.Nx, int(self.Ny/2), self.Nphi, self.r, _option_dd = "dd", _X=self.X, _Y=Y_top, _coeff = _coeff_top)

        return bottom_grid, top_grid