import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, List
from matplotlib import pyplot as plt
import matplotlib

class ElectricField:
    def __init__(self, field: ArrayLike, xlist: ArrayLike, ylist: ArrayLike) -> None:
        self.field = field
        self.xlist = xlist
        self.ylist = ylist
        self.udated = False

    """This class stores Ez electric field map on rectangilar grid and handles basic operations: plot, plot_slice, append, crop, etc.
    
        Args:
            field (ArrayLike): 2D-array of Ez field points.
            xlist (ArrayLike): 1D-array of x coordinates points.
            ylist (ArrayLike): 1D-array of y coordinates points.
    """

    def plot_field(self, ax=None, coor: Optional[List[float]] = [0, 0], dxdy: List[float] = [1, 2],
                          figsize: tuple[float, float] = (7, 4), show_minimum: bool = False, plot_contours: bool = False,
                          clim: Optional[tuple] = None) -> None:
        """Plot the Ez electric field as function of (x,y)
        Args:
            ax (_type_, optional): Matplotlib axes object. Defaults to None.
            coor (List[float, float], optional): Center of the solution window (in microns), this should include the potential minimum. Defaults to [0,0].
            dxdy (List[float, float], optional): width of the solution window for x and y (measured in microns). Defaults to [1, 2].
            figsize (tuple[float, float], optional): Figure size that gets passed to matplotlib.pyplot.figure. Defaults to (7, 4).
            show_minimum (bool, optional): If True, it plots a star where the ratio is smallest. Defaults to False.
            clim (Optional[tuple], optional): Limits for the colorbar. Defaults to None.           
        """

        # Convert field from V/mkm -> V/cm
        zdata = -10000*self.field.T

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            make_colorbar = True
        else:
            make_colorbar = False

        if clim is not None:
            pcm = ax.pcolormesh(
                self.xlist, self.ylist, zdata, vmin=clim[0], vmax=clim[1], cmap=plt.cm.RdYlBu_r)
        else:
            pcm = ax.pcolormesh(
                self.xlist, self.ylist, zdata, cmap=plt.cm.RdYlBu_r)

        if make_colorbar:
            cbar = plt.colorbar(pcm, fraction=0.046, pad=0.04)
            tick_locator = matplotlib.ticker.MaxNLocator(nbins=4)
            cbar.locator = tick_locator
            cbar.update_ticks()
            cbar.ax.set_ylabel(r"Electric field $Ez(x,y), V/cm$")
        
        if show_minimum:
            xidx, yidx = np.unravel_index(zdata.argmin(), zdata.shape)
            ax.plot(self.xlist[yidx],
                self.ylist[xidx], '*', color='white')

        ax.set_xlim(coor[0] - dxdy[0]/2, coor[0] + dxdy[0]/2)
        ax.set_ylim(coor[1] - dxdy[1]/2, coor[1] + dxdy[1]/2)

        ax.set_aspect('equal')

        if plot_contours:
            contours = [np.round(np.min(zdata), 3) + k*1e-3 for k in range(5)]
            CS = ax.contour(
                self.xlist, self.ylist, zdata, levels=contours)
            ax.clabel(CS, CS.levels, inline=True, fontsize=10)

        ax.set_xlabel("$x$"+f" ({chr(956)}m)")
        ax.set_ylabel("$y$"+f" ({chr(956)}m)")
        ax.locator_params(axis='both', nbins=4)

        if ax is None:
            plt.tight_layout()
            
    def plot_field_slice(self, ax=None, x: ArrayLike = [], y: ArrayLike = [], axlims: Optional[tuple] = None, 
                             figsize: tuple[float, float] = (6, 3), tag: str = 'auto'):
        """Plot a field slice along x or y. To control the dimension, supply arguments in one of the two forms
        - x = [x0], y = np.linspace(ymin, ymax, ...) to plot the field vs. y at x = x0 OR
        - y = [y0], x = np.linspace(xmin, xmax, ...) to plot the field vs. x at y = y0

        Args:
            ax (_type_, optional): Matplotlib axes object. If None, a new instance will be created. Defaults to None.
            x (ArrayLike, optional): x values for the field slice. Must be at least of length 1. Defaults to [].
            y (ArrayLike, optional): y values for the field slice. Must be at least of length 1. Defaults to [].
            axlims (Optional[tuple], optional): Limits in eV of the vertical axis of the plot. Defaults to None.
            figsize (tuple[float, float], optional): Figure size in inches. Defaults to (6, 3).
            tag (str, optional): Label in the legend that goes into the legend. If auto, the label is either x0 or y0. Defaults to 'auto'.

        Raises:
            ValueError: If x and y are not according to the rules above, a ValueError is raised.
        """
        # Convert field from V/mkm -> V/cm
        zdata = -10000*self.field.T
        
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            
        if len(x) == 1: 
            # We are plotting along the y-axis for one particular value of x
            x_idx = np.argmin(np.abs(self.xlist - x[0]))
            label = rf"$x$ = {x[0]:.2f}"+f" {chr(956)}m" if tag == 'auto' else tag
            
            ax.plot(self.ylist, zdata[:, x_idx], label=label)
            ax.set_xlabel("$y$"+f" ({chr(956)}m)")
            ax.set_ylabel(r"Potential energy $-eV(x,y)$")
            if axlims is not None:
                ax.set_ylim(axlims)
                
            ax.set_xlim(np.min(y), np.max(y))
            ax.locator_params(axis='both', nbins=4)
            ax.legend(loc=0, frameon=False)
            
        elif len(y) == 1: 
            # We are plotting along the x-axis for one particular value of y
            y_idx = np.argmin(np.abs(self.ylist - y[0]))
            label=rf"$y$ = {y[0]:.2f}"+f" {chr(956)}m" if tag == 'auto' else tag
            
            ax.plot(self.xlist, zdata[y_idx, :], label=label)
            ax.set_xlabel("$x$"+f" ({chr(956)}m)")
            ax.set_ylabel(r"Potential energy $-eV(x,y)$")
            if axlims is not None:
                ax.set_ylim(axlims)
            ax.set_xlim(np.min(x), np.max(x))
            ax.locator_params(axis='both', nbins=4)
            ax.legend(loc=0, frameon=False)
            
        else:
            raise ValueError("At least one of 'x' or 'y' must contain only 1 element to indicate a slice along 'x' or 'y'.")
            