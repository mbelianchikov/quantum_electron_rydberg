from numpy.typing import ArrayLike
from .utils import r2xy
from matplotlib import pyplot as plt
from matplotlib import patheffects as pe

def Lorenz(x, x0, G):
    return 0.5*G/((x-x0)**2+(0.5*G)**2)


def plot_electron_positions(state: ArrayLike, ax=None, color: str = 'mediumseagreen', marker_size: float = 10.0) -> None:
    """Plot electron positions obtained from get_electron_positions

    Args:
        res (dict): Results dictionary from scipy.optimize.minimize
        ax (_type_, optional): Matplotlib axes object. Defaults to None.
        color (str, optional): Color of the markers representing the electrons. Defaults to 'mediumseagreen'.
    """
    x, y = r2xy(state)

    if ax is None:
        plt.plot(x*1e6, y*1e6, 'ok', mfc=color, mew=0.5, ms=marker_size,
                     path_effects=[pe.SimplePatchShadow(), pe.Normal()])
    else:
        ax.plot(x*1e6, y*1e6, 'ok', mfc=color, mew=0.5, ms=marker_size,
                path_effects=[pe.SimplePatchShadow(), pe.Normal()])



def StarkShift(el_field_array: ArrayLike) -> ArrayLike:
    """Plot electron positions obtained from get_electron_positions

    Args:
        res (dict): Results dictionary from scipy.optimize.minimize
        ax (_type_, optional): Matplotlib axes object. Defaults to None.
        color (str, optional): Color of the markers representing the electrons. Defaults to 'mediumseagreen'.
    """
    data = np.loadtxt('stark_tab.txt')
    C_int = CubicSpline(data[0,:], data[1,:])
    return C_int(el_field_array*10000)