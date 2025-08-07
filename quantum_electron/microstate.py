import numpy as np
from .utils import r2xy
from matplotlib import pyplot as plt
from matplotlib import patheffects as pe
from numpy.typing import ArrayLike


class Microstate:
    def __init__(self,n_electrons: int = 0, helium_z: float = 1.0)-> None:
        """Class to store coordinates and electric field of elecrons. Plts electron positions

        """
        self.positions = None
        self.helium_z = helium_z
        self.n = n_electrons

    def plot_positions(self, ax=None, color: str = 'mediumseagreen', marker_size: float = 8.0) -> None:
        """Plot electron positions obtained from get_electron_positions

        Args:
            res (dict): Results dictionary from scipy.optimize.minimize
            ax (_type_, optional): Matplotlib axes object. Defaults to None.
            color (str, optional): Color of the markers representing the electrons. Defaults to 'mediumseagreen'.
        """
        x, y = r2xy(self.positions)

        if ax is None:
            plt.plot(x*1e6, y*1e6, 'ok', mfc=color, mew=0.5, ms=marker_size,
                     path_effects=[pe.SimplePatchShadow(), pe.Normal()])
        else:
            ax.plot(x*1e6, y*1e6, 'ok', mfc=color, mew=0.5, ms=marker_size,
                    path_effects=[pe.SimplePatchShadow(), pe.Normal()])

    def field(self) -> ArrayLike:
        # ke [V*m/C]
        ke = 8.987551792e9
        # kee [V*m*(1e6 mkm/m)= V*mkm]
        kee = (ke*1.602176e-19)*1e6
        x, y = r2xy(self.positions)
        charges = np.stack((x*1e6, y*1e6, np.full(x.size, self.helium_z)), axis=1)
        images = np.stack((x*1e6, y*1e6, np.full(x.size, -self.helium_z)), axis=1)
        field  = np.zeros(x.size)
        for i,charge in enumerate(charges):
            efield_tmp = np.zeros(3)
            for j, image in enumerate(images):
                r = charge-image
                # efield [(V*mkm)*mkm/(mkm**3)= V*mkm]
                efield = kee*r/(np.sqrt(r.dot(r)))**3
                efield_tmp = efield_tmp + efield
            field[i] = efield_tmp[2]
        # return result in V//mkm *(10000 mkm/cm) = V/cm
        return field*10000