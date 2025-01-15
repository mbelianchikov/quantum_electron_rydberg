import numpy as np
from typing import Dict, Optional
from numpy.typing import ArrayLike

from quantum_electron.utils import make_potential, r2xy
from quantum_electron.electric_field import ElectricField
from quantum_electron.rydberg_utils import plot_electron_positions, Lorenz
from quantum_electron import FullModel
from quantum_electron.microstate import Microstate

from scipy.interpolate import RectBivariateSpline, CubicSpline

from quantum_electron.initial_condition import InitialCondition

from quantum_electron.field_data_3 import tab_data


class RydbergDeviceSimulator:  
    def __init__(self, p_coupling_coeff: Dict[str, ArrayLike], f_coupling_coeff: Dict[str, ArrayLike], helium_z: float) -> None:
        """
        Class integrates FullModel object with microstate, device_field and charge_field objects.
        FullModel handles potential data and solve electrons ground state. Microstate keeps electrons postions, fields and handles position plotting. 
        Device_field and charge_field keeps 2D Ez field data from device electrostatics and image charge, respectively, and also handles field plotting and manipulations routines.

        """

        self.device_potential_coupl = p_coupling_coeff
        self.device_electrodes = list(self.device_potential_coupl.keys())[:-2]
        self.device_field_coupl = f_coupling_coeff
        self.device_voltage = dict(zip(self.device_electrodes, np.zeros(len(self.device_electrodes))))
        
        
        self.device_field = ElectricField(make_potential(self.device_field_coupl, voltages=self.device_voltage),self.device_field_coupl['xlist'],self.device_field_coupl['ylist'])
        self.device_field.updated = True

        self.charge_field = None

        self.helium_z = helium_z

        #Initilize simulator wihtout electrons for plotting
        self.microstate = Microstate(helium_z = self.helium_z)
        

    def load(self, typ = "", **kwarg):
        """Create microstate through InitialCondition object. 

        Args:

        Returns:
            ArrayLike: microstate of optimized charge ground state
        """
        charge_source = InitialCondition(potential_dict=self.device_potential_coupl, voltage_dict=self.device_voltage)
        
        if typ == "rectangular":
            self.microstate.positions = charge_source.make_rectangular(**kwarg)
            
        if typ == "chemical_potential":
            self.microstate.positions = harge_source.make_by_chemical_potential(**kwarg)

        self.microstate.n = len(self.microstate.positions) //2

    
    
    def set_voltage(self, **voltage_set):
        """Method to update self.device_voltage and recalculate device potential and field. Charge field remains not updated due to computational overhood.
        Device field and device potential marked as updated; microstate, charge field and charge vector are marked as not updated. 

        Args:

        Returns:
            None
        """   
        for key in voltage_set.keys():
            self.device_voltage[key] = voltage_set[key]
        
        self.device_field.field = make_potential(self.device_field_coupl, voltages=self.device_voltage)
        self.device_field.updated = True

        
        
    def solve(self):
        """Method to solve groundstate and returns microstate. Microstate, and charge vector are marked as updated.

        Args:
        
        Returns:
            ArrayLike: microstate of optimized charge ground state
        """
        solver = FullModel(potential_dict=self.device_potential_coupl, voltage_dict=self.device_voltage)
        res = solver.get_electron_positions(n_electrons=self.microstate.n, electron_initial_positions=self.microstate.positions)
        self.microstate.positions = res.x
        self.microstate.n = len(res.x)//2


    def device_field_vector(self, kx=3, ky=3, s=0) -> ArrayLike:

        """Vector of z component of electric field for every electron.

        Args:
            xi (ArrayLike): electron x-positions np.array([x0, x1, ...])
            yi (ArrayLike): electron y-positions np.array([y0, y1, ...])

        Returns:
            Vector of float: Ez electric field for each electron in V/mkm units.
        """
        x, y = r2xy(self.microstate.positions)
        interpolator = RectBivariateSpline(self.device_field.xlist, self.device_field.ylist, self.device_field.field,
                                        kx=kx, ky=ky, s=s)
        V = interpolator.ev(x*1e6, y*1e6)
    
        return -V*10000

    def total_field_vector(self) -> ArrayLike:
        return self.device_field_vector()+self.microstate.field()

    def freq(self) -> ArrayLike:
        C_int = CubicSpline(tab_data[0,:], tab_data[1,:])
        return C_int(self.total_field_vector())

    def generate_spectra(self, freq_start=100, freq_stop=1000, step = 0.5, G=1.0)-> ArrayLike:
        freq_vector = np.arange(freq_start, freq_stop, step)
        ypoints = np.zeros(freq_vector.size)
        for f in self.freq():
            ypoints = ypoints + Lorenz(freq_vector, f, G)
        return ypoints
        