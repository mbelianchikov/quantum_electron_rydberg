import numpy as np
from typing import Dict, Optional
from numpy.typing import ArrayLike

from quantum_electron.utils import make_potential
from quantum_electron.electric_field import ElectricField
from quantum_electron import FullModel

class RydbergDeviceSimulator:
    def __init__(self, potential_dict: Dict[str, ArrayLike], field_dict: Dict[str, ArrayLike], voltage_dict: Dict[str, float],
                 include_screening: bool = False, screening_length: float = np.inf,
                 potential_smoothing: float = 5e-4, remove_unbound_electrons: bool = False, remove_bounds: Optional[tuple] = None,
                 trap_annealing_steps: list = [0.1] * 5, max_x_displacement: float = 0.2e-6, max_y_displacement: float = 0.2e-6) -> None:
        
        self.device_voltage = voltage_dict
        self.device_field = ElectricField(make_potential(field_dict, voltages=voltage_dict),field_dict['xlist'],field_dict['ylist'])
        self.device_field.updated = True

        #Initilize simulator wihtout electrons for plotting
        self.n_electrons = 0
        self.microstate = None
        
        self.potential = FullModel(potential_dict=potential_dict, voltage_dict=self.device_voltage)
