import numpy as np
from dataclasses import dataclass

@dataclass
class physical_params:
    # Liquid constants we'll need
    mu: float = 0.03222 * 100 # g/(mm*min) # 100 * that of chloroform
    gamma: float = .001 * 97560 # Chloroform W|A, (g/min^2)

    # Gas phase constants
    c_l: float = .001492 / 119.378 # mol / mm^3 (from density and mm)
    c_g: float = 4.0874e-8 # mol / cm^3 (ideal gas)
    D: float = .2 * 540 # mm^2/min, https://cdnsciencepub.com/doi/pdf/10.1139/v71-010
    w_eq: float = .25 # 1, eq mole fraction

    # CASR constants
    theta_a: float = np.deg2rad(120) # rad
    theta_r: float = np.deg2rad(50) # rad
    k_a: float = .05 * (.001 * 97560/(.03222*100))/np.deg2rad(30) # (mm/s)/rad
    k_r: float = .05 * (.001 * 97560/(.03222*100))/np.deg2rad(30) # (mm/s)/rad
    
