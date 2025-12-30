from dataclasses import dataclass

@dataclass
class PHYSICAL_PARAMS:
    mu: float = 0.3222 * 100 # g/(cm*min)
    c_l: float = 1.49 / 119.378 # mol / cm^3 (from density and mm)
    c_g: float = 4.464e-5 # mol / cm^3 (ideal gas)
    D: float = 5.4 # cm^2/min, https://cdnsciencepub.com/doi/pdf/10.1139/v71-010
    w_eq: float = .25 # 1, eq mole fraction
