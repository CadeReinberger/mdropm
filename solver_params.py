import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class solver_params:

    ### Gas Phase FEM Params ###
    #------------------------------------------#
    GAS_PHASE_DYNAMIC_LC = True
    GAS_PHASE_DEFAULT_LC = .6
    # For slide outside, slide-to-drop, and drop inside. Can be Nones
    GAS_PHASE_LC_KS = (.4, .4, .4)
    GAS_PHASE_MESH_FILE = 'internal/gas_msh.msh'
    #------------------------------------------#



    ### Liquid Phase FEM Params ####
    #------------------------------------------#
    LIQUID_PHASE_DEFAULT_LC = .6  
    LIQUID_PHASE_DYNAMIC_LC_K = .4 #.4 # Can be None
    LIQUID_PHASE_MESH_FILE = 'internal/liquid_msh.msh'
    #------------------------------------------#


    ### Overall FEM Params ###
    #------------------------------------------#
    FEM_ELT_ORDER = 3 # idk what's best 
    FEM_SHAPE_ORDER = 2 # For splines and shapes
    PROJECT_OUTSIDE_MESH = True
    GMSH_VERBOSE = False
    # Evaluate FEM gradients just inside their domain rather than ambiguously
    # on a boundary shared by multiple cells. Expressed as a fraction of lc.
    GRADIENT_EVAL_OFFSET_LC = 0.1
    #------------------------------------------#


    # Time Integration Paramaters
    #------------------------------------------#
    RADAU_DT = 2 # Minutes
    SUBDIV_RADAU = True
    RADAU_EVAL_LINSPACE_N = 10 
    RADAU_OUT_EVERY = 1
    ODE_METHOD = 'LSODA'
    ODE_RTOL = 1e-3
    ODE_ATOL = 1e-6
    ODE_THETA_JACOBIAN = False
    ODE_JAC_EPS = 1e-5
    T_FIN = 9.9
    VERBOSE = True

    END_AREA_RATIO = .5 # Make this 0 to not use
    CHECK_SELF_INTERSECTION_DT = .5 # Can be None
    #-------------------------------------------#
    

    # Some overall solver parameters
    #-------------------------------------------#
    OUTPUT_DIR = 'out/'
    PLOTTING_LIMS = ((-5, 5), (-5, 5)) # XLIM, YLIM
    #-------------------------------------------#

    def to_pickle(self):
        # These settings are class attributes, so dataclasses.asdict is empty.
        return {name: getattr(self, name) for name in dir(self)
                if name.isupper() and not callable(getattr(self, name))}
