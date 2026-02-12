import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class solver_params:

    ### Gas Phase FEM Params ###
    #------------------------------------------#
    GAS_PHASE_DYNAMIC_LC = True
    GAS_PHASE_DEFAULT_LC = 1
    # For slide outside, slide-to-drop, and drop inside. Can be Nones
    GAS_PHASE_LC_KS = (.4, .4, .4)
    GAS_PHASE_MESH_FILE = 'internal/gas_msh.msh'
    #------------------------------------------#



    ### Liquid Phase FEM Params ####
    #------------------------------------------#
    LIQUID_PHASE_DEFAULT_LC = 1  
    LIQUID_PHASE_DYNAMIC_LC_K = .4 #.4 # Can be None
    LIQUID_PHASE_MESH_FILE = 'internal/liquid_msh.msh'
    #------------------------------------------#


    ### Overall FEM Params ###
    #------------------------------------------#
    FEM_ELT_ORDER = 3
    FEM_SHAPE_ORDER = 2 # For splines and shapes
    PROJECT_OUTSIDE_MESH = True
    #------------------------------------------#


    # Time Integration Paramaters
    #------------------------------------------#
    RADAU_DT = .4 # Minutes
    SUBDIV_RADAU = True
    RADAU_EVAL_LINSPACE_N = 10 
    RADAU_OUT_EVERY = 1
    T_FIN = 3.5
    VERBOSE = True

    END_AREA_RATIO = .75 # Make this 0 to not use
    CHECK_SELF_INTERSECTION_DT = .5 # Can be None
    #-------------------------------------------#
    

    # Some overall solver parameters
    #-------------------------------------------#
    OUTPUT_DIR = 'out/'
    PLOTTING_LIMS = ((-5, 5), (-5, 5)) # XLIM, YLIM
    #-------------------------------------------#

    def to_pickle(self):
        return asdict(self)


