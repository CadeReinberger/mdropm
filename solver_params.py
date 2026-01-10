import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class solver_params:

    ### Gas Phase FEM Params ###
    #------------------------------------------#
    GAS_PHASE_DYNAMIC_LC = True
    GAS_PHASE_DEFAULT_LC = .02
    # For slide outside, slide-to-drop, and drop inside. Can be Nones
    GAS_PHASE_LC_KS = (.18, .18, .18)
    GAS_PHASE_MESH_FILE = 'internal/gas_msh.msh'
    #------------------------------------------#



    ### Liquid Phase FEM Params ####
    #------------------------------------------#
    LIQUID_PHASE_DEFAULT_LC = .02
    LQIUID_PHASE_DYNAMUC_LC_K = .2 # Can be None
    LIQUID_PHASE_MESH_FILE = 'internal/liquid_msh.msh'
    #------------------------------------------#


    ### Overall FEM Params ###
    #------------------------------------------#
    FEM_ELT_ORDER = 3
    FEM_SHAPE_ORDER = 2 # For splines and shapes
    #------------------------------------------#


    # Time Integration Paramaters
    #------------------------------------------#
    RADAU_DT = .05 # Minutes
    RADAU_EVAL_LINSPACE_N = 6 
    T_FIN = 30
    # All of these can be None to not use
    END_AREA_RATIO = .333 
    END_VOLUME_RATIO = .25 
    REMESH_SPLINE_TIMESTEP = None # Okay, this one's a pretty big deal
    CHECK_SELF_INTERSECTION_DT = .1
    #-------------------------------------------#
    

    # Some overall solver parameters
    #-------------------------------------------#
    OUTPUT_DIR = 'out/'
    PLOTTING_LIMS = ((-5, 5), (-5, 5)) # XLIM, YLIM
    #-------------------------------------------#

    def to_pickle(self):
        return asdict(self)


