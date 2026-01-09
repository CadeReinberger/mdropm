import numpy as np

class EXTERNAL_BC_TYPES:
    HOMOGENEOUS_NEUMANN = 'HOMOGENEOUS_NEUMANN'
    HOMOGENEOUS_DIRICHLET = 'HOMOGENEOUS_DIRICHLET'
    
class BC_TYPES:
    HOMOGENEOUS_NEUMANN = 1
    HOMOGENEOUS_DIRICHLET = 2
    INHOMOGENEOUS_DIRICHLET = 3

class EXTERNAL_BC_SEGMENT:
    def __init__(self, start_pt, end_pt, bc_type):
        self.start_pt = start_pt
        self.end_pt = end_pt
        self.bc_type = bc_type

        def produce_bc_marker(self):
            start = self.start_pt
            diff_vec = self.end_pt - self.start_pt
            orth_vec = np.array([[0, -1], [1, 0]]) @ diff_vec
            # TODO: This will break if we have two separate line segments on the same line
            marker = lambda x : np.isclose(orth_vec[0] * (x[0]-start[0]) + orth_vec[1] * (x[1]-start[1]), 0)
            return marker

class EXTERNAL_BCS:
    def __init__(self, segments):
        self.segments = segments

    def picklable(self):
        n = len(self.segments)
        xs = np.array([seg.start_pt[0] for seg in self.segments])
        ys = np.array([seg.start_pt[1] for seg in self.segments])
        types = [seg.bc_type for seg in self.segments]
        picklable_dict = {'n' : n,
                          'xs' : xs,
                          'ys' : ys,
                          'types' : types}
        return picklable_dict


