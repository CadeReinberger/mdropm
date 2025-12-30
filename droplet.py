from dataclasses import dataclass
import numpy as np

@datclass
class DROPLET:
    n: int # Number of Nodes on the droplet
    L: float # S^1 considered as [0, L) for our purposes
    s: np.array # s-values of samples points
    x: np.array # x-values of sampled points
    y: np.array # y-values of sampled points
    theta: np.array # theta-values of sampled points

    def to_pickle_dict(self):
        return {'n' : self.n,
                'L' : self.L, 
                's' : self.s,
                'x' : self.x, 
                'y' : self.y,
                'theta' : self.theta}
    
    def to_vec(self):
        vec = np.zeros(3 * self.n)
        vec[0:n] = self.x
        vec[n:2*n] = self.y
        vec[2*n:3*n] = self.theta
        return (self.n, self.L, self.s, self.vec)

    def from_vec(n, L, s, vec):
        x = vec[0:n]
        y = vec[n:2*n]
        theta = vec[2*n:3*n]
        return DROPLET(n, L, s, x, y, theta)


