from dataclasses import dataclass
import numpy as np

@dataclass
class droplet:
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

    
class constructors:
    def make_circular_flat_drop(n, R, hs):
        theta = np.linspace(0, 2*np.pi, num=n+1)[:-1]
        x, y = R*np.cos(theta), R*np.sin(theta)
        dh_dn = np.array([np.cos(theta[ind])*hs.hx(x[ind], y[ind]) + np.sin(theta[ind])*hs.hy(x[ind], y[ind]) for ind in range(n)])
        psi = np.arctan(dh_dn)
        ca_theta = np.pi/2 - psi
        return droplet(n, 2*np.pi, theta, x, y, ca_theta)

        
