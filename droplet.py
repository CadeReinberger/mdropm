from dataclasses import dataclass
import numpy as np
from scipy.optimize import root_scalar
from ca_bdry_computer import compute_p_star_over_gamma

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

    def make_drop_about_to_drop(n, R, HL, hs, pps):
        N = 4*n
        theta = np.linspace(0, 2*np.pi, num=N+1)[:-1]
        x, y = R*np.cos(theta), R*np.sin(theta)
        h = np.array([hs.h(xv, yv) for (xv, yv) in zip(x, y)])
        dh_dn = np.array([np.cos(theta[ind])*hs.hx(x[ind], y[ind]) + np.sin(theta[ind])*hs.hy(x[ind], y[ind]) for ind in range(N)])
        psi = np.arctan(dh_dn)
        if np.isclose(max(h), min(h)):
            # Constant Height case must be treated separately
            ca_theta = pps.theta_r
            return droplet(N, 2*np.pi, theta, x, y, ca_theta)
        # Otherwise we use the standard logic
        p_star_over_gamma = compute_p_star_over_gamma(hs, R, HL, pps) 
        ca_theta = -psi + np.arccos(-p_star_over_gamma * h)
        print(f'ca_theta: {ca_theta}')
        return droplet(N, 2*np.pi, theta, x, y, ca_theta)

