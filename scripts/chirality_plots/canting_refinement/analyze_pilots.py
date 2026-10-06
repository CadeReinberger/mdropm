"""Exploratory comparisons; no fitted/smoothed values are substituted for data."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from canting_plotter import load_pickle, estimate_slope, _huber_location


def samples(path):
    p = json.loads((path.parent / 'manifest.json').read_text())['physical_parameters']
    out = load_pickle(path)
    tt = np.asarray(out['out_t'])
    states = np.asarray(out['out_x'])
    keep = (tt >= 1/18) & (tt <= 2.84)
    x, y, theta = np.split(states[keep], 3, axis=1)
    width = p['casr_smoothing_rad']
    sigma = (p['k_a'] * width * np.logaddexp(0, (theta-p['theta_a'])/width)
             - p['k_r'] * width * np.logaddexp(0, (p['theta_r']-theta)/width))
    D = lambda a: (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1))/2
    dx, dy = D(x), D(y)
    tangent = np.hypot(dx, dy)
    vx, vy = sigma*dy/tangent, -sigma*dx/tangent
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = .2*(dx*D(vy)-dy*D(vx))/(tangent**2*np.abs(sigma))
    mirror = (x.shape[1]//2-np.arange(x.shape[1])) % x.shape[1]
    xm, ym = (x-x[:, mirror])/2, (y+y[:, mirror])/2
    valid = ((xm > .25) & np.isfinite(alpha) & np.isfinite(alpha[:, mirror])
             & (np.abs(sigma) > 1e-6) & (np.abs(sigma[:, mirror]) > 1e-6))
    vals = np.rad2deg((alpha-alpha[:, mirror])[valid]/(2*xm[valid]))
    distance_sq = ((xm-1.35)**2+(ym-2.7*np.cos(np.pi/6))**2)[valid]
    area = (np.abs(sigma)*tangent*np.gradient(tt[keep])[:, None])[valid]
    return vals, distance_sq, area, np.c_[xm[valid], ym[valid]]


def weighted_huber(values, weights):
    good = np.isfinite(values) & (weights > 1e-12*weights.max())
    values, weights = values[good], weights[good]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    median = np.interp(.5, np.cumsum(weights)/sum(weights), values)
    deviation = abs(values-median)
    order = np.argsort(deviation)
    scale = max(1e-12, 1.4826*np.interp(.5, np.cumsum(weights[order])/sum(weights), deviation[order]))
    location = median
    for _ in range(60):
        rw = np.minimum(1, 1.5*scale/np.maximum(abs(values-location), 1e-20))
        new = np.average(values, weights=weights*rw)
        if abs(new-location) < 1e-10:
            return new
        location = new
    return location


def analyze(path):
    vals, d2, area, xy = samples(path)
    gaussian = np.exp(-d2/(2*.3**2))
    compact = np.maximum(0, 1-d2/.3**2)**2
    field = LinearNDInterpolator(xy, vals)
    return dict(spline=estimate_slope(path)[0], direct=_huber_location(vals, gaussian),
                area_mean=np.average(vals, weights=gaussian*area),
                area_huber=weighted_huber(vals, gaussian*area),
                compact_mean=np.average(vals, weights=compact*area) if compact.sum() else float('nan'),
                compact_huber=weighted_huber(vals, compact*area) if compact.sum() else float('nan'),
                point=float(field([[1.35, 2.7*np.cos(np.pi/6)]])[0]))


if __name__ == '__main__':
    for name in sys.argv[1:] or ('pilot20', 'pilot40', 'pilot80', 'pilot160', 'mesh80', 'time80'):
        print(name, flush=True)
        for path in sorted((Path(__file__).resolve().parent/name).glob('*/results.pkl'),
                           key=lambda p: float(p.parent.name[7:])):
            print(path.parent.name, {k: round(v, 7) for k, v in analyze(path).items()}, flush=True)
