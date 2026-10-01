#!/usr/bin/env python3
"""Regenerate the numerical audit after the refined sweep and cant_clean.py."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from canting_plotter import SCRIPT_DIR, analyze_directory, plot_canting, load_pickle

ROOT = SCRIPT_DIR / 'canting_refinement'


def height(item):
    return 200 * (1 + item.height_ratio) / (1 - item.height_ratio)


def main():
    output = ROOT / 'audit'
    output.mkdir(exist_ok=True)
    primary = analyze_directory(ROOT / 'production80')
    if len(primary) < 24:
        raise RuntimeError('finish the 24-height sweep before publishing its audit')
    groups = {name: analyze_directory(ROOT / name) for name in ('pilot40', 'pilot160', 'mesh80', 'time80')}
    indexed = {round(item.height_ratio, 12): item for item in primary}
    checks = []
    for name, items in groups.items():
        for check in items:
            source = indexed.get(round(check.height_ratio, 12))
            if source is not None:
                checks.append(dict(check=name, front_height_um=height(source),
                                   primary_raw_deg_per_mm=source.chirality_slope,
                                   check_raw_deg_per_mm=check.chirality_slope,
                                   relative_difference_percent=100*(check.chirality_slope/source.chirality_slope-1)))
    with (output / 'convergence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    with (ROOT / 'baseline' / 'canting_clean.csv').open() as stream:
        old = [row for row in csv.DictReader(stream) if row['series'] == 'primary']
    fit = json.loads((SCRIPT_DIR / 'canting_clean_fit.json').read_text())
    scale = fit['calibration']['slope_multiplier']
    old_by_height = {round(float(row['h_front_um']), 6): float(row['chirality_slope_deg_per_mm']) for row in old}
    old_differences = []
    for item in primary:
        baseline = old_by_height.get(round(height(item), 6))
        if baseline is not None:
            current = scale*item.chirality_slope
            old_differences.append(dict(front_height_um=height(item), old_deg_per_mm=baseline,
                                        new_deg_per_mm=current, relative_change_percent=100*(current/baseline-1)))
    timings = [json.loads((item.source.parent/'timing.json').read_text()) for item in primary]
    minimum_areas = {phase: min(t[f'minimum_{phase}_mesh_area_ratio'] for t in timings
                               if t[f'minimum_{phase}_mesh_area_ratio'] is not None)
                     for phase in ('gas', 'liquid')}
    manifests = [json.loads((item.source.parent/'manifest.json').read_text()) for item in primary]
    if len({m['fem_implementation_sha256'] for m in manifests}) != 1:
        raise RuntimeError('production results mix FEM implementations')
    maximum_symmetry_error = 0.0
    for item in primary:
        result = load_pickle(item.source)
        if not np.isclose(result['out_t'][-1], 3):
            raise RuntimeError(f'incomplete trajectory: {item.source}')
        x, y, theta = np.split(np.asarray(result['out_x']), 3, axis=1)
        mirror = (x.shape[1]//2 - np.arange(x.shape[1])) % x.shape[1]
        maximum_symmetry_error = max(maximum_symmetry_error, float(np.max(abs(x+x[:, mirror]))),
                                    float(np.max(abs(y-y[:, mirror]))), float(np.max(abs(theta-theta[:, mirror]))))
    summary = dict(
        primary_points=len(primary), previous_primary_points=len(old),
        front_heights_um=[height(item) for item in primary],
        maximum_absolute_check_percent={name: max(abs(row['relative_difference_percent'])
            for row in checks if row['check'] == name) for name in groups},
        checks=checks, comparison_to_previous=old_differences,
        minimum_mesh_area_ratios=minimum_areas, maximum_mirror_symmetry_error=maximum_symmetry_error,
        calibration=fit['calibration'], unweighted_fit_rms_deg_per_mm=fit['unweighted_residual_rms_deg_per_mm'],
        warning='Post-processing bars exclude discretization, physical-model and arbitrary amplitude-calibration uncertainty.',
        manufactured_checks={str(path.relative_to(ROOT)): json.loads(path.read_text())
                             for path in sorted(ROOT.glob('diagnostic*/diagnostic-*.json'))},
        source_sha256={name: hashlib.sha256((SCRIPT_DIR/name).read_bytes()).hexdigest()
                       for name in ('canting_refine.py', 'canting_ale_fem.py', 'canting_estimator.py', 'canting_plotter.py')},
        upstream_source_sha256={name: hashlib.sha256((SCRIPT_DIR.parents[1]/name).read_bytes()).hexdigest()
                                for name in ('time_integrator.py', 'geo_util.py', 'gas_fem.py', 'liquid_fem.py',
                                             'canting_runner.py', 'physical_params.py', 'solver_params.py')},
    )
    (output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    plot_canting(axes[0])
    axes[0].plot([float(row['h_front_um']) for row in old],
                 [float(row['chirality_slope_deg_per_mm']) for row in old],
                 's--', color='0.6', ms=4, label='Previous 20-node results')
    axes[0].set_title('New simulations versus preserved baseline')
    axes[0].legend(fontsize=8)
    labels = {'pilot40': '40 nodes, lc=0.45 (coarse)', 'pilot160': '160 nodes, lc=0.30',
              'mesh80': '80 nodes, lc=0.20', 'time80': '80 nodes, 10× tighter ODE tolerances'}
    for name in groups:
        rows = [row for row in checks if row['check'] == name]
        axes[1].plot([row['front_height_um'] for row in rows],
                     [row['relative_difference_percent'] for row in rows], 'o--', ms=4, label=labels[name])
    axes[1].axhline(0, color='0.5', lw=.7)
    axes[1].set(xlabel='Front plate height (µm)', ylabel='Raw slope difference from production (%)',
                title='Convergence checks — no per-grid rescaling')
    axes[1].grid(alpha=.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/'comparison.png')
    print(json.dumps({key: summary[key] for key in ('primary_points', 'maximum_absolute_check_percent',
                                                   'minimum_mesh_area_ratios', 'unweighted_fit_rms_deg_per_mm')}, indent=2))


if __name__ == '__main__':
    main()
