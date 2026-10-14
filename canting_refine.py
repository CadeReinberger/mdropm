#!/usr/bin/env python3
"""Run reproducible moving-mesh canting simulations in a separate data directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/canting-refinement-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/canting-clean-mpl")

SCRIPT_DIR = Path(__file__).resolve().parent

import numpy as np
from canting_runner import build_problem, atomic_json, read_checkpoint, run_manifest, valid_result
from pickler import atomic_pickle, pickle_canting, pickle_output
import time_integrator
from canting_ale_fem import MovingFEM
from mpi4py import MPI

DEFAULT_RATIOS = "0,0.01,0.02,0.04,0.06,0.08,0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.375,0.45,0.55,0.65,0.8,1,1.1,1.25,1.5,1.75,2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratios", default=DEFAULT_RATIOS, help="(h_front/h_back)-1; displayed height is 200*(1+ratio) microns")
    parser.add_argument("--nodes", type=int, default=80)
    parser.add_argument("--lc", type=float, default=0.3)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "runs" / "canting")
    parser.add_argument("--workers", type=int, default=1, help="independent serial FEM processes; 6 is a reasonable starting point")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--diagnostic-results", type=Path, help="test the final geometry of this saved result instead of the initial circle")
    args = parser.parse_args()
    if MPI.COMM_WORLD.size != 1:
        parser.error("use --workers for independent serial cases, not mpirun")
    if args.nodes < 12 or args.nodes % 4:
        parser.error("nodes must be a multiple of four and at least 12")
    ratios = list(dict.fromkeys(map(float, args.ratios.split(","))))
    if not ratios or any(not np.isfinite(r) or r < 0 for r in ratios):
        parser.error("ratios must be finite and nonnegative")
    if args.diagnostic_results is not None and (not args.diagnostic or len(ratios) != 1):
        parser.error("--diagnostic-results requires --diagnostic and exactly one ratio")
    if not all(np.isfinite(v) and v > 0 for v in (args.lc, args.rtol, args.atol, args.duration)) or args.workers < 1 or args.order < 1:
        parser.error("mesh size, tolerances, duration, workers, and order must be positive")
    if args.workers > 1:
        def launch(ratio):
            command = [sys.executable, str(Path(__file__).resolve()), "--ratios", str(ratio),
                       "--nodes", str(args.nodes), "--lc", str(args.lc), "--order", str(args.order),
                       "--rtol", str(args.rtol), "--atol", str(args.atol), "--duration", str(args.duration),
                       "--output-dir", str(args.output_dir.resolve())]
            if args.diagnostic:
                command.append("--diagnostic")
            if args.diagnostic_results is not None:
                command.extend(["--diagnostic-results", str(args.diagnostic_results.resolve())])
            subprocess.run(command, check=True)
        with ThreadPoolExecutor(max_workers=min(args.workers, len(ratios))) as pool:
            list(pool.map(launch, ratios))
        return

    for ratio in ratios:
        front, back, drop, universe = build_problem(ratio, args.nodes, "production")
        sps = universe.sol_ps
        sps.FEM_ELT_ORDER = args.order
        sps.GAS_PHASE_DEFAULT_LC = sps.LIQUID_PHASE_DEFAULT_LC = args.lc
        sps.ODE_RTOL, sps.ODE_ATOL = args.rtol, args.atol
        sps.T_FIN = args.duration
        sps.END_AREA_RATIO = 0.0
        sps.RADAU_EVAL_LINSPACE_N = 26
        fem = MovingFEM(drop.x, drop.y, universe)
        calls = [0]

        def derivative_builder(starting_drop, pu):
            raw = time_integrator.compute_deriv_function(starting_drop, pu, fem=fem)
            mirror = (args.nodes // 2 - np.arange(args.nodes)) % args.nodes

            def symmetric_rhs(t, state):
                calls[0] += 1
                dx, dy, dtheta = np.split(raw(t, state), 3)
                return np.r_[0.5 * (dx - dx[mirror]), 0.5 * (dy + dy[mirror]), 0.5 * (dtheta + dtheta[mirror])]

            return symmetric_rhs

        if args.diagnostic:
            rhs = derivative_builder(drop, universe)
            state = np.r_[drop.x, drop.y, np.broadcast_to(drop.theta, drop.x.shape)]
            diagnostic_time = 0.0
            if args.diagnostic_results is not None:
                with args.diagnostic_results.open('rb') as stream:
                    saved = pickle.load(stream)
                state = np.asarray(saved['out_x'][-1], dtype=float)
                diagnostic_time = float(saved['out_t'][-1])
                if state.shape != (3*args.nodes,):
                    raise ValueError('diagnostic result has the wrong number of nodes')
            measurements = []
            for index, step in enumerate((0, 1e-8, 1e-6, 1e-4, 0)):
                altered = state.copy()
                altered[0] += step
                started = perf_counter()
                value = rhs(0, altered)
                if index == 0:
                    baseline = value.copy()
                measurement = {"delta_x_mm": step, "seconds": perf_counter() - started,
                               "rhs_inf": float(abs(value).max()),
                               "difference_inf": float(abs(value - baseline).max())}
                measurements.append(measurement)
                print(measurement, flush=True)
            test_x, test_y, _ = np.split(state, 3)
            constant_error = float(np.max(np.abs(fem.pressure(test_x, test_y, np.ones(args.nodes), universe))))
            linear_gradient = fem.pressure(test_x, test_y, test_x, universe)
            linear_error = float(np.max(np.abs(linear_gradient - np.array([1.0, 0.0]))))
            # Nontrivial manufactured solution: h(y)^3 * p'(y) is constant.
            h0, hy = float(universe.htscp.h(0, 0)), float(universe.htscp.hy(0, 0))
            heights = np.broadcast_to(universe.htscp.h(test_x, test_y), test_y.shape)
            vertical_pressure = test_y if abs(hy) < 1e-14 else h0**3/(2*hy)*(h0**-2-heights**-2)
            expected_gradient = np.c_[np.zeros(args.nodes), (h0/heights)**3]
            vertical_gradient = fem.pressure(test_x, test_y, vertical_pressure, universe)
            vertical_error = float(np.max(np.abs(vertical_gradient-expected_gradient))/np.max(np.abs(expected_gradient)))
            print("manufactured pressure-gradient errors: constant", constant_error,
                  "linear x", linear_error, "vertical relative", vertical_error, flush=True)
            if constant_error > 1e-9 or linear_error > 0.02 or vertical_error > 0.02:
                raise AssertionError("manufactured pressure-gradient check failed")
            if measurements[1]["difference_inf"] > 1e-5:
                raise AssertionError("RHS is not continuous under a tiny geometry perturbation")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            atomic_json({"nodes": args.nodes, "lc": args.lc, "order": args.order,
                         "geometry_time": diagnostic_time,
                         "measurements": measurements, "constant_gradient_max_error": constant_error,
                         "linear_gradient_max_error": linear_error,
                         "vertical_gradient_relative_max_error": vertical_error}, args.output_dir / f"diagnostic-{ratio:g}.json")
            continue

        run_dir = args.output_dir.resolve() / f"height-{ratio:.12g}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path, checkpoint_path = run_dir / "results.pkl", run_dir / "checkpoint.pkl"
        manifest = run_manifest(ratio, args.nodes, front, back, drop, universe)
        manifest["numerical_method"] = "persistent_radial_mesh_l2_gradient_periodic_pressure_symmetric_rhs_v1"
        manifest["fem_implementation_sha256"] = hashlib.sha256((SCRIPT_DIR / "canting_ale_fem.py").read_bytes()).hexdigest()
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f"settings differ from existing run in {run_dir}")
        atomic_json(manifest, manifest_path)
        pickle_canting(front, back, 3, run_dir / "setup.pkl")
        if result_path.exists() and valid_result(result_path):
            with result_path.open('rb') as stream:
                existing_result = pickle.load(stream)
            if np.isclose(existing_result['out_t'][-1], args.duration, rtol=0, atol=1e-10):
                print(f"[done] {run_dir}", flush=True)
                continue
        started = perf_counter()

        def checkpoint(state):
            atomic_pickle(state, checkpoint_path)
            print(f"[progress] ratio={ratio:g} nodes={args.nodes} t={state['cur_t']:.3f} "
                  f"rhs={calls[0]} elapsed={perf_counter() - started:.1f}s", flush=True)

        times, states = time_integrator.solve_problem(
            drop, universe, resume_state=read_checkpoint(checkpoint_path), checkpoint_callback=checkpoint,
            derivative_function=derivative_builder(drop, universe))
        if len(times) < 2 or times[-1] < args.duration - 1e-10:
            raise RuntimeError(f"run stopped early at t={times[-1] if len(times) else None}; checkpoint retained in {run_dir}")
        pickle_output(times, states, result_path)
        atomic_json({"wall_seconds": perf_counter() - started, "rhs_evaluations": calls[0],
                     "samples": len(times), "t_final": float(times[-1]),
                     "minimum_gas_mesh_area_ratio": fem.gas.minimum_area_ratio if fem.gas else None,
                     "minimum_liquid_mesh_area_ratio": fem.liquid.minimum_area_ratio if fem.liquid else None}, run_dir / "timing.json")
        print(f"[done] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
