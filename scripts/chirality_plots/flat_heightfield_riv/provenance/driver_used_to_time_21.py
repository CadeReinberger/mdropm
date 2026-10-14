#!/usr/bin/env python3
"""Long, checkpointed flat-heightfield ALE run for the RIV map.

The equations and physical parameters come from the current canting study.
Meshes are rebuilt only between integration segments. Within each segment,
the existing ALE solver retains fixed topology and L2 gradient recovery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from time import perf_counter

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/canting-refinement-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/canting-clean-mpl")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.interpolate import CubicSpline
from mpi4py import MPI
from dolfinx.mesh import entities_to_geometry, locate_entities_boundary

from canting_ale_fem import MovingPhase
from canting_runner import atomic_json, build_problem, read_checkpoint, run_manifest
from geo_util import compute_area_shoelace
from pickler import atomic_pickle, pickle_output
import time_integrator


class RebasedPhase(MovingPhase):
    """Give the existing ALE displacement its material labels on a new mesh.

    MovingPhase originally labels a circular reference boundary by polar angle.
    A receded reference boundary is not circular: invert its polar angle to
    recover the original, equally spaced material-node parameter instead.
    """

    def __init__(self, x, y, universe, *, gas):
        super().__init__(x, y, universe, gas=gas)
        center = self.initial_boundary.mean(axis=0)
        relative = self.initial_boundary - center
        phi = np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))
        phi = np.r_[phi, phi[0] + 2 * np.pi]
        if np.any(np.diff(phi) <= 0):
            raise ValueError("reference boundary must be star-shaped about its node mean")
        correction = self.parameter - (phi - phi[0])
        correction[-1] = correction[0]
        material_correction = CubicSpline(phi, correction, bc_type="periodic")
        radii = np.hypot(relative[:, 0], relative[:, 1])
        radius_at = CubicSpline(phi, np.r_[radii, radii[0]], bc_type="periodic")

        def labels(points):
            offset = points[:, :2] - center
            angle = phi[0] + np.mod(np.arctan2(offset[:, 1], offset[:, 0]) - phi[0], 2*np.pi)
            parameter = np.mod(angle - phi[0] + material_correction(angle), 2*np.pi)
            return angle, parameter, np.linalg.norm(offset, axis=1)

        angle, self.angle, radial = labels(self.reference)
        boundary_radius = radius_at(angle)
        if np.any(boundary_radius <= 0):
            raise ValueError("nonpositive reference radius")
        if gas:
            direction = np.c_[np.cos(angle), np.sin(angle)]
            outer = np.full_like(direction, np.inf)
            for axis in (0, 1):
                active = np.abs(direction[:, axis]) > 1e-14
                wall = np.where(direction[:, axis] > 0, 5.0, -5.0)
                outer[active, axis] = (wall[active] - center[axis]) / direction[active, axis]
            outer_radius = outer.min(axis=1)
            self.weight = np.clip((outer_radius-radial)/(outer_radius-boundary_radius), 0, 1)
        else:
            self.weight = np.minimum(radial/boundary_radius, 1.0)**2
            self.dof_angle = labels(self.V.tabulate_dof_coordinates())[1]
        # Make boundary displacement exact and hold every outer gas wall fixed.
        facets = locate_entities_boundary(self.mesh, 1, lambda q: np.full(q.shape[1], True))
        boundary = np.unique(entities_to_geometry(self.mesh, 1, facets).ravel())
        points = self.reference[boundary, :2]
        exterior = np.any(np.isclose(np.abs(points), 5.0, atol=1e-9, rtol=0), axis=1) if gas else np.zeros(len(boundary), bool)
        self.weight[boundary[~exterior]] = 1.0
        self.weight[boundary[exterior]] = 0.0


class SegmentFEM:
    def __init__(self, state, universe):
        self.x, self.y, _ = np.split(np.asarray(state), 3)
        self.universe = universe
        self.gas = self.liquid = None

    def concentration(self, x, y, universe):
        if self.gas is None:
            self.gas = RebasedPhase(self.x, self.y, self.universe, gas=True)
        return self.gas.gradients(x, y)

    def pressure(self, x, y, pressures, universe):
        if self.liquid is None:
            self.liquid = RebasedPhase(self.x, self.y, self.universe, gas=False)
        return self.liquid.gradients(x, y, pressures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "flat_heightfield_work/full80")
    parser.add_argument("--nodes", type=int, default=80)
    parser.add_argument("--lc", type=float, default=0.3)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--duration", type=float, default=21.0)
    parser.add_argument("--segment-duration", type=float, default=3.0)
    parser.add_argument("--seed", type=Path, help="completed fresh canting_refine case with identical parameters")
    args = parser.parse_args()
    if MPI.COMM_WORLD.size != 1:
        parser.error("run in serial; do not use mpirun")
    if args.nodes < 12 or args.nodes % 4:
        parser.error("nodes must be a multiple of four and at least 12")
    if not all(np.isfinite(v) and v > 0 for v in (args.lc, args.rtol, args.atol, args.duration, args.segment_duration)) or args.order < 1:
        parser.error("numerical parameters must be finite and positive")
    front, back, drop, universe = build_problem(0.0, args.nodes, "production")
    sps = universe.sol_ps
    sps.FEM_ELT_ORDER = args.order
    sps.GAS_PHASE_DEFAULT_LC = sps.LIQUID_PHASE_DEFAULT_LC = args.lc
    sps.ODE_RTOL, sps.ODE_ATOL = args.rtol, args.atol
    sps.T_FIN, sps.END_AREA_RATIO = args.duration, 0.0
    sps.RADAU_EVAL_LINSPACE_N = 26
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = run_manifest(0.0, args.nodes, front, back, drop, universe)
    manifest.update(numerical_method="segmented_persistent_ale_l2_material_labels_symmetric_rhs_v1",
                    segment_duration=args.segment_duration, height_field_mm="h(x,y) = 0.1",
                    plate_gap_um=200.0, seed=None,
                    source_sha256={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                   for directory, names in [(HERE, ("flat_heightfield_run.py", "canting_ale_fem.py")),
                                                            (ROOT, ("time_integrator.py", "canting_runner.py", "gas_fem.py", "liquid_fem.py", "physical_params.py", "geo_util.py"))]
                                   for name in names})
    if args.seed:
        seed_manifest = json.loads((args.seed / "manifest.json").read_text())
        for field in ("height_front", "height_back", "nodes", "physical_parameters", "initial_state_sha256"):
            if seed_manifest[field] != manifest[field]:
                raise ValueError(f"seed differs in {field}")
        for field, value in manifest["solver_parameters"].items():
            if field != "T_FIN" and seed_manifest["solver_parameters"][field] != value:
                raise ValueError(f"seed differs in solver parameter {field}")
        if seed_manifest["fem_implementation_sha256"] != manifest["source_sha256"]["canting_ale_fem.py"]:
            raise ValueError("seed uses a different ALE implementation")
        manifest["seed"] = {"results_sha256": hashlib.sha256((args.seed / "results.pkl").read_bytes()).hexdigest(),
                            "manifest": seed_manifest}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"settings differ from existing run in {output}")
    atomic_json(manifest, manifest_path)
    atomic_pickle({"heightscape": universe.htscp.to_pickle_dict(), "tapescape": universe.tpscp.picklable(),
                   "solver_params": manifest["solver_parameters"], "physical_params": manifest["physical_parameters"],
                   "start_drop": drop.to_pickle_dict(), "ht_front": front, "ht_back": back, "radius": 3.0}, output / "setup.pkl")
    checkpoint_path = output / "checkpoint.pkl"
    resume = read_checkpoint(checkpoint_path)
    if resume is None and args.seed:
        resume = read_checkpoint(args.seed / "checkpoint.pkl")
        with (args.seed / "results.pkl").open("rb") as stream:
            seed_result = pickle.load(stream)
        if resume is None or not np.array_equal(seed_result["out_x"][-1], resume["cur_x"]) or seed_result["out_t"][-1] != resume["cur_t"]:
            raise ValueError("seed checkpoint does not match its completed result")
    initial = np.r_[drop.x, drop.y, np.broadcast_to(drop.theta, drop.x.shape)]
    starting_area = compute_area_shoelace(drop.x, drop.y)
    original_derivative = time_integrator.compute_deriv_function
    calls = [0]
    started = perf_counter()
    segments_path = output / "mesh_segments.json"
    segments = json.loads(segments_path.read_text()) if segments_path.exists() else []
    while resume is None or resume["cur_t"] < args.duration - 1e-10:
        current_t = float(resume["cur_t"]) if resume else 0.0
        if resume and resume.get("segment_end", -1) > current_t + 1e-10:
            reference = np.asarray(resume["reference_state"])
            segment_start, segment_end = resume["segment_start"], resume["segment_end"]
        else:
            reference = np.asarray(resume["cur_x"]) if resume else initial
            segment_start, segment_end = current_t, min(current_t + args.segment_duration, args.duration)
        sps.T_FIN = segment_end
        fem = SegmentFEM(reference, universe)
        test_x, test_y, _ = np.split(reference, 3)
        constant_error = float(np.max(np.abs(fem.pressure(test_x, test_y, np.ones(args.nodes), universe))))
        linear_x_error = float(np.max(np.abs(fem.pressure(test_x, test_y, test_x, universe) - [1.0, 0.0])))
        linear_y_error = float(np.max(np.abs(fem.pressure(test_x, test_y, test_y, universe) - [0.0, 1.0])))
        checks = dict(reference_time=segment_start, constant_error=constant_error,
                      linear_x_error=linear_x_error, linear_y_error=linear_y_error)
        atomic_json(checks, output / f"reference_check_{segment_start:g}.json")
        if constant_error > 1e-9 or max(linear_x_error, linear_y_error) > 0.02:
            raise AssertionError(f"rebased manufactured-solution check failed: {checks}")
        time_integrator.compute_concentration_gradients = fem.concentration
        time_integrator.compute_pressure_gradients = fem.pressure
        mirror = (args.nodes // 2 - np.arange(args.nodes)) % args.nodes

        def derivative_builder(starting_drop, pu):
            raw = original_derivative(starting_drop, pu)
            def symmetric_rhs(t, state):
                calls[0] += 1
                dx, dy, dt = np.split(raw(t, state), 3)
                return np.r_[0.5*(dx-dx[mirror]), 0.5*(dy+dy[mirror]), 0.5*(dt+dt[mirror])]
            return symmetric_rhs
        time_integrator.compute_deriv_function = derivative_builder

        def checkpoint(state):
            state.update(reference_state=reference, segment_start=segment_start, segment_end=segment_end)
            atomic_pickle(state, checkpoint_path)
            x, y, _ = np.split(state["cur_x"], 3)
            progress = dict(t=float(state["cur_t"]), target_t=args.duration, rhs_evaluations=calls[0],
                            wall_seconds=perf_counter()-started, area_ratio=compute_area_shoelace(x, y)/starting_area,
                            top_y_mm=float(y.max()),
                            gas_minimum_area_ratio=fem.gas.minimum_area_ratio if fem.gas else None,
                            liquid_minimum_area_ratio=fem.liquid.minimum_area_ratio if fem.liquid else None)
            atomic_json(progress, output / "progress.json")
            print(json.dumps(progress), flush=True)
        times, states = time_integrator.solve_problem(drop, universe, resume_state=resume, checkpoint_callback=checkpoint)
        resume = read_checkpoint(checkpoint_path)
        if times[-1] < segment_end - 1e-10:
            raise RuntimeError(f"run stopped early at {times[-1]}; checkpoint retained")
        segments.append(dict(start=segment_start, end=segment_end,
                             gas_minimum_area_ratio=fem.gas.minimum_area_ratio,
                             liquid_minimum_area_ratio=fem.liquid.minimum_area_ratio))
        atomic_json(segments, segments_path)
    pickle_output(resume["out_t"], resume["out_x"], output / "results.pkl")
    print(f"[done] {output / 'results.pkl'}", flush=True)


if __name__ == "__main__":
    main()
