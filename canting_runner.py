#!/usr/bin/env python3
"""Crash-resumable canting parameter sweep.

Each height gets a deterministic directory.  ``checkpoint.pkl`` is atomically
replaced after every outer LSODA interval and is automatically resumed on the
next invocation.  Completed heights are skipped, so rerunning the same command
is safe after a process, machine, or power failure.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np

import droplet
import heightscape
import physical_params
import solver_params
import tapescape
from pickler import atomic_pickle, pickle_canting, pickle_output
from problem_universe import problem_universe
from time_integrator import solve_problem


def parse_ratios(text: str) -> list[float]:
    values = [float(value) for value in text.split(",") if value.strip()]
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("ratios must be a nonempty comma-separated list of nonnegative numbers")
    return values


def run_name(height_ratio: float) -> str:
    # Human-readable and deterministic; 12 significant digits avoids collisions
    # for ordinary parameter sweeps.
    return f"height-{height_ratio:.12g}".replace("+", "")


def build_problem(height_ratio: float, nodes: int, profile: str = "production"):
    height_back = 0.1
    height_front = height_back * (1.0 + height_ratio)
    heights = (height_back, height_front, height_front, height_back)
    hs = heightscape.constructors.rect_interp_htscape((10, 10), heights)
    ts = tapescape.constructors.make_rectangle_top_open(-5, -5, 5, 5)
    pps = physical_params.physical_params()
    sps = solver_params.solver_params()
    # RHS evaluations are numerous and FEM-dominated; per-evaluation console
    # output can itself become a major bottleneck on long sweeps.
    sps.VERBOSE = False
    if profile in {"production", "production_lsoda", "reference", "mesh_reference"}:
        pps.casr_smoothing_rad = 1e-3
        sps.FEM_ELT_ORDER = 2
        sps.GAS_PHASE_DYNAMIC_LC = False
        sps.GAS_PHASE_DEFAULT_LC = 0.6
        sps.LIQUID_PHASE_DYNAMIC_LC_K = None
        sps.LIQUID_PHASE_DEFAULT_LC = 0.6
        sps.GRADIENT_EVAL_OFFSET_LC = 0.0
        sps.ODE_METHOD = "BDF"
        # Ten times tighter than scipy's defaults. Going below this did not
        # improve FEM-level convergence, but greatly amplified remeshing noise.
        sps.ODE_RTOL = 1e-4
        sps.ODE_ATOL = 1e-6
        sps.ODE_THETA_JACOBIAN = True
        sps.RADAU_DT = 0.25
        sps.SUBDIV_RADAU = True
        sps.RADAU_EVAL_LINSPACE_N = 10
        sps.T_FIN = 3.0
        sps.END_AREA_RATIO = 0.85
        if profile == "production_lsoda":
            sps.ODE_METHOD = "LSODA"
            sps.ODE_THETA_JACOBIAN = False
        elif profile == "reference":
            # Mesh/order study: this combination agreed with lc=0.25 to about
            # 3% at the RHS level without the pathological lc=0.35 topology.
            sps.FEM_ELT_ORDER = 3
            sps.GAS_PHASE_DEFAULT_LC = 0.45
            sps.LIQUID_PHASE_DEFAULT_LC = 0.45
            sps.ODE_THETA_JACOBIAN = False
        elif profile == "mesh_reference":
            sps.GAS_PHASE_DEFAULT_LC = 0.25
            sps.LIQUID_PHASE_DEFAULT_LC = 0.25
            sps.GRADIENT_EVAL_OFFSET_LC = 0.1
    elif profile != "default":
        raise ValueError(f"unknown solver profile: {profile}")
    pu = problem_universe(hs, ts, sps, pps)
    # make_drop_about_to_drop creates 4*n nodes.
    drop = droplet.constructors.make_drop_about_to_drop(nodes // 4, 3, 5, hs, pps)
    return height_front, height_back, drop, pu


def read_checkpoint(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as stream:
        checkpoint = pickle.load(stream)
    if checkpoint.get("version") != 1:
        raise ValueError(f"unsupported checkpoint version in {path}")
    return checkpoint


def atomic_json(data: dict, path: Path) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def run_manifest(height_ratio: float, nodes: int, height_front: float, height_back: float, drop, universe) -> dict:
    solver = {
        name: getattr(universe.sol_ps, name)
        for name in dir(universe.sol_ps)
        if name.isupper() and not callable(getattr(universe.sol_ps, name))
    }
    initial_state = np.concatenate((drop.x, drop.y, np.broadcast_to(drop.theta, drop.x.shape))).astype(float)
    data = {
        "format_version": 1,
        "height_ratio": height_ratio,
        "height_front": height_front,
        "height_back": height_back,
        "nodes": nodes,
        "solver_parameters": solver,
        "physical_parameters": asdict(universe.phys_ps),
        "initial_state_sha256": hashlib.sha256(initial_state.tobytes()).hexdigest(),
    }
    # Normalize tuples/numpy scalar-like values to exactly what json.load returns.
    return json.loads(json.dumps(data))


def valid_result(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            result = pickle.load(stream)
        times = np.asarray(result["out_t"], dtype=float)
        states = np.asarray(result["out_x"], dtype=float)
        return len(times) > 1 and len(times) == len(states) and np.all(np.isfinite(times)) and np.all(np.isfinite(states))
    except Exception:
        return False


def run_one(output_dir: Path, height_ratio: float, nodes: int, profile: str = "production") -> None:
    run_dir = output_dir / run_name(height_ratio)
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "results.pkl"
    checkpoint_path = run_dir / "checkpoint.pkl"
    manifest_path = run_dir / "manifest.json"
    height_front, height_back, drop, universe = build_problem(height_ratio, nodes, profile)
    expected = run_manifest(height_ratio, nodes, height_front, height_back, drop, universe)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != expected:
            raise ValueError(f"run settings conflict in {run_dir}: {existing} != {expected}")
    else:
        atomic_json(expected, manifest_path)
    if not (run_dir / "setup.pkl").exists():
        pickle_canting(height_front, height_back, 3, run_dir / "setup.pkl")
    if result_path.exists() and valid_result(result_path):
        print(f"[done]   height ratio {height_ratio:g}: {result_path}")
        return
    if result_path.exists():
        print(f"[repair] invalid result at {result_path}; recovering from its checkpoint")

    resume_state = read_checkpoint(checkpoint_path)
    action = "resume" if resume_state else "start"
    print(f"[{action}] height ratio {height_ratio:g} in {run_dir}")
    out_t, out_x = solve_problem(
        drop,
        universe,
        resume_state=resume_state,
        checkpoint_callback=lambda state: atomic_pickle(state, checkpoint_path),
    )
    pickle_output(out_t, out_x, result_path)
    # Keep the last checkpoint: it is useful provenance and makes completion
    # recoverable even if results.pkl is later damaged.
    print(f"[done]   height ratio {height_ratio:g}: {result_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("canting_runs"))
    parser.add_argument(
        "--ratios",
        type=parse_ratios,
        default=parse_ratios("0,0.222222222222,0.444444444444,0.666666666667,0.888888888889,1.11111111111,1.33333333333,1.55555555556,1.77777777778,2"),
        help="comma-separated raw front/back height changes (default: 10 points from 0 to 2)",
    )
    parser.add_argument("--nodes", type=int, default=20, choices=range(8, 81, 4), metavar="N")
    parser.add_argument(
        "--profile",
        choices=("production", "production_lsoda", "reference", "mesh_reference", "default"),
        default="production",
        help="production_lsoda is the robust fallback for cases where BDF stalls near a CASR transition",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for height_ratio in args.ratios:
        run_one(args.output_dir, height_ratio, args.nodes, args.profile)


if __name__ == "__main__":
    main()
