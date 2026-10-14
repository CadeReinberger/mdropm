#!/usr/bin/env python3
"""Shared problem construction and checkpoint metadata for the ALE runners."""

from __future__ import annotations

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
from problem_universe import problem_universe


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


if __name__ == "__main__":
    # Retain the familiar command name, using the refined solver by default.
    from canting_refine import main
    main()
