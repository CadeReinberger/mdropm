#!/usr/bin/env python3

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from chirality_map import make_alpha_interpolator_loc

# --- Global constants (as requested) ---
K_ALPHA = 25 * .03
R_EFF = 0.85 * 3
ANG_EVAL = np.deg2rad(50) #np.arccos(1/3)

DEFAULT_DRYING_TIME = 28.12065827516264 
DEFAULT_D = .2 * 540

dt_riv = np.array([11.857707311723846, 25.395265969668696, 29.84189730620701, 33.79447256690575, 42.58893457931299, 60.0790684979566, 121.14625706346621])
kg_riv = np.array([0.06439461869983874, 0.08044843349760893, 0.057309416377203395, 0.0505829594157291, 0.00762333353619936, 0.003946201367648286, 0.004125558481529563])

#dt_riv = dt_riv[1:]
#kg_riv = kg_riv[1:]
dt_riv = 9 * np.pi / dt_riv

def load_pickle(p: Path):
    with p.open("rb") as f:
        return pickle.load(f)

def main():
    base_dir = (Path(__file__).resolve().parent / "../../speed_tests").resolve()

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # Evaluation points
    x1 = R_EFF * np.cos(np.pi / 2 - ANG_EVAL)
    y1 = R_EFF * np.sin(np.pi / 2 - ANG_EVAL)
    x2 = R_EFF * np.cos(np.pi / 2 + ANG_EVAL)
    y2 = R_EFF * np.sin(np.pi / 2 + ANG_EVAL)

    drying_times = []
    k_gs = []
    labels = []

    subdirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])

    for sd in subdirs:
        setup_pkl = sd / "setup.pkl"
        results_pkl = sd / "results.pkl"

        if not setup_pkl.exists() or not results_pkl.exists():
            # silently skip incomplete dirs
            continue

        try:
            setup = load_pickle(setup_pkl)
            print(f'LOADED SETUP: {setup_pkl}')
            D_USED = float(setup["D"])

            drying_time = DEFAULT_DRYING_TIME * DEFAULT_D / D_USED

            print(f'{sd} : {drying_time}')


            alpha_interpolator = make_alpha_interpolator_loc(
                str(results_pkl),
                k_alpha=K_ALPHA,
            )

            a1 = float(alpha_interpolator(x1, y1))
            a2 = float(alpha_interpolator(x2, y2))

            denom = (x2 - x1)
            if denom == 0:
                continue

            k_g = np.abs(a2 - a1) / 2

            drying_times.append(drying_time)
            k_gs.append(k_g)
            labels.append(sd.name)

        except Exception as e:
            print(f"[skip] {sd.name}: {type(e).__name__}: {e}")
            continue

    drying_times = np.asarray(drying_times, dtype=float) 
    # CHANGE FROM DRYING TIMES TO DRYING RATE
    drying_times = 9 * np.pi / drying_times

    k_gs = np.asarray(k_gs, dtype=float)

    sorted_inds = sorted(list(range(len(drying_times))), key=lambda i: drying_times[i])
    sorted_inds = sorted_inds[:-2]

    drying_times_sorted = [drying_times[si] for si in sorted_inds]
    k_gs_sorted = [k_gs[si] for si in sorted_inds]

    if drying_times.size == 0:
        raise RuntimeError(
            f"No valid data points found under {base_dir}. "
            "Check that each subdir has setup.pkl/results.pkl and setup['ht_front'] exists."
        )

    # --- Plot ---
    plt.figure(figsize=(8, 5.5), dpi=160)
    plt.plot(drying_times_sorted, k_gs_sorted, 'k--', alpha=.85)
    plt.scatter(drying_times_sorted, k_gs_sorted, s=55, alpha=0.85, label='Simulated')

    plt.scatter(dt_riv, kg_riv, color='m', s=55, alpha=.85, label='Experimental')

    plt.xlabel("Drying Time (min)")
    plt.legend()
    plt.ylabel("Max g-factor")
    plt.title("Speed Chirality Series")
    
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    out_name = "speed_chirality_series.png"  # lowercase, no spaces
    plt.savefig(out_name, bbox_inches="tight")
    print(f"Saved: {out_name}  (N={len(drying_times)} points)")


if __name__ == "__main__":
    main()

