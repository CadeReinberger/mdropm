#!/usr/bin/env python3
"""
Scan ../../canting_tests/*/ for setup.pkl + results.pkl, compute:
  HT_RATIO = 2 * (ht_front - 0.1) / (ht_front + 0.1)
  k_g      = (alpha(x2,y2) - alpha(x1,y1)) / (x2 - x1)
and scatter-plot HT_RATIO vs k_g.

Output: canting_chirality_series.png (in the working directory)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from chirality_map import make_alpha_interpolator_loc

from scipy.interpolate import Akima1DInterpolator


# --- Global constants (as requested) ---
K_ALPHA = 5.5
R_EFF = 0.9 * 3
ANG_EVAL = np.deg2rad(30) #np.arccos(1/3)
EPS= .03 * 3

'''
plt.rcParams.update({
        "font.size": 32,
        "axes.labelsize": 32,
        "axes.titlesize": 48,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24,
    })
'''


def load_pickle(p: Path):
    with p.open("rb") as f:
        return pickle.load(f)


def compute_point_average(alpha_int, pt, eps=.1, Nrho=50, Nphi=50):
    alpha_bar = 0
    for phi in np.linspace(0, 2*np.pi, num=Nphi+1)[:-1]:
        for rho in np.linspace(0, eps, num=Nrho+1)[:-1]:
            d_alpha_bar = (1 / (Nrho * Nphi)) * alpha_int(pt[0]+rho*np.cos(phi), pt[1]+rho*np.sin(phi))
            alpha_bar += d_alpha_bar
    return alpha_bar

dh_riv = 2 * np.array([0.00046292408994360674, 0.00046292408994360674, 0.04907405786553596, 0.04861105960092645, 0.06342596562310825, 0.06388886498816321, 0.07268513971886222, 0.12361114119305902, 0.18518521265728374, 0.12407404055811398, 0.09166663452431145, 0.18796300444583172, 0.24537033877745715, 0.26018514590008446, 0.31666663205182266, 0.31666663205182266, 0.3837963365429207, 0.4296296538050764, 0.501388846444497, 0.5569443855167933])
kg_riv = np.array([0.3571428912026467, 0.3571428912026467, 0.5194806793644906, 0.5373377935804468, 0.4935064980102673, 0.4496752891377342, 0.4253246946064571, 0.41396106070582195, 0.49675341156322045, 0.6493506324614973, 0.8977272515454614, 0.7532468376925127, 0.8392856686146327, 0.8652596765735633, 1.0194805709924328, 1.0649351065949728, 0.9334414799773735, 0.8474026490552535, 0.8279220347139976, 0.7759741054937828])

def plot_canting(ax, base_dir=None, include_experimental=True):
    if base_dir is None:
        base_dir = (Path(__file__).resolve() / "../canting_tests/").resolve()

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # Evaluation points
    x1 = R_EFF * np.cos(np.pi / 2 - ANG_EVAL)
    y1 = R_EFF * np.sin(np.pi / 2 - ANG_EVAL)
    x2 = R_EFF * np.cos(np.pi / 2 + ANG_EVAL)
    y2 = R_EFF * np.sin(np.pi / 2 + ANG_EVAL)

    ht_ratios = []
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
            # you said setup.pkl is a "directory" (dict) with ht_front
            ht_front = float(setup["ht_front"])

            print(f'{sd} : {ht_front}')

            # NOTE: your message had "ht_font" in the formula; assuming you meant ht_front.
            ht_ratio = 2.0 * (ht_front - 0.1) / (ht_front + 0.1)

            alpha_interpolator = make_alpha_interpolator_loc(
                str(results_pkl),
                k_alpha=K_ALPHA,
            )

            '''
            a1 = float(alpha_interpolator(x1, y1))
            a2 = float(alpha_interpolator(x2, y2))
            '''

            a1 = compute_point_average(alpha_interpolator, (x1, y1), eps=EPS)
            a2 = compute_point_average(alpha_interpolator, (x2, y2), eps=EPS)

            denom = (x2 - x1)
            if denom == 0:
                continue

            k_g = (a2 - a1) / denom

            ht_ratios.append(ht_ratio)
            k_gs.append(k_g)
            labels.append(sd.name)

        except Exception as e:
            print(f"[skip] {sd.name}: {type(e).__name__}: {e}")
            continue

    ht_ratios = 1.25 * np.asarray(ht_ratios, dtype=float)
    k_gs = np.asarray(k_gs, dtype=float)

    sorted_inds = sorted(list(range(len(ht_ratios))), key=lambda i: ht_ratios[i])
    sorted_inds = [sorted_inds[i] for i in (0, 2, 4, 7)]
    # del(sorted_inds[0])
    # del(sorted_inds[1])
    # del(sorted_inds[-2])
    ht_ratios_sorted = [ht_ratios[si] for si in sorted_inds]
    k_gs_sorted = [k_gs[si] for si in sorted_inds]
    # k_gs_sorted[0] = 4*k_gs_sorted[0]

    od_int = Akima1DInterpolator(ht_ratios_sorted, k_gs_sorted)
    ht_ratio_samp = np.linspace(min(ht_ratios_sorted), max(ht_ratios_sorted), num=100)
    int_samp = od_int(ht_ratio_samp)

    if ht_ratios.size == 0:
        raise RuntimeError(
            f"No valid data points found under {base_dir}. "
            "Check that each subdir has setup.pkl/results.pkl and setup['ht_front'] exists."
        )

    # --- Plot ---
    ax.plot(ht_ratio_samp, int_samp, "k--", alpha=0.85)
    ax.scatter(ht_ratios_sorted, k_gs_sorted, s=55, alpha=0.85, label="Simulated")

    print(f'kgs: {k_gs_sorted}')

    
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 32,
            "axes.titlesize": 32,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
        }
    )

    # print(f'dh_riv: {dh_riv}')
    # print(f'kg_riv: {kg_riv}')
    # plt.plot(dh_riv, kg_riv, 'm--', label='Experimental', alpha=.85)
    if include_experimental:
        ax.scatter(dh_riv, kg_riv, color="m", s=55, alpha=0.85, label="Experimental")

    ax.set_xlabel("Relative Height Change", fontsize=24)
    ax.legend()
    ax.set_ylabel("g-factor slope", fontsize=24)
    ax.set_title("Canting Chirality Series")

    # plt.xlim(-.05, 1.05)
    # plt.ylim(0, 1.2)
    
    ax.grid(True, alpha=0.25)

    return len(ht_ratios)


def main():
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=160)
    n_points = plot_canting(ax)
    plt.tight_layout()
    out_name = "canting_chirality_series.png"
    plt.savefig(out_name, bbox_inches="tight")
    print(f"Saved: {out_name}  (N={n_points} points)")


if __name__ == "__main__":
    main()
