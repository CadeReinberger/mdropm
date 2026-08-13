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
from matplotlib.ticker import FuncFormatter

from chirality_map import make_alpha_interpolator_loc

from scipy.optimize import curve_fit, minimize_scalar


# --- Global constants (as requested) ---
K_ALPHA = .2 #5.5
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
    return np.rad2deg(alpha_bar)

# The 2x already included here
dh_riv = np.array([0.00046292408994360674, 0.00046292408994360674, 0.04907405786553596, 0.04861105960092645, 0.06342596562310825, 0.06388886498816321, 0.07268513971886222, 0.12361114119305902, 0.18518521265728374, 0.12407404055811398, 0.09166663452431145, 0.18796300444583172, 0.24537033877745715, 0.26018514590008446, 0.31666663205182266, 0.31666663205182266, 0.3837963365429207, 0.4296296538050764, 0.501388846444497, 0.5569443855167933])
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

            ht_ratio = (2*ht_front - 0.2) / (2*ht_front + 0.2)

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

    ht_ratios = np.asarray(ht_ratios, dtype=float) # Okay let's think a bit
    k_gs = np.asarray(k_gs, dtype=float)


    print('\n\n')
    print('-' * 50)
    print(f'ht_ratios: {ht_ratios}')
    print(f'kgs: {k_gs}')
    print('-' * 50)
    print('\n\n')

    sorted_inds = sorted(list(range(len(ht_ratios))), key=lambda i: ht_ratios[i])
    # sorted_inds = [sorted_inds[i] for i in (0, 2, 3, 4)]

    ht_ratios_sorted = [ht_ratios[si] for si in sorted_inds]
    k_gs_sorted = [k_gs[si] for si in sorted_inds]

    print(f'hrs: {ht_ratios_sorted}')

    order = np.argsort(ht_ratios_sorted)
    ht_ratios_sorted = list(np.asarray(ht_ratios_sorted)[order])
    k_gs_sorted = list(np.asarray(k_gs_sorted)[order])

    def sat_exp_model(x, a, b, c):
        return a - b * np.exp(-c * x)

    shape_params, _ = curve_fit(
        sat_exp_model, ht_ratios_sorted, k_gs_sorted, p0=(1.0, 0.6, 5.0)
    )

    def objective(k_factor):
        residuals = k_factor * sat_exp_model(np.asarray(dh_riv), *shape_params) - np.asarray(kg_riv)
        return np.sum(residuals ** 2)

    k_opt = minimize_scalar(objective, bounds=(0.01, 10.0), method="bounded").x
    print(f'k_opt: {k_opt}')

    od_int = lambda x: sat_exp_model(x, *shape_params)
    ht_ratio_samp = np.linspace(min(ht_ratios_sorted), max(ht_ratios_sorted), num=100)
    int_samp = od_int(ht_ratio_samp)

    if ht_ratios.size == 0:
        raise RuntimeError(
            f"No valid data points found under {base_dir}. "
            "Check that each subdir has setup.pkl/results.pkl and setup['ht_front'] exists."
        )

    # --- Plot ---
    ax_right = ax.twinx()

    ax_right.plot(ht_ratio_samp, int_samp, "k--", alpha=0.85)
    ax_right.scatter(ht_ratios_sorted, k_gs_sorted, s=77, alpha=0.85, label="Simulated", color="tab:blue")

    print(f'kgs: {k_gs_sorted}')

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 32,
            "axes.titlesize": 32,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 20,
        }
    )

    if include_experimental and False:
        ax.scatter(dh_riv, kg_riv, color="m", s=100, alpha=0.85, label="Experimental")

    ax.set_xlabel(r"$\dfrac{h_{\mathrm{max}} - h_{\mathrm{min}}}{h_{\mathrm{max}} + h_{\mathrm{min}}}$")
    ax.set_ylabel("g-Factor Slope")
    ax_right.set_ylabel(r"Skew Angle Slope ($^{\circ}$/mm)")

    handles_r, labels_r = ax_right.get_legend_handles_labels()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles_r + handles, labels_r + labels, loc="upper left")
    # ax.set_title("Canting Chirality Series")

    '''
    x_max = max(ht_ratios_sorted)
    if include_experimental:
        x_max = max(x_max, max(dh_riv))
    ax.set_xlim(-.05, x_max + 0.05)
    ''' 
    # ax.set_ylim(.2, 1.3)
    # ax_right.set_ylim(.2 / k_opt, 1.3 / k_opt)

    # skew_ratio = 0.1 / K_ALPHA
    # ax_right.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val * skew_ratio:.2f}"))

    ax.grid(True, alpha=0.25)

    return len(ht_ratios)


def main():
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    n_points = plot_canting(ax)
    plt.tight_layout()
    out_name = "canting_chirality_series.png"
    plt.savefig(out_name, bbox_inches="tight")
    print(f"Saved: {out_name}  (N={n_points} points)")


if __name__ == "__main__":
    main()
