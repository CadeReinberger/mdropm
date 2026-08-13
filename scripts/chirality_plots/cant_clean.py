#!/usr/bin/env python3
"""Lightweight re-plot of the canting chirality series from precomputed data."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

K_ALPHA = 5.5
K_OPT = 1.089595041701311
A, B, C = 0.80291084, 0.45570389, 7.28171509

ht_ratios = [0.0, 0.3636363636363637, 0.4999999999999999, 0.6153846153846154]
k_gs = [0.34717317668230663, 0.7730828101006769, 0.7834794996987616, 0.8028278828086988]

dh_riv = np.array([0.00046292408994360674, 0.00046292408994360674, 0.04907405786553596, 0.04861105960092645, 0.06342596562310825, 0.06388886498816321, 0.07268513971886222, 0.12361114119305902, 0.18518521265728374, 0.12407404055811398, 0.09166663452431145, 0.18796300444583172, 0.24537033877745715, 0.26018514590008446, 0.31666663205182266, 0.31666663205182266, 0.3837963365429207, 0.4296296538050764, 0.501388846444497, 0.5569443855167933])
kg_riv = np.array([0.3571428912026467, 0.3571428912026467, 0.5194806793644906, 0.5373377935804468, 0.4935064980102673, 0.4496752891377342, 0.4253246946064571, 0.41396106070582195, 0.49675341156322045, 0.6493506324614973, 0.8977272515454614, 0.7532468376925127, 0.8392856686146327, 0.8652596765735633, 1.0194805709924328, 1.0649351065949728, 0.9334414799773735, 0.8474026490552535, 0.8279220347139976, 0.7759741054937828])

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
ax_r = ax.twinx()

x_samp = np.linspace(min(ht_ratios), max(ht_ratios), 100)
ax_r.plot(x_samp, A - B * np.exp(-C * x_samp), "k--", alpha=0.85)
ax_r.scatter(ht_ratios, k_gs, s=77, alpha=0.85, color="tab:blue", label="Simulated")
ax.scatter(dh_riv, kg_riv, color="m", s=100, alpha=0.85, label="Experimental")

ax.set_xlabel(r"$\dfrac{h_{\mathrm{max}} - h_{\mathrm{min}}}{h_{\mathrm{max}} + h_{\mathrm{min}}}$")
ax.set_ylabel("g-Factor Slope")
ax_r.set_ylabel(r"Skew Angle Slope ($^{\circ}$/mm)")

h_r, l_r = ax_r.get_legend_handles_labels()
h, l = ax.get_legend_handles_labels()
ax.legend(h_r + h, l_r + l, loc="upper left")

ax.set_xlim(-0.05, max(*ht_ratios, *dh_riv) + 0.05)
ax.set_ylim(0.2, 1.3)
ax_r.set_ylim(0.2 / K_OPT, 1.3 / K_OPT)
ax_r.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 0.1 / K_ALPHA:.2f}"))

ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig("canting_clean.png", bbox_inches="tight")
