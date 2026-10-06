import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

FRONT_HEIGHTS = [200, 204, 220, 240, 310, 360, 420, 500, 600]  # microns
G_SLOPES = [0.2456962620374946, 0.3029078994881574, 0.6236824728725682, 0.8477663604619874, 1.0123075510621362, 0.9860404009350147, 0.9903049607163477, 1.0032395844033108, 1.006811864798049]  # degrees/mm

# microns. I backcomputed these and they look kinda off so I assume you have the data here better.
EXPERIMENTAL_FRONT_HEIGHTS = [200.18525539516267, 200.18525539516267, 220.6426413208934, 220.43793344099032, 227.08850055417386, 227.2996923543081, 231.35294939491254, 256.41839918473414, 290.9091074604632, 256.6596019769389, 240.3669568942042, 292.588394605129, 330.06132750199254, 340.675816095439, 385.36582400632193, 385.36582400632193, 449.13603037652393, 501.29873102361506, 602.228343966385, 702.8211965366287]
EXPERIMENTAL_G_SLOPES = [0.3571428912026467, 0.3571428912026467, 0.5194806793644906, 0.5373377935804468, 0.4935064980102673, 0.4496752891377342, 0.4253246946064571, 0.41396106070582195, 0.49675341156322045, 0.6493506324614973, 0.8977272515454614, 0.7532468376925127, 0.8392856686146327, 0.8652596765735633, 1.0194805709924328, 1.0649351065949728, 0.9334414799773735, 0.8474026490552535, 0.8279220347139976, 0.7759741054937828]
AXIS_SCALE = 1.175494751294973  # theory = AXIS_SCALE * experiment, fitted through zero

fig, theory = plt.subplots(figsize=(8, 6))
experiment = theory.twinx()
heights = [FRONT_HEIGHTS[0] + (FRONT_HEIGHTS[-1] - FRONT_HEIGHTS[0]) * i / 499 for i in range(500)]
theory.plot(heights, CubicSpline(FRONT_HEIGHTS, G_SLOPES, bc_type="natural")(heights), color="tab:blue", lw=2, label="Simulation")
theory.scatter(FRONT_HEIGHTS, G_SLOPES, color="tab:blue", s=42, zorder=3)
experiment.scatter(EXPERIMENTAL_FRONT_HEIGHTS, EXPERIMENTAL_G_SLOPES, color="m", s=52, alpha=0.8, label="Experiment")
values = G_SLOPES + [AXIS_SCALE * g for g in EXPERIMENTAL_G_SLOPES]
limits = (min(values) - 0.08 * (max(values) - min(values)), max(values) + 0.08 * (max(values) - min(values)))
theory.set(xlim=(180, 725), ylim=limits, xlabel=r"Front plate height $h_{\mathrm{front}}$ ($\mu$m)", ylabel=r"Chirality slope $d\alpha/dx$ ($^\circ$/mm)")
experiment.set(ylim=[limit / AXIS_SCALE for limit in limits], ylabel="Experimental g-factor slope")
theory.grid(True, alpha=0.25)
theory.legend(handles=[theory.lines[0], experiment.collections[0]], loc="lower right")
fig.savefig("comp_simple.png", dpi=300, bbox_inches="tight")
