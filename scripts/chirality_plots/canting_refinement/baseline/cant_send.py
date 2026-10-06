import matplotlib.pyplot as plt

FRONT_HEIGHTS = [200, 204, 220, 240, 310, 360, 420, 500, 600]  # microns
G_SLOPES = [0.2456962620374946, 0.3029078994881574, 0.6236824728725682, 0.8477663604619874, 1.0123075510621362, 0.9860404009350147, 0.9903049607163477, 1.0032395844033108, 1.006811864798049]  # degrees/mm

# Bounds are (lower, upper); fit uses r=(h-BACK_HEIGHT)/(h+BACK_HEIGHT).
G_BOUNDS = [(0.2434691528069625, 0.2479233712680267), (0.2996525514933249, 0.30616324748298984), (0.6050047903834218, 0.6423601553617145), (0.8123187234617625, 0.8832139974622123), (0.9298952748483101, 1.0947198272759622), (0.8768361074541391, 1.0952446944158902), (0.9077196715045506, 1.0728902499281447), (0.920718833401458, 1.0857603354051637), (0.922110348369146, 1.0915133812269522)]
BASELINE, AMPLITUDE, SCALE, SHAPE = 0.24523329168078872, 0.7548157658961426, 0.06290444264319028, 1.3589286601562192
BACK_HEIGHT = 200  # microns, using the doubled-height convention

plt.figure(figsize=(8, 6))
plt.errorbar(FRONT_HEIGHTS, G_SLOPES, yerr=[[g - lo for g, (lo, hi) in zip(G_SLOPES, G_BOUNDS)], [hi - g for g, (lo, hi) in zip(G_SLOPES, G_BOUNDS)]], fmt="o", color="tab:blue", capsize=3, ms=6, label="Simulation (20 nodes)")
heights = [FRONT_HEIGHTS[0] + (FRONT_HEIGHTS[-1] - FRONT_HEIGHTS[0]) * i / 399 for i in range(400)]
slopes = [BASELINE + AMPLITUDE * (1 - 2.718281828459045 ** (-((((h - BACK_HEIGHT) / (h + BACK_HEIGHT)) / SCALE) ** SHAPE))) for h in heights]
plt.plot(heights, slopes, color="0.2", lw=2, label="Saturating fit")
plt.xlabel(r"Front plate height $h_{\mathrm{front}}$ ($\mu$m)")
plt.ylabel(r"Chirality slope $d\alpha/dx$ ($^\circ$/mm)")
plt.grid(True, alpha=0.25)
plt.legend()
plt.savefig("cant_simple.png", dpi=300, bbox_inches="tight")
