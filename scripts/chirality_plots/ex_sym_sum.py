import numpy as np
from matplotlib import pyplot as plt

R_START = 3.0
R_END = 0.3
N_TIMES = 4
N_S = 200

radii = np.linspace(R_START, R_END, N_TIMES)

def circle_points(r, n=N_S):
    s = np.linspace(0, 2 * np.pi, n + 1)
    return r * np.cos(s), r * np.sin(s)

def make_multi_plot():
    pad = 1.1 * R_START
    lim = (-pad, pad)

    n = len(radii)
    fig, axes = plt.subplots(1, n, figsize=(max(3.0 * n, 4.0), 3.2), squeeze=False)
    axes = axes[0]

    for i, (ax, r) in enumerate(zip(axes, radii)):
        ax.set_aspect('equal')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xticks([])
        ax.set_yticks([])

        x, y = circle_points(r)
        ax.plot(-y, x, 'k', linewidth=2)

        if i > 0:
            ref = plt.Circle((0, 0), R_START, fill=False, linestyle='--', edgecolor='gray', linewidth=1)
            ax.add_patch(ref)

    fig.tight_layout(w_pad=0.3)
    plt.savefig('ex_sym_sum.png')

if __name__ == '__main__':
    make_multi_plot()
