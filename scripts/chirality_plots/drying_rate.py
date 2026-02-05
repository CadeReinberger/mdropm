from matplotlib import pyplot as plt


def plot_drying_rate(ax):
    dt = [12, 25.5, 30, 34, 42.5, 60, 121, 35, 31.5, 66, 82, 25.5, 20.5, 27, 40]
    gfs = [
        0.0157,
        0.0295,
        0.0185,
        0.0112,
        0.0009,
        0.00157,
        0.0004,
        0.0508,
        0.0642,
        -0.0173,
        -0.0039,
        0.0528,
        0.055,
        0.0447,
        -0.0149,
    ]

    inds = sorted(list(range(len(dt))), key=lambda i: dt[i])
    dt = [dt[i] for i in inds]
    gfs = [gfs[i] for i in inds]

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 24,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
        }
    )

    ax.scatter(dt, gfs, label="Measured Data")
    ax.plot(
        (30, 30),
        (min(gfs), max(gfs)),
        "k--",
        label="Predicted Evaporation/Contact-Line Cutoff",
    )

    ax.set_title("Drying Rate Data")
    ax.set_xlabel("Drying time (min)", fontsize=24)
    ax.set_ylabel("Maximum g-factor", fontsize=24)
    ax.legend(fontsize=18)
    ax.grid(alpha=0.25)


def main():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot_drying_rate(ax)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
