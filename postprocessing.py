import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


figsize = (12, 3)
num_to_plot = 8  # subsample interval of times to not clutter profiles plots


def PlotProfiles(z, data, Kt):
    t = data.t  # create views for clarity
    nz = np.size(z)
    u = data.y[0:nz, :]
    v = data.y[nz : 2 * nz, :]
    tke = data.y[2 * nz : 3 * nz, :]

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=figsize, sharey=True)

    times_to_plot = np.linspace(0, len(t) - 1, num=num_to_plot, dtype=int)
    for idx, it in enumerate(times_to_plot):
        alpha = (idx + 1) / len(times_to_plot)

        axs[0].plot(
            u[:, it],
            z / 1000.0,
            "C0",
            label=r"Time {:3.2f}".format(t[it] / 3600.0),
            alpha=alpha,
        )

        axs[1].plot(
            v[:, it],
            z / 1000.0,
            "C0",
            label=r"Time {:3.2f}".format(t[it] / 3600.0),
            alpha=alpha,
        )

        axs[2].plot(
            tke[:, it],
            z / 1000.0,
            "C0",
            label=r"Time {:3.2f}".format(t[it] / 3600.0),
            alpha=alpha,
        )

        Kt_loc, _ = Kt(z=z, vel=(u[:, it], v[:, it]), tke=tke[:, it], time=t[it])
        axs[3].plot(
            Kt_loc,
            z / 1000.0,
            "C0",
            label=r"Time {:3.2f}".format(t[it] / 3600.0),
            alpha=alpha,
        )

    axs[0].set_xlabel(r"Streamwise velocity (m s$^{-1}$)")
    axs[1].set_xlabel(r"Spanwise velocity (m s$^{-1}$)")
    axs[2].set_xlabel(r"Turbulence kinetic energy (m$^2$ s$^{-2}$)")
    axs[3].set_xlabel(r"Turbulent viscosity (m$^2$ s$^{-1}$)")
    axs[0].set_ylabel(r"Height (km)")
    # axs[1].set_ylabel(r"Height (km)")
    axs[0].legend(loc="best")

    for ax in axs:
        ax.spines["left"].set_position(("axes", -0.02))
        ax.spines["bottom"].set_position(("axes", -0.02))
        ax.set_ylim([0, None])

    plt.tight_layout(pad=0.5)

    return fig, axs


def PlotHovmoeller(z, data):
    t = data.t  # create views for clarity
    nz = np.size(z)
    u = data.y[0:nz, :]
    v = data.y[nz : 2 * nz, :]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)

    axs[0].pcolormesh(t / 3600.0, z / 1000.0, u)
    axs[0].set_xlabel(r"Time (h)")
    axs[0].set_ylabel(r"Height (km)")
    axs[0].set_title(r"Streamwise velocity")

    axs[1].pcolormesh(t / 3600.0, z / 1000.0, v)
    axs[1].set_xlabel(r"Time (h)")
    # axs[1].set_ylabel(r"Height (km)")
    axs[1].set_title(r"Spanwise velocity")

    for ax in axs:
        ax.spines["left"].set_position(("axes", -0.01))
        ax.spines["bottom"].set_position(("axes", -0.02))
        ax.set_ylim([0, None])
        ax.set_xlim([0, None])

    plt.tight_layout(pad=0.5)

    return fig, axs


def PlotHodograph(z, data):
    t = data.t  # create views for clarity
    nz = np.size(z)
    u = data.y[0:nz, :]
    v = data.y[nz : 2 * nz, :]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))

    times_to_plot = np.linspace(0, len(t) - 1, num=num_to_plot, dtype=int)
    for idx, it in enumerate(times_to_plot):
        alpha = (idx + 1) / len(times_to_plot)
        ax.plot(
            u[:, it],
            v[:, it],
            "C0",
            label=r"Time {:3.2f}".format(t[it] / 3600.0),
            alpha=alpha,
        )

    ax.set_xlabel(r"Streamwise velocity (m s$^{-1}$)")
    ax.set_ylabel(r"Spanwise velocity (m s$^{-1}$)")
    ax.legend(loc="best")

    ax.spines["left"].set_position(("axes", -0.02))
    ax.spines["bottom"].set_position(("axes", -0.02))

    plt.tight_layout(pad=0.5)

    return fig, ax
