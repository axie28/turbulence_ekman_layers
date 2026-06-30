# Basic RANS model for the Ekman layer to explore various models
# Near-surface region is only treated by clipping the turbulent viscosity to a minimum
# The stability function is only treated as a step function

import numpy as np
from scipy import special
from scipy.integrate import solve_ivp, RK23
import matplotlib.pyplot as plt
import Ktheory
from postprocessing import *

# Define the temporal grid
tmin = 0.0
tmax = 3600.0 * 48.0  # s, integration time
tcheck = 60.0 * 20.0  # s, time interval to checkpoint data

# Define the spatial grid, uniformly spaced
zmin = 0.0
zmax = 1000.0 * 2.0  # m, vertical size of the computational domain
nz = 200  # number of grid points in the vertical direction (levels)

# Define the problem
f = 1.0e-4  # Coriolis frequency, 1/s
ug = 10.0  # geostrophic wind velocity, m/s

Kmodel = Ktheory.Kmodel()
Hekman = np.sqrt(2.0 * Kmodel.K0 / f)

Kmodel = Ktheory.Kmodel(lm_outer=Hekman)  # redefine with outer scale to Ekman height
Kt = Kmodel.Blackadar  # define turbulence model
tag = "Blackadar"  # tags in postprocessing files

# check whether computational domain is larger than Ekman height
print("Thickness of Ekman layer {:3.2f} km.".format(Hekman / 1000.0))

# check whether integration time is larger than inertial period
t_inertial = 2.0 * np.pi / f
print("Period of inertial oscillation {:3.2f} h.".format(t_inertial / 3600.0))


###########################################################
def Ekman(eta):
    # Normalized Ekman solution
    u = 1.0 - np.exp(-eta) * np.cos(eta)
    v = np.exp(-eta) * np.sin(eta)

    return u, v


def preprocessing():
    # Construct grid
    z = np.linspace(zmin, zmax, num=nz)  # grid; vertical coordinate, m

    # Construct initial state vector (initial condition)
    K0_ini = Kmodel.K0  # change to study a transient
    H_ini = np.sqrt(2.0 * K0_ini / f)  # initial Ekman thickness
    u, v = ug * Ekman(z / H_ini)[0], ug * Ekman(z / H_ini)[1]
    # u, v = ug * special.erf(z / H_ini), np.zeros_like(z)

    # tke[:] = 0.
    tke = (
        0.1 * ug * 4.0 * (1.0 - special.erf(z / H_ini)) * special.erf(z / H_ini)
    )  # 10% of geostrophic wind velocity

    return z, np.array([u, v, tke])


def simulation(z, state0):
    # create array with the checkpointing times (times at which we save data)
    times = np.arange(0.0, tmax, tcheck)
    times = np.append(times, tmax)  # include the final time

    state = state0.flatten()  # solve_ivp works with 1da arrays
    sol = solve_ivp(Rhs, [tmin, tmax], state, RK23, t_eval=times)
    print(sol.message)

    return sol


def postprocessing(z, data):

    # reference in some of the plots below
    uref, vref = ug * Ekman(z / Hekman)[0], ug * Ekman(z / Hekman)[1]

    fig, axs = PlotProfiles(z, data, Kt)
    # additional information
    axs[0].plot(uref, z / 1000.0, "k", label=r"Ekman solution")
    axs[1].plot(vref, z / 1000.0, "k", label=r"Ekman solution")
    #
    fig.savefig("rans-ekman-" + tag + "-1.pdf", bbox_inches="tight")

    fig, axs = PlotHovmoeller(z, data)
    # additional information
    for ti in np.arange(t_inertial, tmax, t_inertial) / 3600.0:
        axs[0].plot((ti, ti), (z[0] / 1000.0, z[-1] / 1000.0), "k")
        axs[1].plot((ti, ti), (z[0] / 1000.0, z[-1] / 1000.0), "k")
    #
    fig.savefig("rans-ekman-" + tag + "-2.pdf", bbox_inches="tight")

    fig, ax = PlotHodograph(z, data)
    # additional information
    ax.plot(uref, vref, "k", label=r"Ekman solution")
    ax.legend(loc="best")
    #
    fig.savefig("rans-ekman-" + tag + "-3.pdf", bbox_inches="tight")

    plt.show()


###########################################################
def Rhs(t, state):
    u = state[0:nz]  # solve_ivp uses 1d array: create views for clarity
    v = state[nz : 2 * nz]
    tke = state[2 * nz : 3 * nz]

    Kt_loc, lm_loc = Kt(z=z, vel=(u, v), tke=tke, time=t)
    Ruw = Kt_loc * np.gradient(u, z)  # Reynolds stresses
    Rvw = Kt_loc * np.gradient(v, z)

    du_dt = np.gradient(Ruw, z) + f * v
    dv_dt = np.gradient(Rvw, z) - f * (u - ug)
    dk_dt = np.full_like(z, 0.0)  # to be done

    # no-slip boundary conditions
    du_dt[0], du_dt[-1] = 0.0, 0.0
    dv_dt[0], dv_dt[-1] = 0.0, 0.0
    dk_dt[0], dk_dt[-1] = 0.0, 0.0

    return np.array([du_dt, dv_dt, dk_dt]).flatten()


###########################################################
if __name__ == "__main__":
    z, state = preprocessing()
    data = simulation(z, state)
    # save(x, data)
    postprocessing(z, data)
