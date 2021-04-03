from assembly import AssemblyGCA
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.io import loadmat
from utils import *


def setup_model_pullin():
    model = AssemblyGCA()
    model.gca.x0 = model.gca.x0_pullin()
    model.gca.terminate_simulation = model.gca.pulled_in
    return model


def setup_model_release(**kwargs):
    u = [kwargs["V"], kwargs["Fext"]]
    model = AssemblyGCA()
    model.gca.x0 = model.gca.x0_release(u)
    model.gca.terminate_simulation = model.gca.released
    return model


def setup_inputs(**kwargs):
    V = kwargs["V"]
    Fext = kwargs["Fext"]
    return lambda t, x: np.array([V, Fext])


def sim_gca(model, u, t_span, verbose=False):
    f = lambda t, x: model.dx_dt(t, x, u, verbose=verbose)
    x0 = model.x0()
    terminate_simulation = lambda t, x: model.terminate_simulation(t, x)
    terminate_simulation.terminal = True

    sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True, max_step=0.5e-6)
    return sol


def setup_plot(plt_title=None):
    fig = plt.figure()
    if plt_title is not None:
        fig.title(plt_title)
    ax = fig.get_axes()
    return fig, ax


def plot_data(fig, ax, Vs, ts, labels, colors, markers):
    for i in range(len(Vs)):
        plt.plot(Vs[i], ts[i], label=labels[i], )
    plt.legend()


if __name__ == "__main__":
    model = setup_model_pullin()
    t_span = [0, 100e-6]
    colors = ['k', 'r', 'b']
    markers = ['x', 'o', '^']

    data = loadmat("data/craig_gamma_V_Fext_pullin.mat")
    Fs = [0, 50e-6, 100e-6]
    Vs = [data['V_0'], data['V_50'], data['V_100']]
    ts = [data['F_0'], data['F_50'], data['F_100']]

    V_values = np.arange(20, 100, 5)
    # latexify(fig_width=6, columns=3)
    fig, ax = setup_plot()

    labels = []
    for F in Fs:
        labels.append(r"F=%d$\mu$N" % int(F*1e6))

    plot_data(fig, ax, Vs, ts, labels)

    nx, ny = 3, 3

    # Pullin measurements
    for idy in range(len(Fs)):
        Fext = Fs[idy]

        V_converged = []
        times_converged = []

        V_test = []
        for V in V_values:
            # V_test.append(V - 0.1)
            V_test.append(V)
            # V_test.append(V + 0.2)
            V_test.append(V + 0.5)
            V_test.append(V + 1)
            V_test.append(V + 1.5)
            V_test.append(V + 2.5)
        # (adds a lot of compute time, since failed simulations take time)
        for V in V_test:
            u = setup_inputs(V=V, Fext=Fext)
            sol = sim_gca(model, u, t_span)

            if len(sol.t_events[0]) > 0:
                V_converged.append(V)
                times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
        print(times_converged)

        # ax = plt.subplot(nx, ny, idy+1)
        # plt.plot(V_converged, times_converged)
        plt.plot(V_converged, times_converged)
        # ax.text(0.8*ax.get_xlim()[-1], 0.8*ax.get_ylim()[-1], "w={}um\nL={}um".format(fingerW*1e6, fingerL*1e6))
        # ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

    plt.tight_layout()
    plt.show()
