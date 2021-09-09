from assembly import AssemblyGCA
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.io import loadmat
from datetime import datetime
from utils import *


def setup_model_pullin():
    model = AssemblyGCA("../layouts/gamma.csv")
    model.gca.x0 = model.gca.x0_pullin()
    model.gca.terminate_simulation = model.gca.pulled_in
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

    sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True, max_step=0.5e-6, method="LSODA")
    return sol


def setup_plot(plt_title=None):
    fig = plt.figure()
    if plt_title is not None:
        fig.title(plt_title)
    ax = fig.get_axes()
    return fig, ax


def plot_data(fig, ax, Vs, ts, labels, colors, markers):
    print(len(Vs))
    for i in range(len(Vs)):
        plt.scatter(Vs[i], ts[i], label=labels[i], c=colors[i], marker=markers[i])
    plt.legend()


if __name__ == "__main__":
    now = datetime.now()
    name_clarifier = "_V_Fext_pullin"
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    model = setup_model_pullin()
    t_span = [0, 200e-6]
    colors = ['k', 'r', 'b']
    markers = ['x', 'o', '^']
    support_spring_widths = [0e-6, 5e-6, 6.04e-6]

    data = loadmat("../data/20190718_craig_gamma_V_Fext_pullin.mat")
    Fs = [0e-6, 50e-6, 100e-6]
    Vs = [data['V_0'][0].astype('float64'), data['V_50'][0].astype('float64'), data['V_100'][0].astype('float64')]
    ts = [data['F_0'][0].astype('float64'), data['F_50'][0].astype('float64'), data['F_100'][0].astype('float64')]
    print(Vs)
    print(ts)

    # latexify(fig_width=6, columns=3)
    fig, ax = setup_plot()
    plt.xlabel("Voltage (V)")
    plt.ylabel(r"Pull-in Time ($\mu$s)")

    labels = []
    for F in Fs:
        labels.append(r"F=%d$\mu$N"%int(F*1e6))

    plot_data(fig, ax, Vs, ts, labels, colors, markers)

    # Pullin measurements
    for idy in range(len(Fs)):
        Fext = Fs[idy]

        V_converged = []
        times_converged = []

        V_values = Vs[idy]
        V_test = V_values
        # V_test = np.arange(min(V_values), max(V_values))

        if idy != 0:
            model.gca.add_support_spring(springW=support_spring_widths[idy], springL=594.995e-6, nBeams=16,
                                         endcapW=22.889e-6, endcapL=49.441e-6,
                                         etchholeSize=8e-6, nEtchHoles=3, nEndCaps=8*2,
                                         k=Fext/(385.33e-6 + 2*model.gca.process.overetch))

        # (adds a lot of compute time, since failed simulations take time)
        for V in V_test:
            u = setup_inputs(V=V, Fext=Fext)
            sol = sim_gca(model, u, t_span)

            if len(sol.t_events[0]) > 0:
                V_converged.append(V)
                times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
                print(V, Fext, '|', sol.t_events[0][0]*1e6, 'us')
        print(times_converged)

        plt.plot(V_converged, times_converged, color=colors[idy])

    plt.tight_layout()
    plt.savefig("../figures/" + timestamp + ".png")
    plt.savefig("../figures/" + timestamp + ".pdf")
    plt.show()
