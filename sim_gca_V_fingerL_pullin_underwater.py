from assembly import AssemblyGCA
import numpy as np

np.set_printoptions(precision=3, suppress=True)
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from datetime import datetime
from utils import *
from sklearn.metrics import r2_score
from process import *


def setup_model_pullin():
    model = AssemblyGCA(drawn_dimensions_filename="fawn.csv", process=SOIwater())
    # model.gca.x0 = model.gca.x0_pullin()
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

    # speed saving measure
    if u(0, 0)[0] < 2.5:  # long pull-in times
        sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True,
                        max_step=15e-6)  # , method="LSODA")
    else:
        sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True,
                        max_step=10e-6)  # , method="LSODA")
    return sol


def setup_plot(len_x, len_y, plt_title=None, x_label="", y_label=""):
    fig, axs = plt.subplots(len_x, len_y)
    if plt_title is not None:
        fig.suptitle(plt_title)

    # fig.text(0.5, 0.04, x_label, ha='center')
    # fig.text(0.04, 0.5, y_label, va='center', rotation='vertical')
    return fig, axs


def plot_data(fig, axs, pullin_V, pullin_avg, labels):
    nx, ny = 2, 2
    for idx in range(nx):
        for idy in range(ny):
            i = ny*idx + idy
            ax = axs[idx, idy]
            ax.plot(pullin_V[i], pullin_avg[i], 'b.')
            ax.annotate(labels[i], xy=(1, 1), xycoords='axes fraction', fontsize=10,
                        xytext=(-2, -2), textcoords='offset points',
                        ha='right', va='top')


if __name__ == "__main__":
    now = datetime.now()
    name_clarifier = "_V_fingerL_pullin_underwater"
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    model = setup_model_pullin()
    t_span = [0, 30e-3]
    Fext = 0
    nx, ny = 2, 2

    data = loadmat("data/20180208_fawn_gca_V_fingerL_pullin_underwater.mat")
    fingerL_values = np.ndarray.flatten(data["LARR"])*1e-6  # Length = 4

    # V_values = np.arange(1, 7, 0.1)
    # latexify(fig_width=6, columns=3)
    fig, axs = setup_plot(nx, ny, x_label="Voltage (V)", y_label="Pull-in Time (us)")
    # axs[2, 1].set_xlabel("Voltage (V)")
    # axs[1, 0].set_ylabel("Time (us)")

    pullin_V = []
    pullin_avg = []
    pullin_std = []
    labels = []
    for i in range(1, len(fingerL_values) + 1):
        pullin_V.append(np.ndarray.flatten(data["V{}".format(i)]))
        pullin_avg.append(np.ndarray.flatten(data["t{}".format(i)]))
        labels.append(r"L=%0.1f$\mu$m"%(fingerL_values[i - 1]*1e6))

    plot_data(fig, axs, pullin_V, pullin_avg, labels)

    # Pullin measurements
    for idy in range(len(fingerL_values)):
        fingerL = fingerL_values[idy]
        model.gca.fingerL = fingerL - model.gca.process.overetch
        model.gca.update_dependent_variables()
        model.gca.x0 = model.gca.x0_pullin()

        V_converged = []
        times_converged = []

        # V_test = np.sort(np.append(V_values, [pullin_V[idy], pullin_V[idy]+0.2]))  # Add some extra values to test
        V_test = []
        V_values = pullin_V[idy]
        # V_test = list(np.arange(min(V_values), max(V_values) + 1, 1.))
        # V_test = list(np.arange(min(V_values), min(V_values) + 0.5, 0.1))
        V_test = V_values
        # for V in V_values:
        #     # V_test.append(V - 0.1)
        #     V_test.append(V)
        #     # V_test.append(V + 0.2)
        #     # V_test.append(V + 0.5)
        #     # V_test.append(V + 1)
        #     # V_test.append(V + 1.5)
        #     # V_test.append(V + 2)
        for V in V_test:
            u = setup_inputs(V=V, Fext=Fext)
            sol = sim_gca(model, u, t_span)

            if len(sol.t_events[0]) > 0:
                V_converged.append(V)
                times_converged.append(sol.t_events[0][0]*1e3)  # ms conversion
        print(fingerL, V_converged, times_converged)

        axs[idy//ny, idy%ny].plot(V_converged, times_converged)

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Time (us)")

    plt.tight_layout()
    plt.savefig("figures/" + timestamp + ".png")
    plt.savefig("figures/" + timestamp + ".pdf")
    plt.show()
