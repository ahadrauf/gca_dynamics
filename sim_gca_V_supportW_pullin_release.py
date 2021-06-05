from assembly import AssemblyGCA
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from datetime import datetime
from utils import *


def setup_model_pullin():
    model = AssemblyGCA()
    model.gca.x0 = model.gca.x0_pullin()
    model.gca.terminate_simulation = model.gca.pulled_in
    return model


def setup_model_release(**kwargs):
    u = [kwargs["V"], kwargs["Fext"]]
    model = AssemblyGCA()
    # model.gca.k_support = 10.303975
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

    sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True, max_step=0.5e-6) #, method="LSODA")
    return sol


def setup_plot(len_x, len_y, plt_title=None, x_label="", y_label=""):
    fig, axs = plt.subplots(len_x, len_y)
    if plt_title is not None:
        fig.suptitle(plt_title)

    # fig.text(0.5, 0.04, x_label, ha='center')
    # fig.text(0.04, 0.5, y_label, va='center', rotation='vertical')
    return fig, axs


def plot_data(fig, axs, pullin_V, pullin_avg, pullin_std, release_V, release_avg, release_std, labels):
    nx, ny = 4, 2
    for idx in range(nx):
        for idy in range(ny):
            i = ny*idx + idy
            print(idx, idy, i)
            ax = axs[idx, idy]
            ax.errorbar(pullin_V[i], pullin_avg[i], pullin_std[i], fmt='b.', capsize=3)
            ax.errorbar(release_V[i], release_avg[i], release_std[i], fmt='r.', capsize=3)
            ax.annotate(labels[i], xy=(1, 1), xycoords='axes fraction', fontsize=10,
                        xytext=(-2, -2), textcoords='offset points',
                        ha='right', va='top')

    # plt.legend()


if __name__ == "__main__":
    now = datetime.now()
    name_clarifier = "_V_supportW_release"
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    model = setup_model_pullin()
    t_span = [0, 300e-6]
    Fext = 0

    data = loadmat("data/20180208_fawn_gca_V_fingerL_pullin_release.mat")
    # fingerL_values = np.ndarray.flatten(data["LARR"])*1e-6  # Length = 9
    springW_values = np.array([2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5])*1e-6

    V_values = np.arange(20, 105, 5)
    # latexify(fig_width=6, columns=3)
    fig, axs = setup_plot(4, 2, x_label="Voltage (V)", y_label="Pull-in Time (us)")
    # axs[2, 1].set_xlabel("Voltage (V)")
    # axs[1, 0].set_ylabel("Time (us)")

    pullin_V = []
    pullin_avg = []
    pullin_std = []
    release_V = []
    release_avg = []
    release_std = []
    labels = []
    for i in range(1, len(springW_values) + 1):
        pullin_V.append(np.ndarray.flatten(data["V{}_Arr2".format(i)]))
        pullin_avg.append(np.ndarray.flatten(data["tmeas{}_Arr2".format(i)]))
        pullin_std.append(np.ndarray.flatten(data["err{}_Arr2".format(i)]))
        release_V.append(np.ndarray.flatten(data["V{}_Arr2_r".format(i)]))
        release_avg.append(np.ndarray.flatten(data["tmeas{}_Arr2_r".format(i)]))
        release_std.append(np.ndarray.flatten(data["err{}_Arr2_r".format(i)]))
        labels.append(r"$w_{s}$=%0.1f$\mu$m"%(springW_values[i - 1]*1e6))
        # if i < len(fingerL_values):
        #     labels.append("$L_{ol}$=%0.1f$L_{0}$" % (0.1*i+0.1))
        # else:
        #     labels.append(r"$L_{ol}=L_{0}$=%0.1f$\mu$m"%(fingerL_values[i - 1]*1e6))

    plot_data(fig, axs, pullin_V, pullin_avg, pullin_std, release_V, release_avg, release_std, labels)

    nx, ny = 4, 2

    # Pullin measurements
    for idy in range(len(springW_values)):
        springW = springW_values[idy]
        # model.gca.fingerL = fingerL - model.gca.process.overetch
        model.gca.supportW = springW - 2 * model.gca.process.overetch
        model.gca.update_dependent_variables()

        V_converged = []
        times_converged = []

        # V_test = np.sort(np.append(V_values, [pullin_V[idy], pullin_V[idy]+0.2]))  # Add some extra values to test
        V_test = []
        V_values = pullin_V[idy]
        for V in V_values:
            # V_test.append(V - 0.1)
            V_test.append(V)
            # V_test.append(V + 0.2)
            # V_test.append(V + 0.5)
            # V_test.append(V + 1)
            # V_test.append(V + 1.5)
            # V_test.append(V + 2)
        # V_test = np.arange(min(V_values), max(V_values) + 3)
        # V_test = V_values
        # (adds a lot of compute time, since failed simulations take time)
        for V in V_test:
            u = setup_inputs(V=V, Fext=Fext)
            sol = sim_gca(model, u, t_span)

            if len(sol.t_events[0]) > 0:
                V_converged.append(V)
                times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
        print(springW, V_converged, times_converged)

        axs[idy//ny, idy%ny].plot(V_converged, times_converged)

    # Release measurements
    for idy in range(len(springW_values)):
        springW = springW_values[idy]

        V_converged = []
        times_converged = []

        V_values = release_V[idy]
        V_test = V_values
        # V_test = np.arange(min(V_values), max(V_values) + 3)
        # V_test = np.sort(np.append(V_values, [release_V[idy], release_V[idy]+0.2]))  # Add some extra values to test
        # (adds a lot of compute time, since failed simulations take time)
        for V in V_test:
            model = setup_model_release(V=V, Fext=Fext)
            model.gca.supportW = springW - 2*model.gca.process.overetch
            model.gca.update_dependent_variables()
            u = [V, Fext]
            model.gca.x0 = model.gca.x0_release(u)
            u = setup_inputs(V=0, Fext=Fext)  # Changed for release
            sol = sim_gca(model, u, t_span)

            if len(sol.t_events[0]) > 0:
                V_converged.append(V)
                times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
        print(times_converged)
        axs[idy//ny, idy%ny].plot(V_converged, times_converged, 'r')

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
