import time
from assembly import AssemblyGCA
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error
from process import *

np.set_printoptions(precision=3, suppress=True)


def setup_model_pullin(process):
    model = AssemblyGCA(drawn_dimensions_filename="../layouts/fawn_underwater.csv", process=process)
    # model.gca.x0 = model.gca.x0_pullin()
    model.gca.terminate_simulation = model.gca.pulled_in
    return model


def setup_inputs(**kwargs):
    V = kwargs["V"]
    Fext = kwargs["Fext"]
    return lambda t, x: np.array([V, Fext])


def sim_gca(model, u, t_span, verbose=False):
    f = lambda t, x: model.dx_dt(t, x, u, verbose=verbose, Fes_calc_method=2, Fb_calc_method=2)
    x0 = model.x0()
    terminate_simulation = lambda t, x: model.terminate_simulation(t, x)
    terminate_simulation.terminal = True

    # speed saving measure
    sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True,
                    max_step=10e-6)  # , method="LSODA")
    return sol


def setup_plot(len_x, len_y, plt_title=None, x_label="", y_label=""):
    fig, axs = plt.subplots(len_x, len_y)
    if plt_title is not None:
        fig.suptitle(plt_title)
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

    t_span = [0, 30e-3]
    Fext = 0
    nx, ny = 2, 2

    data = loadmat("../data/20180208_fawn_gca_V_fingerL_pullin_underwater.mat")
    fingerL_values = np.ndarray.flatten(data["LARR"])*1e-6  # Length = 4

    pullin_V = []
    pullin_avg = []
    pullin_std = []
    labels = []
    r2_scores = []
    for i in range(1, len(fingerL_values) + 1):
        pullin_V.append(np.ndarray.flatten(data["V{}".format(i)]))
        pullin_avg.append(np.ndarray.flatten(data["t{}".format(i)]))
        labels.append(r"L=%0.1f$\mu$m"%(fingerL_values[i - 1]*1e6))

    # latexify(fig_width=6, columns=3)
    fig, axs = setup_plot(nx, ny)
    plot_data(fig, axs, pullin_V, pullin_avg, labels)

    # Simulation metrics
    pullin_V_results = []
    pullin_t_results = []
    r2_scores_pullin = []
    rmse_pullin = []
    legend_pullin = None

    # Pullin measurements
    # undercut = [4.6000000000000015e-07, 4.7000000000000016e-07, 3.7000000000000006e-07, 3.4000000000000003e-07]  # R2 min
    undercut = [4.6249999999999986e-07, 4.674999999999998e-07, 3.649999999999997e-07, 3.3499999999999997e-07]  # R2 min (finer precision on undercut)
    for idy in range(len(fingerL_values)):
        uc = undercut[idy]
        process = SOIwater()
        process.undercut = uc

        fingerL = fingerL_values[idy]
        model = setup_model_pullin(process=process)
        model.gca.fingerL = fingerL - model.gca.process.undercut
        model.gca.update_dependent_variables()
        model.gca.x0 = model.gca.x0_pullin()

        V_converged = []
        times_converged = []

        # V_test = np.sort(np.append(V_values, [pullin_V[idy], pullin_V[idy]+0.2]))  # Add some extra values to test
        V_values = pullin_V[idy]
        # V_test = V_values[:4]
        # V_test = list(np.arange(min(V_values), max(V_values) + 1, 1.))
        # V_test = list(np.arange(min(V_values)-0.01, min(V_values) + 0.15, 0.02))
        # V_test = np.sort(np.append(V_test, V_values))
        num_points = 5
        if idy == 1:
            lists = []
            lists.append(np.linspace(V_values[0] - 0.05, V_values[0], 3, endpoint=False))
            for idx in range(len(V_values) - 1):
                endpoint = idx == len(V_values) - 2
                if idx == 0:
                    num_points_local = 8
                elif idx >= len(V_values) // 2:
                    num_points_local = 3
                else:
                    num_points_local = num_points
                lists.append(np.linspace(V_values[idx], V_values[idx+1], num_points_local, endpoint=endpoint))
            V_test = np.sort(np.hstack(lists))
        else:
            lists = []
            for idx in range(len(V_values) - 1):
                endpoint = idx == len(V_values) - 2
                if idx == 0:
                    num_points_local = 8
                elif idx >= len(V_values) // 2:
                    num_points_local = 3
                else:
                    num_points_local = num_points
                lists.append(np.linspace(V_values[idx], V_values[idx + 1], num_points_local, endpoint=endpoint))
            V_test = np.sort(np.hstack(lists))
        # if idy == 0:
        #     # V_test = [min(V_values), max(V_values)]
        #     V_test = list(np.arange(min(V_values)+0.01, min(V_values) + 0.05, 0.01))
        #     V_test = np.sort(np.append(V_test, V_values[1:]))
        #     # V_test = [min(V_values), min(V_values) + 0.01]
        # elif idy == 1:
        #     # V_test = [min(V_values)-0.05, max(V_values)]
        #     V_test = list(np.arange(min(V_values)+0.003, min(V_values) + 0.05, 0.01))
        #     V_test = np.sort(np.append(V_test, V_values[1:]))
        #     # V_test = [min(V_values) + 0.002, min(V_values) + 0.003]
        # else:
        #     # V_test = [min(V_values)-0.2, max(V_values)]
        #     V_test = list(np.arange(min(V_values)-0.13, min(V_values) + 0.05, 0.01))
        #     V_test = np.sort(np.append(V_test, V_values))
        #     # V_test = [min(V_values) - 0.13, min(V_values) - 0.11]
        # V_test = list(np.arange(min(V_values) - 0.01, min(V_values) + 0.15, 0.02))
        # V_test = V_values[:2]
        # V_test = [min(V_values) - 0.02, min(V_values) - 0.01]
        print(V_values)
        print(V_test)
        # V_test = V_test[:4]
        for V in V_test:
            start_time = time.process_time()
            u = setup_inputs(V=V, Fext=Fext)
            sol = sim_gca(model, u, t_span)

            if len(sol.t_events[0]) > 0:
                V_converged.append(V)
                times_converged.append(sol.t_events[0][0]*1e3)  # ms conversion

            end_time = time.process_time()
            print("Runtime for L=", fingerL, ", V=", V, "=", end_time - start_time, ", undercut =", uc, "=",
                  '-->', {v: t for v, t in zip(V_converged, times_converged)})
        print(fingerL, {a: b for a, b in zip(V_converged, times_converged)})

        line, = axs[idy//ny, idy%ny].plot(V_converged, times_converged)
        if idy == ny - 1:
            legend_pullin = line
        pullin_V_results.append(V_converged)
        pullin_t_results.append(times_converged)

        # Calculate the r2 score
        actual = []
        pred = []
        for V in V_converged:
            if V in pullin_V[idy]:
                idx = np.where(pullin_V[idy] == V)[0][0]
                actual.append(pullin_avg[idy][idx])
                idx = np.where(V_converged == V)[0][0]
                pred.append(times_converged[idx])
        r2 = r2_score(actual, pred)
        r2_scores.append(r2)
        print("Pullin Pred:", pred, "Actual:", actual)
        print("R2 score for L=", fingerL, "=", r2)
        r2_scores_pullin.append(r2)
        rmse = mean_squared_error(actual, pred, squared=False)
        rmse_pullin.append(rmse)
        print("RMSE score for L=", fingerL, "=", rmse)

        np.save('../data/' + timestamp + '.npy', np.array([model.process, fingerL_values, pullin_V, pullin_avg, pullin_std,
                                                           pullin_V_results, pullin_t_results,
                                                           r2_scores_pullin, rmse_pullin,
                                                           fig], dtype=object),
            allow_pickle=True)

    print(pullin_V_results)
    print(pullin_t_results)
    print(r2_scores_pullin)
    print(rmse_pullin)

    # R2 scores
    # print("R2 scores:", r2_scores, np.mean(r2_scores), np.std(r2_scores))
    print("Finger L values:", [L*1e6 for L in fingerL_values])
    print("Pullin R2 scores:", r2_scores_pullin, np.mean(r2_scores_pullin), np.std(r2_scores_pullin))
    print("Pullin RMSE scores:", rmse_pullin, np.mean(rmse_pullin), np.std(rmse_pullin))

    # fig.legend([legend_pullin], ['Pull-in'], loc='lower right', ncol=2)

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Time (ms)")

    plt.tight_layout()
    plt.savefig("../figures/" + timestamp + ".png")
    plt.savefig("../figures/" + timestamp + ".pdf")

    plt.show()
