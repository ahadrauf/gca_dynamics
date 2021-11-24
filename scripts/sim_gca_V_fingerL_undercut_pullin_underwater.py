"""
One of the core files from the paper, which generated Fig. 10. Simulates the effect of varying finger overlap length and
voltage on the pull-in underwater, and varies the undercut to minimize the error squared between the simulation
and data.
"""

import os
file_location = os.path.abspath(os.path.dirname( __file__))
dir_location = os.path.abspath(os.path.join(file_location, '..'))
import sys
sys.path.append(file_location)
sys.path.append(dir_location)

from assembly import AssemblyGCA
from process import *
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error
import time


def setup_model_pullin(process):
    model = AssemblyGCA(drawn_dimensions_filename="../layouts/fawn_underwater.csv", process=process)
    # model.gca.x0 = model.gca.x0_pullin()
    model.gca.terminate_simulation = model.gca.pulled_in
    return model


def setup_model_release(V, Fext, process, **kwargs):
    u = [V, Fext]
    model = AssemblyGCA(drawn_dimensions_filename="../layouts/fawn_underwater.csv", process=process)
    model.gca.terminate_simulation = model.gca.released
    return model


def setup_inputs(**kwargs):
    V = kwargs["V"]
    Fext = kwargs["Fext"]
    return lambda t, x: np.array([V, Fext])


def sim_gca(model, u, t_span, verbose=False, Fes_calc_method=2, Fb_calc_method=2):
    f = lambda t, x: model.dx_dt(t, x, u, verbose=verbose, Fes_calc_method=Fes_calc_method,
                                 Fb_calc_method=Fb_calc_method)
    x0 = model.x0()
    terminate_simulation = lambda t, x: model.terminate_simulation(t, x)
    terminate_simulation.terminal = True

    sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True,
                    max_step=10e-6)  # , method="LSODA")
    return sol


def setup_plot(len_x, len_y, plt_title=None):
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
    Fes_calc_method, Fb_calc_method = 2, 2
    name_clarifier = "_V_fingerL_pullin_undercut_underwater_Fes=v{}_Fb=v{}".format(Fes_calc_method, Fb_calc_method)
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    t_span = [0, 40e-3]
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

    nx, ny = 2, 2
    legend_pullin, legend_release = None, None

    # Simulation metrics
    # undercut_range = np.arange(0.2e-6, 0.501e-6, 0.01e-6)
    # undercut_range = np.arange(0.3e-6, 0.501e-6, 0.01e-6)
    undercut_range = np.append(np.arange(0.33e-6, 0.3801e-6, 0.0025e-6),
                               np.arange(0.45e-6, 0.481e-6, 0.0025e-6))
    best_undercut_pullin = []
    best_undercut_release = []
    pullin_V_results = {undercut: [] for undercut in undercut_range}
    pullin_t_results = {undercut: [] for undercut in undercut_range}
    r2_scores_pullin = {undercut: [] for undercut in undercut_range}
    rmse_pullin = {undercut: [] for undercut in undercut_range}

    # Pullin measurements
    for undercut in undercut_range:
        process = SOIwater()
        process.undercut = undercut

        for idy in range(len(fingerL_values)):
            fingerL = fingerL_values[idy]
            model = setup_model_pullin(process=process)
            model.gca.fingerL = fingerL - model.gca.process.undercut
            model.gca.update_dependent_variables()
            model.gca.x0 = model.gca.x0_pullin()

            V_converged = []
            times_converged = []

            V_values = pullin_V[idy]
            V_test = V_values[:5]
            # V_test = list(np.arange(min(V_values), max(V_values) + 1, 1.))
            for V in V_test:
                start_time = time.process_time()
                u = setup_inputs(V=V, Fext=Fext)
                sol = sim_gca(model, u, t_span, Fes_calc_method=Fes_calc_method, Fb_calc_method=Fb_calc_method)

                if len(sol.t_events[0]) > 0:
                    V_converged.append(V)
                    times_converged.append(sol.t_events[0][0]*1e3)  # us conversion

                end_time = time.process_time()
                print("Runtime pullin for L =", fingerL, ", V =", V, ", undercut =", undercut, "=",
                      end_time - start_time, '-->', {v: t for v, t in zip(V_converged, times_converged)})
            # print(fingerL, V_converged, times_converged)
            print({V: t for V, t in zip(V_converged, times_converged)})

            pullin_V_results[undercut].append(V_converged)
            pullin_t_results[undercut].append(times_converged)

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
            r2_scores_pullin[undercut].append(r2)
            rmse = mean_squared_error(actual, pred, squared=False)
            rmse_pullin[undercut].append(rmse)
            print("R2 =", r2, ", RMSE =", rmse)

    print("Pullin V results:", pullin_V_results)
    print("Pullin t results:", pullin_t_results)
    print("R2 scores pullin:", r2_scores_pullin)

    r2_scores_pullin_agg = []
    for idy in range(len(fingerL_values)):
        best_uc = undercut_range[np.argmax([r2_scores_pullin[uc][idy] for uc in undercut_range])]
        best_undercut_pullin.append(best_uc)
        best_undercut_release.append(best_uc)
        r2_scores_pullin_agg.append(r2_scores_pullin[best_uc][idy])
        print("Results", idy, ", L =", fingerL_values[idy], ", r2 pullin =",
              np.max([r2_scores_pullin[uc][idy] for uc in undercut_range]), r2_scores_pullin[best_uc][idy],
              ", undercut =", best_uc)
        V_converged_pullin = pullin_V_results[best_uc][idy]
        times_converged_pullin = pullin_t_results[best_uc][idy]
        line_pullin, = axs[idy//ny, idy%ny].plot(V_converged_pullin, times_converged_pullin, 'b')
        if idy == ny - 1:
            legend_pullin = line_pullin

    print("Finger L values:", [L*1e6 for L in fingerL_values])
    print("Pullin V results", pullin_V_results)
    print("Pullin t results", pullin_t_results)
    print("Pullin R2 scores:", r2_scores_pullin)  #, np.mean(r2_scores_pullin), np.std(r2_scores_pullin))
    print("Pullin R2 scores aggregate:", r2_scores_pullin_agg, np.nanmean(r2_scores_pullin_agg), np.nanstd(r2_scores_pullin_agg))
    print("Pullin RMSE scores:", rmse_pullin)  #, np.mean(rmse_pullin), np.std(rmse_pullin))
    print("Best undercut pullin:", best_undercut_pullin)

    # axs[0, ny-1].legend([legend_pullin, legend_release], ['Pull-in', 'Release'])
    # fig.legend([legend_pullin], ['Pull-in'], loc='lower right', ncol=2)
    # fig.legend([legend_release], ['Release'], loc='lower right', ncol=2)
    # fig.legend([legend_pullin, legend_release], ['Pull-in', 'Release'], loc='lower right', ncol=2)

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Pull-in Time (us)")

    plt.tight_layout()
    plt.savefig("../figures/" + timestamp + ".png")
    plt.savefig("../figures/" + timestamp + ".pdf")

    np.save('../data/' + timestamp + '.npy', np.array([fingerL_values, pullin_V, pullin_avg, pullin_std,
                                                       pullin_V_results, pullin_t_results,
                                                       r2_scores_pullin, rmse_pullin,
                                                       best_undercut_pullin, fig], dtype=object),
            allow_pickle=True)
    plt.show()
