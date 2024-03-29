"""
Pre-decessor to the _undercut variation of this same file, which was ultimately used for the final paper.
Simulate the pull-in and release of a GCA vs. voltage while sweeping over finger lengths. Uses the fixed
undercut from whichever process you import (or you can specify it manually).
"""

import os

file_location = os.path.abspath(os.path.dirname(__file__))
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

np.set_printoptions(precision=3, suppress=True)


def setup_model_pullin(process):
    model = AssemblyGCA(process=process)
    # model.gca.x0 = model.gca.x0_pullin()
    model.gca.terminate_simulation = model.gca.pulled_in
    # model.gca.Fescon = 1.1
    return model


def setup_model_release(V, Fext, process, **kwargs):
    u = [V, Fext]
    model = AssemblyGCA(process=process)
    model.gca.terminate_simulation = model.gca.released
    return model


def setup_inputs(**kwargs):
    V = kwargs["V"]
    Fext = kwargs["Fext"]
    return lambda t, x: np.array([V, Fext])


def sim_gca(model, u, t_span, verbose=False, Fes_calc_method="trap", Fb_calc_method=2, max_step=0.075e-6):
    f = lambda t, x: model.dx_dt(t, x, u, verbose=verbose, Fes_calc_method=Fes_calc_method,
                                 Fb_calc_method=Fb_calc_method)
    x0 = model.x0()
    terminate_simulation = lambda t, x: model.terminate_simulation(t, x)
    terminate_simulation.terminal = True

    sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True,
                    max_step=max_step)  # , method="LSODA")
    return sol


def setup_plot(len_x, len_y, plt_title=None, x_label="", y_label=""):
    fig, axs = plt.subplots(len_x, len_y)
    if plt_title is not None:
        fig.suptitle(plt_title)

    # fig.text(0.5, 0.04, x_label, ha='center')
    # fig.text(0.04, 0.5, y_label, va='center', rotation='vertical')
    return fig, axs


def plot_data(fig, axs, labels):
    nx, ny = 3, 3
    for idx in range(nx):
        for idy in range(ny):
            i = ny * idx + idy
            # print(idx, idy, i)  # Confirms that I've ordered the data plots correctly
            ax = axs[idx, idy]
            # ax.grid(True)
            # ax.errorbar(pullin_V[i], pullin_avg[i], pullin_std[i], fmt='b.', capsize=3)
            # ax.errorbar(release_V[i], release_avg[i], release_std[i], fmt='r.', capsize=3)
            ax.annotate(labels[i], xy=(1, 1), xycoords='axes fraction', fontsize=10,
                        xytext=(-2, -2), textcoords='offset points',
                        ha='right', va='top')


if __name__ == "__main__":
    now = datetime.now()
    undercut = SOI().undercut
    Fes_calc_method, Fb_calc_method = "trap", 2
    # name_clarifier = "_V_fingerL_pullin_release_undercut={:.3f}_Fes=v{}_Fb=v{}".format(undercut*1e6, Fes_calc_method, Fb_calc_method)
    name_clarifier = "_vacuum_V_fingerL_pullin_release_undercut=fixedtmax1000e6_Fes=v{}_Fb=v{}".format(Fes_calc_method,
                                                                                                Fb_calc_method)
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)
    total_start_time = time.process_time()

    t_span = [0, 200e-6]
    Fext = 0

    # data = loadmat("../data/20180208_fawn_gca_V_fingerL_pullin_release.mat")
    # fingerL_values = np.ndarray.flatten(data["LARR"])*1e-6  # Length = 9
    fingerL_values = np.array([25.3, 32.97, 40.59, 48.49, 55.88, 63.53, 71.18, 78.824, 86.472]) * 1e-6
    fingerLbuff_values = np.array([9.5, 10, 10, 10.3, 10, 10, 10, 10, 10]) * 1e-6
    fingerWtip_values = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2]) * 1e-6
    fingerWbase_values = np.array([3.82, 4.474, 5.132, 5.79, 6.448, 7.106, 7.764, 8.42, 8.999]) * 1e-6

    # Calculate back gap distance
    distance_between_flats = 17.75e-6
    m = -fingerL_values / (fingerWbase_values - fingerWtip_values)
    b1 = -m * fingerWbase_values
    b2 = fingerLbuff_values - m * (distance_between_flats - fingerWtip_values)
    gb_values = np.abs(b2 - b1) / np.sqrt(m * m + 1)

    # latexify(fig_width=6, columns=3)
    fig, axs = setup_plot(3, 3)
    axs[2, 1].set_xlabel("Voltage (V)")
    axs[1, 0].set_ylabel(r"Time ($\mu$s)")

    labels = [r"$w_{f,base}$" + "={:.2f}".format(fingerWbase * 1e6) + r"$\mu$m" + "\n" + \
              r"$L_{ol}$" + "={:.1f}".format(fingerL * 1e6 - fingerLbuff * 1e6) + r"$\mu$m" for
              fingerL, fingerLbuff, fingerWbase in zip(fingerL_values, fingerLbuff_values, fingerWbase_values)]

    plot_data(fig, axs, labels)

    nx, ny = 3, 3
    legend_pullin, legend_release = None, None

    # Simulation metrics
    pullin_V_results = []
    pullin_t_results = []
    release_V_results = []
    release_t_results = []
    r2_scores_pullin = []
    r2_scores_release = []
    rmse_pullin = []
    rmse_release = []

    # Pullin measurements
    # process = SOI()
    # process = SOIvacuum()
    V_values = np.arange(30, 101, 5)
    undercut = [0.4e-6] * len(fingerL_values)
    # undercut = [4.5e-07, 4.0e-07, 3.4e-07, 3.0e-07, 2.9e-07, 2.8e-07, 2.4e-07, 2.8e-07, 2.6e-07]  # based on UC min/UC avg
    # undercut = [4.5000000000000024e-07, 4.000000000000002e-07, 3.4000000000000013e-07, 3.000000000000001e-07, 2.900000000000001e-07, 2.8000000000000007e-07, 2.4000000000000003e-07, 2.8000000000000007e-07, 3.000000000000001e-07]  # based on padded UC average
    # undercut = [4.5000000000000024e-07, 4.000000000000002e-07, 3.4000000000000013e-07, 3.300000000000001e-07, 2.900000000000001e-07, 2.8000000000000007e-07, 2.900000000000001e-07, 3.300000000000001e-07, 3.100000000000001e-07]  # based on padded UC min
    # undercut = [4.5000000000000024e-07, 4.000000000000002e-07, 3.4000000000000013e-07, 3.300000000000001e-07, 2.900000000000001e-07, 2.8000000000000007e-07, 2.7000000000000006e-07, 3.300000000000001e-07, 3.100000000000001e-07]  # based on padded UC min (corrected t_max)
    # undercut = [4.5000000000000024e-07, 4.000000000000002e-07, 3.4000000000000013e-07, 3.8000000000000017e-07, 2.900000000000001e-07, 2.8000000000000007e-07, 4.100000000000002e-07, 3.4000000000000013e-07, 3.100000000000001e-07]  # based on padded UC min (t max = 1e-3)
    # undercut = [4.5000000000000024e-07, 4.000000000000002e-07, 3.4000000000000013e-07, 3.8000000000000017e-07, 2.900000000000001e-07, 2.8000000000000007e-07, 3.6000000000000015e-07, 3.4000000000000013e-07, 3.100000000000001e-07]
    # undercut = [5.000000000000003e-07, 4.200000000000002e-07, 4.000000000000002e-07, 3.200000000000001e-07, 2.5000000000000004e-07, 3.100000000000001e-07, 2.8000000000000007e-07, 3.000000000000001e-07, 2.6000000000000005e-07]  # based on RMSE max
    for idy in range(len(fingerL_values)):
        uc = undercut[idy]
        process = SOIvacuum()
        process.undercut = uc
        model = setup_model_pullin(process=process)
        fingerL = fingerL_values[idy]
        fingerLbuff = fingerLbuff_values[idy]
        fingerWtip = fingerWtip_values[idy]
        fingerWbase = fingerWbase_values[idy]
        gb = gb_values[idy]
        model.gca.fingerL = fingerL - model.gca.process.undercut
        model.gca.fingerL_buffer = fingerLbuff
        model.gca.fingerWtip = fingerWtip - 2 * model.gca.process.undercut
        model.gca.fingerWbase = fingerWbase - 2 * model.gca.process.undercut
        model.gca.gb = gb + 2 * model.gca.process.undercut
        model.gca.update_dependent_variables()
        model.gca.x0 = model.gca.x0_pullin()

        V_converged = []
        times_converged = []

        # V_test = np.sort(np.append(V_values, [pullin_V[idy], pullin_V[idy]+0.2]))  # Add some extra values to test
        # V_values = pullin_V[idy]
        V_test = V_values
        # V_test = list(np.arange(min(V_values), max(V_values) + 1, 0.1))
        # V_test = V_test[:20]
        for V in V_test:
            start_time = time.process_time()
            u = setup_inputs(V=V, Fext=Fext)
            try:
                sol = sim_gca(model, u, t_span, Fes_calc_method=Fes_calc_method, Fb_calc_method=Fb_calc_method)

                if len(sol.t_events[0]) > 0:
                    V_converged.append(V)
                    times_converged.append(sol.t_events[0][0] * 1e6)  # us conversion
            except Exception as e:
                print("Simulation failed for idy={}, V={}".format(idy, V), str(e))

            end_time = time.process_time()
            print("Runtime for L=", fingerL, ", V =", V, "=", end_time - start_time, ", undercut =", uc, "=",
                  end_time - start_time, '-->', {v: t for v, t in zip(V_converged, times_converged)})
        print(fingerL, V_converged, times_converged)

        line, = axs[idy // ny, idy % ny].plot(V_converged, times_converged)
        if idy == ny - 1:
            legend_pullin = line
        pullin_V_results.append(V_converged)
        pullin_t_results.append(times_converged)

        # Calculate the r2 score
        # actual = []
        # pred = []
        # for V in V_converged:
        #     if V in pullin_V[idy]:
        #         idx = np.where(pullin_V[idy] == V)[0][0]
        #         actual.append(pullin_avg[idy][idx])
        #         idx = np.where(V_converged == V)[0][0]
        #         pred.append(times_converged[idx])
        # r2 = r2_score(actual, pred)
        # print("Pullin Pred:", pred, "Actual:", actual)
        # ratios = [p/a for p, a in zip(pred, actual)]
        # print("Pullin Ratios:", np.max(ratios), np.min(ratios), ratios)
        # print("R2 score for L=", fingerL, "=", r2)
        # r2_scores_pullin.append(r2)
        # rmse = mean_squared_error(actual, pred, squared=False)
        # rmse_pullin.append(rmse)
        # print("RMSE score for L=", fingerL, "=", rmse)

    # Release measurements
    for idy in range(len(fingerL_values)):
        uc = undercut[idy]
        process = SOIvacuum()
        process.undercut = uc

        fingerL = fingerL_values[idy]
        fingerLbuff = fingerLbuff_values[idy]
        fingerWtip = fingerWtip_values[idy]
        fingerWbase = fingerWbase_values[idy]

        V_converged = []
        times_converged = []

        # V_values = release_V[idy]
        V_test = V_values
        # V_test = list(np.arange(min(V_values), max(V_values) + 1, 0.1))
        # V_test = V_test[:20]
        for V in V_test:
            start_time = time.process_time()
            model = setup_model_release(V=V, Fext=Fext, process=process)
            gb = gb_values[idy]
            model.gca.fingerL = fingerL - model.gca.process.undercut
            model.gca.fingerL_buffer = fingerLbuff
            model.gca.fingerWtip = fingerWtip - 2 * model.gca.process.undercut
            model.gca.fingerWbase = fingerWbase - 2 * model.gca.process.undercut
            model.gca.gb = gb + 2 * model.gca.process.undercut
            model.gca.update_dependent_variables()
            u = [V, Fext]
            model.gca.x0 = model.gca.x0_release(u)
            u = setup_inputs(V=0, Fext=Fext)  # Changed for release
            try:
                sol = sim_gca(model, u, t_span, Fes_calc_method=Fes_calc_method, Fb_calc_method=Fb_calc_method)

                if len(sol.t_events[0]) > 0:
                    V_converged.append(V)
                    times_converged.append(sol.t_events[0][0] * 1e6)  # us conversion
            except Exception as e:
                print("Simulation failed for idy={}, V={}".format(idy, V), str(e))

            end_time = time.process_time()
            print("Runtime for L=", fingerL, ", V =", V, "=", end_time - start_time, ", undercut =", uc, "=",
                  end_time - start_time, '-->', {v: t for v, t in zip(V_converged, times_converged)})
        print(times_converged)

        line, = axs[idy // ny, idy % ny].plot(V_converged, times_converged, 'r')
        if idy == ny - 1:
            legend_release = line
        release_V_results.append(V_converged)
        release_t_results.append(times_converged)

        # Calculate the r2 score
        # actual = []
        # pred = []
        # for V in V_converged:
        #     if V in release_V[idy]:
        #         idx = np.where(release_V[idy] == V)[0][0]
        #         actual.append(release_avg[idy][idx])
        #         idx = np.where(V_converged == V)[0][0]
        #         pred.append(times_converged[idx])
        # r2 = r2_score(actual, pred)
        # print("Release Pred:", pred, "Actual:", actual)
        # ratios = [p/a for p, a in zip(pred, actual)]
        # print("Release Ratios:", np.max(ratios), np.min(ratios), ratios)
        # print("R2 score for L=", fingerL, "=", r2)
        # r2_scores_release.append(r2)
        # rmse = mean_squared_error(actual, pred, squared=False)
        # rmse_release.append(rmse)
        # print("RMSE score for L=", fingerL, "=", rmse)

    print("Finger L values:", [L * 1e6 for L in fingerL_values])

    # print("Pullin R2 scores:", r2_scores_pullin, np.mean(r2_scores_pullin), np.std(r2_scores_pullin))
    # print("Release R2 scores:", r2_scores_release, np.mean(r2_scores_release), np.std(r2_scores_release))
    # print("Pullin RMSE scores:", rmse_pullin, np.mean(rmse_pullin), np.std(rmse_pullin))
    # print("Release RMSE scores:", rmse_release, np.mean(rmse_release), np.std(rmse_release))

    # axs[0, ny-1].legend([legend_pullin, legend_release], ['Pull-in', 'Release'])
    # fig.legend([legend_pullin], ['Pull-in'], loc='lower right', ncol=2)
    # fig.legend([legend_release], ['Release'], loc='lower right', ncol=2)
    fig.legend([legend_pullin, legend_release], ['Pull-in', 'Release'], loc='lower right', ncol=2)

    plt.tight_layout()
    plt.savefig("../figures/" + timestamp + ".png")
    plt.savefig("../figures/" + timestamp + ".pdf")

    np.save('../data/' + timestamp + '.npy', np.array([model.process, V_values, fingerL_values, fingerLbuff_values,
                                                       fingerWtip_values, fingerWbase_values, gb_values,
                                                       pullin_V_results, pullin_t_results, release_V_results,
                                                       release_t_results,
                                                       fig], dtype=object),
            allow_pickle=True)

    total_end_time = time.process_time()
    print("Total Runtime:", total_end_time - total_start_time)
    plt.show()
