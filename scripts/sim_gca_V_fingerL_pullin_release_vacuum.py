"""
Setting up the simulation code for in-vacuum simulations, which we might do in the future! Who knows, we couldn't get
the vacuum chamber set up in time for this paper unfortunately because of COVID-19 delays. :(
Simulates the effect of varying finger overlap length and voltage on the pull-in for a GCA in a vaccum. The intention
was to model the dynamics when there's no damping term to mess up switch bounce, release behavior, and the like.
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
    model = AssemblyGCA(drawn_dimensions_filename="../layouts/fawn.csv", process=process)
    # model.gca.x0 = model.gca.x0_pullin()
    model.gca.terminate_simulation = model.gca.pulled_in
    # model.gca.Fescon = 1.1
    return model


def setup_model_release(V, Fext, process):
    u = [V, Fext]
    model = AssemblyGCA(drawn_dimensions_filename="../layouts/fawn.csv", process=process)
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

    sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True,
                    max_step=0.01e-6)  # , method="LSODA")
    return sol


def setup_plot(len_x, len_y, plt_title=None, x_label="", y_label=""):
    fig, axs = plt.subplots(len_x, len_y)
    if plt_title is not None:
        fig.suptitle(plt_title)

    # fig.text(0.5, 0.04, x_label, ha='center')
    # fig.text(0.04, 0.5, y_label, va='center', rotation='vertical')
    return fig, axs


def plot_data(fig, axs, pullin_V, pullin_avg, pullin_std, release_V, release_avg, release_std, labels):
    nx, ny = 3, 3
    for idx in range(nx):
        for idy in range(ny):
            # i = nx*idy + idx
            # ax = axs[idy, idx]
            i = ny * idx + idy
            # print(idx, idy, i)  # Confirms that I've ordered the data plots correctly
            ax = axs[idx, idy]
            ax.errorbar(pullin_V[i], pullin_avg[i], pullin_std[i], fmt='b.', capsize=3)
            ax.errorbar(release_V[i], release_avg[i], release_std[i], fmt='r.', capsize=3)
            ax.annotate(labels[i], xy=(1, 1), xycoords='axes fraction', fontsize=10,
                        xytext=(-2, -2), textcoords='offset points',
                        ha='right', va='top')


if __name__ == "__main__":
    pullin = True

    now = datetime.now()
    name_clarifier = "_vacuum_V_fingerL"
    name_clarifier += "_pullin" if pullin else "_release"
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    t_span = [0, 300e-6]
    Fext = 0
    nx, ny = 3, 3

    data = loadmat("../data/20180208_fawn_gca_V_fingerL_pullin_release.mat")
    fingerL_values = np.ndarray.flatten(data["LARR"]) * 1e-6  # Length = 9

    # latexify(fig_width=6, columns=3)
    fig, axs = setup_plot(nx, ny, x_label="Voltage (V)", y_label="Pull-in Time (us)")
    axs[2, 1].set_xlabel("Voltage (V)")
    axs[1, 0].set_ylabel("Time (us)")

    pullin_V = []
    pullin_avg = []
    pullin_std = []
    release_V = []
    release_avg = []
    release_std = []
    labels = []
    r2_scores_pullin = []
    r2_scores_release = []
    rmse_pullin = []
    rmse_release = []
    save_data_vacuum = {fingerL: () for fingerL in fingerL_values}
    save_data_air = {fingerL: () for fingerL in fingerL_values}
    for i in range(1, len(fingerL_values) + 1):
        pullin_V.append(np.ndarray.flatten(data["V{}_Arr1".format(i)]))
        pullin_avg.append(np.ndarray.flatten(data["tmeas{}_Arr1".format(i)]))
        pullin_std.append(np.ndarray.flatten(data["err{}_Arr1".format(i)]))
        release_V.append(np.ndarray.flatten(data["V{}_Arr1_r".format(i)]))
        release_avg.append(np.ndarray.flatten(data["tmeas{}_Arr1_r".format(i)]))
        release_std.append(np.ndarray.flatten(data["err{}_Arr1_r".format(i)]))
        labels.append(r"L=%0.1f$\mu$m" % (fingerL_values[i - 1] * 1e6))

    # plot_data(fig, axs, pullin_V, pullin_avg, pullin_std, release_V, release_avg, release_std, labels)
    for idx in range(nx):
        for idy in range(ny):
            i = ny * idx + idy
            ax = axs[idx, idy]
            ax.annotate(labels[i], xy=(1, 1), xycoords='axes fraction', fontsize=10,
                        xytext=(-2, -2), textcoords='offset points',
                        ha='right', va='top')

    legend_vacuum, legend_air = None, None

    # Vacuum
    for idy in range(len(fingerL_values)):
        fingerL = fingerL_values[idy]
        V_converged = []
        times_converged = []

        V_test = np.arange(20, 90.1, 0.5)
        # V_test = np.arange(20, 90.1, 0.25) if idy != 0 else np.arange(20, 90.05, 0.05)
        for V in V_test:
            start_time = time.process_time()
            if pullin:  # Pullin
                model = setup_model_pullin(process=SOIvacuum())
                model.gca.fingerL = fingerL - model.gca.process.undercut
                model.gca.update_dependent_variables()
                model.gca.x0 = model.gca.x0_pullin()
                u = setup_inputs(V=V, Fext=Fext)
            else:  # Release
                model = setup_model_release(V=V, Fext=Fext, process=SOIvacuum())
                model.gca.fingerL = fingerL - model.gca.process.undercut
                model.gca.update_dependent_variables()
                u = [V, Fext]
                model.gca.x0 = model.gca.x0_release(u)
                u = setup_inputs(V=0, Fext=Fext)  # Changed for release

            sol = sim_gca(model, u, t_span)

            if len(sol.t_events[0]) > 0:
                V_converged.append(V)
                times_converged.append(sol.t_events[0][0] * 1e6)  # us conversion

            end_time = time.process_time()
            print("Runtime for L=", fingerL, ", V=", V, "=", end_time - start_time, "->", times_converged)
        print(fingerL, V_converged, times_converged)

        line, = axs[idy // ny, idy % ny].plot(V_converged, times_converged)
        save_data_vacuum[fingerL] = (V_converged, times_converged)
        if idy == ny - 1:
            legend_vacuum = line

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
        # print("R2 score for L=", fingerL, "=", r2)
        # r2_scores_pullin.append(r2)
        # rmse = mean_squared_error(actual, pred, squared=False)
        # rmse_pullin.append(rmse)
        # print("RMSE score for L=", fingerL, "=", rmse)

    # Air
    for idy in range(len(fingerL_values)):
        fingerL = fingerL_values[idy]

        V_converged = []
        times_converged = []

        V_test = np.arange(20, 90.1, 0.5)
        for V in V_test:
            start_time = time.process_time()
            if pullin:  # Pullin
                model = setup_model_pullin(process=SOI())
                model.gca.fingerL = fingerL - model.gca.process.undercut
                model.gca.update_dependent_variables()
                model.gca.x0 = model.gca.x0_pullin()
                u = setup_inputs(V=V, Fext=Fext)
            else:  # Release
                model = setup_model_release(V=V, Fext=Fext, process=SOI())
                model.gca.fingerL = fingerL - model.gca.process.undercut
                model.gca.update_dependent_variables()
                u = [V, Fext]
                model.gca.x0 = model.gca.x0_release(u)
                u = setup_inputs(V=0, Fext=Fext)  # Changed for release

            sol = sim_gca(model, u, t_span)
            if len(sol.t_events[0]) > 0:
                V_converged.append(V)
                times_converged.append(sol.t_events[0][0] * 1e6)  # us conversion

            end_time = time.process_time()
            print("Runtime for L=", fingerL, ", V=", V, "=", end_time - start_time, "->", times_converged)
        print(times_converged)

        line, = axs[idy // ny, idy % ny].plot(V_converged, times_converged, 'r')
        save_data_air[fingerL] = (V_converged, times_converged)
        if idy == ny - 1:
            legend_air = line

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
        # print("R2 score for L=", fingerL, "=", r2)
        # r2_scores_release.append(r2)
        # rmse = mean_squared_error(actual, pred, squared=False)
        # rmse_release.append(rmse)
        # print("RMSE score for L=", fingerL, "=", rmse)

    # print("Finger L values:", [L*1e6 for L in fingerL_values])
    # print("Pullin R2 scores:", r2_scores_pullin, np.mean(r2_scores_pullin), np.std(r2_scores_pullin))
    # print("Release R2 scores:", r2_scores_release, np.mean(r2_scores_release), np.std(r2_scores_release))
    # print("Pullin RMSE scores:", rmse_pullin, np.mean(rmse_pullin), np.std(rmse_pullin))
    # print("Release RMSE scores:", rmse_release, np.mean(rmse_release), np.std(rmse_release))

    # axs[0, ny-1].legend([legend_vacuum, legend_air], ['Pull-in', 'Release'])
    if pullin:
        fig.legend([legend_vacuum, legend_air], ['Pull-in Time, Vacuum', 'Pull-in Time, Air'], loc='lower right',
                   ncol=2)
    else:
        fig.legend([legend_vacuum, legend_air], ['Release Time, Vacuum', 'Release Time, Air'], loc='lower right',
                   ncol=2)

    plt.tight_layout()
    plt.savefig("../figures/" + timestamp + ".png")
    plt.savefig("../figures/" + timestamp + ".pdf")
    np.save("../data/simulation_results/" + timestamp + ".npy", (save_data_vacuum, save_data_air))
    plt.show()
