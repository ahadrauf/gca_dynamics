from assembly import AssemblyGCA
from process import *
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from datetime import datetime
from utils import *
from sklearn.metrics import r2_score, mean_squared_error
import time



def setup_model_pullin():
    model = AssemblyGCA(drawn_dimensions_filename="../layouts/fawn.csv", process=SOI())
    model.gca.x0 = model.gca.x0_pullin()
    model.gca.terminate_simulation = model.gca.pulled_in
    return model


def setup_model_release(**kwargs):
    u = [kwargs["V"], kwargs["Fext"]]
    model = AssemblyGCA(drawn_dimensions_filename="../layouts/fawn.csv", process=SOI())
    # model.gca.k_support = 10.303975
    model.gca.terminate_simulation = model.gca.released
    return model


def setup_inputs(**kwargs):
    V = kwargs["V"]
    Fext = kwargs["Fext"]
    return lambda t, x: np.array([V, Fext])


def sim_gca(model, u, t_span, verbose=False, **kwargs):
    f = lambda t, x: model.dx_dt(t, x, u, verbose=verbose, **kwargs)
    x0 = model.x0()
    terminate_simulation = lambda t, x: model.terminate_simulation(t, x)
    terminate_simulation.terminal = True

    sol = solve_ivp(f, t_span, x0, events=terminate_simulation, dense_output=True, max_step=0.25e-6) #, method="LSODA")
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


if __name__ == "__main__":
    now = datetime.now()
    name_clarifier = "_V_supportW_release"
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    model = setup_model_pullin()
    t_span = [0, 400e-6]
    Fext = 0

    data = loadmat("../data/20180208_fawn_gca_V_fingerL_pullin_release.mat")
    # fingerL_values = np.ndarray.flatten(data["LARR"])*1e-6  # Length = 9
    supportW_values = np.array([2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5])*1e-6

    V_values = np.arange(20, 105, 5)
    # latexify(fig_width=6, columns=3)
    fig, axs = setup_plot(4, 2, x_label="Voltage (V)", y_label="Pull-in Time (us)")

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
    for i in range(1, len(supportW_values) + 1):
        pullin_V.append(np.ndarray.flatten(data["V{}_Arr2".format(i)]))
        pullin_avg.append(np.ndarray.flatten(data["tmeas{}_Arr2".format(i)]))
        pullin_std.append(np.ndarray.flatten(data["err{}_Arr2".format(i)]))
        release_V.append(np.ndarray.flatten(data["V{}_Arr2_r".format(i)]))
        release_avg.append(np.ndarray.flatten(data["tmeas{}_Arr2_r".format(i)]))
        release_std.append(np.ndarray.flatten(data["err{}_Arr2_r".format(i)]))
        labels.append(r"$w_{s}$=%0.1f$\mu$m"%(supportW_values[i - 1]*1e6))

    plot_data(fig, axs, pullin_V, pullin_avg, pullin_std, release_V, release_avg, release_std, labels)

    nx, ny = 4, 2
    legend_pullin, legend_release = None, None

    # Pullin measurements
    for idy in range(len(supportW_values)):
        supportW = supportW_values[idy]
        model.gca.supportW = supportW - 2*model.gca.process.overetch
        # if supportW < 3.5e-6:
        #     model.gca.supportW = supportW - 2*model.gca.process.small_overetch
        # else:
        #     model.gca.supportW = supportW - 2*model.gca.process.overetch
        model.gca.update_dependent_variables()

        V_converged = []
        times_converged = []

        # V_test = np.sort(np.append(V_values, [pullin_V[idy], pullin_V[idy]+0.2]))  # Add some extra values to test
        V_test = []
        V_values = pullin_V[idy]
        # V_test = list(np.arange(min(V_values), max(V_values) + 1, 1.))
        for V in V_values:
            # V_test.append(V - 0.1)
            V_test.append(V)
            # V_test.append(V + 0.2)
            # V_test.append(V + 0.5)
            # V_test.append(V + 1)
            # V_test.append(V + 1.5)
            # V_test.append(V + 2)
        for V in V_test:
            start_time = time.process_time()
            u = setup_inputs(V=V, Fext=Fext)
            sol = sim_gca(model, u, t_span, Fes_calc_method=2, Fb_calc_method=1)

            if len(sol.t_events[0]) > 0:
                V_converged.append(V)
                times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
            end_time = time.process_time()
            print("Runtime for L=", supportW, ", V=", V, "=", end_time - start_time)
        print(supportW, V_converged, times_converged)

        # axs[idy//ny, idy%ny].plot(V_converged, times_converged)
        line, = axs[idy//ny, idy%ny].plot(V_converged, times_converged)
        if idy == ny - 1:
            legend_pullin = line

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
        print("Pullin Pred:", pred, "Actual:", actual)
        print("R2 score for supportW=", supportW, "=", r2)
        r2_scores_pullin.append(r2)
        rmse = mean_squared_error(actual, pred, squared=False)
        rmse_pullin.append(rmse)
        print("RMSE score for supportW=", supportW, "=", rmse)

    # Release measurements
    for idy in range(len(supportW_values)):
        supportW = supportW_values[idy]

        V_converged = []
        times_converged = []

        V_values = release_V[idy]
        V_test = V_values
        # V_test = list(np.arange(min(V_values), max(V_values) + 1, 1.))
        for V in V_test:
            start_time = time.process_time()
            model = setup_model_release(V=V, Fext=Fext)
            model.gca.supportW = supportW - 2*model.gca.process.overetch
            # if supportW < 3.5e-6:
            #     model.gca.supportW = supportW - 2*model.gca.process.small_overetch
            # else:
            #     model.gca.supportW = supportW - 2*model.gca.process.overetch
            model.gca.update_dependent_variables()
            u = [V, Fext]
            model.gca.x0 = model.gca.x0_release(u)
            u = setup_inputs(V=0, Fext=Fext)  # Changed for release
            sol = sim_gca(model, u, t_span, Fes_calc_method=2, Fb_calc_method=1)

            if len(sol.t_events[0]) > 0:
                V_converged.append(V)
                times_converged.append(sol.t_events[0][0]*1e6)  # us conversion

            end_time = time.process_time()
            print("Runtime for L=", supportW, ", V=", V, "=", end_time - start_time)
        print(times_converged)
        # axs[idy//ny, idy%ny].plot(V_converged, times_converged, 'r')
        line, = axs[idy//ny, idy%ny].plot(V_converged, times_converged, 'r')
        if idy == ny - 1:
            legend_release = line

        # Calculate the r2 score
        actual = []
        pred = []
        for V in V_converged:
            if V in release_V[idy]:
                idx = np.where(release_V[idy] == V)[0][0]
                actual.append(release_avg[idy][idx])
                idx = np.where(V_converged == V)[0][0]
                pred.append(times_converged[idx])
        r2 = r2_score(actual, pred)
        print("R2 score for supportW=", supportW, "=", r2)
        r2_scores_release.append(r2)
        rmse = mean_squared_error(actual, pred, squared=False)
        rmse_release.append(rmse)
        print("RMSE score for supportW=", supportW, "=", rmse)

    print("Support W values:", supportW_values)
    print("Pullin R2 scores:", r2_scores_pullin, np.mean(r2_scores_pullin), np.std(r2_scores_pullin))
    print("Release R2 scores:", r2_scores_release, np.mean(r2_scores_release), np.std(r2_scores_release))
    print("Pullin RMSE scores:", rmse_pullin, np.mean(rmse_pullin), np.std(rmse_pullin))
    print("Release RMSE scores:", rmse_release, np.mean(rmse_release), np.std(rmse_release))

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Time (us)")
    fig.legend([legend_pullin, legend_release], ['Pull-in', 'Release'], loc='lower right', ncol=2)

    plt.tight_layout()
    plt.savefig("figures/" + timestamp + ".png")
    plt.savefig("figures/" + timestamp + ".pdf")
    plt.show()
