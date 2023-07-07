import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from datetime import datetime

plt.rc('font', size=11)


def setup_plot(len_x, len_y, plt_title=None, x_label="", y_label=""):
    fig, axs = plt.subplots(len_x, len_y)
    if plt_title is not None:
        fig.suptitle(plt_title)
    return fig, axs


def plot_data(fig, axs, frequency, velocity_avg, velocity_std, velocity_fitted, V_labels, line_labels):
    nx, ny = 2, 2
    for idx in range(nx):
        for idy in range(ny):
            i = ny * idx + idy
            ax = axs[idx, idy]
            ax.errorbar(frequency[i], velocity_avg[i], velocity_std[i], color='tab:blue', marker='.',
                        capsize=3, zorder=1)
            # ax.plot(frequency[i], velocity_fitted[i], color='r', linewidth=2,
            #         zorder=2)  # plot the line above the errorbar
            ax.annotate(V_labels[i], xy=(0.035, 0.98), xycoords='axes fraction', fontsize=11,
                        xytext=(-2, -2), textcoords='offset points',
                        ha='left', va='top')
            # ax.annotate(line_labels[i], xy=(0.035, 0.8), xycoords='axes fraction', fontsize=11,
            #             xytext=(-2, -2), textcoords='offset points',
            #             ha='left', va='top', color='red')


if __name__ == "__main__":
    now = datetime.now()
    name_clarifier = "_inchworm_velocity_sim_vs_data_air"
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    Nsteps = 10
    Fext_shuttle = 0
    nx, ny = 2, 2
    # latexify(fig_width=6, columns=3)
    fig, axs = setup_plot(nx, ny)

    # Load data
    data = loadmat("../data/frequency_vs_velocity.mat")

    V_values = np.ndarray.flatten(data["V"])
    frequency = []
    velocity_avg = []
    velocity_std = []
    velocity_fitted = []
    V_labels = []
    line_labels = []
    r2_scores_pullin = []
    r2_scores_release = []
    rmse_pullin = []
    rmse_release = []
    for i in range(1, len(V_values) + 1):
        frequency.append(np.ndarray.flatten(data["f{}".format(i)]))
        velocity_avg.append(np.ndarray.flatten(data["t{}".format(i)]))
        velocity_std.append(np.ndarray.flatten(data["dt{}".format(i)]))
        velocity_fitted.append(np.ndarray.flatten(data["t{}_line".format(i)]))
        line_labels.append("Slope = \n" + data["label{}".format(i)][0])
        V_labels.append(str(V_values[i - 1]) + ' V')
        # labels.append(r"L=%0.1f$\mu$m"%(fingerL_values[i - 1]*1e6))

    plot_data(fig, axs, frequency, velocity_avg, velocity_std, velocity_fitted, V_labels, line_labels)

    data = np.load("../data/simulation_results")