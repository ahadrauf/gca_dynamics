"""
Comparing GCA velocity data vs. heuristic approximation

Uncomment the pawl's spring force calculation in gca.py/GCA/Fk() to model the pawl's spring force.
"""

import os

file_location = os.path.abspath(os.path.dirname(__file__))
dir_location = os.path.abspath(os.path.join(file_location, '..'))
import sys

sys.path.append(file_location)
sys.path.append(dir_location)

from sim_inchworm_transient import *
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.io import loadmat, savemat
from datetime import datetime

plt.rc('font', size=11)
colors = mcolors.TABLEAU_COLORS.keys()


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

    Nsteps = 20
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

    for i in range(len(V_values)):
        slopes = []
        freq = frequency[i]
        velocity = velocity_avg[i]
        for j in range(1, len(freq)):
            slopes.append((velocity[j] - velocity[0]) / (freq[j] - freq[0]) * 1e3)
        print(V_values[i], freq, velocity)
        # print(V_values[i], ['{:.2f}'.format(s) for s in slopes])

    data = {V: {} for V in V_values}
    for i in range(len(V_values)):
        V = V_values[i]
        vel_sim_all = []

        frequencies_to_plot = []
        for j in range(len(frequency[i])):
            # if j % 3 == 1 or j % 3 == 2:
            #     continue
            frequencies_to_plot.append(frequency[i][j])
            drive_freq = frequency[i][j] * 1e3

            t_sim, x_sim, F_shuttle_all, step_counts = sim_inchworm(Nsteps=Nsteps, V=V, drive_freq=drive_freq,
                                                                    Fext_shuttle=0.,
                                                                    drawn_dimensions_filename="../layouts/fawn_velocity.csv",
                                                                    process=SOI())
            # midway_point = np.size(t_sim) // 2
            midway_point = 0
            vel = (x_sim[-1][4] - x_sim[midway_point][4]) / (t_sim[-1] - t_sim[midway_point])
            print("Avg. speed (m/s):", vel, "Avg. step size:", x_sim[-1][4] / Nsteps)
            vel_sim_all.append(vel)
            data[V][frequency[i][j]] = (t_sim, x_sim, F_shuttle_all, step_counts, vel)

        axs[i // ny][i % ny].plot(frequencies_to_plot, vel_sim_all, color='tab:orange',
                                  linestyle="--")  # convert to kHz
        # axs[i // ny][i % ny].set_yticks(np.arange(0, np.max(velocity_avg[i]) + 0.1, 0.1))

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Velocity (m/s)")

    plt.tight_layout()
    plt.savefig("../figures/" + timestamp + ".png")
    plt.savefig("../figures/" + timestamp + ".pdf")
    np.save("../data/simulation_results/" + timestamp + ".npy", data)
    print({V: {f: datum[2] for f, datum in value.items()} for V, value in data.items()})
    print("Total runtime:", datetime.now() - now)
    plt.show()
