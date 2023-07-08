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
colors = list(mcolors.TABLEAU_COLORS.keys())
markers = ['.', '^', 's', 'D', '<']


def setup_plot(len_x, len_y, plt_title=None, x_label="", y_label=""):
    fig, axs = plt.subplots(len_x, len_y)
    if plt_title is not None:
        fig.suptitle(plt_title)
    return fig, axs


if __name__ == "__main__":
    now = datetime.now()
    name_clarifier = "_inchworm_velocity_sim_vs_data_vacuum"
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    Nsteps = 20
    Fext_shuttle = 0
    nx, ny = 2, 2
    # latexify(fig_width=6, columns=3)
    fig, ax = plt.subplots(1, 1)

    V_values = [50, 55, 60, 65]
    frequencies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
                   2, 3, 4, 5, 6, 7, 8, 9, 10]  # [1, 5, 10, 15, 20, 25, 30, 35, 40]

    data = {V: {} for V in V_values}
    for i in range(len(V_values)):
        V = V_values[i]
        vel_sim_all = []

        frequencies_to_plot = []
        for j in range(len(frequencies)):
            drive_freq = frequencies[j] * 1e3

            # print('start sim')
            t_sim, x_sim, F_shuttle_all, step_counts = sim_inchworm(Nsteps=Nsteps, V=V, drive_freq=drive_freq,
                                                                    Fext_shuttle=0.,
                                                                    drawn_dimensions_filename="../layouts/fawn_velocity.csv",
                                                                    process=SOIvacuum())
            # print('end sim')
            vel = (x_sim[-1][4] - x_sim[0][4]) / (t_sim[-1] - t_sim[0])
            print("Avg. speed (m/s):", vel, "Avg. step size:", x_sim[-1][4] / Nsteps)
            frequencies_to_plot.append(frequencies[j])
            vel_sim_all.append(vel)
            data[V][frequencies[j]] = (t_sim, x_sim, F_shuttle_all, step_counts, vel)

        ax.plot(frequencies_to_plot, vel_sim_all, color=colors[i], marker=markers[i],
                label="V = {}".format(V))  # convert to kHz
        # axs[i // ny][i % ny].set_yticks(np.arange(0, np.max(velocity_avg[i]) + 0.1, 0.1))

    # add a big axis, hide frame
    # fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("../figures/" + timestamp + ".png")
    plt.savefig("../figures/" + timestamp + ".pdf")
    np.save("../data/simulation_results/" + timestamp + ".npy", data)
    print({V: {f: datum[2] for f, datum in value.items()} for V, value in data.items()})
    print("Total runtime:", datetime.now() - now)
    plt.show()
