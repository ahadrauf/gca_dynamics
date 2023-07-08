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

    file_name = "20230707_09_37_21_inchworm_velocity_sim_vs_data_air"
    file_name = "20230707_12_34_31_inchworm_velocity_sim_vs_data_air"
    file_name = "20230707_22_41_22_inchworm_velocity_sim_vs_data_air"
    data = np.load("../data/simulation_results/" + file_name + ".npy",
                   allow_pickle=True).item()
    for i in range(len(V_values)):
        V = V_values[i]
        frequencies = list(data[V].keys())
        data_to_plot = []
        for j in range(len(frequencies)):
            freq = frequencies[j]
            t_sim, x_sim, avg_vel = data[V][freq]
            print("V = {}, freq = {} kHz --> {}".format(V, freq, np.shape(x_sim)))

            N = np.size(t_sim)
            # if V == 50 or (V == 55 and freq < 35) or freq < 10:
            #     min_point = 0
            #     max_point = np.size(t_sim) * 8 // 15
            # elif freq < 15 or (V == 50):
            #     min_point = np.size(t_sim) * 1 // 15
            #     max_point = np.size(t_sim) * 9 // 15
            # else:
            #     min_point = np.size(t_sim) * 2 // 15
            #     max_point = np.size(t_sim) * 10 // 15
            min_point = 0
            max_point = N // 2
            avg_vel = (x_sim[max_point][4] - x_sim[min_point][4]) / (t_sim[max_point] - t_sim[min_point])
            data_to_plot.append(avg_vel)
        axs[i // ny][i % ny].plot(frequencies, data_to_plot, color='tab:orange',
                                  linestyle="--")  # convert to kHz

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Velocity (m/s)")

    plt.tight_layout()
    # plt.savefig("../figures/" + timestamp + ".png")
    # plt.savefig("../figures/" + timestamp + ".pdf")
    # np.save("../data/simulation_results/" + timestamp + ".npy", data)
    print({V: {f: datum[2] for f, datum in value.items()} for V, value in data.items()})
    print("Total runtime:", datetime.now() - now)
    plt.show()
