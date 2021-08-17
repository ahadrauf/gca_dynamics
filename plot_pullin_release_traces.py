import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

np.set_printoptions(precision=3, suppress=True)


def setup_plot(len_x, len_y, x_data, y_data, x_label="", y_label="", plt_title=None):
    fig, axs = plt.subplots(len_x, len_y)
    if plt_title is not None:
        fig.suptitle(plt_title)
    colors = ['b', 'r', 'g']
    for idx in range(len_x):
        for idy in range(len_y):
            i = len_y*idx + idy
            if len(np.shape(axs)) == 1:
                ax = axs[i]
            else:
                ax = axs[idx, idy]
            ax.plot(x_data[i], y_data[i], color=colors[i%len(colors)], linewidth=2)
            # ax.annotate(labels[i], xy=(1, 1), xycoords='axes fraction', fontsize=10,
            #             xytext=(-2, -2), textcoords='offset points',
            #             ha='right', va='top')

    if (len_x%2 == 1 or len_y%2 == 1) and (x_label != "" or y_label != ""):
        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
    elif x_label != "" or y_label != "":
        axs[len_x//2, len_y - 1].xlabel(x_label)
        axs[0, len_y//2].ylabel(y_label)

    return fig, axs

if __name__ == '__main__':
    data = loadmat("data/pullin_release_traces.mat")
    t_pullin = np.ndarray.flatten(data["t_pullin"])
    t_release = np.ndarray.flatten(data["t_release"])
    Vsense_pullin = np.ndarray.flatten(data["Vsense_pullin"])
    Vsense_release = np.ndarray.flatten(data["Vsense_release"])
    Vdrive_pullin = np.ndarray.flatten(data["Vdrive_pullin"])
    Vdrive_release = np.ndarray.flatten(data["Vdrive_release"])

    # print(np.where(0 <= t_pullin 1))
    t_pullin_range = [-10e-6, 80e-6]
    idx = np.where(np.all(t_pullin_range[0] <= t_pullin, t_pullin <= t_pullin_range[1]))
    t_pullin_plot = t_pullin[idx]
    Vdrive_pullin_plot = Vdrive_pullin[idx]
    Vsense_pullin_plot = Vsense_pullin[idx]
    fig, axs = setup_plot(2, 1, [t_pullin_plot, t_pullin_plot], [Vdrive_pullin_plot, Vsense_pullin_plot],
                          "Time (us)", "Voltage (V)", "Pull-in Scope Trace")

    plt.show()
