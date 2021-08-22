import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from datetime import datetime

np.set_printoptions(precision=3, suppress=True)


def setup_plot(len_x, len_y, x_data, y_data, x_label="", y_label="", plt_title=None):
    fig, axs = plt.subplots(len_x, len_y)
    if plt_title is not None:
        fig.suptitle(plt_title)
    colors = ['b', 'r', 'r', 'g']
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
    now = datetime.now()
    name_clarifier = "_release_scope_trace"
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    data = loadmat("../data/pullin_release_traces.mat")
    t_pullin = np.ndarray.flatten(data["t_pullin"])
    t_release = np.ndarray.flatten(data["t_release"])
    Vsense_pullin = np.ndarray.flatten(data["Vsense_pullin"])
    Vsense_release = np.ndarray.flatten(data["Vsense_release"])
    Vdrive_pullin = np.ndarray.flatten(data["Vdrive_pullin"])
    Vdrive_release = np.ndarray.flatten(data["Vdrive_release"])

    ##################### Pullin-in Scope Trace #####################
    # t_pullin_range = [-10, 80]  # us
    # idx = np.where((t_pullin_range[0] <= t_pullin) & (t_pullin <= t_pullin_range[1]))
    # t_pullin_plot = t_pullin[idx]
    # Vdrive_pullin_plot = Vdrive_pullin[idx]
    # Vsense_pullin_plot = Vsense_pullin[idx]
    # fig, axs = setup_plot(2, 1, [t_pullin_plot, t_pullin_plot], [Vdrive_pullin_plot, Vsense_pullin_plot],
    #                       r"Time ($\mu$s)", "Voltage (V)", "Pull-in Scope Trace")
    # axs[0].annotate("Actuation Signal", xy=(0.98, 0.035), xycoords='axes fraction', fontsize=14, color='blue',
    #                 xytext=(-2, -2), textcoords='offset points', ha='right', va='bottom')
    # min_pos_t_idx = np.argmin(np.abs(t_pullin_plot))
    # axs[0].annotate("Pull-in Voltage", xy=(t_pullin_plot[min_pos_t_idx], Vdrive_pullin_plot[min_pos_t_idx]),
    #                 xytext=(20, 20), textcoords='offset points', ha='left', va='bottom', fontsize=14,
    #                 arrowprops=dict(arrowstyle='->'))
    #
    # # Labels for Sense Signal
    # axs[1].annotate("Sense Signal", xy=(0.035, 0.035), xycoords='axes fraction', fontsize=14, color='red',
    #                 xytext=(-2, -2), textcoords='offset points', ha='left', va='bottom')
    # axs[1].annotate("Coupling Rise", xy=(t_pullin_plot[min_pos_t_idx], Vsense_pullin_plot[min_pos_t_idx]),
    #                 xytext=(10, -20), textcoords='offset points', ha='left', va='top', fontsize=14,
    #                 arrowprops=dict(arrowstyle='->'))
    # max_sense = np.max(Vsense_pullin_plot)
    # signal_drop_idx = np.where(Vsense_pullin_plot <= 0.9*max_sense)[0][0]
    # print(t_pullin_plot[signal_drop_idx], Vsense_pullin_plot[signal_drop_idx])
    # axs[1].annotate("Signal Drop", xy=(t_pullin_plot[signal_drop_idx], Vsense_pullin_plot[signal_drop_idx]),
    #                 xytext=(-20, -30), textcoords='offset points', ha='right', va='top', fontsize=14,
    #                 arrowprops=dict(arrowstyle='->'))
    # idx_range_switch_bounce = np.where((65 <= t_pullin_plot) & (t_pullin_plot <= 70))
    # idx_switch_bounce = np.argmax(Vsense_pullin_plot[idx_range_switch_bounce]) + idx_range_switch_bounce[0][0]
    # axs[1].annotate("Switch\nBounce", xy=(t_pullin_plot[idx_switch_bounce], Vsense_pullin_plot[idx_switch_bounce]),
    #                 xytext=(5, 20), textcoords='offset points', ha='left', va='bottom', fontsize=14,
    #                 arrowprops=dict(arrowstyle='->'))
    #
    # # Pull-in Time Indicator in the Middle
    # # axs[0].text(0.5, -0.12, "Pull-in Time", size=14, ha="center", va="top", transform=axs[0].transAxes, color='green')
    # x_frac_min_pos_t = (0.0 - t_pullin_plot[0])/(t_pullin_plot[-1] - t_pullin_plot[0])
    # x_frac_signal_drop = (t_pullin_plot[signal_drop_idx] - t_pullin_plot[0])/(t_pullin_plot[-1] - t_pullin_plot[0])
    # print(x_frac_min_pos_t, x_frac_signal_drop)
    # axs[0].annotate("Pull-in Time", xy=(0.7539, -0.3), xycoords='axes fraction',
    #                 xytext=(-73, 0), textcoords='offset points', ha='right', va='center', fontsize=14,
    #                 arrowprops=dict(arrowstyle='->', color='green'), color='green')
    # axs[0].annotate("", xy=(0.1460, -0.3), xycoords='axes fraction',
    #                 xytext=(73, 0), textcoords='offset points', ha='left', va='center', fontsize=14,
    #                 arrowprops=dict(arrowstyle='->', color='green'), color='green')

    ##################### Release Scope Trace #####################
    t_release_range = [-10, 30]  # us
    t_release_zoomed_out_range = [-220*(10/30), 220]
    idx = np.where((t_release_range[0] <= t_release) & (t_release <= t_release_range[1]))
    idx_zoomed_out = np.where((t_release_zoomed_out_range[0] <= t_release) & (t_release <= t_release_zoomed_out_range[1]))
    t_release_plot = t_release[idx]
    Vdrive_release_plot = Vdrive_release[idx]
    Vsense_release_plot = Vsense_release[idx]
    t_release_zoomed_out_plot = t_release[idx_zoomed_out]
    Vsense_release_zoomed_out_plot = Vsense_release[idx_zoomed_out]
    fig, axs = setup_plot(3, 1, [t_release_plot, t_release_plot, t_release_zoomed_out_plot],
                          [Vdrive_release_plot, Vsense_release_plot, Vsense_release_zoomed_out_plot],
                          r"Time ($\mu$s)", "Voltage (V)", "Release Scope Trace")
    axs[0].annotate("Actuation Signal", xy=(0.98, 0.98), xycoords='axes fraction', fontsize=14, color='blue',
                    xytext=(-2, -2), textcoords='offset points', ha='right', va='top')
    min_pos_t_idx = np.argmin(np.abs(t_release_plot))
    axs[0].annotate("Release Voltage", xy=(t_release_plot[min_pos_t_idx], Vdrive_release_plot[min_pos_t_idx]),
                    xytext=(30, 0), textcoords='offset points', ha='left', va='center', fontsize=14,
                    arrowprops=dict(arrowstyle='->'))
    print("Min t", min_pos_t_idx, t_release_plot[min_pos_t_idx], Vdrive_release_plot[min_pos_t_idx])

    # Labels for Sense Signal
    axs[1].annotate("Sense Signal", xy=(0.035, 0.035), xycoords='axes fraction', fontsize=14, color='red',
                    xytext=(-2, -2), textcoords='offset points', ha='left', va='bottom')
    axs[1].annotate("Ringing", xy=(t_release_plot[min_pos_t_idx], Vsense_release_plot[min_pos_t_idx]),
                    xytext=(-20, -20), textcoords='offset points', ha='center', va='top', fontsize=14,
                    arrowprops=dict(arrowstyle='->'))
    max_sense = Vsense_release_plot[0]  # np.max(Vsense_release_plot)
    signal_drop_idx = np.where(Vsense_release_plot <= 0.9*max_sense)[0][2]
    print(t_release_plot[signal_drop_idx], Vsense_release_plot[signal_drop_idx])
    axs[1].annotate("Signal Drop", xy=(t_release_plot[signal_drop_idx], Vsense_release_plot[signal_drop_idx]),
                    xytext=(-20, -30), textcoords='offset points', ha='right', va='top', fontsize=14,
                    arrowprops=dict(arrowstyle='->'))

    # Release Time Indicator in the Middle
    x_frac_min_pos_t = (0.0 - axs[0].get_xlim()[0])/(axs[0].get_xlim()[1] - axs[0].get_xlim()[0])
    x_frac_signal_drop = (t_release_plot[signal_drop_idx] - axs[0].get_xlim()[0])/(axs[0].get_xlim()[1] - axs[0].get_xlim()[0])
    axs[0].annotate("Release Time", xy=(x_frac_signal_drop, -0.3), xycoords='axes fraction',
                    xytext=(-47, 0), textcoords='offset points', ha='right', va='center', fontsize=14,
                    arrowprops=dict(arrowstyle='->', color='green'), color='green')
    axs[0].annotate("", xy=(x_frac_min_pos_t, -0.3), xycoords='axes fraction',
                    xytext=(47, 0), textcoords='offset points', ha='left', va='center', fontsize=14,
                    arrowprops=dict(arrowstyle='->', color='green'), color='green')

    # Zoomed out plot
    axs[2].annotate("Sense Signal\n(Zoomed Out)", xy=(0.035, 0.035), xycoords='axes fraction', fontsize=14, color='red',
                    xytext=(-2, -2), textcoords='offset points', ha='left', va='bottom')
    idx_switch_bounce = np.where((30 <= t_release) & (t_release <= 60))
    first_switch_bounce = np.argmax(Vsense_release[idx_switch_bounce]) + np.min(idx_switch_bounce)
    print(first_switch_bounce, t_release[first_switch_bounce], Vsense_release[first_switch_bounce])
    axs[2].annotate("Switch Bounce", xy=(t_release[first_switch_bounce], Vsense_release[first_switch_bounce]),
                    xytext=(10, 30), textcoords='offset points', ha='left', va='center', fontsize=14,
                    arrowprops=dict(arrowstyle='->'))

    ##################### Release Scope Trace (Zoomed Out) #####################

    fig.set_figheight(7.3)
    fig.tight_layout()
    plt.savefig("figures/" + timestamp + ".png")
    plt.savefig("figures/" + timestamp + ".pdf")
    plt.show()
