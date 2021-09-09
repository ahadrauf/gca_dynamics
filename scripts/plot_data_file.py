import numpy as np
import matplotlib.pyplot as plt
from process import *
from datetime import datetime


def setup_plot(len_x, len_y, plt_title=None):
    fig, axs = plt.subplots(len_x, len_y)
    if plt_title is not None:
        fig.suptitle(plt_title)
    return fig, axs


def plot_data(fig, axs, pullin_V, pullin_avg, pullin_std, release_V, release_avg, release_std, labels):
    nx, ny = 3, 3
    # nx, ny = 4, 2
    for idx in range(nx):
        for idy in range(ny):
            i = ny*idx + idy
            ax = axs[idx, idy]
            # ax.grid(True)
            ax.errorbar(pullin_V[i], pullin_avg[i], pullin_std[i], fmt='b.', capsize=3)
            ax.errorbar(release_V[i], release_avg[i], release_std[i], fmt='r.', capsize=3)
            ax.annotate(labels[i], xy=(1, 1), xycoords='axes fraction', fontsize=10,
                        xytext=(-2, -2), textcoords='offset points',
                        ha='right', va='top')


if __name__ == '__main__':
    now = datetime.now()
    Fes_calc_method, Fb_calc_method = 2, 2
    # name_clarifier = "_V_supportW_pullin_release_undercut=custom_Fes=v{}_Fb=v{}_modified_20210903_23_08_34_20210904_00_33_13".format(Fes_calc_method, Fb_calc_method)
    name_clarifier = "_V_supportW_pullin_release_undercut=custom_Fes=v{}_Fb=v{}_modified_20210908_23_12_51".format(Fes_calc_method, Fb_calc_method)
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    filename = "../data/20210908_23_12_51_V_fingerL_pullin_release_undercut=fixedtmax1000e6_Fes=v2_Fb=v2.npy"
    data = np.load(filename, allow_pickle=True)
    process, supportW_values, pullin_V, pullin_avg, pullin_std, release_V, release_avg, release_std, pullin_V_results, pullin_t_results, release_V_results, release_t_results, r2_scores_pullin, r2_scores_release, rmse_pullin, rmse_release, fig = data
    plt.close()

    # filename = "../data/20210904_00_33_13_V_fingerL_pullin_release_undercut=padded_uc_min_Fes=v2_Fb=v2.npy"
    # data = np.load(filename, allow_pickle=True)
    # _, _, _, _, _, _, _, _, pullin_V_results2, pullin_t_results2, release_V_results2, release_t_results2, r2_scores_pullin2, r2_scores_release2, rmse_pullin2, rmse_release2, _ = data
    # plt.close()
    #
    # for i in range(len(supportW_values)):
    #     pullin_V_results[i].extend(pullin_V_results2[i])
    #     pullin_t_results[i].extend(pullin_t_results2[i])
    #     release_V_results[i].extend(release_V_results2[i])
    #     release_t_results[i].extend(release_t_results2[i])
    #
    #     idx = np.argsort(pullin_V_results[i])
    #     pullin_V_results[i] = list(np.array(pullin_V_results[i])[idx])
    #     pullin_t_results[i] = list(np.array(pullin_t_results[i])[idx])
    #
    #     idx = np.argsort(release_V_results[i])
    #     release_V_results[i] = list(np.array(release_V_results[i])[idx])
    #     release_t_results[i] = list(np.array(release_t_results[i])[idx])

    # nx, ny = 4, 2
    nx, ny = 3, 3
    fig, axs = setup_plot(nx, ny)
    axs[2, 1].set_xlabel("Voltage (V)")
    axs[1, 0].set_ylabel(r"Time ($\mu$s)")
    # # add a big axis, hide frame
    # fig.add_subplot(111, frameon=False)
    # # hide tick and tick label of the big axis
    # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # plt.xlabel("Voltage (V)")
    # plt.ylabel("Time (us)")

    labels = []
    for i in range(1, len(supportW_values) + 1):
        labels.append(r"L=%0.1f$\mu$m"%(supportW_values[i - 1]*1e6))
        # labels.append(r"$w_{s}$=%0.1f$\mu$m"%(supportW_values[i - 1]*1e6))
    plot_data(fig, axs, pullin_V, pullin_avg, pullin_std, release_V, release_avg, release_std, labels)

    for idy in range(len(supportW_values)):
        V = pullin_V_results[idy]
        t = pullin_t_results[idy]
        sorted_itr = np.argsort(V)
        V = np.array(V)[sorted_itr]
        t = np.array(t)[sorted_itr]
        # if idy == 3:
        #     V, t = V[5:], t[5:]
        # if idy == 2 or idy == 4 or idy == 5:
        #     V, t = V[10:], t[10:]
        # if idy == 6:
        #     V, t = V[3:], t[3:]
        # if idy == 7 or idy == 8:
        #     V, t = V[20:], t[20:]
        line, = axs[idy//ny, idy%ny].plot(V, t, 'b')
        if idy == ny - 1:
            legend_pullin = line

        V = release_V_results[idy]
        t = release_t_results[idy]
        sorted_itr = np.argsort(V)
        V = np.array(V)[sorted_itr]
        t = np.array(t)[sorted_itr]
        line, = axs[idy//ny, idy%ny].plot(V, t, 'r')
        if idy == ny - 1:
            legend_release = line

    fig.legend([legend_pullin, legend_release], ['Pull-in', 'Release'], loc='lower right', ncol=2)

    print("R2 pullin", r2_scores_pullin, np.mean(r2_scores_pullin))
    print("R2 release", r2_scores_release, np.mean(r2_scores_release))

    plt.tight_layout()
    # plt.savefig("../figures/" + timestamp + ".png")
    # plt.savefig("../figures/" + timestamp + ".pdf")
    plt.show()
