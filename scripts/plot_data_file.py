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
    name_clarifier = "_V_fingerL_pullin_release_undercut=custom_Fes=v{}_Fb=v{}_modified_20210901_14_45_57".format(Fes_calc_method, Fb_calc_method)
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    filename = "../data/20210901_14_45_57_V_fingerL_pullin_release_undercut=0.300_Fes=v2_Fb=v2.npy"
    data = np.load(filename, allow_pickle=True)
    process, fingerL_values, pullin_V, pullin_avg, pullin_std, release_V, release_avg, release_std, pullin_V_results, pullin_t_results, release_V_results, release_t_results, r2_scores_pullin, r2_scores_release, rmse_pullin, rmse_release, fig = data
    # print(pullin_V_results)

    nx, ny = 3, 3
    plt.close()
    fig, axs = setup_plot(nx, ny)
    axs[2, 1].set_xlabel("Voltage (V)")
    axs[1, 0].set_ylabel(r"Time ($\mu$s)")

    labels = []
    for i in range(1, len(fingerL_values) + 1):
        labels.append(r"L=%0.1f$\mu$m"%(fingerL_values[i - 1]*1e6))
    plot_data(fig, axs, pullin_V, pullin_avg, pullin_std, release_V, release_avg, release_std, labels)

    for idy in range(len(fingerL_values)):
        V = pullin_V_results[idy]
        t = pullin_t_results[idy]
        if idy == 2 or idy == 4 or idy == 5 or idy == 7 or idy == 8:
            V, t = V[1:], t[1:]

        line, = axs[idy//ny, idy%ny].plot(V, t, 'b')
        if idy == ny - 1:
            legend_pullin = line
        V = release_V_results[idy]
        t = release_t_results[idy]
        line, = axs[idy//ny, idy%ny].plot(V, t, 'r')
        if idy == ny - 1:
            legend_release = line

    fig.legend([legend_pullin, legend_release], ['Pull-in', 'Release'], loc='lower right', ncol=2)

    plt.tight_layout()
    plt.savefig("../figures/" + timestamp + ".png")
    plt.savefig("../figures/" + timestamp + ".pdf")
    plt.show()
