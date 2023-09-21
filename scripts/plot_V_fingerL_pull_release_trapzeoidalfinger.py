import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def setup_plot(len_x, len_y, plt_title=None, x_label="", y_label=""):
    fig, axs = plt.subplots(len_x, len_y)
    if plt_title is not None:
        fig.suptitle(plt_title)

    # fig.text(0.5, 0.04, x_label, ha='center')
    # fig.text(0.04, 0.5, y_label, va='center', rotation='vertical')
    return fig, axs


def plot_data(fig, axs, labels):
    nx, ny = 3, 3
    for idx in range(nx):
        for idy in range(ny):
            i = ny * idx + idy
            ax = axs[idx, idy]



if __name__ == '__main__':
    pullin = False

    now = datetime.now()
    fileloc_air = "../data/20230911_23_59_15_V_fingerL_pullin_release_undercut=fixedtmax1000e6_Fes=vtrap_Fb=v2.npy"
    fileloc_vacuum = "../data/20230912_08_42_13_vacuum_V_fingerL_pullin_release_undercut=fixedtmax1000e6_Fes=vtrap_Fb=v2.npy"
    date_air = fileloc_air[8:25]
    date_vacuum = fileloc_vacuum[8:25]
    name_clarifier = "_V_fingerL_{}_air={}+vac={}".format("pullin" if pullin else "release", date_air, date_vacuum)
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    fig, axs = setup_plot(3, 3)
    axs[2, 1].set_xlabel("Voltage (V)")
    axs[1, 0].set_ylabel(r"Time ($\mu$s)")

    process, V_values, fingerL_values, fingerLbuff_values, fingerWtip_values, fingerWbase_values, gb_values, \
        pullin_V_results, pullin_t_results, release_V_results, release_t_results, _ = np.load(
        fileloc_vacuum, allow_pickle=True)
    labels = [r"$w_{f,base}$" + "={:.2f}".format(fingerWbase * 1e6) + r"$\mu$m" + "\n" + \
              r"$L_{ol}$" + "={:.1f}".format(fingerL * 1e6 - fingerLbuff * 1e6) + r"$\mu$m" for
              fingerL, fingerLbuff, fingerWbase in zip(fingerL_values, fingerLbuff_values, fingerWbase_values)]

    nx, ny = 3, 3
    for i in range(len(fingerL_values)):
        fingerL = fingerL_values[i]
        ax = axs[i // ny, i % ny]
        if pullin:
            line_vacuum, = ax.plot(pullin_V_results[i], pullin_t_results[i], 'tab:blue')
        else:
            line_vacuum, = ax.plot(release_V_results[i], release_t_results[i], 'tab:blue')

        ax.annotate(labels[i], xy=(1, 1), xycoords='axes fraction', fontsize=10,
                    xytext=(-2, -2), textcoords='offset points',
                    ha='right', va='top')

    axs[2, 1].set_xlabel("Voltage (V)", )
    axs[1, 0].set_ylabel(r"Time ($\mu$s)")

    process, V_values, fingerL_values, fingerLbuff_values, fingerWtip_values, fingerWbase_values, gb_values, \
        pullin_V_results, pullin_t_results, release_V_results, release_t_results, _ = np.load(
        fileloc_air, allow_pickle=True)
    for i in range(len(fingerL_values)):
        fingerL = fingerL_values[i]
        ax = axs[i // ny, i % ny]
        if pullin:
            line_air, = ax.plot(pullin_V_results[i], pullin_t_results[i], 'tab:orange')
        else:
            line_air, = ax.plot(release_V_results[i], release_t_results[i], 'tab:orange')

    if pullin:
        fig.legend([line_vacuum], ['Pull-in Time, Vacuum'], loc='lower left')
        fig.legend([line_air], ['Pull-in Time, Air'], loc='lower right')
    else:
        fig.legend([line_vacuum], ['Release Time, Vacuum'], loc='lower left')
        fig.legend([line_air], ['Release Time, Air'], loc='lower right')

    fig.tight_layout()
    [plt.close(f) for f in plt.get_fignums() if f != fig.number]  # Close all figures except the main one
    # plt.savefig("../figures/" + timestamp + ".png")
    # plt.savefig("../figures/" + timestamp + ".svg")
    # plt.savefig("../figures/" + timestamp + ".pdf")
    plt.show()
