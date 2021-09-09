from assembly import AssemblyGCA
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from datetime import datetime
from process import *


def setup_model_pullin():
    model = AssemblyGCA(drawn_dimensions_filename="../layouts/fawn.csv", process=SOI())
    model.gca.x0 = model.gca.x0_pullin()
    model.gca.terminate_simulation = model.gca.pulled_in
    return model


def setup_model_release(**kwargs):
    u = [kwargs["V"], kwargs["Fext"]]
    model = AssemblyGCA(drawn_dimensions_filename="../layouts/fawn.csv", process=SOI())
    # model.gca.k_support = 10.303975
    if "Fescon" in kwargs:
        model.gca.Fescon = kwargs["Fescon"]
    if "Fkcon" in kwargs:
        model.gca.Fkcon = kwargs["Fkcon"]
    model.gca.x0 = model.gca.x0_release(u)
    model.gca.terminate_simulation = model.gca.released
    return model


def setup_inputs(**kwargs):
    V = kwargs["V"]
    Fext = kwargs["Fext"]
    return lambda t, x: np.array([V, Fext])


def sim_gca(model, u, t_span, verbose=False):
    f = lambda t, x: model.dx_dt(t, x, u, verbose=verbose)
    x0 = model.x0()
    terminate_simulation = lambda t, x: model.terminate_simulation(t, x)
    terminate_simulation.terminal = True

    sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True,
                    max_step=0.25e-6)  # , method="LSODA")
    return sol


def setup_plot(len_x, len_y, plt_title=None, x_label="", y_label=""):
    fig, axs = plt.subplots(len_x, len_y)
    if plt_title is not None:
        fig.suptitle(plt_title)

    if x_label or y_label:
        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
    return fig, axs


def display_stats(x_converged, times_converged, label):
    # Get helpful stats for paper
    print(label, x_converged, times_converged)
    time_50_up = times_converged[np.where(np.isclose(x_converged, 1.5))[0][0]]
    time_nominal = times_converged[np.where(np.isclose(x_converged, 1.))[0][0]]
    time_50_down = times_converged[np.where(np.isclose(x_converged, 0.5))[0][0]]
    print("{}: con=0.5: {} (Ratio: {}), con=1: {}, con=1.5: {} (Ratio: {})".format(
        label, time_50_down, time_nominal/time_50_down, time_nominal, time_50_up, time_nominal/time_50_up
    ))


if __name__ == "__main__":
    now = datetime.now()
    undercut = SOI().undercut
    Fes_calc_method, Fb_calc_method = 2, 2
    name_clarifier = "_vary_multipliers_undercut={:.3f}_Fes=v{}_Fb=v{}".format(undercut*1e6, Fes_calc_method, Fb_calc_method)
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    model = setup_model_pullin()
    t_span = [0, 300e-6]
    Fext = 0.

    V = 60
    mcon_range = np.arange(0.1, 2.1, 0.01)
    Fescon_range = np.arange(0.1, 2.1, 0.01)
    Fbcon_range = np.arange(0.1, 2.1, 0.01)
    Fkcon_range = np.arange(0.1, 2.1, 0.01)
    # latexify(fig_width=6, columns=3)
    fig, axs = setup_plot(2, 2, x_label="Scaling Variable", y_label="Time (us)")

    nx, ny = 2, 2

    ##### ax[0, 0] = Varying mass
    # Pullin
    print("Nominal mass:", model.gca.spineA*model.gca.process.t_SOI*model.gca.process.density)
    x_converged = []
    times_converged = []
    for mcon in mcon_range:
        model = setup_model_pullin()
        model.gca.mcon = mcon
        u = setup_inputs(V=V, Fext=Fext)
        sol = sim_gca(model, u, t_span)

        if len(sol.t_events[0]) > 0:
            x_converged.append(mcon)
            times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
    axs[0, 0].plot(x_converged, times_converged, 'b')
    display_stats(x_converged, times_converged, "Pullin mcon")

    # Release
    x_converged = []
    times_converged = []
    for mcon in mcon_range:
        model = setup_model_release(V=V, Fext=Fext)
        model.gca.mcon = mcon
        u = setup_inputs(V=0., Fext=Fext)
        sol = sim_gca(model, u, t_span)

        if len(sol.t_events[0]) > 0:
            x_converged.append(mcon)
            times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
    axs[0, 0].plot(x_converged, times_converged, 'r')
    # axs[0, 0].legend(["Pull-in", "Release"])
    axs[0, 0].set_title(r"Varying $m_{GCA}$", fontsize=12)
    axs[0, 0].axvline(1., color='k', linestyle='--')
    display_stats(x_converged, times_converged, "Release mcon")

    model = setup_model_pullin()
    m_nom = model.gca.spineA*model.gca.process.t_SOI*model.gca.process.density
    label = r"$m_{GCA}=$" + "{:0.1f}".format(m_nom*1e9) + r' $\mu$g'
    axs[0, 0].annotate(label, xy=(0.52, 0.96), xycoords='axes fraction', color='k',
                       xytext=(0, 0), textcoords='offset points', ha='left', va='top')

    ##### ax[0, 1] = Varying Fes
    # Pullin
    x_converged = []
    times_converged = []
    for Fescon in Fescon_range:
        model = setup_model_pullin()
        model.gca.Fescon = Fescon
        u = setup_inputs(V=V, Fext=Fext)
        sol = sim_gca(model, u, t_span)

        if len(sol.t_events[0]) > 0:
            x_converged.append(Fescon)
            times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
    axs[0, 1].plot(x_converged, times_converged, 'b')
    display_stats(x_converged, times_converged, "Pullin Fescon")

    # Release
    x_converged = []
    times_converged = []
    for Fescon in Fescon_range:
        model = setup_model_release(V=V, Fext=Fext, Fescon=Fescon)
        u = setup_inputs(V=0., Fext=Fext)
        sol = sim_gca(model, u, t_span)
        if len(sol.t_events[0]) > 0:
            x_converged.append(Fescon)
            times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
    axs[0, 1].plot(x_converged, times_converged, 'r')
    # axs[0, 1].legend(["Pull-in", "Release"])
    axs[0, 1].set_title(r"Varying $F_{es}$", fontsize=12)
    axs[0, 1].axvline(1., color='k', linestyle='--')
    display_stats(x_converged, times_converged, "Release Fescon")

    model = setup_model_pullin()
    label = r"$\epsilon_r$=1"
    axs[0, 1].annotate(label, xy=(0.52, 0.96), xycoords='axes fraction', color='k',
                       xytext=(0, 0), textcoords='offset points', ha='left', va='top')

    ##### ax[1, 0] = Varying Fb
    # Pullin
    x_converged = []
    times_converged = []
    for Fbcon in Fbcon_range:
        model = setup_model_pullin()
        model.gca.Fbcon = Fbcon
        u = setup_inputs(V=V, Fext=Fext)
        sol = sim_gca(model, u, t_span)

        if len(sol.t_events[0]) > 0:
            x_converged.append(Fbcon)
            times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
    axs[1, 0].plot(x_converged, times_converged, 'b')
    display_stats(x_converged, times_converged, "Pullin Fbkcon")

    # Release
    x_converged = []
    times_converged = []
    for Fbcon in Fbcon_range:
        model = setup_model_release(V=V, Fext=Fext)
        model.gca.Fbcon = Fbcon
        u = setup_inputs(V=0., Fext=Fext)
        sol = sim_gca(model, u, t_span)

        if len(sol.t_events[0]) > 0:
            x_converged.append(Fbcon)
            times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
    axs[1, 0].plot(x_converged, times_converged, 'r')
    # axs[1, 0].legend(["Pull-in", "Release"])
    axs[1, 0].set_title(r"Varying $F_b$", fontsize=12)
    axs[1, 0].axvline(1., color='k', linestyle='--')
    display_stats(x_converged, times_converged, "Release Fbcon")

    model = setup_model_pullin()
    label = r"$\mu$=18.5 $\mu$Pa$\cdot$s"
    axs[1, 0].annotate(label, xy=(0.45, 0.96), xycoords='axes fraction', color='k',
                       xytext=(0, 0), textcoords='offset points', ha='right', va='top')

    ##### ax[1, 1] = Varying Fk
    # Pullin
    x_converged = []
    times_converged = []
    for Fkcon in Fkcon_range:
        model = setup_model_pullin()
        model.gca.Fkcon = Fkcon
        u = setup_inputs(V=V, Fext=Fext)
        sol = sim_gca(model, u, t_span)

        if len(sol.t_events[0]) > 0:
            x_converged.append(Fkcon)
            times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
    l1 = axs[1, 1].plot(x_converged, times_converged, 'b')
    display_stats(x_converged, times_converged, "Pullin Fkcon")

    # Release
    x_converged = []
    times_converged = []
    for Fkcon in Fkcon_range:
        model = setup_model_release(V=V, Fext=Fext, Fkcon=Fkcon)
        # model.gca.Fkcon = Fkcon
        u = setup_inputs(V=0., Fext=Fext)
        sol = sim_gca(model, u, t_span)

        if len(sol.t_events[0]) > 0:
            x_converged.append(Fkcon)
            times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
    l2 = axs[1, 1].plot(x_converged, times_converged, 'r')
    axs[1, 1].legend(["Pull-in", "Release"])
    axs[1, 1].set_title(r"Varying $F_k$", fontsize=12)
    axs[1, 1].axvline(1., color='k', linestyle='--')
    display_stats(x_converged, times_converged, "Release Fkcon")

    model = setup_model_pullin()
    k_nom = model.gca.k_support
    label = r"$k=$" + "{:0.1f}".format(k_nom) + ' N/m'
    axs[1, 1].annotate(label, xy=(0.52, 0.6), xycoords='axes fraction', color='k',
                       xytext=(0, 0), textcoords='offset points', ha='left', va='top')

    plt.tight_layout()
    plt.savefig("../figures/" + timestamp + ".png")
    plt.savefig("../figures/" + timestamp + ".pdf")
    plt.show()
