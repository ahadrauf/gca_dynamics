import sys
sys.path.append(r"C:\Users\ahadrauf\Desktop\Research\Pister\gca_dynamics")

from assembly import AssemblyGCA
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from datetime import datetime
from process import *
import time


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
                    max_step=0.005e-6)  # , method="LSODA")
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
    # time_50_up = times_converged[np.where(np.isclose(x_converged, 1.2))[0][0]]
    # time_nominal = times_converged[np.where(np.isclose(x_converged, 1.))[0][0]]
    # time_50_down = times_converged[np.where(np.isclose(x_converged, 0.8))[0][0]]
    # print("{}: con=0.5: {} (Ratio: {}), con=1: {}, con=1.5: {} (Ratio: {})".format(
    #     label, time_50_down, 1 - time_50_down/time_nominal, time_nominal, time_50_up, 1 - time_50_up/time_nominal
    # ))


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
    fingerLnom = 76.472e-6
    fingerWnom = 5.005e-6
    supportWnom = 3e-6
    gfnom = 4.83e-6
    fingerLcon_range = np.arange(0.1, 1.6, 0.025)
    fingerWcon_range = np.arange(0.2, 2.0, 0.025)
    supportWcon_range = np.arange(0.3, 2.0, 0.025)
    gfcon_range = np.arange(0.1, 1.6, 0.025)
    # latexify(fig_width=6, columns=3)
    # fig, axs = setup_plot(2, 2, x_label="Scaling Variable", y_label="Time (us)")
    fig, axs = setup_plot(2, 2, y_label="Time (us)")

    nx, ny = 2, 2
    data = {'fingerLpullin': None,
            'fingerLrelease': None,
            'fingerWpullin': None,
            'fingerWrelease': None,
            'supportWpullin': None,
            'supportWrelease': None,
            'gfpullin': None,
            'gfrelease': None}

    ##### ax[0, 0] = Varying fingerL
    # Pullin
    # print("Nominal mass:", model.gca.spineA*model.gca.process.t_SOI*model.gca.process.density)
    x_converged = []
    times_converged = []
    for con in fingerLcon_range:
        start_time = time.process_time()
        model = setup_model_pullin()
        model.gca.fingerL = fingerLnom*con - model.gca.process.undercut
        model.gca.update_dependent_variables()
        u = setup_inputs(V=V, Fext=Fext)
        sol = sim_gca(model, u, t_span)

        if len(sol.t_events[0]) > 0:
            # # x_converged.append(con)
            x_converged.append(fingerLnom*con*1e6)
            times_converged.append(sol.t_events[0][0]*1e6)  # us conversion

        end_time = time.process_time()
        print("Runtime for pullin fingerL con =", con, ", V =", V, "=", end_time - start_time, "=",
              end_time - start_time, '-->', {v: "{:0.2f}".format(t) for v, t in zip(x_converged, times_converged)})
    axs[0, 0].plot(x_converged, times_converged, 'b')
    display_stats(x_converged, times_converged, "Pullin fingerL")
    data['fingerLpullin'] = (x_converged, times_converged)
    np.save('../data/' + timestamp + '.npy', np.array([model.process, fingerLcon_range, fingerWcon_range,
                                                       gfcon_range, supportWcon_range, fingerLnom, fingerWnom, gfnom,
                                                       supportWnom, data, fig], dtype=object), allow_pickle=True)

    # Release
    x_converged = []
    times_converged = []
    for con in fingerLcon_range:
        start_time = time.process_time()
        model = setup_model_release(V=V, Fext=Fext)
        model.gca.fingerL = fingerLnom*con - model.gca.process.undercut
        model.gca.update_dependent_variables()
        u = [V, Fext]
        model.gca.x0 = model.gca.x0_release(u)
        u = setup_inputs(V=0., Fext=Fext)
        sol = sim_gca(model, u, t_span)

        if len(sol.t_events[0]) > 0:
            # x_converged.append(con)
            x_converged.append(fingerLnom*con*1e6)
            times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
        end_time = time.process_time()
        print("Runtime for release fingerLcon =", con, ", V =", V, "=", end_time - start_time, "=",
              end_time - start_time, '-->', {v: "{:0.2f}".format(t) for v, t in zip(x_converged, times_converged)})
    axs[0, 0].plot(x_converged, times_converged, 'r')
    # axs[0, 0].legend(["Pull-in", "Release"])
    axs[0, 0].set_title(r"Varying $L_{ol}$", fontsize=12)
    axs[0, 0].axvline(fingerLnom*1e6, color='k', linestyle='--')
    display_stats(x_converged, times_converged, "Release fingerL")
    data['fingerLrelease'] = (x_converged, times_converged)
    np.save('../data/' + timestamp + '.npy', np.array([model.process, fingerLcon_range, fingerWcon_range,
                                                       gfcon_range, supportWcon_range, fingerLnom, fingerWnom, gfnom,
                                                       supportWnom, data, fig], dtype=object), allow_pickle=True)

    model = setup_model_pullin()
    m_nom = model.gca.spineA*model.gca.process.t_SOI*model.gca.process.density
    label = r"$L_{ol}=$" + "{:0.1f}".format(fingerLnom*1e6) + r' $\mu$m'
    axs[0, 0].annotate(label, xy=(0.52, 0.96), xycoords='axes fraction', color='k',
                       xytext=(0, 0), textcoords='offset points', ha='left', va='top')
    axs[0, 0].set_xlabel(r'$L_{ol} [\mu$m]')

    ##### ax[0, 1] = Varying fingerW
    # Pullin
    x_converged = []
    times_converged = []
    for con in fingerWcon_range:
        start_time = time.process_time()
        model = setup_model_pullin()
        model.gca.fingerW = fingerWnom*con - 2*model.gca.process.undercut
        model.gca.update_dependent_variables()
        u = setup_inputs(V=V, Fext=Fext)
        sol = sim_gca(model, u, t_span)

        if len(sol.t_events[0]) > 0:
            # x_converged.append(con)
            x_converged.append(fingerWnom*con*1e6)
            times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
        end_time = time.process_time()
        print("Runtime for pullin fingerWcon =", con, ", V =", V, "=", end_time - start_time, "=",
              end_time - start_time, '-->', {v: "{:0.2f}".format(t) for v, t in zip(x_converged, times_converged)})
    axs[0, 1].plot(x_converged, times_converged, 'b')
    display_stats(x_converged, times_converged, "Pullin fingerW")
    data['fingerWpullin'] = (x_converged, times_converged)
    np.save('../data/' + timestamp + '.npy', np.array([model.process, fingerLcon_range, fingerWcon_range,
                                                       gfcon_range, supportWcon_range, fingerLnom, fingerWnom, gfnom,
                                                       supportWnom, data, fig], dtype=object), allow_pickle=True)

    # Release
    x_converged = []
    times_converged = []
    for con in fingerWcon_range:
        start_time = time.process_time()
        model = setup_model_release(V=V, Fext=Fext)
        model.gca.fingerW = fingerWnom*con - 2*model.gca.process.undercut
        model.gca.update_dependent_variables()
        u = [V, Fext]
        model.gca.x0 = model.gca.x0_release(u)
        u = setup_inputs(V=0., Fext=Fext)
        sol = sim_gca(model, u, t_span)
        if len(sol.t_events[0]) > 0:
            # x_converged.append(con)
            x_converged.append(fingerWnom*con*1e6)
            times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
        end_time = time.process_time()
        print("Runtime for release fingerWcon =", con, ", V =", V, "=", end_time - start_time, "=",
              end_time - start_time, '-->', {v: "{:0.2f}".format(t) for v, t in zip(x_converged, times_converged)})
    axs[0, 1].plot(x_converged, times_converged, 'r')
    # axs[0, 1].legend(["Pull-in", "Release"])
    axs[0, 1].set_title(r"Varying $w_f$", fontsize=12)
    axs[0, 1].axvline(fingerWnom*1e6, color='k', linestyle='--')
    display_stats(x_converged, times_converged, "Release fingerW")
    data['fingerWrelease'] = (x_converged, times_converged)
    np.save('../data/' + timestamp + '.npy', np.array([model.process, fingerLcon_range, fingerWcon_range,
                                                       gfcon_range, supportWcon_range, fingerLnom, fingerWnom, gfnom,
                                                       supportWnom, data, fig], dtype=object), allow_pickle=True)

    model = setup_model_pullin()
    label = r"$w_f=$" + "{:0.1f}".format(fingerWnom*1e6) + r' $\mu$m'
    axs[0, 1].annotate(label, xy=(0.52, 0.96), xycoords='axes fraction', color='k',
                       xytext=(0, 0), textcoords='offset points', ha='left', va='top')
    axs[0, 1].set_xlabel(r'$w_f [\mu$m]')

    ##### ax[1, 0] = Varying gf
    # Pullin
    x_converged = []
    times_converged = []
    for con in supportWcon_range:
        start_time = time.process_time()
        model = setup_model_pullin()
        model.gca.gf = gfnom*con + 2*model.gca.process.undercut
        model.gca.x_GCA = model.gca.gf - 1e-6
        model.gca.update_dependent_variables()
        u = setup_inputs(V=V, Fext=Fext)
        sol = sim_gca(model, u, t_span)

        if len(sol.t_events[0]) > 0:
            # x_converged.append(con)
            x_converged.append(gfnom*con*1e6)
            times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
        end_time = time.process_time()
        print("Runtime for pullin gfcon =", con, ", V =", V, "=", end_time - start_time, "=",
              end_time - start_time, '-->', {v: "{:0.2f}".format(t) for v, t in zip(x_converged, times_converged)})
    axs[1, 0].plot(x_converged, times_converged, 'b')
    display_stats(x_converged, times_converged, "Pullin gfcon")
    data['gfpullin'] = (x_converged, times_converged)
    np.save('../data/' + timestamp + '.npy', np.array([model.process, fingerLcon_range, fingerWcon_range,
                                                       gfcon_range, supportWcon_range, fingerLnom, fingerWnom, gfnom,
                                                       supportWnom, data, fig], dtype=object), allow_pickle=True)

    # Release
    x_converged = []
    times_converged = []
    for con in supportWcon_range:
        model = setup_model_release(V=V, Fext=Fext)
        model.gca.gf = gfnom*con + 2*model.gca.process.undercut
        model.gca.x_GCA = model.gca.gf - 1e-6
        print(gfnom, con, model.gca.gf, model.gca.x_GCA)
        model.gca.update_dependent_variables()
        u = [V, Fext]
        model.gca.x0 = model.gca.x0_release(u)
        u = setup_inputs(V=0., Fext=Fext)
        sol = sim_gca(model, u, t_span)

        if len(sol.t_events[0]) > 0:
            start_time = time.process_time()
            # x_converged.append(con)
            x_converged.append(gfnom*con*1e6)
            times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
        end_time = time.process_time()
        print("Runtime for release gfcon =", con, ", V =", V, "=", end_time - start_time, "=",
              end_time - start_time, '-->', {v: "{:0.2f}".format(t) for v, t in zip(x_converged, times_converged)})
    axs[1, 0].plot(x_converged, times_converged, 'r')
    # axs[1, 0].legend(["Pull-in", "Release"])
    axs[1, 0].set_title(r"Varying $g_0$", fontsize=12)
    axs[1, 0].axvline(gfnom*1e6, color='k', linestyle='--')
    display_stats(x_converged, times_converged, "Release gfcon")
    data['gfrelease'] = (x_converged, times_converged)
    np.save('../data/' + timestamp + '.npy', np.array([model.process, fingerLcon_range, fingerWcon_range,
                                                       gfcon_range, supportWcon_range, fingerLnom, fingerWnom, gfnom,
                                                       supportWnom, data, fig], dtype=object), allow_pickle=True)

    model = setup_model_pullin()
    label = r"$g_0=$" + "{:0.2f}".format(gfnom*1e6) + r' $\mu$m'
    axs[1, 0].annotate(label, xy=(0.44, 0.96), xycoords='axes fraction', color='k',
                       xytext=(0, 0), textcoords='offset points', ha='right', va='top')
    # axs[1, 0].annotate(label, xy=(0.52, 0.38), xycoords='axes fraction', color='k',
    #                    xytext=(0, 0), textcoords='offset points', ha='left', va='top')
    axs[1, 0].set_xlabel(r'$g_0 [\mu$m]')

    ##### ax[1, 1] = Varying supportW
    # Pullin
    x_converged = []
    times_converged = []
    for con in supportWcon_range:
        start_time = time.process_time()
        model = setup_model_pullin()
        model.gca.supportW = supportWnom*con - 2*model.gca.process.undercut
        model.gca.update_dependent_variables()
        u = setup_inputs(V=V, Fext=Fext)
        sol = sim_gca(model, u, t_span)

        if len(sol.t_events[0]) > 0:
            # x_converged.append(con)
            x_converged.append(supportWnom*con*1e6)
            times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
        end_time = time.process_time()
        print("Runtime for pullin supportWcon =", con, ", V =", V, "=", end_time - start_time, "=",
                  end_time - start_time, '-->', {v: "{:0.2f}".format(t) for v, t in zip(x_converged, times_converged)})
    axs[1, 1].plot(x_converged, times_converged, 'b')
    display_stats(x_converged, times_converged, "Pullin supportWcon")
    data['supportWpullin'] = (x_converged, times_converged)
    np.save('../data/' + timestamp + '.npy', np.array([model.process, fingerLcon_range, fingerWcon_range,
                                                       gfcon_range, supportWcon_range, fingerLnom, fingerWnom, gfnom,
                                                       supportWnom, data, fig], dtype=object), allow_pickle=True)

    # Release
    x_converged = []
    times_converged = []
    for con in supportWcon_range:
        model = setup_model_release(V=V, Fext=Fext)
        model.gca.supportW = supportWnom*con - 2*model.gca.process.undercut
        model.gca.update_dependent_variables()
        u = [V, Fext]
        model.gca.x0 = model.gca.x0_release(u)
        u = setup_inputs(V=0., Fext=Fext)
        sol = sim_gca(model, u, t_span)

        if len(sol.t_events[0]) > 0:
            start_time = time.process_time()
            # x_converged.append(con)
            x_converged.append(supportWnom*con*1e6)
            times_converged.append(sol.t_events[0][0]*1e6)  # us conversion
        end_time = time.process_time()
        print("Runtime for release supportWcon =", con, ", V =", V, "=", end_time - start_time, "=",
                  end_time - start_time, '-->', {v: "{:0.2f}".format(t) for v, t in zip(x_converged, times_converged)})
    axs[1, 1].plot(x_converged, times_converged, 'r')
    axs[1, 1].legend(["Pull-in", "Release"])
    axs[1, 1].set_title(r"Varying $w_s$", fontsize=12)
    axs[1, 1].axvline(supportWnom*1e6, color='k', linestyle='--')
    display_stats(x_converged, times_converged, "Release supportWcon")
    data['supportWrelease'] = (x_converged, times_converged)

    model = setup_model_pullin()
    label = r"$w_s=$" + "{:0.1f}".format(supportWnom*1e6) + r' $\mu$m'
    axs[1, 1].annotate(label, xy=(0.44, 0.96), xycoords='axes fraction', color='k',
                       xytext=(0, 0), textcoords='offset points', ha='right', va='top')
    axs[1, 1].set_xlabel(r'$w_s [\mu$m]')
    # axs[1, 1].annotate(label, xy=(0.52, 0.38), xycoords='axes fraction', color='k',
    #                    xytext=(0, 0), textcoords='offset points', ha='left', va='top')

    plt.tight_layout()
    plt.savefig("../figures/" + timestamp + ".png")
    plt.savefig("../figures/" + timestamp + ".pdf")

    np.save('../data/' + timestamp + '.npy', np.array([model.process, fingerLcon_range, fingerWcon_range, gfcon_range, supportWcon_range,
                                                       fingerLnom, fingerWnom, gfnom, supportWnom, data,
                                                       fig], dtype=object),
            allow_pickle=True)
    plt.show()
