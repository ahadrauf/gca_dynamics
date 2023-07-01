"""
Similarly inspired compared to sim_gca_transient.py, except with extra plotting/saving functionality. It takes a little
longer to run in exchange for better fidelity, making this version a little better for production-quality photos
while sim_gca_transient.py is the quick and dirty debugging tool.
"""

import os
file_location = os.path.abspath(os.path.dirname( __file__))
dir_location = os.path.abspath(os.path.join(file_location, '..'))
import sys
sys.path.append(file_location)
sys.path.append(dir_location)

from assembly import AssemblyGCA
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from process import *
from datetime import datetime
plt.rc('font', size=12)


def setup_model_pullin():
    model = AssemblyGCA(drawn_dimensions_filename="../layouts/fawn.csv", process=SOI())
    model.gca.x0 = model.gca.x0_pullin()
    model.gca.terminate_simulation = model.gca.pulled_in
    return model


def setup_model_release(**kwargs):
    u = [kwargs["V"], kwargs["Fext"]]
    model = AssemblyGCA(drawn_dimensions_filename="../layouts/fawn.csv", process=SOI())
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


def sim_gca(model, u, t_span):
    f = lambda t, x: model.dx_dt(t, x, u, verbose=False, Fes_calc_method=2, Fb_calc_method=2)
    x0 = model.x0()
    terminate_simulation = lambda t, x: model.terminate_simulation(t, x)
    terminate_simulation.terminal = True

    sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True, max_step=0.05e-6)
    return sol


def plot_solution(sol, t_sim, model, plt_title=None):
    fig = plt.figure()
    if plt_title is not None:
        fig.suptitle(plt_title)

    ax1 = plt.subplot(121)
    x_sim = sol.sol(t_sim)
    plt.plot(t_sim*1e6, x_sim[0, :]*1e6)
    plt.xlabel('t (us)')
    plt.ylabel('x (um)')
    ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')

    ax1_right = fig.add_subplot(121, sharex=ax1, frameon=False)
    ax1_right.plot(t_sim*1e6, x_sim[1, :], 'r-')
    ax1_right.yaxis.tick_right()
    ax1_right.yaxis.set_label_position("right")
    plt.ylabel("dx/dt (m/s)")
    ax1_right.yaxis.label.set_color('red')
    ax1_right.tick_params(axis='y', colors='red')
    ax1_right.set_aspect(1.0/ax1_right.get_data_ratio(), adjustable='box')

    ax2 = plt.subplot(122)
    t = model.gca.sim_log['t']
    t = t[t <= max(t_sim)]
    Fes = np.abs(model.gca.sim_log['Fes'][:len(t)])
    Fb = np.abs(model.gca.sim_log['Fb'][:len(t)])
    Fbsf = np.abs(model.gca.sim_log['Fbsf'][:len(t)])
    Fbcf = np.abs(model.gca.sim_log['Fbcf'][:len(t)])
    Fk = np.abs(model.gca.sim_log['Fk'][:len(t)])
    Ftot = np.abs([a + b + c for a, b, c in zip(model.gca.sim_log['Fes'][:len(t)],
                                                model.gca.sim_log['Fb'][:len(t)],
                                                model.gca.sim_log['Fk'][:len(t)])])
    eps = 1e-10
    if np.max(Fes) > eps:
        ax2.plot(t*1e6, Fes + eps, 'b', label='|Fes|', marker='.')
    # ax2.plot(t*1e6, Fb + eps, 'orange', label='|Fb|', marker='.')
    ax2.plot(t*1e6, Fbsf + eps, 'orange', label='|Fb_sf|', marker='.')
    ax2.plot(t*1e6, Fbcf + eps, 'pink', label='|Fb_cf|', marker='.')
    ax2.plot(t*1e6, Fk + eps, 'g', label='|Fk|', marker='.')
    ax2.plot(t*1e6, Ftot + eps, 'r', label='|Ftot|', marker='.')
    ax2.legend()
    ax2.set_xlabel('t (us)')
    ax2.set_ylabel('Force (N)')
    ax2.semilogy(True)
    ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')

    # fig.tight_layout()
    # plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    now = datetime.now()
    name_clarifier = "_plot_gca_transient"
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    V = 60
    Fext = 0.
    model = setup_model_pullin()
    u = setup_inputs(V=V, Fext=0.)  # Change V=V for pullin, V=0 for release
    t_span = [0, 300e-6]
    sol_pullin = sim_gca(model, u, t_span)
    print('End time:', sol_pullin.t_events[0]*1e6)
    t_sim_pullin = np.linspace(t_span[0], sol_pullin.t_events[0], 30)
    t_sim_pullin = np.ndarray.flatten(t_sim_pullin)

    model = setup_model_release(V=V, Fext=Fext)  # Change for pullin/release
    model.gca.x0 = model.gca.x0_release([V, Fext])
    u = setup_inputs(V=0, Fext=Fext)  # Change V=V for pullin, V=0 for release
    t_span = [0, 300e-6]
    sol_release = sim_gca(model, u, t_span)
    print('End time:', sol_release.t_events[0]*1e6)
    t_sim_release = np.linspace(t_span[0], sol_release.t_events[0], 30)
    t_sim_release = np.ndarray.flatten(t_sim_release)

    t_sim = np.append(t_sim_pullin, t_sim_release + max(t_sim_pullin))
    x = np.append(sol_pullin.sol(t_sim_pullin)[0, :], sol_release.sol(t_sim_release)[0, :])
    v = np.append(sol_pullin.sol(t_sim_pullin)[1, :], sol_release.sol(t_sim_release)[1, :])
    offset = max(t_sim_pullin)

    title = "GCA Simulation, V = {}".format(V)
    fig = plt.figure()

    ax1 = plt.subplot(111)
    # line1, = plt.plot(t_sim_pullin*1e6, sol_pullin.sol(t_sim_pullin)[0, :]*1e6, 'b-', label="x")
    # plt.plot((t_sim_release + offset)*1e6, sol_release.sol(t_sim_release)[0, :]*1e6, 'b-', label='_nolegend_')
    line1, = plt.plot(t_sim*1e6, x*1e6, 'b-', label="x")
    plt.xlabel('t (us)')
    plt.ylabel('x (um)')
    # plt.legend()

    ax1_right = fig.add_subplot(111, sharex=ax1, frameon=False)
    line2, = ax1_right.plot(t_sim_pullin*1e6, sol_pullin.sol(t_sim_pullin)[1, :], 'r-', label="dx/dt")
    ax1_right.plot((t_sim_release + offset)*1e6, sol_release.sol(t_sim_release)[1, :], 'r-', label='_nolegend_')
    ax1_right.yaxis.tick_right()
    ax1_right.yaxis.set_label_position("right")
    plt.ylabel("dx/dt (m/s)")
    ax1_right.yaxis.label.set_color('red')
    ax1_right.tick_params(axis='y', colors='red')

    ax1.axvline(max(t_sim_pullin)*1e6, color='k', linestyle='--')
    # ax1.axvline(offset*1e6, color='k', linestyle='--')
    plt.title(title)
    plt.legend([line1, line2], [line1.get_label(), line2.get_label()], loc='center right')

    ax1.annotate("Pull-in", xy=(0.04, 0.96), xycoords='axes fraction', color='k',
                 xytext=(0, 0), textcoords='offset points', ha='left', va='top')

    ax1.annotate("Release", xy=(0.96, 0.96), xycoords='axes fraction', color='k',
                 xytext=(0, 0), textcoords='offset points', ha='right', va='top')
    # plt.legend(handles=[line1, line2])
    plt.tight_layout()
    plt.show()
    # plt.savefig("../figures/" + timestamp + ".png")
    # plt.savefig("../figures/" + timestamp + ".pdf")
    # plot_solution(sol, t_sim, model, plt_title=title)
