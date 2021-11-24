"""
One of the most useful helpful files in this whole library. Runs a pull-in/release simulation of a GCA, and plots
both the [spine position, spine velocity] curve on the left but also the various forces on the right. This is a very
helpful debugging tool if you're trying to understand why your motor is moving differentthan you expected.
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


def setup_model_pullin():
    model = AssemblyGCA(drawn_dimensions_filename="../layouts/fawn.csv", process=SOI())
    model.gca.x0 = model.gca.x0_pullin()
    model.gca.terminate_simulation = model.gca.pulled_in
    return model


def setup_model_release(**kwargs):
    u = [kwargs["V"], kwargs["Fext"]]
    model = AssemblyGCA(drawn_dimensions_filename="../layouts/fawn.csv", process=SOI())
    # model.gca.fingerL = 15.6e-6
    # model.gca.update_dependent_variables()
    if "Fescon" in kwargs:
        model.gca.Fescon = kwargs["Fescon"]
    if "Fkcon" in kwargs:
        model.gca.Fkcon = kwargs["Fkcon"]
    model.gca.x0 = model.gca.x0_release(u)
    # model.gca.x0 = [4.630000e-6, -0.349416] #model.gca.x0_release(u)
    print("x0 = ", model.gca.x0, "ours:", model.gca.x0_release(u))
    print("k_support = ", model.gca.k_support)
    print("mass = ", model.gca.m_total, model.gca.mainspineA*model.gca.process.t_SOI*model.gca.process.density)
    print("Lol =", model.gca.fingerL)
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
    V = 60
    # Fext = 50e-6
    Fext = 0.
    # model = setup_model_pullin()
    # u = setup_inputs(V=V, Fext=0.)  # Change V=V for pullin, V=0 for release
    model = setup_model_release(V=V, Fext=Fext)  # Change for pullin/release
    # model.gca.gf = 2e-6 + 2*model.gca.process.undercut  # 4.83e-6 + 2*model.gca.process.undercut
    # model.gca.x_GCA = model.gca.gf - 1e-6
    # model.gca.fingerW = 3.425e-6 - 2*model.gca.process.undercut  # 5.005e-6 + 2*model.gca.process.undercut
    # model.gca.fingerW = 3.5e-6 - 2*model.gca.process.undercut
    # model.gca.update_dependent_variables()
    # model.gca.x0 = model.gca.x0_release([V, Fext])
    # print("x0 = ", model.gca.x0, "ours:", model.gca.x0_release([V, Fext]))
    u = setup_inputs(V=0, Fext=Fext)  # Change V=V for pullin, V=0 for release

    t_span = [0, 300e-6]
    sol = sim_gca(model, u, t_span)
    print('End time:', sol.t_events[0]*1e6)

    print(model.gca.m_total)
    print(model.gca.supportW*model.gca.supportL*model.gca.process.t_SOI*model.gca.process.density)

    t_sim = np.linspace(t_span[0], sol.t_events[0], 30)
    t_sim = np.ndarray.flatten(t_sim)
    title = "GCA Simulation, x0 = {}, V = {}".format(model.x0(), V)
    plot_solution(sol, t_sim, model, plt_title=title)
