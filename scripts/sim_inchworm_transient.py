"""
Similarly inspired compared to sim_gca_transient.py, except with extra plotting/saving functionality. It takes a little
longer to run in exchange for better fidelity, making this version a little better for production-quality photos
while sim_gca_transient.py is the quick and dirty debugging tool.
"""

import os

file_location = os.path.abspath(os.path.dirname(__file__))
dir_location = os.path.abspath(os.path.join(file_location, '..'))
import sys

sys.path.append(file_location)
sys.path.append(dir_location)

from assembly import AssemblyInchwormMotor
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from process import *
from datetime import datetime

plt.rc('font', size=12)


def setup_model(terminate_on_pullin, x_prev=None, u=(0., 0.)):
    model = AssemblyInchwormMotor(drawn_dimensions_filename="../layouts/fawn_velocity.csv", process=SOI())
    if x_prev is None:
        model.gca.x0 = model.gca.x0_pullin()
        model.inchworm.x0 = np.array([0., 0.])
    else:
        model.gca.x0 = model.gca.x0_release(u, x_curr=x_prev[0], v_curr=x_prev[1])
        model.inchworm.x0 = x_prev[2:]
    if terminate_on_pullin:
        model.gca.terminate_simulation = model.gca.pulled_in
    else:
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

    sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True, max_step=0.1e-6)
    return sol


if __name__ == "__main__":
    now = datetime.now()
    name_clarifier = "_plot_inchworm_transient"
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    V = 60
    Fext = 0.
    drive_freq = 1e3
    t_max = 0.5 / drive_freq
    model = setup_model(terminate_on_pullin=True)
    u = setup_inputs(V=V, Fext=0.)  # Change V=V for pullin, V=0 for release
    t_span = [0, min(t_max, 300e-6)]
    sol_pullin = sim_gca(model, u, t_span)
    print('End time:', sol_pullin.t_events[0] * 1e6)
    t_sim_pullin = np.linspace(t_span[0], sol_pullin.t_events[0], 30)
    t_sim_pullin = np.ndarray.flatten(t_sim_pullin)

    print("Final position:", sol_pullin.sol(t_sim_pullin)[:, -1])
    if sol_pullin.t_events[0] < t_span[1]:
        model = setup_model(x_prev=[model.x])
        model = setup_model(x_prev=sol_pullin.sol(t_sim_pullin)[:, -1], u=[V, Fext], terminate_on_pullin=False)
    u = setup_inputs(V=0, Fext=Fext)
    sol_release = sim_gca(model, u, t_span)
    print('End time:', sol_release.t_events[0]*1e6)
    t_sim_release = np.linspace(t_span[0], sol_release.t_events[0], 30)
    t_sim_release = np.ndarray.flatten(t_sim_release)
    print(model.gca.x0, model.inchworm.x0)

    t_sim = np.append(t_sim_pullin, t_sim_release + max(t_sim_pullin))
    x = np.append(sol_pullin.sol(t_sim_pullin)[0, :], sol_release.sol(t_sim_release)[0, :])
    v = np.append(sol_pullin.sol(t_sim_pullin)[1, :], sol_release.sol(t_sim_release)[1, :])
    x_shuttle = np.append(sol_pullin.sol(t_sim_pullin)[2, :], sol_release.sol(t_sim_release)[3, :])
    v_shuttle = np.append(sol_pullin.sol(t_sim_pullin)[2, :], sol_release.sol(t_sim_release)[3, :])
    offset = 30e-6

    title = "GCA Simulation, V = {}".format(V)
    fig = plt.figure()

    ax1 = plt.subplot(111)
    # line1, = plt.plot(t_sim_pullin * 1e6, sol_pullin.sol(t_sim_pullin)[0, :] * 1e6, 'b-', label="x")
    # plt.plot((t_sim_release + offset) * 1e6, sol_release.sol(t_sim_release)[0, :] * 1e6, 'b-', label='_nolegend_')
    # plt.plot(t_sim_pullin * 1e6, sol_pullin.sol(t_sim_pullin)[2, :] * 1e6, 'k--', label='_nolegend_')
    # plt.plot((t_sim_release + offset) * 1e6, sol_release.sol(t_sim_release)[2, :] * 1e6, 'k--', label='_nolegend_')
    line1, = plt.plot(t_sim * 1e6, x * 1e6, 'b-', label="x_GCA")
    line12, = plt.plot(t_sim*1e6, x_shuttle*1e6, 'b--', label='x_shuttle')
    plt.xlabel('t (us)')
    plt.ylabel('x (um)')
    # plt.legend()

    ax1_right = fig.add_subplot(111, sharex=ax1, frameon=False)
    line2, = plt.plot(t_sim * 1e6, v * 1e6, 'r-', label="v_GCA")
    line22, = plt.plot(t_sim * 1e6, v_shuttle * 1e6, 'r--', label='v_shuttle')
    ax1_right.yaxis.tick_right()
    ax1_right.yaxis.set_label_position("right")
    plt.ylabel("dx/dt (m/s)")
    ax1_right.yaxis.label.set_color('red')
    ax1_right.tick_params(axis='y', colors='red')

    ax1.axvline(max(t_sim_pullin) * 1e6, color='k', linestyle='--')
    # ax1.axvline(offset * 1e6, color='k', linestyle='--')
    plt.title(title)
    plt.legend([line1, line2, line12, line22], [line1.get_label(), line2.get_label(), line12.get_label(),
                                                line22.get_label()], loc='center right')

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
