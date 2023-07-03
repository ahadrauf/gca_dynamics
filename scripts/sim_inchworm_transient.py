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


def setup_model(period, x_prev=None, u=(0., 0.)):
    model = AssemblyInchwormMotor(period, drawn_dimensions_filename="../layouts/fawn_velocity.csv", process=SOI())
    if x_prev is None:
        model.gca_pullin.x0 = model.gca_pullin.x0_pullin()
        model.gca_release.x0 = model.gca_release.x0_release(u[:2])  # assume ideal behavior for the first step
        model.inchworm.x0 = np.array([0., 0.])
    else:
        xprev_release, xprev_pullin, xprev_shuttle = model.unzip_state(x_prev)  # switched (pullin pawl --> release)
        # if model.gca_release.impacted_shuttle(xprev_release[0]):
        #     model.gca_release.x0 = model.gca_release.x0_release(u[:2], x_curr=xprev_release[0], v_curr=0.)
        # else:
        model.gca_release.x0 = model.gca_release.x0_release(u[:2], x_curr=xprev_release[0], v_curr=xprev_release[1])
        if xprev_pullin[0] < 0.2e-6:  # arbitrary boundary
            model.gca_pullin.x0 = model.gca_pullin.x0_pullin()
        else:
            model.gca_pullin.x0 = xprev_pullin
        model.inchworm.x0 = xprev_shuttle
    model.gca_pullin.terminate_simulation = lambda t, x: model.gca_pullin.pulled_in(t, x[:2])
    return model


def setup_inputs(V, Fext_pullin=0., Fext_release=0., Fext_shuttle=0.):
    return [lambda t, x: np.array([V, Fext_pullin]),
            lambda t, x: np.array([0., Fext_release]),
            lambda t, x: np.array([Fext_shuttle, ])]


def sim_gca(model, u, t_span):
    f = lambda t, x: model.dx_dt(t, x, u, verbose=False, Fes_calc_method=2, Fb_calc_method=2)
    x0 = model.x0()
    terminate_simulation = lambda t, x: model.terminate_simulation(t, x)
    terminate_simulation.terminal = True

    sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True, max_step=0.1e-6)
    if sol.t_events[0]:
        t_sim = np.linspace(t_span[0], sol.t_events[0][0], 30)
    else:
        t_sim = np.linspace(t_span[0], t_span[1], 30)
    x_sim = np.transpose(sol.sol(t_sim))
    return t_sim, x_sim


if __name__ == "__main__":
    now = datetime.now()
    name_clarifier = "_plot_inchworm_transient"
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    V = 60
    Fext = 0.
    drive_freq = 2e4
    t_max = 0.25 / drive_freq
    model = setup_model(period=1 / drive_freq, u=(V, 0.))
    u = setup_inputs(V=V)  # Change V=V for pullin, V=0 for release
    t_span = [0, min(t_max, 200e-6)]

    t_all = []
    x_all = []
    default_step_size = 2e-6


    def postprocess_sim_data(T, X):
        global t_all, x_all
        print(np.shape(X))
        if T[-1] < 0.99 * t_span[1]:
            T = np.append(T, t_max)
            X = np.vstack([X, [model.gca_pullin.x_GCA, 0.,
                               0., 0.,
                               default_step_size * np.ceil(X[-1][4] / default_step_size), 0.]])
        if not t_all:
            t_all.append(T)
        else:
            t_all.append(T + t_all[-1][-1])
        x_all.append(X)
        return T, X


    Nsteps = 6

    # First step
    T, X = postprocess_sim_data(*sim_gca(model, u, t_span))

    # Second step
    for i in range(Nsteps - 1):
        model = setup_model(period=1 / drive_freq, u=(V, 0.), x_prev=X[-1, :])
        T, X = postprocess_sim_data(*sim_gca(model, u, t_span))

    t_sim = np.hstack(t_all)
    x_sim = np.vstack(x_all)
    x = x_sim[:, 0]
    v = x_sim[:, 1]
    xr = x_sim[:, 2]
    vr = x_sim[:, 3]
    x_shuttle = x_sim[:, 4]
    v_shuttle = x_sim[:, 5]

    title = "GCA Simulation, V = {}".format(V)
    fig = plt.figure()

    ax1 = plt.subplot(111)
    # line1, = plt.plot(t_sim_pullin * 1e6, sol_pullin.sol(t_sim_pullin)[0, :] * 1e6, 'b-', label="x")
    # plt.plot((t_sim_release + offset) * 1e6, sol_release.sol(t_sim_release)[0, :] * 1e6, 'b-', label='_nolegend_')
    # plt.plot(t_sim_pullin * 1e6, sol_pullin.sol(t_sim_pullin)[2, :] * 1e6, 'k--', label='_nolegend_')
    # plt.plot((t_sim_release + offset) * 1e6, sol_release.sol(t_sim_release)[2, :] * 1e6, 'k--', label='_nolegend_')
    line1, = plt.plot(t_sim * 1e6, x * 1e6, '-', c='tab:blue', label="x_GCA_pullin")
    line11, = plt.plot(t_sim * 1e6, xr * 1e6, '-', c='tab:green', label="x_GCA_release")
    line12, = plt.plot(t_sim * 1e6, x_shuttle * 1e6, '-', c='tab:orange', label='x_shuttle')
    plt.xlabel('t (us)')
    plt.ylabel('x (um)')
    # plt.legend()

    ax1_right = fig.add_subplot(111, sharex=ax1, frameon=False)
    line2, = plt.plot(t_sim * 1e6, v * 1e6, '--', c='tab:blue', label="v_GCA_pullin")
    line21, = plt.plot(t_sim * 1e6, vr * 1e6, '--', c='tab:green', label='v_GCA_release')
    line22, = plt.plot(t_sim * 1e6, v_shuttle * 1e6, '--', c='tab:orange', label='v_shuttle')
    ax1_right.yaxis.tick_right()
    ax1_right.yaxis.set_label_position("right")
    plt.ylabel("dx/dt (m/s)")
    ax1_right.yaxis.label.set_color('red')
    ax1_right.tick_params(axis='y', colors='red')

    # ax1.axvline(max(t_sim_pullin) * 1e6, color='k', linestyle='--')
    # ax1.axvline(offset * 1e6, color='k', linestyle='--')
    print("Contact line:", model.gca_pullin.x_impact + 2 * 0.2e-6)
    line_contact = ax1.axhline((model.gca_pullin.x_impact + 2 * 0.2e-6) * 1e6, color='k', linestyle='--',
                               label='Pawl-Shuttle Contact')
    for i in range(0, Nsteps, 2):
        ax1.axvline(i * t_span[1] * 1e6, color='k', linestyle='--', label="Step {}".format(i))
    plt.title(title)
    plt.legend([line1, line2, line11, line21, line12, line22, line_contact],
               [line1.get_label(), line2.get_label(), line11.get_label(), line21.get_label(),
                line12.get_label(), line22.get_label(), line_contact.get_label()], loc='upper right')

    # ax1.annotate("Pull-in", xy=(0.04, 0.96), xycoords='axes fraction', color='k',
    #              xytext=(0, 0), textcoords='offset points', ha='left', va='top')
    #
    # ax1.annotate("Release", xy=(0.96, 0.96), xycoords='axes fraction', color='k',
    #              xytext=(0, 0), textcoords='offset points', ha='right', va='top')
    # plt.legend(handles=[line1, line2])
    plt.tight_layout()
    plt.show()
    # plt.savefig("../figures/" + timestamp + ".png")
    # plt.savefig("../figures/" + timestamp + ".pdf")
    # plot_solution(sol, t_sim, model, plt_title=title)
