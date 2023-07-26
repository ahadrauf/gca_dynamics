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


def setup_model(period, x_prev=None, u=(0., 0.), drawn_dimensions_filename="../layouts/fawn_velocity.csv",
                process=SOI()):
    model = AssemblyInchwormMotor(period, drawn_dimensions_filename=drawn_dimensions_filename, process=process)
    if x_prev is None:
        model.gca_pullin.x0 = model.gca_pullin.x0_pullin()
        model.gca_release.x0 = model.gca_release.x0_pullin()  # model.gca_release.x0_release(u[:2])  # assume ideal behavior for the first step
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
    # model.gca_pullin.terminate_simulation = lambda t, x: model.gca_pullin.impacted_shuttle(t, x[0])
    return model


def setup_inputs(V, period, Fext_pullin=0., Fext_release=0., Fext_shuttle=0.):
    R = 205244.8126133541 + 217573.24999999994
    C = 8.043260992499999e-13
    RC = R * C
    # return [lambda t, x: np.array([V if t < 0. else V * (1 - np.exp(-t / RC)), Fext_pullin]),
    #         lambda t, x: np.array([0. if t < period / 4 else V * np.exp(-(t - period / 4) / RC), Fext_release]),
    #         lambda t, x: np.array([Fext_shuttle, ])]
    return [lambda t, x: np.array([V, Fext_pullin]),
            lambda t, x: np.array([0., Fext_release]),
            lambda t, x: np.array([Fext_shuttle, ])]


def sim_gca(model, u, t_span, max_step=0.1e-6):
    f = lambda t, x: model.dx_dt(t, x, u, verbose=False, Fes_calc_method=2, Fb_calc_method=2)
    x0 = model.x0()
    terminate_simulation = lambda t, x: model.terminate_simulation(t, x)
    terminate_simulation.terminal = True

    sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True, max_step=max_step)
    if len(sol.t_events[0]) > 0:
        t_sim = np.linspace(t_span[0], sol.t_events[0][0], 30)
    else:
        t_sim = np.linspace(t_span[0], t_span[1], 30)
    x_sim = np.transpose(sol.sol(t_sim))
    return t_sim, x_sim


def sim_inchworm(Nsteps, V, drive_freq, Fext_shuttle=0., print_every_step=False,
                 drawn_dimensions_filename="../layouts/fawn_velocity.csv",
                 process=SOI()):
    t_max = 0.5 / drive_freq
    period = 1 / drive_freq
    model = setup_model(period=period, u=(V, 0.), drawn_dimensions_filename=drawn_dimensions_filename, process=process)
    u = setup_inputs(V=V, period=period, Fext_shuttle=Fext_shuttle)  # 0.002942 = 1gf * 0.3
    t_span = [0, t_max]
    # max_step = 0.5e-6 if drive_freq < 1e4 else 0.1e-6
    max_step = 0.03e-6
    max_step_inner = 0.05e-6 if drive_freq > 1e4 else 0.5e-6

    t_all = []
    x_all = []
    F_shuttle_all = []
    curr_step = 0
    step_count = []
    default_step_size = 2e-6

    def postprocess_sim_data(T, X):
        nonlocal t_all, x_all, F_shuttle_all, step_count
        # if T[-1] < 0.99 * t_span[1]:
        #     T = np.append(T, t_max)
        #     X = np.vstack([X, [model.gca_pullin.x_GCA, 0.,
        #                        0., 0.,
        #                        default_step_size * np.ceil(X[-1][4] / default_step_size), 0.]])
        # if X[-1][0] >= model.gca_pullin.x_GCA and (X[-1][4] - X[0][4] < default_step_size):
        # if np.max(X[:, 0]) >= model.gca_pullin.x_GCA:
        #     X[-1][4] = default_step_size * np.round(X[-1][4] / default_step_size)
        # else:
        #     X[-1][4] = default_step_size * np.floor(X[-1][4] / default_step_size)
        F_shuttle_all.append([model.dx_dt(t, x, u)[5] * model.inchworm.m_total for (t, x) in zip(T, X)])

        if not t_all:
            t_all.append(T)
        else:
            t_all.append(T + (t_all[-1][-1] - T[0]))
        x_all.append(X)
        step_count.append(curr_step * np.ones_like(T))
        return T, X

    def run_sim(model, u, t_span):
        nonlocal t_all, x_all
        T, X = postprocess_sim_data(*sim_gca(model, u, t_span, max_step=max_step))
        if T[-1] < 0.99 * t_span[1]:
            Estar = model.gca_pullin.process.E / (1 - model.gca_pullin.process.v**2)
            I_pawl = model.gca_pullin.pawlW**3 * model.gca_pullin.process.t_SOI / 12
            k = 3 * Estar * I_pawl / model.gca_pullin.pawlL**3
            v_GCA0 = X[-1][1]
            v_shuttle0 = X[-1][5]
            m_GCA = model.gca_pullin.mainspineA * model.gca_pullin.process.t_SOI * model.gca_pullin.process.density
            m_shuttle = model.inchworm.m_total
            N = model.inchworm.Ngca * 2

            k_spine = model.gca_pullin.process.E * (
                    model.gca_pullin.process.t_SOI * model.gca_pullin.spineW) / model.gca_pullin.spineL
            # Fes, y, Ues = model.gca_pullin.Fes_calc2(X[-1][0], V)
            Fes = model.dx_dt(T[-2], X[-2], u)[1] * model.gca_pullin.m_total
            x_spine = model.gca_pullin.Nfing * Fes / k_spine

            # v_shuttlef = np.sqrt(1 / m_shuttle * (k * ((model.gca_pullin.x_GCA - model.gca_pullin.x_impact) /
            #                                            np.cos(model.gca_pullin.alpha))**2) +
            #                      m_GCA * v_GCA0**2 + v_shuttle0**2)
            v_shuttlef1 = np.sqrt(1 / m_shuttle * (N * k * (model.gca_pullin.x_GCA - model.gca_pullin.x_impact)**2 /
                                                   np.sin(model.gca_pullin.alpha)) / 4 +
                                  v_shuttle0**2)
            # v_shuttlef1 = (N * m_GCA * v_GCA0 + m_shuttle * v_shuttle0) / m_shuttle
            # v_shuttlef1 = np.sqrt(1 / m_shuttle * N * m_GCA * v_GCA0**2 / 50 + v_shuttle0**2)
            # v_shuttlef = np.sqrt(1 / m_shuttle * N * (m_GCA * v_GCA0**2 - k_spine * x_spine**2 / 2) + v_shuttle0**2)
            # v_shuttlef1 = np.sqrt(1 / m_shuttle * N * (k * (model.gca_pullin.x_GCA - model.gca_pullin.x_impact)**2 /
            #                                            np.sin(model.gca_pullin.alpha) / 4 +
            #                                            m_GCA * v_GCA0**2 / 50) + v_shuttle0**2)
            # print("Compare velocity:", X[-1][5], v_shuttlef1, v_shuttlef1 , m_GCA * v_GCA0**2, k_spine * x_spine**2)

            model.gca_pullin.terminate_simulation = lambda t, x: False
            model.gca_pullin.x0 = np.array([model.gca_pullin.x_GCA, 0.])
            model.gca_release.x0 = np.array([X[-1][2], X[-1][3]])
            # model.inchworm.x0 = np.array([X[-1][4], v_shuttlef1])
            model.inchworm.x0 = np.array([X[-1][4], X[-1][5]])
            T2, X2 = postprocess_sim_data(*sim_gca(model, u, [T[-1], t_span[1]], max_step=max_step_inner))
            T = np.hstack([T, T2])
            X = np.vstack([X, X2])

        if np.max(X[:, 0]) >= model.gca_pullin.x_GCA:
            X[-1][4] = default_step_size * np.ceil(X[-1][4] / default_step_size)
        return T, X

    # First step
    # T, X = postprocess_sim_data(*sim_gca(model, u, t_span))
    print("Step 0 / {}, V = {}, drive_freq = {}, Fext_shuttle = {}".format(Nsteps, V, drive_freq, Fext_shuttle),
          end=" --> ", flush=True)
    curr_step = 0
    T, X = run_sim(model, u, t_span)

    # Second step
    start = datetime.now()
    # print("Step", i + 1, "/", 2*Nsteps, end=" = ")
    for curr_step in np.arange(0.5, Nsteps, 0.5):
        if print_every_step:
            print("Step", curr_step, end=" --> ", flush=True)
        model = setup_model(period=1 / drive_freq, u=(V, 0.), x_prev=X[-1, :],
                            drawn_dimensions_filename=drawn_dimensions_filename, process=process)
        # T, X = postprocess_sim_data(*sim_gca(model, u, t_span))
        T, X = run_sim(model, u, t_span)
    print("Runtime =", datetime.now() - start, "s", flush=True)

    t_sim = np.hstack(t_all)
    x_sim = np.vstack(x_all)
    F_shuttle_all = np.hstack(F_shuttle_all)
    step_counts = np.hstack(step_count)
    return t_sim, x_sim, F_shuttle_all, step_counts


if __name__ == "__main__":
    now = datetime.now()
    name_clarifier = "_inchworm_velocity_transient"
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    V = 50
    Fext_shuttle = 0.
    drive_freq = 2e4  # 4.8e3
    t_max = 0.5 / drive_freq
    period = 1 / drive_freq
    t_span = [0, t_max]

    Nsteps = 5
    t_sim, x_sim, F_shuttle_all, step_counts = sim_inchworm(Nsteps, V, drive_freq, Fext_shuttle, print_every_step=True,
                                                            drawn_dimensions_filename="../layouts/fawn_velocity.csv",
                                                            process=SOI())
    xp = x_sim[:, 0]
    vp = x_sim[:, 1]
    xr = x_sim[:, 2]
    vr = x_sim[:, 3]
    x_shuttle = x_sim[:, 4]
    v_shuttle = x_sim[:, 5]
    f_sim = np.hstack(F_shuttle_all)
    print("Force shuttle", list(np.ndarray.flatten(f_sim)), flush=True)

    midway_point = 0  # np.size(t_sim) * 2 // 3
    avg_speed = (x_shuttle[-1] - x_shuttle[midway_point]) / (t_sim[-1] - t_sim[midway_point])
    print("Total shuttle distance:", x_shuttle[-1], "Avg. step size:", x_shuttle[-1] / Nsteps * 1e6,
          "Avg. speed (m/s):", avg_speed, flush=True)

    fig = plt.figure()

    ax1 = plt.subplot(111)
    line1, = plt.plot(t_sim * 1e6, xp * 1e6, '-', c='tab:blue', label="x_GCA_pullin")
    line11, = plt.plot(t_sim * 1e6, xr * 1e6, '-', c='tab:green', label="x_GCA_release")
    line12, = plt.plot(t_sim * 1e6, x_shuttle * 1e6, '-', c='tab:orange', label='x_shuttle')
    plt.xlabel('t (us)')
    plt.ylabel('x (um)')
    # plt.legend()

    ax1_right = fig.add_subplot(111, sharex=ax1, frameon=False)
    line2, = plt.plot(t_sim * 1e6, vp * 1e6, '--', c='tab:blue', label="v_GCA_pullin")
    line21, = plt.plot(t_sim * 1e6, vr * 1e6, '--', c='tab:green', label='v_GCA_release')
    line22, = plt.plot(t_sim * 1e6, v_shuttle * 1e6, '--', c='tab:orange', label='v_shuttle')
    ax1_right.yaxis.tick_right()
    ax1_right.yaxis.set_label_position("right")
    plt.ylabel("dx/dt (m/s)")
    ax1_right.yaxis.label.set_color('red')
    ax1_right.tick_params(axis='y', colors='red')

    # ax1.axvline(max(t_sim_pullin) * 1e6, color='k', linestyle='--')
    # ax1.axvline(offset * 1e6, color='k', linestyle='--')
    # print("Contact line:", model.gca_pullin.x_impact + 2 * 0.2e-6)
    line_contact = ax1.axhline((3e-6 + 2 * 0.2e-6) * 1e6, color='k', linestyle='--',
                               label='Pawl-Shuttle Contact')
    ax1_right.axhline(0., color='r', linestyle='--')
    for i in np.arange(0, Nsteps * 2 + 0.5, 0.5):
        lw = 2 if i % 2 == 0 else 1
        ax1.axvline(i * t_span[1] * 1e6, color='k', linestyle='--', lw=lw, label="Step {}".format(i))

    title = "GCA Simulation, V = {}, Nsteps = {}, f = {} kHz, Avg. Vel = {:.1f} mm/s".format(V, Nsteps,
                                                                                             drive_freq / 1e3,
                                                                                             1e3 * avg_speed)
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
    fig.savefig("../figures/" + timestamp + ".png", bbox_inches="tight")
    # plt.savefig("../figures/" + timestamp + ".pdf")

    fig2, ax2 = plt.subplots(1, 1)
    ax2.plot(t_sim * 1e6, F_shuttle_all, lw=3)
    ax2.set_xlabel("Time (us)")
    ax2.set_ylabel("Force (N)")
    print(list(t_sim))
    print(list(F_shuttle_all))
    for i in np.arange(0, Nsteps * 2 + 0.5, 0.5):
        lw = 2 if i % 2 == 0 else 1
        ax2.axvline(i * t_span[1] * 1e6, color='k', linestyle='--', lw=lw, label="Step {}".format(i))

    plt.show()
    # plot_solution(sol, t_sim, model, plt_title=title)
