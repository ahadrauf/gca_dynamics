"""
Simulate switch bounce behavior assuming a perfectly elastic collisions (i.e., no energy loss)
"""

import os
file_location = os.path.abspath(os.path.dirname( __file__))
dir_location = os.path.abspath(os.path.join(file_location, '..'))
import sys
sys.path.append(file_location)
sys.path.append(dir_location)

from assembly import AssemblyGCA
from process import *
import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from datetime import datetime


def setup_model_pullin():
    model = AssemblyGCA(drawn_dimensions_filename="../layouts/fawn.csv", process=SOI())
    model.gca.x0 = model.gca.x0_pullin()
    model.gca.terminate_simulation = model.gca.pulled_in
    return model


def setup_model_release(**kwargs):
    u = [kwargs["V"], kwargs["Fext"]]
    model = AssemblyGCA(drawn_dimensions_filename="../layouts/fawn.csv", process=SOI())
    # 61.2 and 4.0 align well with data
    # so fo 76.5 and 4.0 (with overetch = 0.4)
    model.gca.fingerL = 76.5e-6 - model.gca.process.overetch
    model.gca.supportW = 4.e-6 - 2*model.gca.process.overetch
    model.gca.update_dependent_variables()
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
    f = lambda t, x: model.dx_dt(t, x, u, verbose=False)
    x0 = model.x0()
    terminate_simulation = lambda t, x: model.terminate_simulation(t, x)
    terminate_simulation.terminal = True

    sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True, max_step=0.5e-6)
    return sol


def plot_solution(fig, axs, sol, t_sim, model, t_start, voltage_func, add_legend=True):
    ax1 = axs[0]
    ax1_right = axs[1]
    ax2 = axs[2]

    x_sim = sol.sol(t_sim)
    # print(t_sim*1e6)
    # print(x_sim*1e6)
    # ax1.plot(t_sim*1e6 + t_start, x_sim[0, :]*1e6, 'b-')

    # ax1_right.plot(t_sim*1e6 + t_start, x_sim[1, :], 'r-')
    V = voltage_func(t_sim)
    ax1_right.plot(t_sim*1e6 + t_start, V, 'r-')
    last_V = (t_sim[-1]*1e6 + t_start, V[-1])

    t = model.gca.sim_log['t']
    t = t[t <= max(t_sim)]
    Fes = np.abs(model.gca.sim_log['Fes'][:len(t)])
    Fb = np.abs(model.gca.sim_log['Fb'][:len(t)])
    Fk = np.abs(model.gca.sim_log['Fk'][:len(t)])
    eps = 1e-10
    if np.max(Fes) > eps:
        lineFes, = ax2.plot(t*1e6 + t_start, Fes + eps, 'b', label='|Fes|', marker='.')
    lineFb, = ax2.plot(t*1e6 + t_start, Fb + eps, 'orange', label='|Fb|', marker='.')
    lineFk, = ax2.plot(t*1e6 + t_start, Fk + eps, 'g', label='|Fk|', marker='.')
    if add_legend:
        ax2.legend([lineFb, lineFk], ['|Fb|', '|Fk|'])
    return last_V


if __name__ == "__main__":
    now = datetime.now()
    name_clarifier = "_switchbounce"
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    data = loadmat("../data/pullin_release_traces.mat")
    t_release = np.ndarray.flatten(data["t_release"])
    Vsense_release = np.ndarray.flatten(data["Vsense_release"])
    Vdrive_release = np.ndarray.flatten(data["Vdrive_release"])

    V = 50
    # Fext = 50e-6
    Fext = 0.
    # model = setup_model_pullin()
    # u = setup_inputs(V=V, Fext=0.)  # Change V=V for pullin, V=0 for release
    # model = setup_model_release(V=V, Fext=Fext)  # Change for pullin/release
    # u = setup_inputs(V=0, Fext=Fext)  # Change V=V for pullin, V=0 for release

    n = 7
    t_span = [0, 100e-6]
    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax1_right = fig.add_subplot(121, sharex=ax1, frameon=False)
    ax2 = plt.subplot(122)

    ax1.set_xlabel('t (us)')
    ax1.set_ylabel('x (um)')
    ax1_right.yaxis.tick_right()
    ax1_right.yaxis.set_label_position("right")
    ax1_right.set_ylabel("V (V)")
    ax1_right.yaxis.label.set_color('red')
    ax1_right.tick_params(axis='y', colors='red')
    ax2.set_xlabel('t (us)')
    ax2.set_ylabel('Force (N)')
    ax2.semilogy(True)

    t_release_zoomed_out_range = [0, 250]  # [-220*(10/30), 220]
    idx_zoomed_out = np.where(
        (t_release_zoomed_out_range[0] <= t_release) & (t_release <= t_release_zoomed_out_range[1]))
    t_release_zoomed_out_plot = t_release[idx_zoomed_out]
    Vsense_release_zoomed_out_plot = Vsense_release[idx_zoomed_out]
    ax1_right.plot(t_release_zoomed_out_plot, Vsense_release_zoomed_out_plot, 'g-')

    t_start = 0.
    x0 = [0., 0.]
    last_V = None
    for itr in range(n):
        model = setup_model_release(V=V, Fext=Fext)  # Change for pullin/release
        if itr != 0:
            model.gca.x0 = x0
        u = setup_inputs(V=0, Fext=Fext)  # Change V=V for pullin, V=0 for release
        sol = sim_gca(model, u, t_span)
        print('End time:', sol.t_events[0]*1e6)

        t_sim = np.linspace(t_span[0], sol.t_events[0], 30)
        t_sim = np.ndarray.flatten(t_sim)
        if itr == 0:
            voltage_func = lambda t: 4.*np.ones_like(t)
        else:
            voltage_func = lambda t: 4. - 4*np.exp(-t/8.2e-5) - 0.223  # old RC constant = 6.6e-5

        if last_V is not None:
            ax1_right.plot([last_V[0], last_V[0]], [last_V[1], -0.223], 'r-')

        last_V = plot_solution(fig, [ax1, ax1_right, ax2], sol, t_sim, model, t_start, voltage_func, add_legend=(itr == 0))
        t_start += sol.t_events[0]*1e6
        x_sim = np.ndarray.flatten(sol.sol(sol.t_events[0]))
        print('Final state:', n, x_sim*1e6)
        x0 = [1e-9, -x_sim[1]]

    ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
    ax1_right.set_aspect(1.0/ax1_right.get_data_ratio(), adjustable='box')
    ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')
    plt.suptitle("Switch-Bounce Simulation, V={}".format(V))
    # plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    plt.savefig("../figures/" + timestamp + ".png")
    plt.savefig("../figures/" + timestamp + ".pdf")
    plt.show()
