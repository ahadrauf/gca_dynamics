from assembly import AssemblyGCA
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def setup_model():
    model = AssemblyGCA()
    # model.gca.x0 = np.array([0, 0])
    model.gca.terminate_simulation = model.gca.pulled_in
    return model


def setup_inputs(**kwargs):
    V = kwargs["V"]
    return lambda t, x: np.array([V])


def sim_gca(model, u, t_span):
    f = lambda t, x: model.dx_dt(t, x, u, verbose=True)
    x0 = model.x0()
    terminate_simulation = lambda t, x: model.terminate_simulation(t, x)
    terminate_simulation.terminal = True

    sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True, max_step=1e-6)
    return sol


def plot_solution(sol, t_sim, model, plt_title=None):
    fig = plt.figure()
    if plt_title is not None:
        fig.suptitle(plt_title)

    ax1 = plt.subplot(121)
    x_sim = sol.sol(t_sim)
    plt.plot(t_sim, x_sim[0, :]*1e6)
    plt.xlabel('t')
    plt.ylabel('x (um)')
    ax1.set_aspect(1.0 / ax1.get_data_ratio(), adjustable='box')

    ax1_right = fig.add_subplot(121, sharex=ax1, frameon=False)
    ax1_right.plot(t_sim, x_sim[1, :], 'r-')
    ax1_right.yaxis.tick_right()
    ax1_right.yaxis.set_label_position("right")
    plt.ylabel("xdot (m/s)")
    ax1_right.yaxis.label.set_color('red')
    ax1_right.set_aspect(1.0/ax1_right.get_data_ratio(), adjustable='box')

    ax2 = plt.subplot(122)
    t = model.gca.sim_log['t']
    t = t[t <= max(t_sim)]
    Fes = model.gca.sim_log['Fes'][:len(t)]
    Fb = model.gca.sim_log['Fb'][:len(t)]
    Fk = model.gca.sim_log['Fk'][:len(t)]
    plt.plot(t, Fes, label='Fes', marker='.')
    plt.plot(t, Fb, label='Fb', marker='.')
    plt.plot(t, Fk, label='Fk', marker='.')
    plt.legend()
    plt.xlabel('t (s)')
    plt.ylabel('Force (N)')
    ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')

    # fig.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    plt.show()


if __name__ == "__main__":
    model = setup_model()
    t_span = [0, 100e-6]

    fingerW_values = [3e-6, 4e-6, 7e-6]
    fingerL_values = [30e-6, 50e-6, 90e-6]
    V_values = np.arange(20, 100, 10)

    for fingerW in fingerW_values:
        for fingerL in fingerL_values:
            for V in V_values:
                model.gca.fingerW = fingerW - 2*model.gca.process.overetch
                model.gca.fingerL = fingerL - model.gca.process.overetch
                model.gca.fingerL_total = fingerL + model.gca.fingerL_buffer
                u = setup_inputs(V=V)
                sol = sim_gca(model, u, t_span)

    t_sim = np.linspace(t_span[0], sol.t_events[0], 30)
    t_sim = np.ndarray.flatten(t_sim)
    # title = "GCA Simulation, x0 = {}, V = {}".format(model.x0(), V)
    plot_solution(sol, t_sim, model, plt_title=None)
