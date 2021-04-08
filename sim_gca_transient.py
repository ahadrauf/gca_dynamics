from assembly import AssemblyGCA
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def setup_model_pullin():
    model = AssemblyGCA("gamma.csv")
    model.gca.x0 = model.gca.x0_pullin()
    # model.gca.x0[0] = -(2e-6 + 2*model.gca.process.overetch)
    # print(model.gca.x0)
    # print(2e-6 + 2*model.gca.process.overetch)
    # print(model.gca.x0[0])
    model.gca.terminate_simulation = model.gca.pulled_in
    return model


def setup_model_release(**kwargs):
    u = [kwargs["V"], kwargs["Fext"]]
    model = AssemblyGCA()
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

    sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True, max_step=1e-6)
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
    ax1_right.set_aspect(1.0/ax1_right.get_data_ratio(), adjustable='box')

    ax2 = plt.subplot(122)
    t = model.gca.sim_log['t']
    t = t[t <= max(t_sim)]
    Fes = model.gca.sim_log['Fes'][:len(t)]
    Fb = model.gca.sim_log['Fb'][:len(t)]
    Fk = model.gca.sim_log['Fk'][:len(t)]
    plt.plot(t*1e6, Fes, label='Fes', marker='.')
    plt.plot(t*1e6, Fb, label='Fb', marker='.')
    plt.plot(t*1e6, Fk, label='Fk', marker='.')
    plt.legend()
    plt.xlabel('t (us)')
    plt.ylabel('Force (N)')
    ax2.semilogy(True)
    ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')

    # fig.tight_layout()
    # plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    V = 40
    Fext = 50e-6
    model = setup_model_release(V=V, Fext=Fext)  # Change for pullin/release
    model.gca.add_support_spring(springW=5e-6, springL=594.995e-6, nBeams=16,
                                 endcapW=22.889e-6, endcapL=49.441e-6,
                                 etchholeSize=8e-6, nEtchHoles=3, nEndCaps=8*2,
                                 k=Fext/(385.33e-6 + 2*model.gca.process.overetch))
    print(model.gca.k_support)
    print(Fext/(385.33e-6 + 2*model.gca.process.overetch))
    u = setup_inputs(V=0, Fext=0.)  # Change V=V for pullin, V=0 for release
    t_span = [0, 200e-6]
    sol = sim_gca(model, u, t_span)
    print('End time:', sol.t_events[0])

    t_sim = np.linspace(t_span[0], sol.t_events[0], 30)
    t_sim = np.ndarray.flatten(t_sim)
    title = "GCA Simulation, x0 = {}, V = {}".format(model.x0(), V)
    plot_solution(sol, t_sim, model, plt_title=title)
