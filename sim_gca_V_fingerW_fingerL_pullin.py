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
    Fext = kwargs["Fext"]
    return lambda t, x: np.array([V, Fext])


def sim_gca(model, u, t_span, verbose=False):
    f = lambda t, x: model.dx_dt(t, x, u, verbose=verbose)
    x0 = model.x0()
    terminate_simulation = lambda t, x: model.terminate_simulation(t, x)
    terminate_simulation.terminal = True

    sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True, max_step=0.5e-6)
    return sol


def setup_plot(len_x, len_y, plt_title=None, x_label="", y_label=""):
    fig = plt.figure()
    if plt_title is not None:
        fig.suptitle(plt_title)

    fig.text(0.5, 0.04, x_label, ha='center')
    fig.text(0.04, 0.5, y_label, va='center', rotation='vertical')
    return fig


if __name__ == "__main__":
    model = setup_model()
    t_span = [0, 100e-6]
    Fext = 0

    fingerW_values = [3e-6, 4e-6, 7e-6]
    fingerL_values = [30e-6, 50e-6, 90e-6]
    V_values = np.arange(20, 100, 10)
    fig = setup_plot(len(fingerW_values), len(fingerL_values), x_label="Voltage (V)", y_label="Time (us)")
    nx, ny = len(fingerL_values), len(fingerW_values)

    for idx in range(len(fingerW_values)):
        for idy in range(len(fingerL_values)):
            fingerW = fingerW_values[idx]
            fingerL = fingerL_values[idy]
            model.gca.fingerW = fingerW-2*model.gca.process.overetch
            model.gca.fingerL = fingerL-model.gca.process.overetch
            model.gca.update_dependent_variables()

            V_converged = []
            times_converged = []
            for V in V_values:
                u = setup_inputs(V=V, Fext=Fext)
                sol = sim_gca(model, u, t_span)

                if len(sol.t_events[0]) > 0:
                    V_converged.append(V)
                    times_converged.append(sol.t_events[0][0] * 1e6)  # us conversion
            print(times_converged)

            ax = plt.subplot(nx, ny, nx*idy+idx+1)
            plt.plot(V_converged, times_converged)
            ax.text(0.8*ax.get_xlim()[-1], 0.8*ax.get_ylim()[-1], "w={}um\nL={}um".format(fingerW*1e6, fingerL*1e6))
            ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

    plt.tight_layout()
    plt.show()
