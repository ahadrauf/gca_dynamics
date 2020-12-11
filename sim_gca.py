from assembly import AssemblyGCA
import numpy as np
from scipy.integrate import solve_ivp


def setup_model():
    model = AssemblyGCA()
    # model.gca.x0 = np.array([0, 0])
    model.gca.terminate_simulation = model.gca.pulled_in
    return model


def setup_inputs():
    V = 45
    return lambda t, x: np.array([V])


def sim_gca(model, u, t_span):
    f = lambda t, x: model.dx_dt(t, x, u)
    x0 = model.x0()
    terminate_simulation = lambda t, x: model.terminate_simulation(t, x)
    terminate_simulation.terminal = True

    sol = solve_ivp(f, t_span, x0, events=terminate_simulation)
    return sol.t, sol.y, sol.t_events


if __name__ == "__main__":
    model = setup_model()
    u = setup_inputs()
    t_span = [0, 100e-6]
    t, x, t_events = sim_gca(model, u, t_span)
    print(t)
    print(x)
    print(t_events)
