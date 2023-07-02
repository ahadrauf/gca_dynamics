"""
Comparing GCA velocity data vs. heuristic approximation

Uncomment the pawl's spring force calculation in gca.py/GCA/Fk() to model the pawl's spring force.
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
from scipy.io import loadmat, savemat
from datetime import datetime
plt.rc('font', size=11)


def setup_model_pullin():
    model = AssemblyGCA(drawn_dimensions_filename="../layouts/fawn_velocity.csv")
    model.gca.x0 = model.gca.x0_pullin()
    model.gca.terminate_simulation = model.gca.pulled_in
    return model


def setup_model_pullin_to_shuttle():
    model = AssemblyGCA(drawn_dimensions_filename="../layouts/fawn_velocity.csv")
    model.gca.x0 = model.gca.x0_pullin()
    model.gca.terminate_simulation = lambda t, x: x[0] >= (3e-6 + 2*0.2e-6)
    return model


def setup_model_release(**kwargs):
    u = [kwargs["V"], kwargs["Fext"]]
    model = AssemblyGCA(drawn_dimensions_filename="../layouts/fawn_velocity.csv")
    model.gca.x0 = model.gca.x0_release(u)
    model.gca.terminate_simulation = model.gca.released
    return model


def setup_model_release_from_shuttle(**kwargs):
    u = [kwargs["V"], kwargs["Fext"]]
    model = AssemblyGCA(drawn_dimensions_filename="../layouts/fawn_velocity.csv")
    model.gca.x0 = model.gca.x0_release(u)
    model.gca.terminate_simulation = lambda t, x: x[0] <= (3e-6 + 2*0.2e-6)
    return model


def setup_inputs(**kwargs):
    V = kwargs["V"]
    Fext = kwargs["Fext"]
    return lambda t, x: np.array([V, Fext])


def setup_inputs_RC_pullin(**kwargs):
    R = 205244.8126133541 + 217573.24999999994
    C = 8.043260992499999e-13
    RC = R*C
    print("RC", RC)
    V = kwargs["V"]
    Fext = kwargs["Fext"]
    return lambda t, x: np.array([V*(1-np.exp(-t/RC)), Fext])
    # return lambda t, x: np.array([V, Fext])  # simulate infinite RC constants


def setup_inputs_RC_release(**kwargs):
    R = 205244.8126133541 + 217573.24999999994
    C = 8.043260992499999e-13
    RC = R*C
    V = kwargs["V"]
    Fext = kwargs["Fext"]
    return lambda t, x: np.array([V*np.exp(-t/RC), Fext])
    # return lambda t, x: np.array([0, Fext])  # simulate infinite RC constants


def sim_gca(model, u, t_span, verbose=False):
    f = lambda t, x: model.dx_dt(t, x, u, verbose=verbose)
    x0 = model.x0()
    terminate_simulation = lambda t, x: model.terminate_simulation(t, x)
    terminate_simulation.terminal = True

    sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True,
                    max_step=0.1e-6)  # , method="LSODA")
    return sol


def setup_plot(len_x, len_y, plt_title=None, x_label="", y_label=""):
    fig, axs = plt.subplots(len_x, len_y)
    if plt_title is not None:
        fig.suptitle(plt_title)
    return fig, axs


def plot_data(fig, axs, frequency, velocity_avg, velocity_std, velocity_fitted, V_labels, line_labels):
    nx, ny = 2, 2
    for idx in range(nx):
        for idy in range(ny):
            i = ny*idx + idy
            ax = axs[idx, idy]
            ax.errorbar(frequency[i], velocity_avg[i], velocity_std[i], fmt='b.', capsize=3, zorder=1)
            ax.plot(frequency[i], velocity_fitted[i], color='r', linewidth=2,
                    zorder=2)  # plot the line above the errorbar
            ax.annotate(V_labels[i], xy=(0.035, 0.98), xycoords='axes fraction', fontsize=11,
                        xytext=(-2, -2), textcoords='offset points',
                        ha='left', va='top')
            ax.annotate(line_labels[i], xy=(0.035, 0.8), xycoords='axes fraction', fontsize=11,
                        xytext=(-2, -2), textcoords='offset points',
                        ha='left', va='top', color='red')


if __name__ == "__main__":
    now = datetime.now()
    name_clarifier = "_frequency_vs_velocity_RC_pawl_ignoreRC"
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    t_span = [0, 400e-6]
    Fext = 0
    nx, ny = 2, 2
    # latexify(fig_width=6, columns=3)
    fig, axs = setup_plot(nx, ny)

    # Load data
    data = loadmat("../data/frequency_vs_velocity.mat")

    V_values = np.ndarray.flatten(data["V"])
    frequency = []
    velocity_avg = []
    velocity_std = []
    velocity_fitted = []
    V_labels = []
    line_labels = []
    r2_scores_pullin = []
    r2_scores_release = []
    rmse_pullin = []
    rmse_release = []
    for i in range(1, len(V_values) + 1):
        frequency.append(np.ndarray.flatten(data["f{}".format(i)]))
        velocity_avg.append(np.ndarray.flatten(data["t{}".format(i)]))
        velocity_std.append(np.ndarray.flatten(data["dt{}".format(i)]))
        velocity_fitted.append(np.ndarray.flatten(data["t{}_line".format(i)]))
        line_labels.append("Slope = \n" + data["label{}".format(i)][0])
        V_labels.append(str(V_values[i - 1]) + ' V')
        # labels.append(r"L=%0.1f$\mu$m"%(fingerL_values[i - 1]*1e6))

    plot_data(fig, axs, frequency, velocity_avg, velocity_std, velocity_fitted, V_labels, line_labels)

    for i in range(len(V_values)):
        slopes = []
        freq = frequency[i]
        velocity = velocity_avg[i]
        for j in range(1, len(freq)):
            slopes.append((velocity[j] - velocity[0])/(freq[j] - freq[0])*1e3)
        print(V_values[i], ['{:.2f}'.format(s) for s in slopes])

    for i in range(len(V_values)):
        V = V_values[i]

        ##################### Calculate pull-in time t_P #####################
        model = setup_model_pullin()
        u = setup_inputs_RC_pullin(V=V, Fext=Fext)
        sol = sim_gca(model, u, t_span)
        t_P = sol.t_events[0][0]

        ##################### Calculate pull-in time to shuttle t_PT #####################
        model = setup_model_pullin_to_shuttle()
        u = setup_inputs_RC_pullin(V=V, Fext=Fext)
        sol = sim_gca(model, u, t_span)
        t_PT = sol.t_events[0][0]

        ##################### Calculate release time t_R #####################
        model = setup_model_release(V=V, Fext=Fext)
        u = setup_inputs_RC_release(V=0, Fext=Fext)
        sol = sim_gca(model, u, t_span)
        t_R = sol.t_events[0][0]

        ##################### Calculate release time from shuttle t_RT #####################
        model = setup_model_release_from_shuttle(V=V, Fext=Fext)
        u = setup_inputs_RC_release(V=0, Fext=Fext)
        sol = sim_gca(model, u, t_span)
        t_RT = sol.t_events[0][0]

        print(V, t_P*1e6, t_PT*1e6, t_R*1e6, t_RT*1e6)

        ##################### Calculate frequency from conditions #####################
        # Condition 1: having pawl A come in contact before pawl B releases contact (tPT < tRT + 0.25T)
        T1 = 4.*(t_PT - t_RT)  # T1 > 4*(t_PT - t_RT) --> upper bound on frequency
        # Condition 2: having pawl A fully pullin before releasing  its  drive  voltage  (tP<0.75T)
        T2 = t_P/0.75  # T2 > t_P/0.75 --> upper bound on frequency
        # Condition 3: having  pawl  B release  the  shuttle  before  its  drive  voltage  is  toggled  again (tRT<0.25T)
        T3 = 4*t_RT  # T3 > 4*t_RT --> upper bound on frequency
        # Condition 4: having pawl A fully settle before pawl B returns again (tP-tPT<0.5T)
        T4 = 2*(t_P - t_PT)  # T4 > 2*(t_P - t_PT) --> upper bound on frequency

        f1, f2, f3, f4 = 1./T1, 1./T2, 1./T3, 1./T4
        f_max = min(f1, f2, f3, f4)
        f_max_idx = np.argmin([f1, f2, f3, f4])
        print("Max frequencies:", f1, f2, f3, f4, f_max, f_max_idx + 1)
        print(V, i//ny, i%ny)
        axs[i//ny][i%ny].axvline(f_max/1e3, linestyle="--", color="grey")  # convert to kHz

        label = r"$f_{ideal,max}$ = " + "\n{:.1f} kHz".format(f_max/1e3)
        x_frac = f_max/40e3
        axs[i//ny][i%ny].annotate(label, xy=(x_frac+0.03, 0.05), xycoords='axes fraction', fontsize=11,
                                  xytext=(-2, -2), textcoords='offset points',
                                  ha='left', va='bottom')
        axs[i//ny][i%ny].set_yticks(np.arange(0, np.max(velocity_avg[i]) + 0.1, 0.1))

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Velocity (m/s)")

    plt.tight_layout()
    # plt.savefig("../figures/" + timestamp + ".png")
    # plt.savefig("../figures/" + timestamp + ".pdf")
    plt.show()
