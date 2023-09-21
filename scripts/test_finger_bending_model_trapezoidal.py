import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
from assembly import AssemblyGCA
from process import *
from scipy.integrate import solve_ivp

# plt.rc('font', size=18)  # controls default text size
plt.rc('axes', labelsize=13)  # fontsize of the x and y labels
# plt.rc('xtick', labelsize=16)  # fontsize of the x tick labels
# plt.rc('ytick', labelsize=16)  # fontsize of the y tick labels
# plt.rc('legend', fontsize=16)  # fontsize of the legend
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

colors = list(mcolors.TABLEAU_COLORS.keys())  # generally, colorblind safe

if __name__ == '__main__':
    now = datetime.now()
    name_clarifier = "_test_finger_bending_model_trapzeoidal"
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    fingerWbases = np.array([5, 6, 7, 8, 9]) * 1e-6
    fingerWtips = np.array([5, 5, 5, 5, 5]) * 1e-6
    fingerLs = np.array([76.5, 76.5, 76.5, 76.5, 76.5]) * 1e-6
    x=3.83e-6
    V = 100

    fig, ax = plt.subplots(1, 1)
    for i in range(len(fingerWbases)):
        fingerWbase, fingerWtip, fingerL = fingerWbases[i], fingerWtips[i], fingerLs[i]
        model = AssemblyGCA(process=SOI())
        model.gca.fingerWbase = fingerWbase - 2 * model.gca.process.undercut
        model.gca.fingerWtip = fingerWtip - 2 * model.gca.process.undercut
        model.gca.fingerW = model.gca.fingerWbase  # (model.gca.fingerWbase + model.gca.fingerWtip) / 2
        model.gca.fingerL = fingerL - model.gca.process.undercut
        model.gca.update_dependent_variables()

        F, y, _ = model.gca.Fes_calc_trapezoidalfinger(x=x, V=V)
        xi_range = np.linspace(0., 1., np.size(y))
        label = r"$w_{f, base}$=" + "{:.1f}um".format(model.gca.fingerWbase*1e6) + \
                r", $w_{f, tip}$" + "={:.1f}um".format(model.gca.fingerWtip*1e6)
        ax.plot(xi_range, y, c=colors[i], ls='-', label=label)
        print('wf_base = {}, wf_tip = {}, Trapezoidal Finger'.format(fingerWbase * 1e6, fingerWtip * 1e6), y)

        F, y, _ = model.gca.Fes_calc3(x=x, V=V)
        xi_range = np.linspace(0., 1., np.size(y))
        # label = "wf_base={}um, wf_tip={}um".format(fingerWbase * 1e6, fingerWtip * 1e6)
        label = r"Straight Finger, $w_f=w_{f,base}$" if i == len(fingerWbases) - 1 else ""
        ax.plot(xi_range, y, c=colors[i], ls='--', label=label)
        print('wf_base = {}, wf_tip = {}, Numerical Integration'.format(fingerWbase*1e6, fingerWtip*1e6), y)

    ax.legend()
    ax.set_xlabel(r"$\tilde{\xi}$ (0 = Finger Base, 1 = Finger Tip) [-]")
    ax.set_ylabel("y (Deflection of Finger) [m]")
    ax.set_title("Deflection of Trapezoidal Finger vs. Straight Finger\nV = {}V, x = {:.2f}um".format(V, x*1e6))
    ax.grid(True)
    fig.tight_layout()

    fig2, ax2 = plt.subplots(1, 1)
    V_range = np.arange(30, 101, 5)
    x = 3.83e-6
    for i in range(len(fingerWbases)):
        fingerWbase, fingerWtip, fingerL = fingerWbases[i], fingerWtips[i], fingerLs[i]
        model = AssemblyGCA(process=SOI())
        model.gca.fingerWbase = fingerWbase - 2 * model.gca.process.undercut
        model.gca.fingerWtip = fingerWtip - 2 * model.gca.process.undercut
        model.gca.fingerW = model.gca.fingerWbase  # (model.gca.fingerWbase + model.gca.fingerWtip) / 2
        model.gca.fingerL = fingerL - model.gca.process.undercut
        model.gca.update_dependent_variables()

        Fs_trapezoidal = []
        Fs_straight = []
        for V in V_range:
            F, y, _ = model.gca.Fes_calc_trapezoidalfinger(x=x, V=V)
            # xi_range = np.linspace(0., 1., np.size(y))
            # print('wf_base = {}, wf_tip = {}, Trapezoidal Finger'.format(fingerWbase * 1e6, fingerWtip * 1e6), y)
            Fs_trapezoidal.append(F)

            F, y, _ = model.gca.Fes_calc3(x=x, V=V)
            # xi_range = np.linspace(0., 1., np.size(y))
            # label = "wf_base={}um, wf_tip={}um".format(fingerWbase * 1e6, fingerWtip * 1e6)
            # print('wf_base = {}, wf_tip = {}, Numerical Integration'.format(fingerWbase*1e6, fingerWtip*1e6), y)
            Fs_straight.append(F)

        label = r"$w_{f, base}$=" + "{:.1f}um".format(model.gca.fingerWbase * 1e6) + \
                r", $w_{f, tip}$" + "={:.1f}um".format(model.gca.fingerWtip * 1e6)
        ax2.plot(V_range, Fs_trapezoidal, c=colors[i], ls='-', label=label)
        label = r"Straight Finger, $w_f=w_{f,base}$" if i == len(fingerWbases) - 1 else ""
        ax2.plot(V_range, Fs_straight, c=colors[i], ls='--', label=label)

        print('wf_base = {}, wf_tip = {}, Trapezoidal Finger'.format(fingerWbase * 1e6, fingerWtip * 1e6), Fs_trapezoidal)
        print('wf_base = {}, wf_tip = {}, Numerical Integration'.format(fingerWbase * 1e6, fingerWtip * 1e6), Fs_straight)


    ax2.legend()
    ax2.set_xlabel(r"V [V]")
    ax2.set_ylabel("Force Between Fingers [N]")
    ax2.set_title("Fes of Trapezoidal Finger vs. Straight Finger\nx = {:.2f}um".format(x*1e6))
    ax2.grid(True)
    fig2.tight_layout()

    # plt.savefig("../figures/" + timestamp + ".png")
    # plt.savefig("../figures/" + timestamp + ".pdf")
    plt.show()