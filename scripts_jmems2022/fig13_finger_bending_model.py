"""
Compare various finger bending metrics to data from a CoventorWare simulation
Data for CoventorWare stored in /data/fingertip_deflection.mat.
"""

from assembly import AssemblyGCA
from process import SOI
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from sklearn.metrics import r2_score
from datetime import datetime
np.set_printoptions(precision=3, suppress=True)
plt.rc('font', size=12)


if __name__ == '__main__':
    now = datetime.now()
    name_clarifier = "_test_finger_bending_model_with_fringefieldmultiplier"
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    data = loadmat("../data/fingertip_deflection.mat")
    dx_coventorware = np.ndarray.flatten(data["dx_coventorware"])
    F_coventorware = np.ndarray.flatten(data["F_coventorware"])
    dx_parallelplate_contreras = np.ndarray.flatten(data["dx_parallelplate_contreras"])  # np.array([0.0331846184498904, 0.3212818489738992])
    F_parallelplate_contreras = np.ndarray.flatten(data["F_parallelplate_contreras"])  # np.array([18.101327043662167, 174.954732673056])

    model = AssemblyGCA(drawn_dimensions_filename="../layouts/coventorware_finger_bending_model.csv", process=SOI())
    I_fing = (model.gca.fingerW**3)*model.gca.process.t_SOI/12
    k_fing = 8*model.gca.process.E*I_fing/(model.gca.fingerL_total**3)
    dx_fingerbending = []
    F_fingerbending = []
    dx_fingerbending_twosided = []
    F_fingerbending_twosided = []
    dx_fingerbending_numerical = []
    F_fingerbending_numerical = []
    dx_parallelplate = []
    F_parallelplate = []
    dx_fingerbending_trapezoidal = []
    F_fingerbending_trapezoidal = []

    V_all_parallelplate = np.arange(20, 115+1, 1)
    V_all_fingerbending = np.arange(30, 90+0.25, 1)
    x = model.gca.x_GCA
    for V in V_all_parallelplate:
        Fes_parallelplate = model.gca.Fes_calc1(x, V)[0]/model.gca.Nfing
        x_parallelplate = Fes_parallelplate/k_fing
        dx_parallelplate.append(x_parallelplate*1e6)
        F_parallelplate.append(Fes_parallelplate*1e6)
    for V in V_all_fingerbending:
        Fes_fingerbending, y, U = model.gca.Fes_calc2(x, V)
        F_fingerbending.append(Fes_fingerbending*1e6)
        dx_fingerbending.append(y[-1]*1e6)

        Fes_fingerbending, y, _ = model.gca.Fes_calc3(x, V)
        F_fingerbending_twosided.append(Fes_fingerbending*1e6)
        dx_fingerbending_twosided.append(y[-1]*1e6)

        Fes_fingerbending, y, _ = model.gca.Fes_calc4(x, V)
        if 0 <= y[-1] < 0.335e-6:
            F_fingerbending_numerical.append(Fes_fingerbending*1e6)
            dx_fingerbending_numerical.append(y[-1]*1e6)

    print("Model difference", F_parallelplate[-1]/F_fingerbending[-1])

    print("Parallel plate DC:", dx_parallelplate_contreras, F_parallelplate_contreras)
    print("Parallel Plate:", dx_parallelplate, F_parallelplate)
    print("CoventorWare:", dx_coventorware, F_coventorware)
    print("Finger Bending:", dx_fingerbending, F_fingerbending)
    print("Finger Bending Two Sided:", dx_fingerbending_twosided, F_fingerbending_twosided)

    # Calculate R2 score
    actual_F = F_coventorware
    pred_F = []
    for actual_dx in dx_coventorware:
        idx = np.argmin(np.abs(np.array(dx_fingerbending) - actual_dx))
        pred_F.append(F_fingerbending[idx])
    print("Actual F", actual_F)
    print("Pred F", pred_F)
    r2 = r2_score(actual_F, pred_F)
    print("R2 score:", r2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.plot(dx_parallelplate_contreras, F_parallelplate_contreras, 'b-')
    plt.plot(dx_parallelplate, F_parallelplate, 'b-')
    plt.plot(dx_coventorware, F_coventorware, 'ko--')

    V_coventorware = np.linspace(20, 90, 8)
    # for i in range(len(dx_coventorware)):
    #     V = V_coventorware[i]
    #     label = "{} V".format(int(V))
    #     if i < 3:
    #         ax.annotate(label, xy=(dx_coventorware[i], F_coventorware[i]), color='k',
    #                  xytext=(15, 0), textcoords='offset points', ha='left', va='center')
    #     else:
    #         ax.annotate(label, xy=(dx_coventorware[i], F_coventorware[i]), color='k',
    #                     xytext=(0, -10), textcoords='offset points', ha='center', va='top')

    plt.plot(dx_fingerbending, F_fingerbending, 'r-')
    plt.plot(dx_fingerbending_twosided, F_fingerbending_twosided, color='purple') #, marker='o')
    plt.plot(dx_fingerbending_numerical, F_fingerbending_numerical, 'g-')
    plt.plot(dx_fingerbending_trapezoidal, F_fingerbending_trapezoidal, 'orange')
    plt.legend(["Parallel Plate Model", "CoventorWare Simulation", "Finger Bending Model",
                "Two-Sided Finger Bending Model", "Numerical Integration of Eq. 20"])
    plt.xlabel(r"Finger Tip Deflection ($\mu$m)")
    plt.ylabel(r"Force on Conductor ($\mu$N)")

    plt.tight_layout()
    # plt.savefig("../figures/" + timestamp + ".png")
    # plt.savefig("../figures/" + timestamp + ".pdf")
    plt.show()
