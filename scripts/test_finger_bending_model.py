from assembly import AssemblyGCA
from process import SOI
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from sklearn.metrics import r2_score
from datetime import datetime
np.set_printoptions(precision=3, suppress=True)


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
    dx_parallelplate = []
    F_parallelplate = []

    V_all_parallelplate = np.arange(20, 115+1, 1)
    V_all_fingerbending = np.arange(30, 90+1, 0.25)
    x = model.gca.x_GCA
    for V in V_all_parallelplate:
        Fes_parallelplate = model.gca.Fes_calc1(x, V)/model.gca.Nfing
        x_parallelplate = Fes_parallelplate/k_fing
        dx_parallelplate.append(x_parallelplate*1e6)
        F_parallelplate.append(Fes_parallelplate*1e6)
    for V in V_all_fingerbending:
        Fes_fingerbending, y = model.gca.Fes_calc2(x, V)
        F_fingerbending.append(Fes_fingerbending*1e6)
        dx_fingerbending.append(y[-1]*1e6)
    # difference = [pp/fb for pp, fb in zip(F_parallelplate, F_fingerbending)]
    # print(np.max(difference), difference)
    print("Model difference", F_parallelplate[-1]/F_fingerbending[-1])

    # print("Parallel plate DC | Parallel Plate | CoventorWare | Finger Bending")
    print("Parallel plate DC:", dx_parallelplate_contreras, F_parallelplate_contreras)
    print("Parallel Plate:", dx_parallelplate, F_parallelplate)
    print("CoventorWare:", dx_coventorware, F_coventorware)
    print("Finger Bending:", dx_fingerbending, F_fingerbending)

    # Calculate rough R2 score
    actual_F = F_coventorware
    pred_F = []
    for actual_dx in dx_coventorware:
        idx = np.argmin(np.abs(np.array(dx_fingerbending) - actual_dx))
        pred_F.append(F_fingerbending[idx])
    print("Actual F", actual_F)
    print("Pred F", pred_F)
    r2 = r2_score(actual_F, pred_F)
    print("R2 score:", r2)

    plt.plot(dx_parallelplate_contreras, F_parallelplate_contreras, 'b-')
    # plt.plot(dx_parallelplate, F_parallelplate, 'b-')
    plt.plot(dx_coventorware, F_coventorware, 'ko--')
    plt.plot(dx_fingerbending, F_fingerbending, 'r-')
    # plt.legend(["Parallel Plate (WPD)", "Parallel Plate", "Coventorware (WPD)", "Finger Bending Model"])
    plt.legend(["Parallel Plate Model", "Coventorware Simulation", "Finger Bending Model"])
    plt.xlabel(r"Finger Tip Deflection ($\mu$m)")
    plt.ylabel(r"Force on Conductor ($\mu$m)")

    plt.tight_layout()
    plt.savefig("../figures/" + timestamp + ".png")
    plt.savefig("../figures/" + timestamp + ".pdf")
    plt.show()
