from assembly import AssemblyGCA
from process import SOI
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from sklearn.metrics import r2_score
from datetime import datetime
np.set_printoptions(precision=3, suppress=True)
plt.rc('font', size=11)


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

    x = model.gca.x_GCA
    V = 70
    Fes_fingerbending, y, _ = model.gca.Fes_calc2(x, V)
    print(Fes_fingerbending)
    print(y)
    plt.plot(np.linspace(0, 1, np.size(y)), y)
    # print(sol.sol(0.))
    # print(sol.sol(model.gca.fingerL_total))
    # print(sol.sol([0., model.gca.fingerL_total])[0])
    # print(sol.x)
    # print(sol.y)
    # print(sol.rms_residuals)
    # print(sol.success)
    Fes_fingerbending, y, _ = model.gca.Fes_calc4(x, V)
    print(Fes_fingerbending)
    print(y)
    plt.plot(np.linspace(0, 1, np.size(y)), y)
    # plt.plot(sol.x, sol.y[0])
    # # plt.ylim([0, 1e-6])
    # print(sol.x)
    # print(sol.y)
    # print(sol.rms_residuals)
    # print(sol.success)
    plt.legend(["Calc 2", "Calc 4"])
    plt.show()
