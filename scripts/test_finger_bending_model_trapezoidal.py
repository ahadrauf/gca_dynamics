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

if __name__ == '__main__':
    fingerWbases = np.array([5, 6, 7, 8, 9]) * 1e-6
    fingerWtips = np.array([5, 5, 5, 5, 5]) * 1e-6
    fingerLs = np.array([76.5, 76.5, 76.5, 76.5, 76.5]) * 1e-6

    for i in range(len(fingerWbases)):
        fingerWbase, fingerWtip, fingerL = fingerWbases[i], fingerWtips[i], fingerLs[i]
        model = AssemblyGCA(process=SOI())
        model.gca.fingerWbase = fingerWbase - 2 * model.gca.process.undercut
        model.gca.fingerWtip = fingerWtip - 2 * model.gca.process.undercut
        model.gca.fingerL = fingerL - model.gca.process.undercut
        model.gca.update_dependent_variables()
