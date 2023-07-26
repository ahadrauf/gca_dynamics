import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

# plt.rc('font', size=18)  # controls default text size
plt.rc('axes', labelsize=13)  # fontsize of the x and y labels
# plt.rc('xtick', labelsize=16)  # fontsize of the x tick labels
# plt.rc('ytick', labelsize=16)  # fontsize of the y tick labels
# plt.rc('legend', fontsize=16)  # fontsize of the legend
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

file_loc = "../data/simulation_results/"
file_name = "20230623_23_26_26_vacuum_V_fingerL_release"
# file_name = "20230624_00_29_51_vacuum_V_fingerL_pullin"

now = datetime.now()
name_clarifier = "_edited_" + file_name
timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
print(timestamp)

colors = list(mcolors.TABLEAU_COLORS.keys())
print(colors)

data = np.load(file_loc + file_name + ".npy", allow_pickle=True)
save_data_vacuum, save_data_air = data
fingerL_values = sorted(save_data_vacuum.keys())
print(fingerL_values, save_data_vacuum.keys())
print(fingerL_values)

nx, ny = 3, 3
fig, axs = plt.subplots(nx, ny)
for i in range(len(fingerL_values)):
    fingerL = fingerL_values[i]
    V_converged, times_converged = save_data_vacuum[fingerL]
    ax = axs[i // ny, i % ny]
    line_vacuum, = ax.plot(V_converged, times_converged, colors[0])

    V_converged, times_converged = save_data_air[fingerL]
    line_air, = ax.plot(V_converged, times_converged, colors[1])



    label = r"L=%0.1f$\mu$m" % (fingerL_values[i - 1] * 1e6)
    ax.annotate(label, xy=(1, 1), xycoords='axes fraction', fontsize=10, xytext=(-2, -2), textcoords='offset points',
                ha='right', va='top')

axs[2, 1].set_xlabel("Voltage (V)", )
axs[1, 0].set_ylabel(r"Time ($\mu$s)")

if "pullin" in file_name:
    fig.legend([line_vacuum], ['Pull-in Time, Vacuum'], loc='lower left')
    fig.legend([line_air], ['Pull-in Time, Air'], loc='lower right')
else:
    fig.legend([line_vacuum], ['Release Time, Vacuum'], loc='lower left')
    fig.legend([line_air], ['Release Time, Air'], loc='lower right')

plt.tight_layout()
# plt.savefig("../figures/" + timestamp + ".png")
# plt.savefig("../figures/" + timestamp + ".svg")
# plt.savefig("../figures/" + timestamp + ".pdf")
plt.show()
