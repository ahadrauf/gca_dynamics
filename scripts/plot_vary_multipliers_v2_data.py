import sys
sys.path.append(r"C:\Users\ahadrauf\Desktop\Research\Pister\gca_dynamics")

from assembly import AssemblyGCA
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from datetime import datetime
from process import *
import time
plt.rc('font', size=11.5)
plt.rcParams['legend.fontsize'] = 11.5


def setup_plot(len_x, len_y, plt_title=None, x_label="", y_label=""):
    fig, axs = plt.subplots(len_x, len_y)
    if plt_title is not None:
        fig.suptitle(plt_title)

    if x_label or y_label:
        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
    return fig, axs

def display_stats(x_converged, times_converged, label, nom):
    # Get helpful stats for paper
    # print(label, x_converged, times_converged)
    try:
        time_50_up = times_converged[np.where(np.isclose(x_converged, nom*1.25e6))[0][0]]
        time_nominal = times_converged[np.where(np.isclose(x_converged, nom*1.e6))[0][0]]
        time_50_down = times_converged[np.where(np.isclose(x_converged, nom*0.8e6))[0][0]]
        print("{}: con=0.75: {} (Ratio: {}), con=1: {}, con=1.25: {} (Ratio: {})".format(
            label, time_50_down, 1 - time_50_down/time_nominal, time_nominal, time_50_up, 1 - time_50_up/time_nominal
        ))
    except Exception as e:
        print(e)

now = datetime.now()
name_clarifier = "_20211024_01_37_55_vary_multipliers_undercut=0.400_Fes=v2_Fb=v2_modified"
timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
print(timestamp)

filename = "../data/20211024_01_37_55_vary_multipliers_undercut=0.400_Fes=v2_Fb=v2.npy"
fileData = np.load(filename, allow_pickle=True)
process, fingerLcon_range, fingerWcon_range, gfcon_range, supportWcon_range, fingerLnom, fingerWnom, gfnom, supportWnom, data, fig = fileData
plt.close()

fig, axs = setup_plot(2, 2, y_label=r"Time ($\mu$s)")


### fingerL
x_converged, times_converged = data["fingerLpullin"]
x_converged = np.array(x_converged)
times_converged = np.array(times_converged)
idx = np.where(x_converged < 95)
x_converged = x_converged[idx]
times_converged = times_converged[idx]
axs[0, 0].plot(x_converged, times_converged, 'b')
display_stats(x_converged, times_converged, "Pullin fingerLcon", fingerLnom)

x_converged, times_converged = data["fingerLrelease"]
x_converged = np.array(x_converged)
times_converged = np.array(times_converged)
idx = np.where(x_converged < 95)
x_converged = x_converged[idx]
times_converged = times_converged[idx]
axs[0, 0].plot(x_converged, times_converged, 'r')
axs[0, 0].set_title(r"Varying $L_{ol}$", fontsize=12)
axs[0, 0].axvline(fingerLnom*1e6, color='k', linestyle='--')
display_stats(x_converged, times_converged, "Release fingerLcon", fingerLnom)
label = r"$L_{ol}=$" + "{:0.1f}".format(fingerLnom*1e6) + r' $\mu$m'
axs[0, 0].annotate(label, xy=(0.73, 0.96), xycoords='axes fraction', color='k',
                   xytext=(0, 0), textcoords='offset points', ha='right', va='top')
axs[0, 0].set_xlabel(r'$L_{ol}  (\mu$m)')

### fingerW
x_converged, times_converged = data["fingerWpullin"]
x_converged = np.array(x_converged)
times_converged = np.array(times_converged)
# idx = np.where(x_converged > 3.425)
idx = np.where(np.array(x_converged) > 3.5)
# idx = np.where(x_converged >= 4.)
x_converged = x_converged[idx]
times_converged = times_converged[idx]
axs[0, 1].plot(x_converged, times_converged, 'b')
display_stats(x_converged, times_converged, "Pullin fingerWcon", fingerWnom)

x_converged, times_converged = data["fingerWrelease"]
x_converged = np.array(x_converged)
times_converged = np.array(times_converged)
# idx = np.where(np.array(x_converged) > 3.425)
idx = np.where(np.array(x_converged) > 3.5)
# idx = np.where(x_converged >= 4.)
x_converged = x_converged[idx]
times_converged = times_converged[idx]
axs[0, 1].plot(x_converged, times_converged, 'r')
axs[0, 1].set_title(r"Varying $w_f$", fontsize=12)
axs[0, 1].axvline(fingerWnom*1e6, color='k', linestyle='--')
display_stats(x_converged, times_converged, "Release fingerWcon", fingerWnom)
label = r"$w_f=$" + "{:0.1f}".format(fingerWnom*1e6) + r' $\mu$m'
axs[0, 1].annotate(label, xy=(0.29, 0.6), xycoords='axes fraction', color='k',
                   xytext=(0, 0), textcoords='offset points', ha='left', va='top')
axs[0, 1].set_xlabel(r'$w_f  (\mu$m)')

### gf0
x_converged, times_converged = data["gfpullin"]
axs[1, 0].plot(x_converged, times_converged, 'b')
display_stats(x_converged, times_converged, "Pullin gf0con", gfnom)
max_x = max(x_converged)

x_converged, times_converged = data["gfrelease"]
x_converged = np.array(x_converged)
times_converged = np.array(times_converged)
idx = np.where(np.array(x_converged) < max_x)
x_converged = x_converged[idx]
times_converged = times_converged[idx]
axs[1, 0].plot(x_converged, times_converged, 'r')
axs[1, 0].set_title(r"Varying $x_0$", fontsize=12)
axs[1, 0].axvline(gfnom*1e6, color='k', linestyle='--')
display_stats(x_converged, times_converged, "Release gf0con", gfnom)
label = r"$x_0=$" + "\n" + "{:0.2f}".format(gfnom*1e6) + r' $\mu$m'
axs[1, 0].annotate(label, xy=(0.57, 0.96), xycoords='axes fraction', color='k',
                   xytext=(0, 0), textcoords='offset points', ha='left', va='top')
axs[1, 0].set_xlabel(r'$x_0  (\mu$m)')

### supportW
x_converged, times_converged = data["supportWpullin"]
axs[1, 1].plot(x_converged, times_converged, 'b')
display_stats(x_converged, times_converged, "Pullin supportWcon", supportWnom)

x_converged, times_converged = data["supportWrelease"]
axs[1, 1].plot(x_converged, times_converged, 'r')
axs[1, 1].set_title(r"Varying $w_{spr}$", fontsize=12)
axs[1, 1].axvline(supportWnom*1e6, color='k', linestyle='--')
display_stats(x_converged, times_converged, "Release supportWcon", supportWnom)
label = r"$w_{spr}=$" + "{:0.1f}".format(supportWnom*1e6) + r' $\mu$m'
axs[1, 1].annotate(label, xy=(0.47, 0.96), xycoords='axes fraction', color='k',
                   xytext=(0, 0), textcoords='offset points', ha='left', va='top')
axs[1, 1].set_xlabel(r'$w_{spr}  (\mu$m)')
axs[1, 1].legend(["Pull-in", "Release"], loc='center right')

plt.tight_layout()
timestamp = "20211024_01_37_55_vary_multipliers_undercut=0.400_Fes=v2_Fb=v2_modified_v4"
# plt.savefig("../figures/" + timestamp + ".png")
# plt.savefig("../figures/" + timestamp + ".pdf")
plt.show()


