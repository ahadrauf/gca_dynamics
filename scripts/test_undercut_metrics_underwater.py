import numpy as np
from scipy.io import loadmat
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error as mape

# R2 scores fingerL
data = np.load("../data/20210909_01_41_42_V_fingerL_pullin_undercut_underwater_Fes=v2_Fb=v2.npy", allow_pickle=True)
# data = np.load("../data/20210909_13_00_35_V_fingerL_pullin_undercut_underwater_Fes=v2_Fb=v2.npy", allow_pickle=True)
[fingerL_values, pullin_V, pullin_avg, pullin_std, pullin_V_results, pullin_t_results, r2_scores_pullin, rmse_pullin, best_undercut_pullin, fig] = data
undercut_range = np.arange(0.3e-6, 0.501e-6, 0.01e-6)
# undercut_range = np.append(np.arange(0.33e-6, 0.3801e-6, 0.0025e-6),
#                                np.arange(0.45e-6, 0.481e-6, 0.0025e-6))
t_span = [0, 40e-3]

pullin_V_results_agg = {uc: [] for uc in undercut_range}
pullin_t_results_agg = {uc: [] for uc in undercut_range}
r2_scores_pullin_agg = {uc: [] for uc in undercut_range}
rmse_scores_pullin_agg = {uc: [] for uc in undercut_range}
mape_scores_pullin_agg = {uc: [] for uc in undercut_range}

def update_sim_data(V_sim, t_sim, V_source, t_source):
    V_sim, V_source = np.copy(V_sim), np.copy(V_source)
    t_sim, t_source = np.copy(t_sim), np.copy(t_source)

    for V in V_source:
        if not np.any(np.isclose(V, V_sim)):
            V_sim = np.append(V_sim, V)
            t_sim = np.append(t_sim, t_span[-1]*1e3)
    idx_sorted = np.argsort(V_sim)
    V_sim = list(np.array(V_sim)[idx_sorted])
    t_sim = list(np.array(t_sim)[idx_sorted])

    # Calculate the r2 score
    actual = []
    pred = []
    for V in V_sim:
        if V in V_source[:5]:
            idx = np.where(V_source == V)[0][0]
            actual.append(t_source[idx])
            idx = np.where(V_sim == V)[0][0]
            pred.append(t_sim[idx])
    r2 = r2_score(actual, pred)
    rmse = mean_squared_error(actual, pred, squared=False)
    MAPE = mape(actual, pred)
    print(MAPE, np.abs(np.divide(np.subtract(actual, pred), actual)))
    return V_sim, t_sim, r2, rmse, MAPE

for uc in undercut_range:
    print(uc)
    for i in range(len(fingerL_values)):
        fingerL = fingerL_values[i]
        idx_sorted = np.argsort(pullin_V[i])
        pullin_V[i] = list(np.array(pullin_V[i])[idx_sorted])
        pullin_avg[i] = list(np.array(pullin_avg[i])[idx_sorted])
        V_updated, t_updated, r2, rmse, MAPE = update_sim_data(pullin_V_results[uc][i], pullin_t_results[uc][i],
                                                         pullin_V[i], pullin_avg[i])
        pullin_V_results_agg[uc].append(V_updated)
        pullin_t_results_agg[uc].append(t_updated)
        r2_scores_pullin_agg[uc].append(r2)
        rmse_scores_pullin_agg[uc].append(rmse)
        mape_scores_pullin_agg[uc].append(MAPE)

print(pullin_V_results)
print(pullin_V_results_agg)
print(pullin_t_results)
print(pullin_t_results_agg)
print("R2 pullin orig", r2_scores_pullin)
print("R2 pullin modd", r2_scores_pullin_agg)
print(mape_scores_pullin_agg)

# import sys
# sys.exit(0)
r2_scores_pullin = r2_scores_pullin_agg
rmse_scores_pullin = rmse_scores_pullin_agg
mape_scores_pullin = mape_scores_pullin_agg

def calc_r2(undercuts):
    r2_pullin = [r2_scores_pullin[uc][idy] for idy, uc in enumerate(undercuts)]
    print("R2 Pullin:", r2_pullin, np.mean(r2_pullin))
    rmse_pullin = [rmse_scores_pullin[uc][idy] for idy, uc in enumerate(undercuts)]
    print("RMSE Pullin:", rmse_pullin, np.mean(rmse_pullin))
    mape_pullin = [mape_scores_pullin[uc][idy] for idy, uc in enumerate(undercuts)]
    print("MAPE Pullin:", mape_pullin, np.mean(mape_pullin))

#########################################################################################################
# R2 Metrics
#########################################################################################################
best_uc_pullin_agg = []
best_uc_release_agg = []
best_uc_avg_agg = []
best_uc_min_agg = []
for idy in range(len(r2_scores_pullin[undercut_range[0]])):
    # best_uc_pullin = undercut_range[np.argmax([r2_scores_pullin[uc][idy]
    #                                         for uc in undercut_range])]
    best_uc_pullin = undercut_range[np.argmin([mape_scores_pullin[uc][idy]
                                               for uc in undercut_range])]
    best_uc_pullin_agg.append(best_uc_pullin)
    print(idy, "|", best_uc_pullin, r2_scores_pullin[best_uc_pullin][idy])

print("MAPE metrics")
print("Best UC Pullin:", best_uc_pullin_agg)
calc_r2(best_uc_pullin_agg)

#########################################################################################################
# RMSE Metrics
#########################################################################################################
best_uc_pullin_agg = []
best_uc_release_agg = []
best_uc_avg_agg = []
best_uc_min_agg = []
for idy in range(len(r2_scores_pullin[undercut_range[0]])):
    best_uc_pullin = undercut_range[np.argmin([rmse_scores_pullin[uc][idy]
                                            for uc in undercut_range])]
    best_uc_pullin_agg.append(best_uc_pullin)
    print(idy, "|", best_uc_pullin, rmse_scores_pullin[best_uc_pullin][idy], mape_scores_pullin_agg[best_uc_pullin][idy])

print("RMSE metrics")
print("Best UC Pullin:", best_uc_pullin_agg)
calc_r2(best_uc_pullin_agg)
