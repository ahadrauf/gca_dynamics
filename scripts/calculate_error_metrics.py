import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score


if __name__ == '__main__':
    now = datetime.now()
    Fes_calc_method, Fb_calc_method = 2, 2
    # name_clarifier = "_V_supportW_pullin_release_undercut=custom_Fes=v{}_Fb=v{}_modified_20210903_23_08_34_20210904_00_33_13".format(Fes_calc_method, Fb_calc_method)
    name_clarifier = "_V_supportW_pullin_release_undercut=custom_Fes=v{}_Fb=v{}_modified_20210903_23_08_34".format(Fes_calc_method, Fb_calc_method)
    timestamp = now.strftime("%Y%m%d_%H_%M_%S") + name_clarifier
    print(timestamp)

    # filename = "../data/20210903_23_08_34_V_fingerL_pullin_release_undercut=padded_uc_min_Fes=v2_Fb=v2.npy"
    filename = "../data/20210909_02_04_42_V_supportW_pullin_release_undercut=custom_r2_min_fixedtmax_Fes=v2_Fb=v2.npy"
    # filename = "../data/20210909_21_27_22_V_fingerL_pullin_underwater.npy"
    data = np.load(filename, allow_pickle=True)
    process, supportW_values, pullin_V, pullin_avg, pullin_std, release_V, release_avg, release_std, pullin_V_results, pullin_t_results, release_V_results, release_t_results, r2_scores_pullin, r2_scores_release, rmse_pullin, rmse_release, fig = data
    # process, supportW_values, pullin_V, pullin_avg, pullin_std, pullin_V_results, pullin_t_results, r2_scores_pullin, rmse_pullin, fig = data
    plt.close()

    r2_scores_pullin_v2 = []
    r2_scores_release_v2 = []
    mape_pullin = []
    mape_release = []
    for idy in range(len(supportW_values)):
        actual = []
        pred = []
        for V in pullin_V_results[idy]:
            if V in pullin_V[idy]:
                idx = np.where(pullin_V[idy] == V)[0][0]
                actual.append(pullin_avg[idy][idx])
                idx = np.where(pullin_V_results[idy] == V)[0][0]
                pred.append(pullin_t_results[idy][idx])
        r2_scores_pullin_v2.append(r2_score(actual, pred))
        mape_pullin.append(mape(actual, pred))

        actual = []
        pred = []
        for V in release_V_results[idy]:
            if V in release_V[idy]:
                idx = np.where(release_V[idy] == V)[0][0]
                actual.append(release_avg[idy][idx])
                idx = np.where(release_V_results[idy] == V)[0][0]
                pred.append(release_t_results[idy][idx])
        r2_scores_release_v2.append(r2_score(actual, pred))
        mape_release.append(mape(actual, pred))

        print(idy)
        print([float("{:0.3f}".format(x)) for x in actual])
        print([float("{:0.3f}".format(x)) for x in pred])

    print("R2 pullin", r2_scores_pullin, np.mean(r2_scores_pullin))
    print("R2 release", r2_scores_release, np.mean(r2_scores_release))
    print("R2 pullin v2", r2_scores_pullin_v2, np.mean(r2_scores_pullin_v2))
    print("R2 release v2", r2_scores_release_v2, np.mean(r2_scores_release_v2))
    print("RMSE pullin", rmse_pullin, np.mean(rmse_pullin))
    print("RMSE release", rmse_release, np.mean(rmse_release))
    print("MAPE pullin", mape_pullin, np.mean(mape_pullin))
    print("MAPE release", mape_release, np.mean(mape_release))
