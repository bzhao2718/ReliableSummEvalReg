from time import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from scoring.wodeutil.nlp.metrics.eval_constants import *
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LassoCV
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel, RFECV
from mlxtend.feature_selection import SequentialFeatureSelector as mlxtend_SFS
from sklearn.feature_selection import SequentialFeatureSelector as sklearn_SFS
from models.models import regr_dict, regr_dict_regularized


def get_features_filter_method(df, corr_type='pearson', use_abs=True, include_labels=False):
    """
    get features sorted by their absolute correlation value
    """
    corr_matrix = df.corr(method=corr_type)
    against = "coherence"
    excludes = ['coherence', 'relevance', 'fluency', 'consistency', 'line_id', 'doc_id']
    if use_abs:
        feature_corrs = corr_matrix[against].abs().sort_values(ascending=False)
    else:
        feature_corrs = corr_matrix[against].sort_values(ascending=False)
    # if not include_labels:
    #     feature_corrs = feature_corrs[~summeval_labels].copy()
    # scatter_matrix(df_abs[attr], figsize=(12, 10))
    # df_mix_path.plot(kind="scatter", x="density", y="coherence")
    # plt.show()


def get_features_mlxtend_sfs(df_metrics, df_label):
    sfs_stats = defaultdict(set)
    num_of_features = 20
    # use_regr = [regr_forest, regr_lin, regr_lasso, regr_grad_boost, regr_adaboost, regr_lin_svr]
    use_regr = []
    subset_list = []
    curr_dict = {regr_forest: regr_dict_regularized[regr_forest],
                 regr_lin: regr_dict_regularized[regr_lin],
                 regr_bagging: regr_dict_regularized[regr_bagging],
                 regr_mlp: regr_dict_regularized[regr_mlp],
                 regr_adaboost: regr_dict_regularized[regr_adaboost],
                 regr_voting: regr_dict_regularized[regr_voting],
                 regr_grad_boost: regr_dict_regularized[regr_grad_boost]
                 }
    # curr_dict = regr_dict_regularized
    for regr_name, regressor in curr_dict.items():
        if regr_name in use_regr or len(
                use_regr) == 0:  # (set use_regr to empty lsit) len(use_regr) ==0 when use all regressors
            sfs = mlxtend_SFS(regressor,
                              k_features=num_of_features,
                              forward=True,
                              floating=False,
                              scoring='r2',
                              cv=5,
                              n_jobs=-1)
            sfs = sfs.fit(df_metrics, df_label)
            features = set(sfs.k_feature_names_)
            sfs_stats[regr_name] = features
            subset_list.append(sfs.subsets_)
    print(sfs_stats)


def min_max_scale(df_metrics, save_to):
    scaler = MinMaxScaler()
    mix_minmax = scaler.fit_transform(df_metrics)
    df_min_minmax = pd.DataFrame(mix_minmax, columns=df_metrics.columns)
    print(f"len of df_min_minmax: {len(df_min_minmax)}")
    df_min_minmax.to_csv(save_to, index=False)


def get_selected_features(df_metrics, df_label):
    lasso = LassoCV().fit(df_metrics, df_label)
    importance = np.abs(lasso.coef_)
    features_names = np.array(df_metrics.columns.values.tolist())
    print(f"feature names: {features_names}")
    print(f"and their importance: {importance} with len {len(importance)}")

    threshold = np.sort(importance)[-1] + 0.0001
    tic = time()
    sfm = SelectFromModel(lasso, threshold=threshold).fit(df_metrics, df_label)
    toc = time()
    print(f"features seleted by SelectFromModel: {features_names[sfm.get_support()]}")
    print(f"done in {toc - tic:.3f}s")


def sequential_feature_selector(df_metrics, df_label):
    forest = RandomForestRegressor()
    lasso = LassoCV().fit(df_metrics, df_label)
    # importance = np.abs(lasso.coef_)
    features_names = np.array(df_metrics.columns.values.tolist())
    sfs_fw = sklearn_SFS(forest, n_features_to_select=10, direction='forward', n_jobs=-1).fit(df_metrics, df_label)
    print(f"features selected by sfs_fw: {features_names[sfs_fw.get_support()]}")
    # sfs_bw = SequentialFeatureSelector(lasso, n_features_to_select=30, direction='backward').fit(df_metrics, df_label)
    # print(f"features selected sfs_bw: {features_names[sfs_bw.get_support()]}")


def RFECV_feature_selection(df_metrics, df_label):
    forest = RandomForestRegressor(n_jobs=-1)
    forest = SVR(kernel='linear')
    rfecv = RFECV(estimator=forest, step=1, n_jobs=-1, min_features_to_select=5, cv=5)
    rfecv.fit(df_metrics, df_label)
    print(f"optimal number of eatures: {rfecv.n_features_}")

    print(f"scores: {rfecv.grid_scores_}")
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    plt.plot(range(5,
                   len(rfecv.grid_scores_) + 5),
             rfecv.grid_scores_)
    plt.show()


if __name__ == '__main__':
    get_features_filter_method(include_labels=True)
