import argparse
import sys

from scoring.wodeutil.nlp.metrics.eval_constants import *

import os
import pandas as pd
from models.models import load_regress_model
from feature_extract import get_features_as_df


def add_regr_stats(regr_stats, mse=None, rmse=None, corr_type=None, corr=None, regr_name="", syst_type="",
                   mask_pval=None, pval_threshold=0.05, use_cv=False, corr_key=None):
    if not regr_stats is None:
        corr_value = corr if use_cv else corr[0]  # (corr, p_value)
        if not use_cv:
            pval = corr[1]
        if corr_key is None:
            corr_key = f"{syst_type}_{regr_name}_{corr_type}"
        # mse_key = f"{syst_type}_{regr_name}_mse"
        # rmse_key = f"{syst_type}_{regr_name}_rmse"
        if mask_pval:
            corr_value = '-' if pval >= pval_threshold else corr_value
            # print(f"regr_name: {regr_name}, corr_type: {corr_type}, syst_type: {syst_type}, pval: {pval}")
        regr_stats[corr_key].append(corr_value)
        # regr_stats[mse_key].append(mse)
        # regr_stats[rmse_key].append(rmse)


def add_stats_to_df(regr_stats=None, save_path=None):
    if not regr_stats is None and save_path:
        stats = pd.DataFrame(regr_stats)
        if save_path:
            stats.to_csv(save_path, index=True)
            print(f"-------- save stats to {save_path} ----------")


def predict_score(df_features, regr_name=regr_nn_keras, col_name="Filter20", is_nn=False,
                  save_to=None):
    print(f"................... {regr_name}.....................")
    if regr_name == regr_nn_keras:
        regr_name = "NNReg"
    regr_type = regr_name + "_" + col_name
    regrsor = load_regress_model(regr_type=regr_type, is_keras_nn=is_nn)
    pred = regrsor.predict(df_features)
    df_pred = pd.DataFrame(pred, columns=['pred'])
    if save_to:
        df_pred.to_csv(save_to, index=False)


def default_args(parser, feature_combine=20, file_path=None):
    feature20 = "rouge,bert_score, mover_score,meteor,bleu,rouge_we,stats,sms"
    feature20 = "bleu,chrf,meteor,cider,bert_score, mover_score,stats,sms"
    feature5 = "stats"
    if feature_combine == 20:
        metrics = feature20
    else:
        metrics = feature5
    parser.add_argument('--config-file', type=str,
                        default="scoring/SummEval/evaluation/examples/basic.config",
                        help='config file with metric parameters')

    parser.add_argument('--metrics', type=str, default=metrics, help='comma-separated string of metrics')
    parser.add_argument('--aggregate', type=bool, help='whether to aggregate scores')
    # parser.add_argument('--jsonl-file', default="scoring/SummEval/external/data_annotations/model_annotations.aligned.paired.jsonl", type=str, help='input jsonl file to score')
    parser.add_argument('--jsonl-file',
                        default=file_path,
                        type=str, help='input jsonl file to score')

    parser.add_argument('--article-file', type=str, help='input article file')
    parser.add_argument('--save_to', type=str, help='save predictions to')
    parser.add_argument('--feature_combine', type=int, help='number of features (20 or 5 for Filter Method)')
    parser.add_argument('--summ-file', type=str, help='input summary file')
    parser.add_argument('--ref-file', type=str, help='input reference file')
    parser.add_argument('--output-file', default="out", type=str, help='output file')
    parser.add_argument('--eos', type=str, help='EOS for ROUGE (if reference not supplied as list)')
    args = parser.parse_args()
    return args


def evaluate(summary_path, feature_combine=20, save_to=None):
    """
        1. add path info and score calculation specs
        2. pass the info to Summeval library and calculate the scores
        3. save the scores and convert it to a dataframe object df_features
        4. load the model and pass df_features and predict the scores
        5. save the result specified to save_to
    """
    parser = argparse.ArgumentParser(description="predictor")
    args = default_args(parser=parser, file_path=summary_path, feature_combine=5)
    print(f"----------- Extract features ----------------")
    df_features = get_features_as_df(args)
    if feature_combine == 20:
        cols = filter_coherence_features[:20]
        col_name = "Filter20"
    elif feature_combine == 5:
        cols = filter_coherence_features[:5]
        col_name = "Filter5"
    df_features = df_features.loc[:, cols].copy()
    print(f"----------- Predict scores ----------------")
    predict_score(df_features=df_features, save_to=save_to, is_nn=True, col_name=col_name)
    print("------------- Prediction Done ---------------")


def evaluate_from_path():
    """
        1. add path info and score calculation specs from args
        2. pass the info to Summeval library and calculate the scores
        3. save the scores and convert it to a dataframe object df_features
        4. load the model and pass df_features and predict the scores
        5. save the result specified to save_to
    """
    parser = argparse.ArgumentParser(description="predictor")
    args = default_args(parser=parser, feature_combine=20)
    print(f"----------- Extract features ----------------")
    df_features = get_features_as_df(args)
    feature_combine = args.feature_combine
    if feature_combine == 20:
        cols = filter_coherence_features[:20]
        col_name = "Filter20"
    elif feature_combine == 5:
        cols = filter_coherence_features[:5]
        col_name = "Filter5"
    df_features = df_features.loc[:, cols].copy()
    print(f"----------- Predict scores ----------------")
    predict_score(df_features=df_features, save_to=args.save_to, is_nn=True, col_name=col_name)
    print("------------- Prediction Done ---------------")


if __name__ == '__main__':
    # summary_path = "scoring/SummEval/external/data_annotations/3sample.aligned.paired.jsonl"
    # save_to = "scoring/SummEval/external/data_annotations/temp_results.csv"
    # evaluate_from_path(summary_path=summary_path, save_to=save_to)
    evaluate_from_path()
