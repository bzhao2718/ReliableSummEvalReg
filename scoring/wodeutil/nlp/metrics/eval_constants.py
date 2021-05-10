# import os

# exp_dir = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments"
# analysis_dir = f"{exp_dir}/analysis"
# data_root_dir = f"{exp_dir}/data"

# summeval_analysis_general_dir = f"{analysis_dir}/general/summeval"
# summeval_analysis_metric_labels_dir = f"{summeval_analysis_general_dir}/all_data_metrics_labels"
# summeval_analysis_system_metric_labels_dir = f"{summeval_analysis_general_dir}/syst_metric_against_labels"
#
# summevalanalysis_all_metrics_labels_dir = f"{summeval_analysis_general_dir}/metrics_labels_attr_general"
# summeval_analysis_total_attr_dir = f"{summevalanalysis_all_metrics_labels_dir}/total_attr_general"
# summeval_analysis_system_metric_labels_dir = f"{summeval_analysis_general_dir}/syst_metric_against_labels"
# summeval_analysis_abs_stats_file_dir = f"{summevalanalysis_all_metrics_labels_dir}/abs_attr_general/stats_files"
# summeval_analysis_ext_stats_file_dir = f"{summevalanalysis_all_metrics_labels_dir}/ext_attr_general/stats_files"
# summeval_analysis_total_stats_file_dir = f"{summeval_analysis_total_attr_dir}/stats_files"
#
# summeval_analysis_abs_metric_labels_dir = f"{summeval_analysis_system_metric_labels_dir}/abs_syst"
# summeval_analysis_ext_metric_labels_dir = f"{summeval_analysis_system_metric_labels_dir}/ext_syst"
# #
# # summeval_analysis_lin_dir = f"{analysis_dir}/lin_combine/summeval"
# # summeval_analysis_abs_lin_dir = f"{summeval_analysis_lin_dir}/abs_syst"
# # summeval_analysis_ext_lin_dir = f"{summeval_analysis_lin_dir}/ext_syst"
# # summeval_analysis_total_lin_dir = f"{summeval_analysis_lin_dir}/total"
# # summeval_analysis_lin_temp = f"{summeval_analysis_lin_dir}/temp"
#
# summeval_analysis_reg_dir = f"{analysis_dir}/regression_model/summeval"
# summeval_analysis_abs_reg_dir = f"{summeval_analysis_reg_dir}/abs_syst"
# summeval_analysis_ext_reg_dir = f"{summeval_analysis_reg_dir}/ext_syst"
# summeval_analysis_mix_reg_dir = f"{summeval_analysis_reg_dir}/mix"
#
# summeval_data_fnn08_dir = f"{summeval_analysis_reg_dir}/fnn08"
# summeval_data_abs_fnn08_train_path = f"{summeval_data_fnn08_dir}/abs_train_fnn08.csv"
# summeval_data_abs_fnn08_test_path = f"{summeval_data_fnn08_dir}/abs_test_fnn08.csv"
# summeval_data_abs_regr_dir = f"{summeval_analysis_abs_reg_dir}/lin08"
# summeval_data_ext_regr_dir = f"{summeval_analysis_ext_reg_dir}/lin08"
# summeval_data_mix_reg_dir = f"{summeval_analysis_mix_reg_dir}/lin08"
#
# summeval_data_ext_fnn08_train_path = f"{summeval_data_fnn08_dir}/ext_train_fnn08.csv"
# summeval_data_ext_fnn08_test_path = f"{summeval_data_fnn08_dir}/ext_test_fnn08.csv"
#
# summeval_data_mix_fnn08_train_path = f"{summeval_data_fnn08_dir}/mix_train_fnn08.csv"
# summeval_data_mix_fnn08_test_path = f"{summeval_data_fnn08_dir}/mix_test_fnn08.csv"
#
# summeval_data_abs_fnn08_crossvalid_train_path = f"{summeval_data_fnn08_dir}/cross_valid/abs_train_crossvalid5.csv"
# summeval_data_abs_fnn08_crossvalid_test_path = f"{summeval_data_fnn08_dir}/cross_valid/abs_test_crossvalid5.csv"
#
# summeval_analysis_abs_metric_labels_dir = f"{summeval_analysis_system_metric_labels_dir}/abs_syst"
# summeval_analysis_ext_metric_labels_dir = f"{summeval_analysis_system_metric_labels_dir}/ext_syst"
#
# summeval_analysis_lin_dir = f"{analysis_dir}/lin_combine/summeval"
# summeval_analysis_abs_lin_dir = f"{summeval_analysis_lin_dir}/abs_syst"
# summeval_analysis_ext_lin_dir = f"{summeval_analysis_lin_dir}/ext_syst"
# summeval_analysis_lin_temp = f"{summeval_analysis_lin_dir}/temp"
# summeval_analysis_lin_total_dir = f"{summeval_analysis_lin_dir}/all_models"
#
# summeval_data_all_dir = f"{exp_dir}/all_data/summeval"
# summeval_all_dir = f"{summeval_data_all_dir}/sumeval_all"
# summeval_data_all_syst_dir = f"{summeval_data_all_dir}/models"
#
# summeval_data_all_abs_model_dir = f"{summeval_data_all_dir}/abs_models"
# summeval_data_all_ext_model_dir = f"{summeval_data_all_dir}/ext_models"
# summeval_data_all_abs_path = f"{summeval_data_all_dir}/sumeval_all/summeval_abs_all.csv"
# summeval_data_all_ext_path = f"{summeval_data_all_dir}/sumeval_all/summeval_ext_all.csv"
# summeval_data_all_path = f"{summeval_data_all_dir}/sumeval_all/summeval_all_scores.csv"
#
# summeval_data_all_metrics_dir = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/summeval/models_with_docid/with_s3/abs_ext_mix"
# summeval_data_all_metrics_abs_path = f"{summeval_data_all_metrics_dir}/summeval_abs_all_metrics.csv"
# summeval_data_all_metrics_ext_path = f"{summeval_data_all_metrics_dir}/summeval_ext_all_metrics.csv"
# summeval_data_all_metrics_mix_path = f"{summeval_data_all_metrics_dir}/summeval_mix_all_metrics.csv"
#
# summeval_data_mix_minmax = f"{summeval_data_all_dir}/models_with_docid/with_s3/summeval_mix_minmax.csv"
#
# summeval_data_all_with_s3_path = f"{summeval_data_all_dir}/models_with_docid/with_s3/summeval_all_scores.csv"
# summeval_data_all_new_JS = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/summeval/models_with_docid/with_s3/abs_ext_mix/summeval_mix_all_metrics.csv"
#
# summeval_analysis_reg_dir = f"{analysis_dir}/regression_model/summeval"
# summeval_analysis_abs_reg_dir = f"{summeval_analysis_reg_dir}/abs_syst"
# summeval_analysis_ext_reg_dir = f"{summeval_analysis_reg_dir}/ext_syst"
#
# summeval_top_combine_ablation_fig_dir = "/Users/jackz/OneDrive/One Drive Sync/Research/Workspace/Multifacet_evaluation_metric/experiment_results/figs/top_feature_combine_ablation"
# summeval_four_quality_bar_char_dir = "/Users/jackz/OneDrive/One Drive Sync/Research/Workspace/Multifacet_evaluation_metric/experiment_results/figs/four_quality_bar_chart"
# data_by_syst_dir = f"{exp_dir}/data_by_system"
# lin_combine_dir = f"{exp_dir}/lin_combine"
# regr_model_dir = f"{exp_dir}/refression_model"
#
# summeval_data_all_original_name = "summeval_all_original"
# summeval_data_all_name = "summeval_all_scores"
# summeval_annotations_paired_jsonl_path = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/data_annotations/model_annotations.aligned.paired.jsonl"
# score_files_dir = f"{summeval_data_all_dir}/score_files"
# summeval_all_data_original_path = f"{summeval_data_all_dir}/summeval_all_original.csv"
# summeval_data_with_all_scores_path = f"{summeval_data_all_dir}/summeval_data_all_with_scores.csv"
# summeval_train_data08_with_all_scores_path = f"{summeval_data_all_dir}/train/train_data08_all_scores.csv"
# summeval_test_data08_with_all_scores_path = f"{summeval_data_all_dir}/test/test_data08_all_scores.csv"
#
# summeval_analysis_reg_train_dir = f"{summeval_analysis_reg_dir}/train_all08"
# summeval_analysis_general_corr_dir = f"{summevalanalysis_all_metrics_labels_dir}/abs_ext_mix_corr"
#
# features_coherence_filter_all = ['percentage_repeated_2-gram_in_summ',
#                                  'percentage_novel_3-gram', 'percentage_novel_2-gram', 'percentage_novel_1-gram',
#                                  'density',
#                                  'percentage_repeated_3-gram_in_summ', 'percentage_repeated_1-gram_in_summ',
#                                  'rouge_su*_precision', 'rouge_s*_precision', 'rouge_1_precision', 'rouge_we_1_p',
#                                  'rouge_l_precision', 'rouge_1_f_score', 'rouge_we_1_f', 'rouge_w_1.2_precision',
#                                  'rouge_s*_f_score', 'rouge_s*_recall', 'rouge_su*_f_score', 'rouge_l_f_score',
#                                  'rouge_su*_recall', 'bert_recall_score', 'rouge_1_recall', 'rouge_we_1_r',
#                                  'rouge_l_recall',
#                                  'glove_sms', 'meteor', 'rouge_we_2_p', 'mover_score', 'rouge_2_precision',
#                                  'bert_f_score',
#                                  'rouge_2_recall', 'rouge_w_1.2_f_score', 'rouge_we_3_p', 'rouge_2_f_score',
#                                  'rouge_we_2_f',
#                                  'rouge_we_2_r', 'rouge_3_recall', 'rouge_we_3_r', 'rouge_3_precision', 'rouge_we_3_f',
#                                  'rouge_3_f_score', 'rouge_w_1.2_recall', 'bleu', 'rouge_4_recall', 'rouge_4_precision',
#                                  'rouge_4_f_score', 'bert_precision_score', 'summary_length', 'compression',
#                                  'S3_Responsive',
#                                  'coverage', 'S3_Pyramid', 'cider', 'JS-1', 'JS-2']
#
# features_coherence_filter_all_newjs = ['percentage_repeated_2-gram_in_summ',
#                                        'percentage_novel_3-gram', 'percentage_novel_2-gram',
#                                        'percentage_novel_1-gram', 'density', 'percentage_repeated_3-gram_in_summ',
#                                        'percentage_repeated_1-gram_in_summ', 'rouge_su*_precision',
#                                        'rouge_s*_precision', 'rouge_1_precision', 'rouge_we_1_p', 'rouge_l_precision',
#                                        'rouge_1_f_score', 'rouge_we_1_f', 'rouge_w_1.2_precision', 'rouge_s*_f_score',
#                                        'rouge_s*_recall', 'rouge_su*_f_score', 'rouge_l_f_score', 'rouge_su*_recall',
#                                        'bert_recall_score', 'rouge_1_recall', 'rouge_we_1_r', 'rouge_l_recall',
#                                        'glove_sms', 'meteor', 'rouge_we_2_p', 'mover_score', 'rouge_2_precision',
#                                        'bert_f_score', 'rouge_2_recall', 'rouge_w_1.2_f_score', 'rouge_we_3_p',
#                                        'rouge_2_f_score', 'rouge_we_2_f', 'rouge_we_2_r', 'rouge_3_recall',
#                                        'rouge_we_3_r', 'rouge_3_precision', 'rouge_we_3_f', 'rouge_3_f_score',
#                                        'rouge_w_1.2_recall', 'bleu', 'JS-1', 'JS-2', 'rouge_4_recall',
#                                        'rouge_4_precision', 'rouge_4_f_score', 'bert_precision_score', 'summary_length',
#                                        'compression', 'coverage', 'cider']
#
# features_coherence_filter_without_s3 = [
#     'percentage_repeated_2-gram_in_summ',
#     'percentage_novel_3-gram', 'percentage_novel_2-gram', 'percentage_novel_1-gram',
#     'density',
#     'percentage_repeated_3-gram_in_summ', 'percentage_repeated_1-gram_in_summ',
#     'rouge_su*_precision', 'rouge_s*_precision', 'rouge_1_precision',
#     'rouge_we_1_p',
#     'rouge_l_precision', 'rouge_1_f_score', 'rouge_we_1_f', 'rouge_w_1.2_precision',
#     'rouge_s*_f_score', 'rouge_s*_recall', 'rouge_su*_f_score', 'rouge_l_f_score',
#     'rouge_su*_recall', 'bert_recall_score', 'rouge_1_recall', 'rouge_we_1_r',
#     'rouge_l_recall',
#     'glove_sms', 'meteor', 'rouge_we_2_p', 'mover_score', 'rouge_2_precision',
#     'bert_f_score',  # 30
#     'rouge_2_recall', 'rouge_w_1.2_f_score', 'rouge_we_3_p', 'rouge_2_f_score',
#     'rouge_we_2_f',
#     'rouge_we_2_r', 'rouge_3_recall', 'rouge_we_3_r', 'rouge_3_precision',
#     'rouge_we_3_f',
#     'rouge_3_f_score', 'rouge_w_1.2_recall', 'bleu', 'rouge_4_recall',
#     'rouge_4_precision',
#     'rouge_4_f_score', 'bert_precision_score', 'summary_length', 'compression',
#     'coverage', 'cider', 'JS-1', 'JS-2']
#
######## columns
col_r1 = ['rouge_1_precision', 'rouge_1_recall', 'rouge_1_f_score']
col_r2 = ['rouge_2_precision', 'rouge_2_recall', 'rouge_2_f_score']
col_r3 = ['rouge_3_precision', 'rouge_3_recall', 'rouge_3_f_score']
col_r4 = ['rouge_4_precision', 'rouge_4_recall', 'rouge_4_f_score']
col_rl = ['rouge_l_precision', 'rouge_l_recall', 'rouge_l_f_score']

col_rs_star = ['rouge_s*_precision', 'rouge_s*_recall', 'rouge_s*_f_score']
col_rsu = ['rouge_su*_precision', 'rouge_su*_recall', 'rouge_su*_f_score']
col_rw = ['rouge_w_1.2_precision', 'rouge_w_1.2_recall', 'rouge_w_1.2_f_score']
col_rwe = ['rouge_we_3_p', 'rouge_we_3_r', 'rouge_we_3_f']
col_rwe12 = ['rouge_we_1_p', 'rouge_we_1_r', 'rouge_we_1_f', 'rouge_we_2_p', 'rouge_we_2_r', 'rouge_we_2_f']
col_bleu_cider_sms_meteor = ['bleu', 'meteor', 'cider', 'glove_sms']
col_syntac_stats = ['coverage', 'density', 'compression', 'summary_length', 'percentage_novel_1-gram',
                    'percentage_repeated_1-gram_in_summ', 'percentage_novel_2-gram',
                    'percentage_repeated_2-gram_in_summ', 'percentage_novel_3-gram',
                    'percentage_repeated_3-gram_in_summ']
col_percent_repeated_gram = ['percentage_repeated_1-gram_in_summ',
                             'percentage_repeated_2-gram_in_summ',
                             'percentage_repeated_3-gram_in_summ']
col_percent_novel_gram = ['percentage_novel_1-gram',
                          'percentage_novel_2-gram',
                          'percentage_novel_3-gram']
# col_summeval_bertscore = ['bert_score_precision', 'bert_score_recall', 'bert_score_f1']
# col_bertscore = ['bert_precision_score', 'bert_recall_score', 'bert_f1_score']
#
# col_js2 = ['js-2']
# # col_litepyramid = ['litepyramid_recall']
# # col_moverscore = ['mover_score']
# col_amr = ['amr_cand', 'amr_ref']
# col_smatch = ['smatch_precision', 'smatch_recall', 'smatch_f_score']
# col_s2match = ['s2match_precision', 's2match_recall', 's2match_f_score']
# col_sema = ['sema_precision', 'sema_recall', 'sema_f_score']
#
# col_expert_annotate_stats = ["coherence", "consistency", "fluency", "relevance"]
# summeval_labels = ["coherence", "consistency", "fluency", "relevance"]
# cap_summeval_labels = ["Coherence", "Consistency", "Fluency", "Relevance"]
#
# ["	bart_out	bottom_up_out	"
#  "fast_abs_rl_out_rerank	"
#  "presumm_out_abs	presumm_out_ext_abs	presumm_out_trans_abs	"
#  "ptr_generator_out_pointer_gen_cov	semsim_out	"
#  "t5_out_11B	t5_out_base	t5_out_large	two_stage_rl_out	"
#  "unilm_out_v1	unilm_out_v2"]
# ["Bart-Abs Bottom-Up fastAbsRL-rank T5-Abs      Unilm-v1, Unilm-v2   twoStateRL, PreSummAbs, "
#  "PreSummExt, PreSummExtAbs, Semsim, Pointer-Generator-Cov"]
#
# ["	banditsumm_out	bart_out	heter_graph_out	matchsumm_out	"
#  "neusumm_out	pnbert_out_bert_lstm_pn	pnbert_out_bert_lstm_pn_rl	"
#  "pnbert_out_bert_tf_pn	pnbert_out_bert_tf_sl	pnbert_out_lstm_pn_rl	"
#  "refresh_out"]
# ["BanditSum, Bart-Ext, HeterGraph, MatchSum, REFRESH, NeuSum, BERT-Tf_pn, BERT-tf-sl,BERT-lstm-pn-rl"]
#
# # M0 - LEAD-3
# # M1 - NEUSUM M2 - BanditSum M3 - LATENT M4 - REFRESH M5 - RNES
# # M6 - JECS M7 - STRASS
# # M8 - Pointer Generator M9 - Fast-abs-rl
# # M10 - Bottom-Up
# # M11 - Improve-abs M12 - Unified-ext-abs M13 - ROUGESal
# # M14 - Multi-task (Ent + QG ) M15 - Closed book decoder M16 - SENECA
# # M17 - T5
# # M18 - NeuralTD
# # M19 - BertSum-abs
# # M20 - GPT-2 (supervised) M21 - UniLM
# # M22 - BART
# # M23 - Pegasus (huge news)
#
# rename_bert_dict = {
#     'bert_score_precision': 'bert_precision_score',
#     'bert_score_recall': 'bert_recall_score',
#     'bert_score_f1': 'bert_f_score'
# }
# regr_mlp = "mlp"
# regr_lin = "lin_regr"
# regr_dtree = "dtree"
# regr_forest = "forest"
# regr_ridge = "ridge"
# regr_lasso = "lasso"
# regr_elastic_net = "elastic_net"
# regr_lin_svr = "svr_lin"
# regr_poly_svr = "svr_poly"
# regr_nn_keras = "nn_keras"
# regr_bagging = "bagging"
# regr_voting = "voting"
# regr_adaboost = "ada_boosting"
# regr_grad_boost = "gradient_boosting"
# regr_stacking = "stacking"
#
# summeval_metrics_dict = {
#     'rouge_1_f_score': 'ROUGE-1',
#     'rouge_2_f_score': 'ROUGE-2',
#     'rouge_3_f_score': 'ROUGE-3',
#     'rouge_4_f_score': 'ROUGE-4',
#     'rouge_l_f_score': 'ROUGE-L',
#     'rouge_s*_f_score': 'ROUGE-s*',
#     'rouge_su*_f_score': 'ROUGE-su*',
#     'rouge_w_1.2_f_score': 'ROUGE-w',
#     'rouge_we_1_f': 'ROUGE-we-1',
#     'rouge_we_2_f': 'ROUGE-we-2',
#     "rouge_we_3_f": 'ROUGE-we-3',
#     # 'bert_f_score': 'BERTScore-F1',
#     # 'bert_score_recall': 'BERTScore-R',
#     # 'bert_score_precision': 'BERTScore-P',
#     'bert_f_score': 'BERTScore-f',
#     'bert_recall_score': 'BERTScore-r',
#     'bert_precision_score': 'BERTScore-p',
#     'JS-2': 'JS-2',
#     'glove_sms': 'SMS',
#     'bleu': 'BLEU',
#     'mover_score': 'MoverScore',
#     # 'sema_recall': 'Sema',
#     # 'smatch_recall': 'Smatch',
#     'cider': 'CIDEr',
#     'meteor': 'METEOR',
#     'summary_length': 'Length',
#     'compression': 'Stats-compression',
#     'coverage': 'Stats-coverage',
#     'density': 'Stats-density',
#     'percentage_novel_1-gram': 'Novel unigram',
#     'percentage_novel_2-gram': 'Novel bi-gram',
#     'percentage_novel_3-gram': 'Novel tri-gram',
#     'percentage_repeated_1-gram_in_summ': 'Repeated unigram',
#     'percentage_repeated_2-gram_in_summ': 'Repeated bi-gram',
#     'percentage_repeated_3-gram_in_summ': 'Repeated tri-gram',
#     "lin0": "Lin0",
#     "lin1": "Lin_top1",
#     "lin2": "Lin_top2",
#     "lin3": "Lin_top3",
#     "lin4": "Lin_top4",
#     "lin5": "Lin_combine5",
#     "lin6": "Lin_combine6",
#     "lin7": "Lin_combine7",
#     "lin8": "Lin_combine8",
#     "lin9": "Lin_combine9",
#     "lin10": "Lin_combine10",
#     "lin11": "Lin_combine11",
#     "dtree": "DecisionTree",
#     "lin_regr": "LinReg",
#     "ridge": "RidgeReg",
#     "lasso": "LassoReg",
#     "elastic_net": "ElasticNetReg",
#     "svr_lin": "LinSVR",
#     "forest": "RandomForest",
#     regr_bagging: "Bagging",
#     regr_adaboost: "AdaBoost",
#     regr_grad_boost: "GradientBoost",
#     regr_voting: "Voting",
#     regr_stacking: "Stacking",
#     "mlp": "MLP",
#     "nn_keras": "NNReg"
# }
# regr_colors_dict = {
#     summeval_metrics_dict[regr_nn_keras]: "Green",
#     summeval_metrics_dict[regr_mlp]: "Lime",
#     summeval_metrics_dict[regr_forest]: "Blue",
#     summeval_metrics_dict[regr_dtree]: "Orange",
#     summeval_metrics_dict[regr_voting]: "Red",
#     summeval_metrics_dict[regr_stacking]: "Fuchsia",
#     summeval_metrics_dict[regr_lin]: "Purple",
#     summeval_metrics_dict[regr_lin_svr]: "OliveDrab",
#     summeval_metrics_dict[regr_adaboost]: "Brown",
#     summeval_metrics_dict[regr_grad_boost]: "Olive",
#     summeval_metrics_dict[regr_lasso]: "Yellow",
#     summeval_metrics_dict[regr_elastic_net]: "Navy",
#     summeval_metrics_dict[regr_bagging]: "Aqua",
#     summeval_metrics_dict[regr_ridge]: "Teal"
# }
#
# regr_colors_step_dict = {
#     summeval_metrics_dict[regr_nn_keras]: "Lime",
#     summeval_metrics_dict[regr_mlp]: "Red",
#     summeval_metrics_dict[regr_forest]: "Blue",
#     summeval_metrics_dict[regr_dtree]: "Orange",
#     summeval_metrics_dict[regr_voting]: "Green",
#     summeval_metrics_dict[regr_stacking]: "Fuchsia",
#     summeval_metrics_dict[regr_lin]: "Purple",
#     summeval_metrics_dict[regr_lin_svr]: "OliveDrab",
#     summeval_metrics_dict[regr_adaboost]: "Brown",
#     summeval_metrics_dict[regr_grad_boost]: "Olive",
#     summeval_metrics_dict[regr_lasso]: "Yellow",
#     summeval_metrics_dict[regr_elastic_net]: "Navy",
#     summeval_metrics_dict[regr_bagging]: "Aqua",
#     summeval_metrics_dict[regr_ridge]: "Teal"
# }
# # extra_cols = ['rouge_1_f_score', 'bert_recall_score', 'density', 'rouge_we_1_f']
# regr_bar_colors_dict = {
#     summeval_metrics_dict[regr_nn_keras]: "Green",
#     summeval_metrics_dict[regr_mlp]: "Lime",
#     summeval_metrics_dict[regr_forest]: "Blue",
#     summeval_metrics_dict["rouge_1_f_score"]: "Orange",
#     summeval_metrics_dict[regr_voting]: "Red",
#     summeval_metrics_dict[regr_stacking]: "Fuchsia",
#     summeval_metrics_dict[regr_lin]: "Purple",
#     summeval_metrics_dict[regr_lin_svr]: "OliveDrab",
#     summeval_metrics_dict["bert_recall_score"]: "Brown",
#     summeval_metrics_dict[regr_grad_boost]: "Olive",
#     summeval_metrics_dict["rouge_we_1_f"]: "Yellow",
#     summeval_metrics_dict["density"]: "Navy",
#     summeval_metrics_dict[regr_bagging]: "Aqua",
#     summeval_metrics_dict[regr_ridge]: "Teal"
# }
# regrs_list = [regr_dtree, regr_lin, regr_ridge, regr_lasso, regr_elastic_net, regr_lin_svr, regr_forest, regr_bagging,
#               regr_adaboost, regr_grad_boost, regr_mlp, regr_voting, regr_stacking,
#               regr_nn_keras]
#
# regr_names_for_combine_ablation = [regr_lin, regr_forest, regr_mlp, regr_grad_boost, regr_nn_keras,
#                                    regr_ridge, regr_lin_svr]
# regr_names_for_pca_features = [regr_dtree, regr_lin, regr_ridge, regr_forest, regr_bagging,
#                                regr_adaboost, regr_grad_boost, regr_mlp, regr_voting, regr_stacking,
#                                regr_nn_keras]
#
# regr_names_for_combine_ablation_lin = [regr_lin, regr_lasso,
#                                        regr_ridge, regr_lin_svr]
# regr_names_for_combine_ablation_nn = [
#     # regr_nn_keras,
#     regr_mlp, regr_ridge, regr_grad_boost,
#     regr_forest, regr_dtree]
# regr_remove_step_ablation = [regr_nn_keras, regr_mlp,
#                              regr_forest, regr_dtree, regr_ridge]
# regr_names_for_combine_ablation4 = [regr_lin, regr_forest, regr_mlp, regr_nn_keras]
# regr_names_four_quality_bar = ['rouge_1_f_score', 'bert_recall_score', 'density', 'rouge_we_1_f', regr_lin, regr_voting,
#                                # regr_lin_svr,
#                                # regr_mlp,
#                                regr_forest, regr_nn_keras, regr_mlp,
#                                # regr_voting,
#                                regr_grad_boost]
# regr_names_four_quality_bar_old = ['rouge_1_f_score', 'bert_recall_score', 'density', 'rouge_we_1_f', regr_lin,
#                                    regr_voting,
#                                    # regr_lin_svr,
#                                    # regr_mlp,
#                                    regr_forest, regr_nn_keras, regr_bagging,
#                                    # regr_voting,
#                                    regr_grad_boost]
# regr_names_four_quality_bar_all = ['rouge_1_f_score', 'bert_recall_score', 'density', 'rouge_we_1_f', regr_lin,
#                                    regr_ridge,
#                                    # regr_lin_svr,
#                                    # regr_stacking,
#                                    regr_nn_keras,
#                                    regr_mlp,
#                                    regr_forest, regr_grad_boost, regr_voting,
#                                    regr_bagging]
# summeval_quality_metric_dict = {
#     'rouge_1_f_score': 'ROUGE-1',
#     'rouge_2_f_score': 'ROUGE-2',
#     'rouge_3_f_score': 'ROUGE-3',
#     'rouge_4_f_score': 'ROUGE-4',
#     'rouge_l_f_score': 'ROUGE-L',
#     'rouge_s*_f_score': 'ROUGE-s*',
#     'rouge_su*_f_score': 'ROUGE-su*',
#     'rouge_w_1.2_f_score': 'ROUGE-w',
#     'rouge_we_1_f': 'ROUGE-we-1',
#     'rouge_we_2_f': 'ROUGE-we-2',
#     "rouge_we_3_f": 'ROUGE-we-3',
#     # 'bert_f_score': 'BERTScore-F1',
#     # 'bert_score_recall': 'BERTScore-R',
#     # 'bert_score_precision': 'BERTScore-P',
#     'bert_f_score': 'BERTScore-f',
#     'bert_recall_score': 'BERTScore-r',
#     'bert_precision_score': 'BERTScore-p',
#     'js-2': 'JS-2',
#     'glove_sms': 'SMS',
#     'bleu': 'BLEU',
#     'mover_score': 'MoverScore',
#     # 'sema_recall': 'Sema',
#     # 'smatch_recall': 'Smatch',
#     'cider': 'CIDEr',
#     'meteor': 'METEOR',
#     'summary_length': 'Length',
#     'compression': 'Stats-compression',
#     'coverage': 'Stats-coverage',
#     'density': 'Stats-density',
#     'percentage_novel_1-gram': 'Novel unigram',
#     'percentage_novel_2-gram': 'Novel bi-gram',
#     'percentage_novel_3-gram': 'Novel tri-gram',
#     'percentage_repeated_1-gram_in_summ': 'Repeated unigram',
#     'percentage_repeated_2-gram_in_summ': 'Repeated bi-gram',
#     'percentage_repeated_3-gram_in_summ': 'Repeated tri-gram',
# }
#
# fn_abbr = {
#     "summeval_abs_all_metrics": "abs",
#     "summeval_ext_all_metrics": "ext",
#     "summeval_mix_all_metrics": "mix"
# }
# idx_unnamed = "Unnamed: 0"
# # output_types = {"summeval_abs_all": "Abstractive", "summeval_ext_all": "Extractive", "summeval_all_scores": "Mix"}
# output_types = {"summeval_abs_all_metrics": "Abstractive", "summeval_ext_all_metrics": "Extractive",
#                 "summeval_mix_all_metrics": "Mix"}
#
# # output_types = {"abs_test_fnn08": "Abstractive", "ext_test_fnn08": "Extractive",
# #                 "mix_test_fnn08": "Mix"}
#
# rouge_metrics_list = [*col_r1, *col_r2, *col_r3, *col_r4, *col_rw, *col_rs_star, *col_rsu, *col_rl]
# summeval_all_metrics = [*rouge_metrics_list, *col_syntac_stats,
#                         *col_js2, *col_summeval_bertscore, *col_rwe, *col_expert_annotate_stats]
#
# all_cols = [*col_r1, *col_r2, *col_r3, *col_r4, *col_rw, *col_rs_star, *col_rsu, *col_rl, *col_syntac_stats, *col_js2,
#             *col_summeval_bertscore, *col_smatch, *col_sema, *col_rwe, *col_expert_annotate_stats]
#
# summeval_metrics_r = [*rouge_metrics_list, *col_rwe]
# summeval_metrics_b_sm_rwe = [*col_summeval_bertscore, *col_bleu_cider_sms_meteor, *col_rwe]
#
# summeval_data_name = "summeval"
# ##### system info
#
# ext_model_codes = ['M0', 'M1', 'M2', 'M5']
#
# import pandas as pd
#
#
# def get_summeval_models_data_as_df(data_type="mix", with_s3=False, new_js=True):
#     if with_s3:
#         df = pd.read_csv(summeval_data_all_with_s3_path)
#     else:
#         df = pd.read_csv(summeval_data_all_path)
#
#     if new_js:
#         df = pd.read_csv(summeval_data_all_new_JS)
#
#     if data_type == "abs":
#         return df[~df.model_id.isin(ext_model_codes)].copy()
#     elif data_type == "ext":
#         return df[df.model_id.isin(ext_model_codes)].copy()
#     elif data_type == "mix":
#         return df.copy()
#     elif data_type == "mix_minmax":
#         return pd.read_csv(summeval_data_mix_minmax)
#
#
# def get_summeval_pca_data(n_components=20):
#     path_xpca = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/summeval/models_with_docid/with_s3/train_xpca20.csv"
#     path_test_xpca = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/summeval/models_with_docid/with_s3/test_xpca20.csv"
#     if n_components == 20:
#         df_xpca20 = pd.read_csv(path_xpca)
#         df_test_xpca20 = pd.read_csv(path_test_xpca)
#         return df_xpca20, df_test_xpca20
#
#
# def get_summeval_data_as_df_split(df=None, data_type='abs', cols=None, label=None, with_s3=True, split_ratio=0.8):
#     """
#     return train_x, train_y, test_x,test_y splitted based on the split_ratio
#     """
#     if cols and label:
#         if df is None:
#             df = get_summeval_models_data_as_df(data_type=data_type, with_s3=with_s3)
#         split = split_ratio * len(df)
#
#         train_x = df.loc[:split, cols].copy()
#         train_y = df.loc[:split, [label]].copy()
#
#         test_x = df.loc[split:, cols].copy()
#         test_y = df.loc[split:, [label]].copy()
#
#         return train_x, train_y, test_x, test_y
#
#
# systems_code = {
#     "M0": "mosystem"
# }
#
# ######
# corr_pearsonr = 'pearsonr'
# corr_spearmanr = 'spearmanr'
# corr_kendalltau = 'kendalltau'
# col_bertscore = ['bert_precision_score', 'bert_recall_score', 'bert_f_score']
#
# ######## features combination
# summeval_features_r12l_b = [*col_r1, *col_r2, *col_rl, *col_bertscore]
# summeval_features_r12l_b_m_bleu_meteor = [*col_r1, *col_r2, *col_rl, *col_bertscore,
#                                           *col_bleu_cider_sms_meteor]
# summeval_features_r12l = [*col_r1, *col_r2, *col_rl]
# summeval_features_b_bleu_meteor_sms_rsu = [*col_bertscore, *col_bleu_cider_sms_meteor, *col_rsu]
# summeval_features_b_rsu = [*col_bertscore, *col_rsu]
# summeval_features_b_rwe = [*col_bertscore, *col_rwe]
# summeval_features_r2 = [*col_r2]
# summeval_features_r1 = [*col_r1]
# summeval_features_rl = [*col_rl]
# summeval_features_r1b = [*col_r1, *col_bertscore]
# summeval_features_r1b_syntac = [*col_r1, *col_bertscore, *col_syntac_stats]
#
# summeval_features_r1lsu_b_rwe = [*col_r1, *col_rl, *col_rwe, *col_rsu]
#
# summeval_cols_lin5_c1 = ['rouge_l_recall', 'rouge_1_f_score', 'rouge_2_recall', 'bert_score_recall', 'rouge_1_recall']
# cols_lin12 = [*col_r1, *col_r2, *col_bertscore]
# cols_lin12_r_b = [*col_rsu, *col_r2, 'bert_recall_score', 'bert_f_score', 'glove_sms']
# cols_lin12_r1lb_syntac = [*col_rwe, *col_rsu, *col_r1, *col_bertscore]
# cols_lin12_systlevel_abs = ['rouge_w_1.2_precision', 'rouge_l_precision', 'rouge_su*_precision', 'rouge_s*_precision',
#                             'rouge_1_precision', 'rouge_we_3_p', 'glove_sms', 'rouge_2_precision', 'bert_recall_score']
# cols_lin12_summlevel_abs = ['rouge_w_1.2_precision', 'rouge_su*_precision', 'rouge_l_precision', 'rouge_1_precision',
#                             'density', 'rouge_s*_precision',
#                             'rouge_1_precision', 'rouge_we_3_p', 'glove_sms', 'rouge_2_precision', 'bert_recall_score']
# cols_lin12_summeval_coherence = ['rouge_w_1.2_precision', 'rouge_su*_precision', 'rouge_l_precision',
#                                  'rouge_1_precision',
#                                  'density', 'rouge_s*_precision',
#                                  'rouge_we_3_p', 'glove_sms', 'rouge_2_precision',
#                                  'bert_recall_score']
# summeval_features_wrapper_common = ['rouge_we_3_r', 'rouge_1_recall', 'rouge_we_3_p', 'rouge_1_f_score',
#                                     'rouge_2_f_score', 'rouge_3_recall', 'percentage_repeated_2-gram_in_summ',
#                                     'bert_recall_score', 'coverage', 'rouge_su*_recall', 'density',
#                                     'percentage_repeated_3-gram_in_summ']
# summeval_features_wrapper_top = ['rouge_1_recall', 'rouge_l_recall', 'rouge_we_3_p', 'rouge_1_f_score',
#                                  'percentage_repeated_2-gram_in_summ', 'bert_recall_score', 'rouge_su*_recall',
#                                  'density', 'percentage_repeated_3-gram_in_summ']
# summeval_features_wrapper_top5 = ['density', 'bert_recall_score', 'percentage_repeated_2-gram_in_summ',
#                                   'rouge_1_recall', 'rouge_we_3_r']
# summeval_features_wrapper_top4 = ['rouge_1_precision', 'rouge_su*_precision', 'rouge_l_precision', 'density']
# summeval_features_wrapper_abs_top7 = ['rouge_1_precision', 'rouge_2_f_score', 'rouge_l_recall', 'bert_precision_score',
#                                       'density', 'rouge_su*_precision', 'rouge_w_1.2_precision']
#
# summeval_features_repeated_grams = ['percentage_repeated_2-gram_in_summ', 'percentage_repeated_3-gram_in_summ',
#                                     'percentage_novel_1-gram', 'rouge_w_1.2_precision', 'rouge_su*_precision',
#                                     'rouge_l_precision',
#                                     'rouge_1_precision',
#                                     'density', 'rouge_s*_precision']
# summeval_features_selected8 = ['bert_recall_score', 'density', 'rouge_1_recall', 'rouge_l_recall', 'rouge_we_3_p',
#                                'percentage_repeated_2-gram_in_summ', 'percentage_novel_1-gram',
#                                'percentage_repeated_3-gram_in_summ']
#
# summeval_features_selected11 = ['bert_recall_score', 'density', 'rouge_1_recall', 'rouge_l_recall', 'rouge_we_3_p',
#                                 'percentage_repeated_2-gram_in_summ', 'percentage_novel_1-gram', 'coverage',
#                                 'rouge_su*_recall', 'rouge_2_f_score',
#                                 'percentage_repeated_3-gram_in_summ']
#
# summeval_features_selected10 = ['bert_recall_score', 'rouge_w_1.2_precision', 'rouge_1_recall', 'rouge_1_precision',
#                                 'rouge_l_recall', 'rouge_we_3_p',
#                                 'rouge_su*_recall', 'rouge_2_f_score', 'rouge_su*_precision', 'rouge_l_precision']
# summeval_features_selected13 = ['bert_recall_score', 'density', 'rouge_1_recall', 'rouge_l_recall', 'rouge_we_3_p',
#                                 'rouge_we_1_p', 'rouge_we_2_p',
#                                 'percentage_repeated_2-gram_in_summ', 'percentage_novel_1-gram', 'coverage',
#                                 'rouge_su*_recall',
#                                 'percentage_repeated_3-gram_in_summ']
# summeval_features_selected18 = ['bert_recall_score', 'density', 'rouge_1_recall', 'rouge_l_recall', 'rouge_we_3_r',
#                                 'rouge_we_1_r', 'rouge_we_2_r', 'rouge_we_3_p', 'rouge_we_1_f', 'rouge_we_2_f',
#                                 'rouge_we_3_f',
#                                 'rouge_we_1_p', 'rouge_we_2_p',
#                                 'percentage_repeated_2-gram_in_summ', 'percentage_novel_1-gram', 'coverage',
#                                 'rouge_su*_recall',
#                                 'percentage_repeated_3-gram_in_summ']
#
# summeval_features_linreg5 = ['density', 'coverage', 'percentage_repeated_2-gram_in_summ', 'percentage_novel_3-gram',
#                              'rouge_l_precision']
# summeval_features_linreg10 = ['percentage_novel_3-gram', 'density', 'percentage_repeated_3-gram_in_summ', 'bleu',
#                               'rouge_2_f_score', 'percentage_novel_1-gram', 'rouge_l_precision', 'bert_recall_score',
#                               'percentage_repeated_2-gram_in_summ', 'coverage']
# summeval_features_linreg20 = ['rouge_3_recall', 'bleu', 'percentage_novel_3-gram', 'rouge_2_recall',
#                               'percentage_novel_1-gram', 'rouge_l_recall', 'percentage_repeated_2-gram_in_summ',
#                               'rouge_l_f_score', 'rouge_2_f_score', 'percentage_repeated_1-gram_in_summ',
#                               'rouge_l_precision', 'rouge_4_recall', 'bert_recall_score', 'JS-2', 'bert_f_score',
#                               'density', 'rouge_2_precision', 'percentage_repeated_3-gram_in_summ',
#                               'bert_precision_score', 'coverage']
# summeval_features_all = [*col_r1, *col_syntac_stats, *col_r2, *col_bertscore, *col_rwe12, *col_rl, *col_rwe, *col_r3,
#                          *col_rs_star,
#                          *col_rsu,
#                          *col_bleu_cider_sms_meteor
#                          ]
#
# # summeval_features_relevance_selected14 = ['rouge_1_recall', 'rouge_we_1_f', 'rouge_su*_recall', 'bert_recall_score',
# #                                           'rouge_1_f_score', 'rouge_l_recall',
# #                                           'rouge_we_1_r', 'rouge_we_2_r',
# #                                           'rouge_2_recall', 'rouge_su*_f_score',
# #                                           'mover_score', 'glove_sms',
# #                                           'meteor', 'bert_f_score']
# summeval_features_relevance_selected14 = ['rouge_1_recall', 'rouge_we_1_f', 'rouge_su*_recall', 'bert_recall_score',
#                                           'rouge_1_f_score', 'rouge_l_recall',
#                                           'rouge_we_1_r', 'rouge_we_2_r',
#                                           'rouge_2_recall', 'rouge_su*_f_score',
#                                           'mover_score',
#                                           'meteor', *col_percent_novel_gram]
#
# summeval_features_consistency_selected14 = ['density', 'meteor', 'bert_recall_score', 'rouge_we_1_f', 'rouge_1_recall',
#                                             'rouge_su*_recall',
#                                             'rouge_1_f_score', 'rouge_l_recall',
#                                             'rouge_we_1_r', 'rouge_we_2_r',
#                                             'rouge_2_recall', 'rouge_su*_f_score',
#                                             'mover_score', 'glove_sms']
# summeval_features_fluency_selected14 = ['density', 'meteor', 'bert_recall_score', 'rouge_we_1_f', 'rouge_1_recall',
#                                         'rouge_su*_recall',
#                                         'rouge_1_f_score', 'rouge_l_recall',
#                                         'rouge_we_1_r', 'rouge_we_2_r',
#                                         'rouge_2_recall', 'rouge_su*_f_score',
#                                         'mover_score', 'glove_sms']
#
# summeval_feature_selected_no_ngram = ['bert_recall_score', 'density', 'rouge_1_recall', 'rouge_l_recall',
#                                       'rouge_we_3_r',
#                                       'rouge_we_1_r', 'rouge_we_2_r', 'rouge_we_3_p', 'rouge_we_1_f', 'rouge_we_2_f',
#                                       'rouge_we_3_f',
#                                       'rouge_we_1_p', 'rouge_we_2_p', 'coverage',
#                                       'rouge_su*_recall', *col_percent_novel_gram]
# # summeval_features_selected_rpf = [*col_bertscore, *col_rwe, *col_rwe12, *col_r1, *col_r2, *col_rl, *col_rsu,
# #                                   *col_percent_repeated_gram, 'density']
#
#
# summeval_features_selected_top10 = ['percentage_repeated_2-gram_in_summ', 'density', 'percentage_novel_2-gram',
#                                     'percentage_novel_3-gram', 'bert_recall_score', 'rouge_we_1_p',
#                                     'rouge_l_recall', 'rouge_su*_recall', 'percentage_novel_1-gram', 'coverage']
# summeval_features_selected_top15 = ['percentage_repeated_2-gram_in_summ', 'density', 'percentage_novel_2-gram',
#                                     'percentage_novel_3-gram', 'bert_recall_score', 'rouge_we_1_p',
#                                     'rouge_l_recall', 'rouge_su*_recall', 'percentage_novel_1-gram', 'coverage',
#                                     'rouge_we_3_f', 'rouge_we_3_r', 'rouge_we_2_r', 'rouge_we_1_f', 'rouge_we_3_p']
#
# summeval_features_selected21 = ['bert_recall_score', 'density', 'rouge_1_recall', 'rouge_l_recall', 'rouge_we_3_r',
#                                 'rouge_we_1_r', 'rouge_we_2_r', 'rouge_we_3_p', 'rouge_we_1_f', 'rouge_we_2_f',
#                                 'rouge_we_3_f',
#                                 'rouge_we_1_p', 'rouge_we_2_p',
#                                 'coverage',
#                                 # 'JS-2',
#                                 'rouge_su*_recall',
#                                 'percentage_repeated_1-gram_in_summ',
#                                 'percentage_repeated_2-gram_in_summ',
#                                 'percentage_repeated_3-gram_in_summ', *col_percent_novel_gram]
# summeval_features_selected21_trained = ['bert_recall_score', 'density', 'rouge_1_recall', 'rouge_l_recall',
#                                         'rouge_we_3_r',
#                                         'rouge_we_1_r', 'rouge_we_2_r', 'rouge_we_3_p', 'rouge_we_1_f', 'rouge_we_2_f',
#                                         'rouge_we_3_f',
#                                         'rouge_we_1_p', 'rouge_we_2_p',
#                                         'percentage_repeated_2-gram_in_summ', 'coverage',
#                                         'rouge_su*_recall',
#                                         'percentage_repeated_3-gram_in_summ', *col_percent_novel_gram]
#
# select_pos = 0
# summeval_features_selected_rpf_1 = [col_bertscore[select_pos], col_rwe[select_pos], col_rwe12[select_pos],
#                                     col_r1[select_pos], col_r2[select_pos], \
#                                     col_rl[select_pos], col_rsu[select_pos],
#                                     col_percent_repeated_gram[select_pos], 'density']
# weights_lin5_c3 = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# weights_lin5_c1 = [0.4, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# weights_lin5_c5_2 = [0.3, 0.3, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# weights_lin5_c0 = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# weights_lin5_c2 = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# weights_lin5_c6 = [0.1, 0.1, 0.0, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# weights_lin5_c10_1 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0]
# weights_lin5_c10_2 = [0.0, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
#
# summeval_example_combines = [
#     (summeval_cols_lin5_c1, weights_lin5_c3),
#     (summeval_cols_lin5_c1, weights_lin5_c1),
#     (summeval_cols_lin5_c1, weights_lin5_c5_2),
#     (summeval_cols_lin5_c1, weights_lin5_c0),
#     (summeval_cols_lin5_c1, weights_lin5_c2),
#     (summeval_cols_lin5_c1, weights_lin5_c6)
# ]
#
# summeval_example_combines_12 = [
#     (cols_lin12, weights_lin5_c3),
#     (cols_lin12, weights_lin5_c1),
#     (cols_lin12, weights_lin5_c5_2),
#     (cols_lin12, weights_lin5_c0),
#     (cols_lin12, weights_lin5_c2),
#     (cols_lin12, weights_lin5_c6),
#     (cols_lin12, weights_lin5_c10_1),
#     (cols_lin12, weights_lin5_c10_2)
# ]
# summeval_example_combines_12_rbs = [
#     (cols_lin12_r_b, weights_lin5_c3),
#     (cols_lin12_r_b, weights_lin5_c1),
#     (cols_lin12_r_b, weights_lin5_c5_2),
#     (cols_lin12_r_b, weights_lin5_c0),
#     (cols_lin12_r_b, weights_lin5_c2),
#     (cols_lin12_r_b, weights_lin5_c6),
#     (cols_lin12_r_b, weights_lin5_c10_1),
#     (cols_lin12_r_b, weights_lin5_c10_2)
# ]
#
# summeval_example_combines_12_rswb = [
#     (cols_lin12_r1lb_syntac, weights_lin5_c3),
#     (cols_lin12_r1lb_syntac, weights_lin5_c1),
#     (cols_lin12_r1lb_syntac, weights_lin5_c5_2),
#     (cols_lin12_r1lb_syntac, weights_lin5_c0),
#     (cols_lin12_r1lb_syntac, weights_lin5_c2),
#     (cols_lin12_r1lb_syntac, weights_lin5_c6),
#     (cols_lin12_r1lb_syntac, weights_lin5_c10_1),
#     (cols_lin12_r1lb_syntac, weights_lin5_c10_2)
# ]
#
# summeval_example_combines_systlevel_abs = [
#     (cols_lin12_systlevel_abs, weights_lin5_c3),
#     (cols_lin12_systlevel_abs, weights_lin5_c1),
#     (cols_lin12_systlevel_abs, weights_lin5_c5_2),
#     (cols_lin12_systlevel_abs, weights_lin5_c0),
#     (cols_lin12_systlevel_abs, weights_lin5_c2),
#     (cols_lin12_systlevel_abs, weights_lin5_c6),
#     (cols_lin12_systlevel_abs, weights_lin5_c10_1),
#     (cols_lin12_systlevel_abs, weights_lin5_c10_2)
# ]
#
# summeval_example_combines_summlevel_abs = [
#     (cols_lin12_summlevel_abs, weights_lin5_c3),
#     (cols_lin12_summlevel_abs, weights_lin5_c1),
#     (cols_lin12_summlevel_abs, weights_lin5_c5_2),
#     (cols_lin12_summlevel_abs, weights_lin5_c0),
#     (cols_lin12_summlevel_abs, weights_lin5_c2),
#     (cols_lin12_summlevel_abs, weights_lin5_c6),
#     (cols_lin12_summlevel_abs, weights_lin5_c10_1),
#     (cols_lin12_summlevel_abs, weights_lin5_c10_2)
# ]
# summeval_example_combines_summlevel_coherence = [
#     (cols_lin12_summeval_coherence, weights_lin5_c0),
#     (cols_lin12_summeval_coherence, weights_lin5_c1),
#     (cols_lin12_summeval_coherence, weights_lin5_c2),
#     (cols_lin12_summeval_coherence, weights_lin5_c3),
#     (cols_lin12_summeval_coherence, weights_lin5_c5_2),  # 4
#     (cols_lin12_summeval_coherence, weights_lin5_c6),  # 5
#     (cols_lin12_summeval_coherence, weights_lin5_c10_1),  # 6
#     # (cols_lin12_summeval_coherence, weights_lin5_c10_2),
#     (summeval_features_repeated_grams, weights_lin5_c0),  # 7
#     (summeval_features_repeated_grams, weights_lin5_c1),  # 8
#     (summeval_features_repeated_grams, weights_lin5_c2),  # 9
#     (summeval_features_selected21, weights_lin5_c0),  # 10
#     (summeval_features_selected21, weights_lin5_c1),  # 11
#     (summeval_features_selected21, weights_lin5_c2),  # 12
# ]
# summeval_example_combines_summlevel_relevance = [
#     (summeval_features_relevance_selected14, weights_lin5_c0),
#     (summeval_features_relevance_selected14, weights_lin5_c1),
#     (summeval_features_relevance_selected14, weights_lin5_c2),
#     (summeval_features_relevance_selected14, weights_lin5_c3),
#     (summeval_features_relevance_selected14, weights_lin5_c5_2),  # 4
#     (summeval_features_relevance_selected14, weights_lin5_c6),  # 5
#     (summeval_features_relevance_selected14, weights_lin5_c10_1),
#     (summeval_features_relevance_selected14, weights_lin5_c10_2)
#
# ]
# summeval_example_combines_summlevel_consistency = [
#     (summeval_features_consistency_selected14, weights_lin5_c0),
#     (summeval_features_consistency_selected14, weights_lin5_c1),
#     (summeval_features_consistency_selected14, weights_lin5_c2),
#     (summeval_features_consistency_selected14, weights_lin5_c3),
#     (summeval_features_consistency_selected14, weights_lin5_c5_2),  # 4
#     (summeval_features_consistency_selected14, weights_lin5_c6),  # 5
#     (summeval_features_consistency_selected14, weights_lin5_c10_1),
#     (summeval_features_consistency_selected14, weights_lin5_c10_2)
#
# ]
# summeval_example_combines_summlevel_fluency = [
#     (summeval_features_fluency_selected14, weights_lin5_c0),
#     (summeval_features_fluency_selected14, weights_lin5_c1),
#     (summeval_features_fluency_selected14, weights_lin5_c2),
#     (summeval_features_fluency_selected14, weights_lin5_c3),
#     (summeval_features_fluency_selected14, weights_lin5_c5_2),  # 4
#     (summeval_features_fluency_selected14, weights_lin5_c6),  # 5
#     (summeval_features_fluency_selected14, weights_lin5_c10_1),
#     (summeval_features_fluency_selected14, weights_lin5_c10_2)
#
# ]
#
# summeval_features_forest20 = ['rouge_we_3_p', 'rouge_1_f_score', 'rouge_2_recall', 'rouge_4_f_score',
#                               'rouge_l_precision', 'rouge_l_f_score',
#                               'rouge_w_1.2_recall', 'rouge_s*_recall', 'bert_recall_score', 'meteor', 'bleu',
#                               'coverage', 'density',
#                               'summary_length', 'percentage_novel_1-gram', 'percentage_repeated_2-gram_in_summ',
#                               'percentage_novel_3-gram',
#                               'percentage_repeated_3-gram_in_summ', 'JS-2', 'rouge_we_1_f']
# summeval_features_wrapper20_dict = {
#     "filter_20": set(features_coherence_filter_without_s3[:20]),
#     'forest': {'JS-2', 'coverage', 'summary_length', 'density', 'percentage_repeated_2-gram_in_summ',
#                'meteor', 'rouge_we_3_p', 'rouge_4_f_score', 'rouge_l_f_score', 'rouge_2_recall', 'bleu',
#                'rouge_we_1_f', 'rouge_s*_recall', 'rouge_l_precision', 'percentage_novel_3-gram',
#                'rouge_1_f_score', 'percentage_novel_1-gram', 'bert_recall_score', 'rouge_w_1.2_recall',
#                'percentage_repeated_3-gram_in_summ'},
#     'lin_regr': {'rouge_2_f_score', 'coverage',
#                  'rouge_4_recall', 'density',
#                  'bert_precision_score',
#                  'rouge_2_precision',
#                  'percentage_repeated_2-gram_in_summ',
#                  'percentage_repeated_1-gram_in_summ',
#                  'rouge_l_f_score', 'rouge_2_recall',
#                  'bert_f_score', 'bleu', 'rouge_3_recall',
#                  'rouge_l_recall', 'rouge_l_precision',
#                  'percentage_novel_3-gram',
#                  'percentage_novel_1-gram',
#                  'bert_recall_score', 'JS-2',
#                  'percentage_repeated_3-gram_in_summ'},
#     'lin_regr_without_JS': {'rouge_2_f_score', 'coverage',
#                             'rouge_4_recall', 'density',
#                             'bert_precision_score',
#                             'rouge_2_precision',
#                             'percentage_repeated_2-gram_in_summ',
#                             'percentage_repeated_1-gram_in_summ',
#                             'rouge_l_f_score', 'rouge_2_recall',
#                             'bert_f_score', 'bleu', 'rouge_3_recall',
#                             'rouge_l_recall', 'rouge_l_precision',
#                             'percentage_novel_3-gram',
#                             'percentage_novel_1-gram',
#                             'bert_recall_score',
#                             'percentage_repeated_3-gram_in_summ'},
#     'mlp': {'density', 'percentage_repeated_2-gram_in_summ', 'percentage_novel_2-gram',
#             'rouge_we_1_p', 'rouge_we_3_p', 'percentage_repeated_1-gram_in_summ', 'rouge_w_1.2_f_score',
#             'rouge_2_recall', 'JS-2', 'compression',
#             'bert_f_score', 'S3_Responsive', 'rouge_1_precision', 'rouge_we_3_r', 'rouge_l_precision',
#             'percentage_novel_3-gram', 'bert_recall_score', 'rouge_su*_recall', 'percentage_repeated_3-gram_in_summ',
#             'rouge_we_2_p'},
#     # 'svr_lin': {'rouge_2_f_score', 'density', 'percentage_repeated_2-gram_in_summ',
#     #             'rouge_w_1.2_precision', 'rouge_we_1_p', 'rouge_4_precision', 'cider',
#     #             'rouge_w_1.2_f_score', 'rouge_l_f_score', 'rouge_su*_precision',
#     #             'rouge_we_1_r',
#     #             'rouge_3_recall', 'mover_score', 'S3_Pyramid', 'rouge_l_precision',
#     #             'percentage_novel_3-gram', 'percentage_novel_1-gram', 'rouge_su*_recall',
#     #             'percentage_repeated_3-gram_in_summ', 'rouge_we_2_p'},
#     # 'ridge': {'JS-2',
#     #           'coverage',
#     #           'density',
#     #           'percentage_repeated_2-gram_in_summ',
#     #           'meteor',
#     #           'JS-1',
#     #           'rouge_we_2_r',
#     #           'rouge_we_2_f',
#     #           'rouge_4_precision',
#     #           'rouge_l_f_score',
#     #           'rouge_su*_precision',
#     #           'bert_f_score',
#     #           'S3_Responsive',
#     #           'bleu',
#     #           'S3_Pyramid',
#     #           'rouge_l_precision',
#     #           'percentage_novel_3-gram',
#     #           'percentage_novel_1-gram',
#     #           'bert_recall_score',
#     #           'percentage_repeated_3-gram_in_summ'},
#     # 'lasso': {
#     #     'rouge_2_f_score', 'compression', 'rouge_4_recall', 'density', 'rouge_3_f_score', 'rouge_2_precision',
#     #     'rouge_we_3_p', 'rouge_4_f_score', 'rouge_we_3_f', 'rouge_4_precision', 'rouge_2_recall', 'rouge_3_precision',
#     #     'rouge_1_precision', 'bleu', 'rouge_3_recall', 'rouge_1_recall', 'rouge_l_recall', 'rouge_we_3_r',
#     #     'rouge_1_f_score', 'glove_sms'},
#     # 'elastic_net': {'rouge_2_f_score', 'compression', 'rouge_4_recall', 'density',
#     #                 'rouge_3_f_score', 'rouge_2_precision', 'rouge_we_3_p',
#     #                 'rouge_we_3_f', 'cider', 'rouge_2_recall', 'rouge_3_precision',
#     #                 'rouge_1_precision', 'bleu', 'rouge_3_recall',
#     #                 'rouge_1_recall',
#     #                 'rouge_we_3_r', 'percentage_novel_3-gram', 'rouge_1_f_score',
#     #                 'percentage_novel_1-gram', 'glove_sms'},
#     'ada_boosting': {
#         'coverage',
#         'rouge_4_recall',
#         'summary_length',
#         'density',
#         'percentage_repeated_2-gram_in_summ',
#         'percentage_novel_2-gram',
#         'rouge_w_1.2_precision',
#         'rouge_4_precision',
#         'cider',
#         'JS-2',
#         'rouge_w_1.2_f_score',
#         'rouge_l_f_score',
#         'bert_f_score',
#         'rouge_3_recall',
#         'rouge_we_1_f',
#         'rouge_l_precision',
#         'percentage_novel_3-gram',
#         'rouge_1_f_score',
#         'bert_recall_score',
#         'rouge_we_2_p'},
#     'bagging': {
#         'JS-2', 'coverage', 'rouge_4_recall', 'summary_length', 'density', 'percentage_repeated_2-gram_in_summ',
#         'rouge_su*_f_score', 'percentage_novel_2-gram', 'rouge_s*_f_score', 'rouge_we_3_f', 'rouge_l_f_score',
#         'rouge_we_1_r', 'rouge_3_precision', 'bleu', 'rouge_l_recall', 'rouge_s*_recall', 'bert_recall_score',
#         'rouge_su*_recall', 'rouge_w_1.2_recall', 'percentage_repeated_3-gram_in_summ'}
# }
#
# trained_regrsor_coherence_cv_dir = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/wodeutil/nlp/metrics/experiments/trained_regressors/coherence_cv"
#
#
# def get_exp_path(top_combine=False, worst_combine=False, curr_remove="", four_quality_bar_path=False,
#                  is_best_remove=False,
#                  is_four_quality=False, four_quality_corr_type=None, path_name=None, is_regr_diff_feature_set=False,
#                  is_worst_remove=False, step=1, end=30):
#     top_combine_dir = "/Users/jackz/OneDrive/One Drive Sync/Research/Workspace/Multifacet_evaluation_metric/experiment_results/figs/top_feature_combine_ablation"
#     top_combine_name = "top_combine_abalation_step" + str(step) + ".csv"
#     top_combine_path = os.path.join(top_combine_dir, top_combine_name)
#     worst_combine_dir = "/Users/jackz/OneDrive/One Drive Sync/Research/Workspace/Multifacet_evaluation_metric/experiment_results/figs/worst_feature_combine_ablation"
#     worst_combine_name = "worst_combine_abalation_step" + str(step) + ".csv"
#     worst_combine_path = os.path.join(worst_combine_dir, worst_combine_name)
#
#     best_worst_dir = "/Users/jackz/OneDrive/One Drive Sync/Research/Workspace/Multifacet_evaluation_metric/experiment_results/figs/best_worst_feature_removal_ablation"
#     best_remove_name = "best_feature_remove_abalation_step" + str(step) + "_end" + str(end) + ".csv"
#     best_remove_path = os.path.join(best_worst_dir, best_remove_name)
#     worst_remove_name = "worst_feature_remove_abalation_step" + str(step) + "_end" + str(end) + ".csv"
#     worst_remove_path = os.path.join(best_worst_dir, worst_remove_name)
#
#     four_quality_path = "/Users/jackz/OneDrive/One Drive Sync/Research/Workspace/Multifacet_evaluation_metric/experiment_results/figs/four_quality_bar_chart/four_quality_bar_corrs.csv"
#     four_quality_dir = "/Users/jackz/OneDrive/One Drive Sync/Research/Workspace/Multifacet_evaluation_metric/experiment_results/figs/four_quality_bar_chart"
#
#     regr_pca_features_path = "/Users/jackz/OneDrive/One Drive Sync/Research/Workspace/Multifacet_evaluation_metric/experiment_results/csvs/regr_test_pca_features/regr_pca_featues_ablation.csv"
#     if not path_name is None:
#         if path_name == "regr_pca_features":
#             return regr_pca_features_path
#     if top_combine:
#         return top_combine_path
#     elif worst_combine:
#         return worst_combine_path
#     elif is_best_remove or curr_remove == "best":
#         return best_remove_path
#     elif is_worst_remove or curr_remove == "worst":
#         return worst_remove_path
#     elif four_quality_bar_path:
#         return four_quality_path
#     elif is_four_quality:
#         fn = f"mix_four_quality_bar_corrs_{four_quality_corr_type}.csv"
#         return os.path.join(four_quality_dir, fn)
#
#
# random_state = 2021
#
# ## bagging
# # defaultdict(<class 'list'>, {'mix_lin_regr_pearsonr': [0.4922230069385817], 'abs_lin_regr_pearsonr': [0.4598527246980132], 'ext_lin_regr_pearsonr': [0.5942955934837418], 'mix_lin_regr_spearmanr': [0.46871129579564313], 'abs_lin_regr_spearmanr': [0.438075987080514], 'ext_lin_regr_spearmanr': [0.5689722708639199], 'mix_lin_regr_kendalltau': [0.33732083496111986], 'abs_lin_regr_kendalltau': [0.31527407293665016], 'ext_lin_regr_kendalltau': [0.4159232317915916], 'mix_ridge_pearsonr': [0.4845087665305471], 'abs_ridge_pearsonr': [0.45055671150215326], 'ext_ridge_pearsonr': [0.5914443298561847], 'mix_ridge_spearmanr': [0.46412121768194375], 'abs_ridge_spearmanr': [0.43254105867559206], 'ext_ridge_spearmanr': [0.5706484614504778], 'mix_ridge_kendalltau': [0.3331154050074397], 'abs_ridge_kendalltau': [0.3102138470625797], 'ext_ridge_kendalltau': [0.4159232317915916], 'mix_lasso_pearsonr': [0.2582552548169802], 'abs_lasso_pearsonr': [0.2086824550088042], 'ext_lasso_pearsonr': [0.39384067580739673], 'mix_lasso_spearmanr': [0.23149729359863463], 'abs_lasso_spearmanr': [0.17645557770679088], 'ext_lasso_spearmanr': [0.3837009365847581], 'mix_lasso_kendalltau': [0.1646128247587572], 'abs_lasso_kendalltau': [0.12481890489373774], 'ext_lasso_kendalltau': [0.2725878675144484], 'mix_elastic_net_pearsonr': [0.2570958670039647], 'abs_elastic_net_pearsonr': [0.2060441543410912], 'ext_elastic_net_pearsonr': [0.3928708781713067], 'mix_elastic_net_spearmanr': [0.23109826034134887], 'abs_elastic_net_spearmanr': [0.1742897930884467], 'ext_elastic_net_spearmanr': [0.3834884579858187], 'mix_elastic_net_kendalltau': [0.16486020783917107], 'abs_elastic_net_kendalltau': [0.12349884597006719], 'ext_elastic_net_kendalltau': [0.2719205777530911], 'mix_svr_lin_pearsonr': [0.2550354821099697], 'abs_svr_lin_pearsonr': [0.1893377890651614], 'ext_svr_lin_pearsonr': [0.4133245067279029], 'mix_svr_lin_spearmanr': [0.2401562677341294], 'abs_svr_lin_spearmanr': [0.17478868301564687], 'ext_svr_lin_spearmanr': [0.413901033289195], 'mix_svr_lin_kendalltau': [0.16749567830221224], 'abs_svr_lin_kendalltau': [0.11902531295096143], 'ext_svr_lin_kendalltau': [0.30185286669718553], 'mix_mlp_pearsonr': [0.4146840595546787], 'abs_mlp_pearsonr': [0.37710217112472144], 'ext_mlp_pearsonr': [0.5369503269758908], 'mix_mlp_spearmanr': [0.4228618899462378], 'abs_mlp_spearmanr': [0.388076566406354], 'ext_mlp_spearmanr': [0.5087710596565592], 'mix_mlp_kendalltau': [0.30173959917655147], 'abs_mlp_kendalltau': [0.2769923641502041], 'ext_mlp_kendalltau': [0.3618899009573992], 'mix_dtree_pearsonr': [0.11037673479624956], 'abs_dtree_pearsonr': [0.10123918652031212], 'ext_dtree_pearsonr': [0.1403358440878793], 'mix_dtree_spearmanr': [0.10783997165036394], 'abs_dtree_spearmanr': [0.10548561155574909], 'ext_dtree_spearmanr': [0.10105871520856237], 'mix_dtree_kendalltau': [0.08071982144133479], 'abs_dtree_kendalltau': [0.0819470819467176], 'ext_dtree_kendalltau': [0.07567795805461253], 'mix_forest_pearsonr': [0.48312364936737523], 'abs_forest_pearsonr': [0.4708881343613588], 'ext_forest_pearsonr': [0.5205450154119289], 'mix_forest_spearmanr': [0.458134431480933], 'abs_forest_spearmanr': [0.45652705785263487], 'ext_forest_spearmanr': [0.4405666286063377], 'mix_forest_kendalltau': [0.3280853809451948], 'abs_forest_kendalltau': [0.3254678612916616], 'ext_forest_kendalltau': [0.3218652114505901], 'mix_bagging_pearsonr': [0.4815545920929268], 'abs_bagging_pearsonr': [0.4678728704659695], 'ext_bagging_pearsonr': [0.526268281336162], 'mix_bagging_spearmanr': [0.45509500317649515], 'abs_bagging_spearmanr': [0.45040681764336], 'ext_bagging_spearmanr': [0.45607729361152827], 'mix_bagging_kendalltau': [0.3289099750537596], 'abs_bagging_kendalltau': [0.32686125682220274], 'ext_bagging_kendalltau': [0.3298701493519519], 'mix_ada_boosting_pearsonr': [0.45912987089157675], 'abs_ada_boosting_pearsonr': [0.4369355359129808], 'ext_ada_boosting_pearsonr': [0.5336148096452036], 'mix_ada_boosting_spearmanr': [0.4348183702326345], 'abs_ada_boosting_spearmanr': [0.4222313715628728], 'ext_ada_boosting_spearmanr': [0.44609341336945096], 'mix_ada_boosting_kendalltau': [0.3312649594019812], 'abs_ada_boosting_kendalltau': [0.3208846716016444], 'ext_ada_boosting_kendalltau': [0.34690727286431533], 'mix_gradient_boosting_pearsonr': [0.4900800524743264], 'abs_gradient_boosting_pearsonr': [0.45086450754802], 'ext_gradient_boosting_pearsonr': [0.6118045232529543], 'mix_gradient_boosting_spearmanr': [0.46701314919298187], 'abs_gradient_boosting_spearmanr': [0.434614747253132], 'ext_gradient_boosting_spearmanr': [0.5488579838252253], 'mix_gradient_boosting_kendalltau': [0.3362076329145575], 'abs_gradient_boosting_kendalltau': [0.31116722295189736], 'ext_gradient_boosting_kendalltau': [0.4092524502071234], 'mix_nn_keras_pearsonr': [0.4436610037595011], 'abs_nn_keras_pearsonr': [0.42716352649256917], 'ext_nn_keras_pearsonr': [0.49951227029769407], 'mix_nn_keras_spearmanr': [0.43202952459816696], 'abs_nn_keras_spearmanr': [0.41419124990094314], 'ext_nn_keras_spearmanr': [0.48430103792462614], 'mix_nn_keras_kendalltau': [0.3100267699676271], 'abs_nn_keras_kendalltau': [0.2944464765854037], 'ext_nn_keras_kendalltau': [0.35521911937293105]})
#
# ##forest
# # defaultdict(<class 'list'>, {'mix_lin_regr_pearsonr': [0.5059534529229717], 'abs_lin_regr_pearsonr': [0.4891111787603767], 'ext_lin_regr_pearsonr': [0.5600878752200107], 'mix_lin_regr_spearmanr': [0.48812986551844056], 'abs_lin_regr_spearmanr': [0.47721331262929845], 'ext_lin_regr_spearmanr': [0.5231721337100848], 'mix_lin_regr_kendalltau': [0.35241090714785456], 'abs_lin_regr_kendalltau': [0.34556209157420226], 'ext_lin_regr_kendalltau': [0.37856685491856973], 'mix_ridge_pearsonr': [0.5026496222456391], 'abs_ridge_pearsonr': [0.4806358751788285], 'ext_ridge_pearsonr': [0.5738361440468714], 'mix_ridge_spearmanr': [0.48673591954336504], 'abs_ridge_spearmanr': [0.46945847681428005], 'ext_ridge_spearmanr': [0.5496960791185042], 'mix_ridge_kendalltau': [0.35084417834158155], 'abs_ridge_kendalltau': [0.33778841124592013], 'ext_ridge_kendalltau': [0.4012475123057615], 'mix_lasso_pearsonr': [0.2582552548169802], 'abs_lasso_pearsonr': [0.2086824550088042], 'ext_lasso_pearsonr': [0.39384067580739673], 'mix_lasso_spearmanr': [0.23149729359863463], 'abs_lasso_spearmanr': [0.17645557770679088], 'ext_lasso_spearmanr': [0.3837009365847581], 'mix_lasso_kendalltau': [0.1646128247587572], 'abs_lasso_kendalltau': [0.12481890489373774], 'ext_lasso_kendalltau': [0.2725878675144484], 'mix_elastic_net_pearsonr': [0.25709586135168744], 'abs_elastic_net_pearsonr': [0.20604420056787043], 'ext_elastic_net_pearsonr': [0.392870982216947], 'mix_elastic_net_spearmanr': [0.23109826034134887], 'abs_elastic_net_spearmanr': [0.1742897930884467], 'ext_elastic_net_spearmanr': [0.3834884579858187], 'mix_elastic_net_kendalltau': [0.16486020783917107], 'abs_elastic_net_kendalltau': [0.12349884597006719], 'ext_elastic_net_kendalltau': [0.2719205777530911], 'mix_svr_lin_pearsonr': [0.3350845049368433], 'abs_svr_lin_pearsonr': [0.2967678524320981], 'ext_svr_lin_pearsonr': [0.4416557861921861], 'mix_svr_lin_spearmanr': [0.34544375092972857], 'abs_svr_lin_spearmanr': [0.3055083193211488], 'ext_svr_lin_spearmanr': [0.44561880868413184], 'mix_svr_lin_kendalltau': [0.24566719979414933], 'abs_svr_lin_kendalltau': [0.21311617956592355], 'ext_svr_lin_kendalltau': [0.32720183671816466], 'mix_mlp_pearsonr': [0.4443974297411014], 'abs_mlp_pearsonr': [0.4122948687771644], 'ext_mlp_pearsonr': [0.5568586031930838], 'mix_mlp_spearmanr': [0.429410121494409], 'abs_mlp_spearmanr': [0.40189446432000586], 'ext_mlp_spearmanr': [0.533359122979095], 'mix_mlp_kendalltau': [0.30854250057221055], 'abs_mlp_kendalltau': [0.2882128650014038], 'ext_mlp_kendalltau': [0.38523763650303794], 'mix_dtree_pearsonr': [0.23494281988895693], 'abs_dtree_pearsonr': [0.23090118090918474], 'ext_dtree_pearsonr': [0.22863319805323912], 'mix_dtree_spearmanr': [0.22894427548308063], 'abs_dtree_spearmanr': [0.2310071002730797], 'ext_dtree_spearmanr': [0.20367511933861593], 'mix_dtree_kendalltau': [0.16884101689726122], 'abs_dtree_kendalltau': [0.17008263367546025], 'ext_dtree_kendalltau': [0.1529801363062034], 'mix_forest_pearsonr': [0.47873588926519506], 'abs_forest_pearsonr': [0.4691152267631568], 'ext_forest_pearsonr': [0.5053504499690645], 'mix_forest_spearmanr': [0.46070761523994275], 'abs_forest_spearmanr': [0.4584850807865107], 'ext_forest_spearmanr': [0.4290811818406981], 'mix_forest_kendalltau': [0.32903366417004426], 'abs_forest_kendalltau': [0.3294280380626733], 'ext_forest_kendalltau': [0.30385410117252604], 'mix_bagging_pearsonr': [0.4986166840573728], 'abs_bagging_pearsonr': [0.492217549585085], 'ext_bagging_pearsonr': [0.5149148388327812], 'mix_bagging_spearmanr': [0.4845177592770826], 'abs_bagging_spearmanr': [0.48765726554097505], 'ext_bagging_spearmanr': [0.45881585851351014], 'mix_bagging_kendalltau': [0.35084417834158155], 'abs_bagging_kendalltau': [0.35502251386050787], 'ext_bagging_kendalltau': [0.3365409309364201], 'mix_ada_boosting_pearsonr': [0.4793497077603637], 'abs_ada_boosting_pearsonr': [0.46369485103684166], 'ext_ada_boosting_pearsonr': [0.5303733848213795], 'mix_ada_boosting_spearmanr': [0.46153674990636306], 'abs_ada_boosting_spearmanr': [0.45336835099251344], 'ext_ada_boosting_spearmanr': [0.4643475577221924], 'mix_ada_boosting_kendalltau': [0.35309581185208416], 'abs_ada_boosting_kendalltau': [0.3474323539299431], 'ext_ada_boosting_kendalltau': [0.360560803296121], 'mix_gradient_boosting_pearsonr': [0.48775527833716514], 'abs_gradient_boosting_pearsonr': [0.4770385002888317], 'ext_gradient_boosting_pearsonr': [0.5219492716201732], 'mix_gradient_boosting_spearmanr': [0.47346894205165674], 'abs_gradient_boosting_spearmanr': [0.4705422420717062], 'ext_gradient_boosting_spearmanr': [0.4650130420201502], 'mix_gradient_boosting_kendalltau': [0.3441649660622072], 'abs_gradient_boosting_kendalltau': [0.34028185587952003], 'ext_gradient_boosting_kendalltau': [0.3485483377884629], 'mix_nn_keras_pearsonr': [0.43277242976174934], 'abs_nn_keras_pearsonr': [0.4289281444901718], 'ext_nn_keras_pearsonr': [0.44914421505146124], 'mix_nn_keras_spearmanr': [0.43763108821267077], 'abs_nn_keras_spearmanr': [0.43658586441081887], 'ext_nn_keras_spearmanr': [0.41171726386304563], 'mix_nn_keras_kendalltau': [0.31225317406075187], 'abs_nn_keras_kendalltau': [0.3104338568831915], 'ext_nn_keras_kendalltau': [0.2998516322218451]})
#
# ##lin_reg
# # defaultdict(<class 'list'>, {'mix_lin_regr_pearsonr': [0.5059534529229717], 'abs_lin_regr_pearsonr': [0.4891111787603767], 'ext_lin_regr_pearsonr': [0.5600878752200107], 'mix_lin_regr_spearmanr': [0.48812986551844056], 'abs_lin_regr_spearmanr': [0.47721331262929845], 'ext_lin_regr_spearmanr': [0.5231721337100848], 'mix_lin_regr_kendalltau': [0.35241090714785456], 'abs_lin_regr_kendalltau': [0.34556209157420226], 'ext_lin_regr_kendalltau': [0.37856685491856973], 'mix_ridge_pearsonr': [0.5026496222456391], 'abs_ridge_pearsonr': [0.4806358751788285], 'ext_ridge_pearsonr': [0.5738361440468714], 'mix_ridge_spearmanr': [0.48673591954336504], 'abs_ridge_spearmanr': [0.46945847681428005], 'ext_ridge_spearmanr': [0.5496960791185042], 'mix_ridge_kendalltau': [0.35084417834158155], 'abs_ridge_kendalltau': [0.33778841124592013], 'ext_ridge_kendalltau': [0.4012475123057615], 'mix_lasso_pearsonr': [0.2582552548169802], 'abs_lasso_pearsonr': [0.2086824550088042], 'ext_lasso_pearsonr': [0.39384067580739673], 'mix_lasso_spearmanr': [0.23149729359863463], 'abs_lasso_spearmanr': [0.17645557770679088], 'ext_lasso_spearmanr': [0.3837009365847581], 'mix_lasso_kendalltau': [0.1646128247587572], 'abs_lasso_kendalltau': [0.12481890489373774], 'ext_lasso_kendalltau': [0.2725878675144484], 'mix_elastic_net_pearsonr': [0.25709586135168744], 'abs_elastic_net_pearsonr': [0.20604420056787043], 'ext_elastic_net_pearsonr': [0.392870982216947], 'mix_elastic_net_spearmanr': [0.23109826034134887], 'abs_elastic_net_spearmanr': [0.1742897930884467], 'ext_elastic_net_spearmanr': [0.3834884579858187], 'mix_elastic_net_kendalltau': [0.16486020783917107], 'abs_elastic_net_kendalltau': [0.12349884597006719], 'ext_elastic_net_kendalltau': [0.2719205777530911], 'mix_svr_lin_pearsonr': [0.3350845049368433], 'abs_svr_lin_pearsonr': [0.2967678524320981], 'ext_svr_lin_pearsonr': [0.4416557861921861], 'mix_svr_lin_spearmanr': [0.34544375092972857], 'abs_svr_lin_spearmanr': [0.3055083193211488], 'ext_svr_lin_spearmanr': [0.44561880868413184], 'mix_svr_lin_kendalltau': [0.24566719979414933], 'abs_svr_lin_kendalltau': [0.21311617956592355], 'ext_svr_lin_kendalltau': [0.32720183671816466], 'mix_mlp_pearsonr': [0.4443974297411014], 'abs_mlp_pearsonr': [0.4122948687771644], 'ext_mlp_pearsonr': [0.5568586031930838], 'mix_mlp_spearmanr': [0.429410121494409], 'abs_mlp_spearmanr': [0.40189446432000586], 'ext_mlp_spearmanr': [0.533359122979095], 'mix_mlp_kendalltau': [0.30854250057221055], 'abs_mlp_kendalltau': [0.2882128650014038], 'ext_mlp_kendalltau': [0.38523763650303794], 'mix_dtree_pearsonr': [0.23494281988895693], 'abs_dtree_pearsonr': [0.23090118090918474], 'ext_dtree_pearsonr': [0.22863319805323912], 'mix_dtree_spearmanr': [0.22894427548308063], 'abs_dtree_spearmanr': [0.2310071002730797], 'ext_dtree_spearmanr': [0.20367511933861593], 'mix_dtree_kendalltau': [0.16884101689726122], 'abs_dtree_kendalltau': [0.17008263367546025], 'ext_dtree_kendalltau': [0.1529801363062034], 'mix_forest_pearsonr': [0.47873588926519506], 'abs_forest_pearsonr': [0.4691152267631568], 'ext_forest_pearsonr': [0.5053504499690645], 'mix_forest_spearmanr': [0.46070761523994275], 'abs_forest_spearmanr': [0.4584850807865107], 'ext_forest_spearmanr': [0.4290811818406981], 'mix_forest_kendalltau': [0.32903366417004426], 'abs_forest_kendalltau': [0.3294280380626733], 'ext_forest_kendalltau': [0.30385410117252604], 'mix_bagging_pearsonr': [0.4986166840573728], 'abs_bagging_pearsonr': [0.492217549585085], 'ext_bagging_pearsonr': [0.5149148388327812], 'mix_bagging_spearmanr': [0.4845177592770826], 'abs_bagging_spearmanr': [0.48765726554097505], 'ext_bagging_spearmanr': [0.45881585851351014], 'mix_bagging_kendalltau': [0.35084417834158155], 'abs_bagging_kendalltau': [0.35502251386050787], 'ext_bagging_kendalltau': [0.3365409309364201], 'mix_ada_boosting_pearsonr': [0.4793497077603637], 'abs_ada_boosting_pearsonr': [0.46369485103684166], 'ext_ada_boosting_pearsonr': [0.5303733848213795], 'mix_ada_boosting_spearmanr': [0.46153674990636306], 'abs_ada_boosting_spearmanr': [0.45336835099251344], 'ext_ada_boosting_spearmanr': [0.4643475577221924], 'mix_ada_boosting_kendalltau': [0.35309581185208416], 'abs_ada_boosting_kendalltau': [0.3474323539299431], 'ext_ada_boosting_kendalltau': [0.360560803296121], 'mix_gradient_boosting_pearsonr': [0.48775527833716514], 'abs_gradient_boosting_pearsonr': [0.4770385002888317], 'ext_gradient_boosting_pearsonr': [0.5219492716201732], 'mix_gradient_boosting_spearmanr': [0.47346894205165674], 'abs_gradient_boosting_spearmanr': [0.4705422420717062], 'ext_gradient_boosting_spearmanr': [0.4650130420201502], 'mix_gradient_boosting_kendalltau': [0.3441649660622072], 'abs_gradient_boosting_kendalltau': [0.34028185587952003], 'ext_gradient_boosting_kendalltau': [0.3485483377884629], 'mix_nn_keras_pearsonr': [0.43277242976174934], 'abs_nn_keras_pearsonr': [0.4289281444901718], 'ext_nn_keras_pearsonr': [0.44914421505146124], 'mix_nn_keras_spearmanr': [0.43763108821267077], 'abs_nn_keras_spearmanr': [0.43658586441081887], 'ext_nn_keras_spearmanr': [0.41171726386304563], 'mix_nn_keras_kendalltau': [0.31225317406075187], 'abs_nn_keras_kendalltau': [0.3104338568831915], 'ext_nn_keras_kendalltau': [0.2998516322218451]})
#
# # defaultdict(<class 'list'>, {'mix_lin_regr_pearsonr': [0.47329343571766913], 'abs_lin_regr_pearsonr': [0.4573161282281985], 'ext_lin_regr_pearsonr': [0.5192162636865496], 'mix_lin_regr_spearmanr': [0.45547448436686244], 'abs_lin_regr_spearmanr': [0.44524027340165495], 'ext_lin_regr_spearmanr': [0.48092504843338985], 'mix_lin_regr_kendalltau': [0.3266771693543525], 'abs_lin_regr_kendalltau': [0.3207009818450735], 'ext_lin_regr_kendalltau': [0.34321171252088833], 'mix_ridge_pearsonr': [0.4815094367808788], 'abs_ridge_pearsonr': [0.4573880073501499], 'ext_ridge_pearsonr': [0.5527688302967825], 'mix_ridge_spearmanr': [0.46510320497674146], 'abs_ridge_spearmanr': [0.4468334039653302], 'ext_ridge_spearmanr': [0.5176950039061209], 'mix_ridge_kendalltau': [0.3329851906721055], 'abs_ridge_kendalltau': [0.31842754703208537], 'ext_ridge_kendalltau': [0.3778997767601229], 'mix_lasso_pearsonr': [0.24646611696127177], 'abs_lasso_pearsonr': [0.22858028834740154], 'ext_lasso_pearsonr': [0.3915141446464107], 'mix_lasso_spearmanr': [0.23074792477623912], 'abs_lasso_spearmanr': [0.2143409237379878], 'ext_lasso_spearmanr': [0.4015396879234923], 'mix_lasso_kendalltau': [0.16725813328834016], 'abs_lasso_kendalltau': [0.1550729589434347], 'ext_lasso_kendalltau': [0.2963236634287337], 'mix_elastic_net_pearsonr': [0.25042295362921496], 'abs_elastic_net_pearsonr': [0.2327986104879445], 'ext_elastic_net_pearsonr': [0.3938329024036327], 'mix_elastic_net_spearmanr': [0.2377028407931573], 'abs_elastic_net_spearmanr': [0.22435643077816994], 'ext_elastic_net_spearmanr': [0.4036503161518476], 'mix_elastic_net_kendalltau': [0.16943679482682844], 'abs_elastic_net_kendalltau': [0.15782037798550155], 'ext_elastic_net_kendalltau': [0.29861216820738223], 'mix_svr_lin_pearsonr': [0.41655311877132467], 'abs_svr_lin_pearsonr': [0.40632487028566117], 'ext_svr_lin_pearsonr': [0.5100202725974687], 'mix_svr_lin_spearmanr': [0.42084699762050315], 'abs_svr_lin_spearmanr': [0.4099251517833099], 'ext_svr_lin_spearmanr': [0.5080392017948228], 'mix_svr_lin_kendalltau': [0.2981467722832082], 'abs_svr_lin_kendalltau': [0.2879195185739215], 'ext_svr_lin_kendalltau': [0.3712289951756547], 'mix_mlp_pearsonr': [0.5162351895047337], 'abs_mlp_pearsonr': [0.5009845369343149], 'ext_mlp_pearsonr': [0.5643988833098317], 'mix_mlp_spearmanr': [0.5079813028465706], 'abs_mlp_spearmanr': [0.49659494624025646], 'ext_mlp_spearmanr': [0.53136422009791], 'mix_mlp_kendalltau': [0.3669165733159664], 'abs_mlp_kendalltau': [0.3580293147422019], 'ext_mlp_kendalltau': [0.38924010545371884], 'mix_dtree_pearsonr': [0.25802761712542843], 'abs_dtree_pearsonr': [0.25944447718456154], 'ext_dtree_pearsonr': [0.23799803976628892], 'mix_dtree_spearmanr': [0.2687806755577865], 'abs_dtree_spearmanr': [0.2699802414249923], 'ext_dtree_spearmanr': [0.24450906998973937], 'mix_dtree_kendalltau': [0.19935417658658244], 'abs_dtree_kendalltau': [0.1993411858471529], 'ext_dtree_kendalltau': [0.1839203546581207], 'mix_forest_pearsonr': [0.4646139823204022], 'abs_forest_pearsonr': [0.4457368965939843], 'ext_forest_pearsonr': [0.52265884806261], 'mix_forest_spearmanr': [0.45131253479286204], 'abs_forest_spearmanr': [0.4409187430698884], 'ext_forest_spearmanr': [0.49034476736348276], 'mix_forest_kendalltau': [0.32177093055165573], 'abs_forest_kendalltau': [0.3120472622343443], 'ext_forest_kendalltau': [0.3605557446405056], 'mix_bagging_pearsonr': [0.4710543883222546], 'abs_bagging_pearsonr': [0.4598647233840558], 'ext_bagging_pearsonr': [0.5019861593573169], 'mix_bagging_spearmanr': [0.46132025173387614], 'abs_bagging_spearmanr': [0.4585130151306409], 'ext_bagging_spearmanr': [0.45267769580217143], 'mix_bagging_kendalltau': [0.33257941883687264], 'abs_bagging_kendalltau': [0.3303080773451203], 'ext_bagging_kendalltau': [0.3285359930350583], 'mix_ada_boosting_pearsonr': [0.46563583542533177], 'abs_ada_boosting_pearsonr': [0.4473523699284606], 'ext_ada_boosting_pearsonr': [0.5308247318345011], 'mix_ada_boosting_spearmanr': [0.44750136574846544], 'abs_ada_boosting_spearmanr': [0.43329983507585346], 'ext_ada_boosting_spearmanr': [0.48659758610267867], 'mix_ada_boosting_kendalltau': [0.34404511352556205], 'abs_ada_boosting_kendalltau': [0.3308099704807145], 'ext_ada_boosting_kendalltau': [0.3793996931207225], 'mix_gradient_boosting_pearsonr': [0.48417890717344736], 'abs_gradient_boosting_pearsonr': [0.4453238338897443], 'ext_gradient_boosting_pearsonr': [0.6130548236689197], 'mix_gradient_boosting_spearmanr': [0.4843671320898156], 'abs_gradient_boosting_spearmanr': [0.45177560050573956], 'ext_gradient_boosting_spearmanr': [0.587646450497262], 'mix_gradient_boosting_kendalltau': [0.351091556574151], 'abs_gradient_boosting_kendalltau': [0.32319442647867347], 'ext_gradient_boosting_kendalltau': [0.4412722018125707], 'mix_nn_keras_pearsonr': [0.5141827068261563], 'abs_nn_keras_pearsonr': [0.5077175063657685], 'ext_nn_keras_pearsonr': [0.5461054228881279], 'mix_nn_keras_spearmanr': [0.507010512732681], 'abs_nn_keras_spearmanr': [0.500828308798362], 'ext_nn_keras_spearmanr': [0.5566369246459412], 'mix_nn_keras_kendalltau': [0.36386563490084406], 'abs_nn_keras_kendalltau': [0.3586893442040372], 'ext_nn_keras_kendalltau': [0.4025816686226552]})
# summeval_features_wrapper20_dict_newJS = {
#     "filter_20": set(features_coherence_filter_all_newjs[:20]),
#     'forest': {'rouge_su*_f_score', 'rouge_3_f_score', 'rouge_s*_recall', 'density',
#                'percentage_repeated_3-gram_in_summ',
#                'bert_recall_score', 'percentage_repeated_2-gram_in_summ', 'percentage_novel_1-gram',
#                'rouge_w_1.2_f_score',
#                'rouge_su*_recall', 'bleu', 'rouge_4_precision', 'rouge_l_f_score', 'rouge_w_1.2_recall', 'coverage',
#                'rouge_we_3_p', 'rouge_1_f_score', 'rouge_3_precision', 'summary_length', 'percentage_novel_3-gram'},
#     'dtree': {'rouge_4_recall', 'rouge_3_f_score', 'cider', 'rouge_s*_recall', 'rouge_3_recall',
#               'percentage_repeated_3-gram_in_summ', 'bert_recall_score', 'percentage_repeated_2-gram_in_summ',
#               'rouge_we_2_f', 'rouge_su*_recall', 'rouge_l_recall', 'JS-1', 'rouge_we_1_r', 'rouge_w_1.2_recall',
#               'coverage', 'rouge_1_f_score', 'percentage_novel_2-gram', 'rouge_1_recall', 'summary_length',
#               'rouge_we_1_f'},
#     'lin_regr': {'rouge_we_2_p', 'rouge_3_f_score', 'density', 'percentage_repeated_3-gram_in_summ',
#                  'bert_recall_score',
#                  'bert_f_score', 'bert_precision_score', 'percentage_repeated_2-gram_in_summ',
#                  'percentage_novel_1-gram',
#                  'rouge_l_precision', 'bleu', 'rouge_4_precision', 'rouge_l_recall', 'rouge_l_f_score', 'JS-1',
#                  'meteor',
#                  'coverage', 'percentage_repeated_1-gram_in_summ', 'rouge_1_recall', 'percentage_novel_3-gram'},
#     'mlp': {'rouge_s*_recall', 'density', 'rouge_3_recall', 'bert_recall_score', 'rouge_we_1_p',
#             'percentage_repeated_2-gram_in_summ', 'rouge_1_precision', 'rouge_we_2_r', 'rouge_l_precision', 'JS-2',
#             'rouge_4_f_score', 'JS-1', 'rouge_w_1.2_precision', 'coverage', 'rouge_we_3_p', 'rouge_1_f_score',
#             'rouge_3_precision', 'percentage_novel_2-gram', 'rouge_1_recall', 'percentage_novel_3-gram'},
#     'svr_lin': {'density', 'percentage_repeated_3-gram_in_summ', 'rouge_we_3_r', 'rouge_2_recall', 'rouge_we_1_p',
#                 'percentage_repeated_2-gram_in_summ', 'percentage_novel_1-gram', 'rouge_l_precision', 'JS-2',
#                 'rouge_4_precision', 'rouge_4_f_score', 'rouge_l_f_score', 'rouge_w_1.2_precision', 'meteor',
#                 'rouge_we_3_p', 'percentage_repeated_1-gram_in_summ', 'rouge_2_f_score', 'rouge_su*_precision',
#                 'rouge_1_recall', 'percentage_novel_3-gram'},
#     'ridge': {'rouge_3_f_score', 'density', 'rouge_3_recall', 'percentage_repeated_3-gram_in_summ',
#               'bert_recall_score',
#               'percentage_repeated_2-gram_in_summ', 'rouge_we_2_r', 'percentage_novel_1-gram', 'rouge_we_2_f',
#               'rouge_l_precision', 'bleu', 'JS-2', 'rouge_4_precision', 'rouge_4_f_score', 'rouge_l_f_score', 'JS-1',
#               'meteor', 'coverage', 'rouge_su*_precision', 'percentage_novel_3-gram'},
#     # 'lasso': {'glove_sms', 'rouge_4_recall', 'rouge_3_f_score', 'cider', 'density', 'rouge_3_recall', 'rouge_we_3_r',
#     #           'rouge_2_recall', 'rouge_1_precision', 'compression', 'percentage_novel_1-gram', 'bleu', 'rouge_we_3_f',
#     #           'rouge_we_3_p', 'rouge_1_f_score', 'percentage_repeated_1-gram_in_summ', 'rouge_2_f_score',
#     #           'rouge_3_precision', 'rouge_2_precision', 'rouge_1_recall'},
#     # 'elastic_net': {'glove_sms', 'rouge_4_recall', 'rouge_3_f_score', 'cider', 'density', 'rouge_3_recall',
#     #                 'rouge_we_3_r',
#     #                 'rouge_2_recall', 'rouge_1_precision', 'compression', 'percentage_novel_1-gram', 'bleu',
#     #                 'rouge_we_3_f', 'rouge_we_3_p', 'rouge_1_f_score', 'rouge_2_f_score', 'rouge_3_precision',
#     #                 'rouge_2_precision', 'rouge_1_recall', 'percentage_novel_3-gram'},
#     'ada_boosting': {'rouge_3_f_score', 'rouge_s*_recall', 'density', 'rouge_3_recall', 'bert_recall_score',
#                      'percentage_repeated_2-gram_in_summ', 'percentage_novel_1-gram', 'rouge_l_precision', 'bleu',
#                      'JS-2',
#                      'rouge_s*_f_score', 'rouge_4_f_score', 'rouge_l_f_score', 'rouge_w_1.2_precision',
#                      'rouge_w_1.2_recall', 'coverage', 'rouge_we_3_f', 'percentage_repeated_1-gram_in_summ',
#                      'rouge_2_f_score', 'summary_length'},
#     'bagging': {'rouge_su*_f_score', 'cider', 'rouge_s*_recall', 'density', 'rouge_3_recall',
#                 'percentage_repeated_3-gram_in_summ', 'bert_recall_score', 'rouge_we_3_r',
#                 'percentage_repeated_2-gram_in_summ', 'percentage_novel_1-gram', 'rouge_w_1.2_f_score',
#                 'rouge_s*_f_score',
#                 'rouge_4_f_score', 'rouge_l_f_score', 'meteor', 'rouge_w_1.2_recall', 'coverage', 'rouge_1_f_score',
#                 'rouge_3_precision', 'percentage_novel_2-gram'},
#     'voting': {'rouge_su*_f_score', 'rouge_3_f_score', 'density', 'bert_recall_score', 'rouge_we_3_r',
#                'rouge_2_recall',
#                'percentage_repeated_2-gram_in_summ', 'rouge_we_2_r', 'rouge_we_2_f', 'rouge_l_precision',
#                'rouge_4_precision', 'rouge_s*_f_score', 'rouge_l_recall', 'rouge_w_1.2_precision', 'coverage',
#                'rouge_we_3_f', 'rouge_s*_precision', 'rouge_3_precision', 'percentage_novel_2-gram',
#                'percentage_novel_3-gram'},
#     'gradient_boosting': {'rouge_we_1_f', 'density', 'percentage_repeated_3-gram_in_summ', 'bert_precision_score',
#                           'rouge_2_recall', 'rouge_we_1_p', 'percentage_repeated_2-gram_in_summ',
#                           'percentage_novel_1-gram', 'rouge_4_precision', 'rouge_s*_f_score', 'rouge_4_f_score',
#                           'rouge_l_f_score', 'rouge_w_1.2_recall', 'coverage', 'rouge_we_3_f', 'rouge_we_3_p',
#                           'rouge_s*_precision', 'rouge_su*_precision', 'rouge_1_recall', 'percentage_novel_3-gram'},
#     'stacking': {'mover_score', 'rouge_su*_f_score', 'rouge_we_2_p', 'rouge_3_f_score', 'density', 'rouge_3_recall',
#                  'percentage_repeated_3-gram_in_summ', 'rouge_we_3_r', 'percentage_repeated_2-gram_in_summ', 'JS-2',
#                  'rouge_s*_f_score', 'rouge_l_recall', 'rouge_l_f_score', 'JS-1', 'rouge_we_3_f', 'rouge_s*_precision',
#                  'percentage_repeated_1-gram_in_summ', 'rouge_2_f_score', 'percentage_novel_2-gram',
#                  'percentage_novel_3-gram'}}
#
# # these are using the new js
# #  regr name: forest
# # features: rouge_1_f_score, rouge_3_precision, percentage_repeated_2-gram_in_summ, density, bleu, rouge_w_1.2_recall, rouge_4_precision, percentage_repeated_3-gram_in_summ, rouge_l_f_score, rouge_su*_f_score, rouge_3_f_score, rouge_s*_recall, percentage_novel_1-gram, coverage, bert_recall_score, rouge_su*_recall, rouge_we_3_p, percentage_novel_3-gram, summary_length, rouge_w_1.2_f_score
# #  regr name: lin_regr
# # features: percentage_repeated_2-gram_in_summ, rouge_l_recall, density, bleu, rouge_l_precision, rouge_4_precision, bert_precision_score, percentage_repeated_3-gram_in_summ, rouge_l_f_score, bert_f_score, rouge_1_recall, rouge_3_f_score, percentage_novel_1-gram, coverage, bert_recall_score, percentage_repeated_1-gram_in_summ, percentage_novel_3-gram, rouge_we_2_p, JS-1, meteor
# #  regr name: bagging
# # features: rouge_1_f_score, percentage_repeated_2-gram_in_summ, percentage_novel_2-gram, density, rouge_w_1.2_recall, cider, bert_precision_score, rouge_l_f_score, mover_score, rouge_s*_recall, rouge_s*_f_score, rouge_we_2_r, rouge_we_1_p, percentage_novel_1-gram, rouge_s*_precision, bert_recall_score, percentage_novel_3-gram, rouge_w_1.2_f_score, JS-1, meteor
# #  regr name: mlp
# # features: rouge_w_1.2_precision, rouge_4_f_score, rouge_1_f_score, rouge_3_precision, percentage_repeated_2-gram_in_summ, percentage_novel_2-gram, density, rouge_l_precision, rouge_1_recall, rouge_3_recall, rouge_s*_recall, rouge_we_2_r, rouge_we_1_p, coverage, rouge_1_precision, JS-2, bert_recall_score, rouge_we_3_p, percentage_novel_3-gram, JS-1
# #  regr name: ada_boosting
# # features: rouge_w_1.2_precision, rouge_4_f_score, percentage_repeated_2-gram_in_summ, density, bleu, rouge_l_precision, rouge_w_1.2_recall, rouge_2_f_score, rouge_l_f_score, rouge_we_3_f, rouge_3_recall, rouge_3_f_score, rouge_s*_recall, rouge_s*_f_score, percentage_novel_1-gram, coverage, JS-2, bert_recall_score, percentage_repeated_1-gram_in_summ, summary_length
# #  regr name: voting
# # features: rouge_3_precision, percentage_repeated_2-gram_in_summ, percentage_novel_2-gram, rouge_we_1_r, density, rouge_4_precision, rouge_l_f_score, rouge_we_3_f, rouge_s*_recall, rouge_we_2_r, coverage, rouge_s*_precision, JS-2, bert_recall_score, percentage_repeated_1-gram_in_summ, rouge_su*_recall, percentage_novel_3-gram, rouge_w_1.2_f_score, JS-1, rouge_su*_precision
# #  regr name: gradient_boosting
# # features: rouge_4_f_score, percentage_repeated_2-gram_in_summ, density, rouge_w_1.2_recall, rouge_4_precision, bert_precision_score, percentage_repeated_3-gram_in_summ, rouge_l_f_score, rouge_we_3_f, rouge_1_recall, rouge_we_1_f, rouge_s*_f_score, rouge_we_1_p, percentage_novel_1-gram, coverage, rouge_s*_precision, rouge_2_recall, rouge_we_3_p, percentage_novel_3-gram, rouge_su*_precision
# curr_feature_sets = {
#     'forest': {'rouge_1_f_score', 'rouge_3_precision', 'percentage_repeated_2-gram_in_summ', 'density', 'bleu',
#                'rouge_w_1.2_recall', 'rouge_4_precision', 'percentage_repeated_3-gram_in_summ', 'rouge_l_f_score',
#                'rouge_su*_f_score', 'rouge_3_f_score', 'rouge_s*_recall', 'percentage_novel_1-gram', 'coverage',
#                'bert_recall_score', 'rouge_su*_recall', 'rouge_we_3_p', 'percentage_novel_3-gram', 'summary_length',
#                'rouge_w_1.2_f_score'},
#     'lin_regr': {'percentage_repeated_2-gram_in_summ', 'rouge_l_recall', 'density', 'bleu', 'rouge_l_precision',
#                  'rouge_4_precision', 'bert_precision_score', 'percentage_repeated_3-gram_in_summ', 'rouge_l_f_score',
#                  'bert_f_score', 'rouge_1_recall', 'rouge_3_f_score', 'percentage_novel_1-gram', 'coverage',
#                  'bert_recall_score', 'percentage_repeated_1-gram_in_summ', 'percentage_novel_3-gram', 'rouge_we_2_p',
#                  'JS-1', 'meteor'},
#     'bagging': {'rouge_1_f_score', 'percentage_repeated_2-gram_in_summ', 'percentage_novel_2-gram', 'density',
#                 'rouge_w_1.2_recall', 'cider', 'bert_precision_score', 'rouge_l_f_score', 'mover_score',
#                 'rouge_s*_recall',
#                 'rouge_s*_f_score', 'rouge_we_2_r', 'rouge_we_1_p', 'percentage_novel_1-gram', 'rouge_s*_precision',
#                 'bert_recall_score', 'percentage_novel_3-gram', 'rouge_w_1.2_f_score', 'JS-1', 'meteor'},
#     'mlp': {'rouge_w_1.2_precision', 'rouge_4_f_score', 'rouge_1_f_score', 'rouge_3_precision',
#             'percentage_repeated_2-gram_in_summ', 'percentage_novel_2-gram', 'density', 'rouge_l_precision',
#             'rouge_1_recall', 'rouge_3_recall', 'rouge_s*_recall', 'rouge_we_2_r', 'rouge_we_1_p', 'coverage',
#             'rouge_1_precision', 'JS-2', 'bert_recall_score', 'rouge_we_3_p', 'percentage_novel_3-gram', 'JS-1'},
#     'ada_boosting': {'rouge_w_1.2_precision', 'rouge_4_f_score', 'percentage_repeated_2-gram_in_summ', 'density',
#                      'bleu',
#                      'rouge_l_precision', 'rouge_w_1.2_recall', 'rouge_2_f_score', 'rouge_l_f_score', 'rouge_we_3_f',
#                      'rouge_3_recall', 'rouge_3_f_score', 'rouge_s*_recall', 'rouge_s*_f_score',
#                      'percentage_novel_1-gram',
#                      'coverage', 'JS-2', 'bert_recall_score', 'percentage_repeated_1-gram_in_summ', 'summary_length'},
#     'voting': {'rouge_3_precision', 'percentage_repeated_2-gram_in_summ', 'percentage_novel_2-gram', 'rouge_we_1_r',
#                'density', 'rouge_4_precision', 'rouge_l_f_score', 'rouge_we_3_f', 'rouge_s*_recall', 'rouge_we_2_r',
#                'coverage', 'rouge_s*_precision', 'JS-2', 'bert_recall_score', 'percentage_repeated_1-gram_in_summ',
#                'rouge_su*_recall', 'percentage_novel_3-gram', 'rouge_w_1.2_f_score', 'JS-1', 'rouge_su*_precision'},
#     'gradient_boosting': {'rouge_4_f_score', 'percentage_repeated_2-gram_in_summ', 'density', 'rouge_w_1.2_recall',
#                           'rouge_4_precision', 'bert_precision_score', 'percentage_repeated_3-gram_in_summ',
#                           'rouge_l_f_score', 'rouge_we_3_f', 'rouge_1_recall', 'rouge_we_1_f', 'rouge_s*_f_score',
#                           'rouge_we_1_p', 'percentage_novel_1-gram', 'coverage', 'rouge_s*_precision', 'rouge_2_recall',
#                           'rouge_we_3_p', 'percentage_novel_3-gram', 'rouge_su*_precision'}}
# feature_format_map = {
#     'percentage_repeated_2-gram_in_summ': 'repeated-bi-gram',
#     'percentage_repeated_3-gram_in_summ': 'repeated-tri-gram',
#     'percentage_repeated_1-gram_in_summ': 'repeated-uni-gram',
#     'percentage_novel_1-gram': 'novel-uni-gram',
#     'percentage_novel_2-gram': 'novel-bi-gram',
#     'percentage_novel_3-gram': 'novel-tri-gram'
# }
#
#
# def format_feature_name(feature_name):
#     import re
#     feature_name = re.sub('\*', '', feature_name)
#     if feature_name in feature_format_map:
#         return feature_format_map[feature_name]
#     else:
#         return feature_name
#
#
# import os
#
# exp_dir = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments"
# analysis_dir = f"{exp_dir}/analysis"
# data_root_dir = f"{exp_dir}/data"
#
# summeval_analysis_general_dir = f"{analysis_dir}/general/summeval"
# summeval_analysis_metric_labels_dir = f"{summeval_analysis_general_dir}/all_data_metrics_labels"
# summeval_analysis_system_metric_labels_dir = f"{summeval_analysis_general_dir}/syst_metric_against_labels"
#
# summevalanalysis_all_metrics_labels_dir = f"{summeval_analysis_general_dir}/metrics_labels_attr_general"
# summeval_analysis_total_attr_dir = f"{summevalanalysis_all_metrics_labels_dir}/total_attr_general"
# summeval_analysis_system_metric_labels_dir = f"{summeval_analysis_general_dir}/syst_metric_against_labels"
# summeval_analysis_abs_stats_file_dir = f"{summevalanalysis_all_metrics_labels_dir}/abs_attr_general/stats_files"
# summeval_analysis_ext_stats_file_dir = f"{summevalanalysis_all_metrics_labels_dir}/ext_attr_general/stats_files"
# summeval_analysis_total_stats_file_dir = f"{summeval_analysis_total_attr_dir}/stats_files"
#
# summeval_analysis_abs_metric_labels_dir = f"{summeval_analysis_system_metric_labels_dir}/abs_syst"
# summeval_analysis_ext_metric_labels_dir = f"{summeval_analysis_system_metric_labels_dir}/ext_syst"
# #
# # summeval_analysis_lin_dir = f"{analysis_dir}/lin_combine/summeval"
# # summeval_analysis_abs_lin_dir = f"{summeval_analysis_lin_dir}/abs_syst"
# # summeval_analysis_ext_lin_dir = f"{summeval_analysis_lin_dir}/ext_syst"
# # summeval_analysis_total_lin_dir = f"{summeval_analysis_lin_dir}/total"
# # summeval_analysis_lin_temp = f"{summeval_analysis_lin_dir}/temp"
#
# summeval_analysis_reg_dir = f"{analysis_dir}/regression_model/summeval"
# summeval_analysis_abs_reg_dir = f"{summeval_analysis_reg_dir}/abs_syst"
# summeval_analysis_ext_reg_dir = f"{summeval_analysis_reg_dir}/ext_syst"
# summeval_analysis_mix_reg_dir = f"{summeval_analysis_reg_dir}/mix"
#
# summeval_data_fnn08_dir = f"{summeval_analysis_reg_dir}/fnn08"
# summeval_data_abs_fnn08_train_path = f"{summeval_data_fnn08_dir}/abs_train_fnn08.csv"
# summeval_data_abs_fnn08_test_path = f"{summeval_data_fnn08_dir}/abs_test_fnn08.csv"
# summeval_data_abs_regr_dir = f"{summeval_analysis_abs_reg_dir}/lin08"
# summeval_data_ext_regr_dir = f"{summeval_analysis_ext_reg_dir}/lin08"
# summeval_data_mix_reg_dir = f"{summeval_analysis_mix_reg_dir}/lin08"
#
# summeval_data_ext_fnn08_train_path = f"{summeval_data_fnn08_dir}/ext_train_fnn08.csv"
# summeval_data_ext_fnn08_test_path = f"{summeval_data_fnn08_dir}/ext_test_fnn08.csv"
#
# summeval_data_mix_fnn08_train_path = f"{summeval_data_fnn08_dir}/mix_train_fnn08.csv"
# summeval_data_mix_fnn08_test_path = f"{summeval_data_fnn08_dir}/mix_test_fnn08.csv"
#
# summeval_data_abs_fnn08_crossvalid_train_path = f"{summeval_data_fnn08_dir}/cross_valid/abs_train_crossvalid5.csv"
# summeval_data_abs_fnn08_crossvalid_test_path = f"{summeval_data_fnn08_dir}/cross_valid/abs_test_crossvalid5.csv"
#
# summeval_analysis_abs_metric_labels_dir = f"{summeval_analysis_system_metric_labels_dir}/abs_syst"
# summeval_analysis_ext_metric_labels_dir = f"{summeval_analysis_system_metric_labels_dir}/ext_syst"
#
# summeval_analysis_lin_dir = f"{analysis_dir}/lin_combine/summeval"
# summeval_analysis_abs_lin_dir = f"{summeval_analysis_lin_dir}/abs_syst"
# summeval_analysis_ext_lin_dir = f"{summeval_analysis_lin_dir}/ext_syst"
# summeval_analysis_lin_temp = f"{summeval_analysis_lin_dir}/temp"
# summeval_analysis_lin_total_dir = f"{summeval_analysis_lin_dir}/all_models"
#
# summeval_data_all_dir = f"{exp_dir}/all_data/summeval"
# summeval_all_dir = f"{summeval_data_all_dir}/sumeval_all"
# summeval_data_all_syst_dir = f"{summeval_data_all_dir}/models"
#
# summeval_data_all_abs_model_dir = f"{summeval_data_all_dir}/abs_models"
# summeval_data_all_ext_model_dir = f"{summeval_data_all_dir}/ext_models"
# summeval_data_all_abs_path = f"{summeval_data_all_dir}/sumeval_all/summeval_abs_all.csv"
# summeval_data_all_ext_path = f"{summeval_data_all_dir}/sumeval_all/summeval_ext_all.csv"
# summeval_data_all_path = f"{summeval_data_all_dir}/sumeval_all/summeval_all_scores.csv"
#
# summeval_data_all_metrics_dir = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/summeval/models_with_docid/with_s3/abs_ext_mix"
# summeval_data_all_metrics_abs_path = f"{summeval_data_all_metrics_dir}/summeval_abs_all_metrics.csv"
# summeval_data_all_metrics_ext_path = f"{summeval_data_all_metrics_dir}/summeval_ext_all_metrics.csv"
# summeval_data_all_metrics_mix_path = f"{summeval_data_all_metrics_dir}/summeval_mix_all_metrics.csv"
#
# summeval_data_mix_minmax = f"{summeval_data_all_dir}/models_with_docid/with_s3/summeval_mix_minmax.csv"
#
# summeval_data_all_with_s3_path = f"{summeval_data_all_dir}/models_with_docid/with_s3/summeval_all_scores.csv"
# summeval_data_all_new_JS = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/summeval/models_with_docid/with_s3/abs_ext_mix/summeval_mix_all_metrics.csv"
#
# summeval_analysis_reg_dir = f"{analysis_dir}/regression_model/summeval"
# summeval_analysis_abs_reg_dir = f"{summeval_analysis_reg_dir}/abs_syst"
# summeval_analysis_ext_reg_dir = f"{summeval_analysis_reg_dir}/ext_syst"
#
# summeval_top_combine_ablation_fig_dir = "/Users/jackz/OneDrive/One Drive Sync/Research/Workspace/Multifacet_evaluation_metric/experiment_results/figs/top_feature_combine_ablation"
# summeval_four_quality_bar_char_dir = "/Users/jackz/OneDrive/One Drive Sync/Research/Workspace/Multifacet_evaluation_metric/experiment_results/figs/four_quality_bar_chart"
# data_by_syst_dir = f"{exp_dir}/data_by_system"
# lin_combine_dir = f"{exp_dir}/lin_combine"
# regr_model_dir = f"{exp_dir}/refression_model"
#
# summeval_data_all_original_name = "summeval_all_original"
# summeval_data_all_name = "summeval_all_scores"
# summeval_annotations_paired_jsonl_path = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/data_annotations/model_annotations.aligned.paired.jsonl"
# score_files_dir = f"{summeval_data_all_dir}/score_files"
# summeval_all_data_original_path = f"{summeval_data_all_dir}/summeval_all_original.csv"
# summeval_data_with_all_scores_path = f"{summeval_data_all_dir}/summeval_data_all_with_scores.csv"
# summeval_train_data08_with_all_scores_path = f"{summeval_data_all_dir}/train/train_data08_all_scores.csv"
# summeval_test_data08_with_all_scores_path = f"{summeval_data_all_dir}/test/test_data08_all_scores.csv"
#
# summeval_analysis_reg_train_dir = f"{summeval_analysis_reg_dir}/train_all08"
# summeval_analysis_general_corr_dir = f"{summevalanalysis_all_metrics_labels_dir}/abs_ext_mix_corr"
#
# features_coherence_filter_all = ['percentage_repeated_2-gram_in_summ',
#                                  'percentage_novel_3-gram', 'percentage_novel_2-gram', 'percentage_novel_1-gram',
#                                  'density',
#                                  'percentage_repeated_3-gram_in_summ', 'percentage_repeated_1-gram_in_summ',
#                                  'rouge_su*_precision', 'rouge_s*_precision', 'rouge_1_precision', 'rouge_we_1_p',
#                                  'rouge_l_precision', 'rouge_1_f_score', 'rouge_we_1_f', 'rouge_w_1.2_precision',
#                                  'rouge_s*_f_score', 'rouge_s*_recall', 'rouge_su*_f_score', 'rouge_l_f_score',
#                                  'rouge_su*_recall', 'bert_recall_score', 'rouge_1_recall', 'rouge_we_1_r',
#                                  'rouge_l_recall',
#                                  'glove_sms', 'meteor', 'rouge_we_2_p', 'mover_score', 'rouge_2_precision',
#                                  'bert_f_score',
#                                  'rouge_2_recall', 'rouge_w_1.2_f_score', 'rouge_we_3_p', 'rouge_2_f_score',
#                                  'rouge_we_2_f',
#                                  'rouge_we_2_r', 'rouge_3_recall', 'rouge_we_3_r', 'rouge_3_precision', 'rouge_we_3_f',
#                                  'rouge_3_f_score', 'rouge_w_1.2_recall', 'bleu', 'rouge_4_recall', 'rouge_4_precision',
#                                  'rouge_4_f_score', 'bert_precision_score', 'summary_length', 'compression',
#                                  'S3_Responsive',
#                                  'coverage', 'S3_Pyramid', 'cider', 'JS-1', 'JS-2']
#
# features_coherence_filter_all_newjs = ['percentage_repeated_2-gram_in_summ',
#                                        'percentage_novel_3-gram', 'percentage_novel_2-gram',
#                                        'percentage_novel_1-gram', 'density', 'percentage_repeated_3-gram_in_summ',
#                                        'percentage_repeated_1-gram_in_summ', 'rouge_su*_precision',
#                                        'rouge_s*_precision', 'rouge_1_precision', 'rouge_we_1_p', 'rouge_l_precision',
#                                        'rouge_1_f_score', 'rouge_we_1_f', 'rouge_w_1.2_precision', 'rouge_s*_f_score',
#                                        'rouge_s*_recall', 'rouge_su*_f_score', 'rouge_l_f_score', 'rouge_su*_recall',
#                                        'bert_recall_score', 'rouge_1_recall', 'rouge_we_1_r', 'rouge_l_recall',
#                                        'glove_sms', 'meteor', 'rouge_we_2_p', 'mover_score', 'rouge_2_precision',
#                                        'bert_f_score', 'rouge_2_recall', 'rouge_w_1.2_f_score', 'rouge_we_3_p',
#                                        'rouge_2_f_score', 'rouge_we_2_f', 'rouge_we_2_r', 'rouge_3_recall',
#                                        'rouge_we_3_r', 'rouge_3_precision', 'rouge_we_3_f', 'rouge_3_f_score',
#                                        'rouge_w_1.2_recall', 'bleu', 'JS-1', 'JS-2', 'rouge_4_recall',
#                                        'rouge_4_precision', 'rouge_4_f_score', 'bert_precision_score', 'summary_length',
#                                        'compression', 'coverage', 'cider']
# features_double_checked = ['rouge_1_recall', 'rouge_we_1_r', 'rouge_l_recall',
#                            'rouge_2_precision', 'rouge_2_recall', 'rouge_2_f_score',
#                            'rouge_we_2_f', 'rouge_we_2_r', 'rouge_3_recall',
#                            'rouge_3_precision', 'rouge_3_f_score', 'rouge_1_f_score',
#                            'rouge_we_1_f', 'meteor']
#
# features_coherence_filter_without_s3 = [
#     'percentage_repeated_2-gram_in_summ',
#     'percentage_novel_3-gram', 'percentage_novel_2-gram', 'percentage_novel_1-gram',
#     'density',
#     'percentage_repeated_3-gram_in_summ', 'percentage_repeated_1-gram_in_summ',
#     'rouge_su*_precision', 'rouge_s*_precision', 'rouge_1_precision',
#     'rouge_we_1_p',
#     'rouge_l_precision', 'rouge_1_f_score', 'rouge_we_1_f', 'rouge_w_1.2_precision',
#     'rouge_s*_f_score', 'rouge_s*_recall', 'rouge_su*_f_score', 'rouge_l_f_score',
#     'rouge_su*_recall', 'bert_recall_score', 'rouge_1_recall', 'rouge_we_1_r',
#     'rouge_l_recall',
#     'glove_sms', 'meteor', 'rouge_we_2_p', 'mover_score', 'rouge_2_precision',
#     'bert_f_score',  # 30
#     'rouge_2_recall', 'rouge_w_1.2_f_score', 'rouge_we_3_p', 'rouge_2_f_score',
#     'rouge_we_2_f',
#     'rouge_we_2_r', 'rouge_3_recall', 'rouge_we_3_r', 'rouge_3_precision',
#     'rouge_we_3_f',
#     'rouge_3_f_score', 'rouge_w_1.2_recall', 'bleu', 'rouge_4_recall',
#     'rouge_4_precision',
#     'rouge_4_f_score', 'bert_precision_score', 'summary_length', 'compression',
#     'coverage', 'cider', 'JS-1', 'JS-2']
#
# ######## columns
# col_r1 = ['rouge_1_precision', 'rouge_1_recall', 'rouge_1_f_score']
# col_r2 = ['rouge_2_precision', 'rouge_2_recall', 'rouge_2_f_score']
# col_r3 = ['rouge_3_precision', 'rouge_3_recall', 'rouge_3_f_score']
# col_r4 = ['rouge_4_precision', 'rouge_4_recall', 'rouge_4_f_score']
# col_rl = ['rouge_l_precision', 'rouge_l_recall', 'rouge_l_f_score']
#
# col_rs_star = ['rouge_s*_precision', 'rouge_s*_recall', 'rouge_s*_f_score']
# col_rsu = ['rouge_su*_precision', 'rouge_su*_recall', 'rouge_su*_f_score']
# col_rw = ['rouge_w_1.2_precision', 'rouge_w_1.2_recall', 'rouge_w_1.2_f_score']
# col_rwe = ['rouge_we_3_p', 'rouge_we_3_r', 'rouge_we_3_f']
# col_rwe12 = ['rouge_we_1_p', 'rouge_we_1_r', 'rouge_we_1_f', 'rouge_we_2_p', 'rouge_we_2_r', 'rouge_we_2_f']
# col_bleu_cider_sms_meteor = ['bleu', 'meteor', 'cider', 'glove_sms']
# col_syntac_stats = ['coverage', 'density', 'compression', 'summary_length', 'percentage_novel_1-gram',
#                     'percentage_repeated_1-gram_in_summ', 'percentage_novel_2-gram',
#                     'percentage_repeated_2-gram_in_summ', 'percentage_novel_3-gram',
#                     'percentage_repeated_3-gram_in_summ']
# col_percent_repeated_gram = ['percentage_repeated_1-gram_in_summ',
#                              'percentage_repeated_2-gram_in_summ',
#                              'percentage_repeated_3-gram_in_summ']
# col_percent_novel_gram = ['percentage_novel_1-gram',
#                           'percentage_novel_2-gram',
#                           'percentage_novel_3-gram']
# col_summeval_bertscore = ['bert_score_precision', 'bert_score_recall', 'bert_score_f1']
# col_bertscore = ['bert_precision_score', 'bert_recall_score', 'bert_f1_score']
#
# col_js2 = ['js-2']
# # col_litepyramid = ['litepyramid_recall']
# # col_moverscore = ['mover_score']
# col_amr = ['amr_cand', 'amr_ref']
# col_smatch = ['smatch_precision', 'smatch_recall', 'smatch_f_score']
# col_s2match = ['s2match_precision', 's2match_recall', 's2match_f_score']
# col_sema = ['sema_precision', 'sema_recall', 'sema_f_score']
#
# col_expert_annotate_stats = ["coherence", "consistency", "fluency", "relevance"]
# summeval_labels = ["coherence", "consistency", "fluency", "relevance"]
# cap_summeval_labels = ["Coherence", "Consistency", "Fluency", "Relevance"]
#
# ["	bart_out	bottom_up_out	"
#  "fast_abs_rl_out_rerank	"
#  "presumm_out_abs	presumm_out_ext_abs	presumm_out_trans_abs	"
#  "ptr_generator_out_pointer_gen_cov	semsim_out	"
#  "t5_out_11B	t5_out_base	t5_out_large	two_stage_rl_out	"
#  "unilm_out_v1	unilm_out_v2"]
# ["Bart-Abs Bottom-Up fastAbsRL-rank T5-Abs      Unilm-v1, Unilm-v2   twoStateRL, PreSummAbs, "
#  "PreSummExt, PreSummExtAbs, Semsim, Pointer-Generator-Cov"]
#
# ["	banditsumm_out	bart_out	heter_graph_out	matchsumm_out	"
#  "neusumm_out	pnbert_out_bert_lstm_pn	pnbert_out_bert_lstm_pn_rl	"
#  "pnbert_out_bert_tf_pn	pnbert_out_bert_tf_sl	pnbert_out_lstm_pn_rl	"
#  "refresh_out"]
# ["BanditSum, Bart-Ext, HeterGraph, MatchSum, REFRESH, NeuSum, BERT-Tf_pn, BERT-tf-sl,BERT-lstm-pn-rl"]
#
# # M0 - LEAD-3
# # M1 - NEUSUM M2 - BanditSum M3 - LATENT M4 - REFRESH M5 - RNES
# # M6 - JECS M7 - STRASS
# # M8 - Pointer Generator M9 - Fast-abs-rl
# # M10 - Bottom-Up
# # M11 - Improve-abs M12 - Unified-ext-abs M13 - ROUGESal
# # M14 - Multi-task (Ent + QG ) M15 - Closed book decoder M16 - SENECA
# # M17 - T5
# # M18 - NeuralTD
# # M19 - BertSum-abs
# # M20 - GPT-2 (supervised) M21 - UniLM
# # M22 - BART
# # M23 - Pegasus (huge news)
#
# rename_bert_dict = {
#     'bert_score_precision': 'bert_precision_score',
#     'bert_score_recall': 'bert_recall_score',
#     'bert_score_f1': 'bert_f_score'
# }
regr_mlp = "mlp"
regr_lin = "lin_regr"
regr_dtree = "dtree"
regr_forest = "forest"
regr_ridge = "ridge"
regr_lasso = "lasso"
regr_elastic_net = "elastic_net"
regr_lin_svr = "svr_lin"
regr_poly_svr = "svr_poly"
regr_nn_keras = "nn_keras"
regr_bagging = "bagging"
regr_voting = "voting"
regr_adaboost = "ada_boosting"
regr_grad_boost = "gradient_boosting"
regr_stacking = "stacking"
#
# summeval_metrics_dict = {
#     'rouge_1_f_score': 'ROUGE-1',
#     'rouge_2_f_score': 'ROUGE-2',
#     'rouge_3_f_score': 'ROUGE-3',
#     'rouge_4_f_score': 'ROUGE-4',
#     'rouge_l_f_score': 'ROUGE-L',
#     'rouge_s*_f_score': 'ROUGE-s*',
#     'rouge_su*_f_score': 'ROUGE-su*',
#     'rouge_w_1.2_f_score': 'ROUGE-w',
#     'rouge_we_1_f': 'ROUGE-we-1',
#     'rouge_we_2_f': 'ROUGE-we-2',
#     "rouge_we_3_f": 'ROUGE-we-3',
#     # 'bert_f_score': 'BERTScore-F1',
#     # 'bert_score_recall': 'BERTScore-R',
#     # 'bert_score_precision': 'BERTScore-P',
#     'bert_f_score': 'BERTScore-f',
#     'bert_recall_score': 'BERTScore-r',
#     'bert_precision_score': 'BERTScore-p',
#     'JS-2': 'JS-2',
#     'glove_sms': 'SMS',
#     'bleu': 'BLEU',
#     'mover_score': 'MoverScore',
#     # 'sema_recall': 'Sema',
#     # 'smatch_recall': 'Smatch',
#     'cider': 'CIDEr',
#     'meteor': 'METEOR',
#     'summary_length': 'Length',
#     'compression': 'Stats-compression',
#     'coverage': 'Stats-coverage',
#     'density': 'Stats-density',
#     'percentage_novel_1-gram': 'Novel unigram',
#     'percentage_novel_2-gram': 'Novel bi-gram',
#     'percentage_novel_3-gram': 'Novel tri-gram',
#     'percentage_repeated_1-gram_in_summ': 'Repeated unigram',
#     'percentage_repeated_2-gram_in_summ': 'Repeated bi-gram',
#     'percentage_repeated_3-gram_in_summ': 'Repeated tri-gram',
#     "lin0": "Lin0",
#     "lin1": "Lin_top1",
#     "lin2": "Lin_top2",
#     "lin3": "Lin_top3",
#     "lin4": "Lin_top4",
#     "lin5": "Lin_combine5",
#     "lin6": "Lin_combine6",
#     "lin7": "Lin_combine7",
#     "lin8": "Lin_combine8",
#     "lin9": "Lin_combine9",
#     "lin10": "Lin_combine10",
#     "lin11": "Lin_combine11",
#     "dtree": "DecisionTree",
#     "lin_regr": "LinReg",
#     "ridge": "RidgeReg",
#     "lasso": "LassoReg",
#     "elastic_net": "ElasticNetReg",
#     "svr_lin": "LinSVR",
#     "forest": "RandomForest",
#     regr_bagging: "Bagging",
#     regr_adaboost: "AdaBoost",
#     regr_grad_boost: "GradientBoost",
#     regr_voting: "Voting",
#     regr_stacking: "Stacking",
#     "mlp": "MLP",
#     "nn_keras": "NNReg"
# }
# regr_colors_dict = {
#     summeval_metrics_dict[regr_nn_keras]: "Green",
#     summeval_metrics_dict[regr_mlp]: "Lime",
#     summeval_metrics_dict[regr_forest]: "Blue",
#     summeval_metrics_dict[regr_dtree]: "Orange",
#     summeval_metrics_dict[regr_voting]: "Red",
#     summeval_metrics_dict[regr_stacking]: "Fuchsia",
#     summeval_metrics_dict[regr_lin]: "Purple",
#     summeval_metrics_dict[regr_lin_svr]: "OliveDrab",
#     summeval_metrics_dict[regr_adaboost]: "Brown",
#     summeval_metrics_dict[regr_grad_boost]: "Olive",
#     summeval_metrics_dict[regr_lasso]: "Yellow",
#     summeval_metrics_dict[regr_elastic_net]: "Navy",
#     summeval_metrics_dict[regr_bagging]: "Aqua",
#     summeval_metrics_dict[regr_ridge]: "Teal"
# }
#
# regr_colors_step_dict = {
#     summeval_metrics_dict[regr_nn_keras]: "Lime",
#     summeval_metrics_dict[regr_mlp]: "Red",
#     summeval_metrics_dict[regr_forest]: "Blue",
#     summeval_metrics_dict[regr_dtree]: "Orange",
#     summeval_metrics_dict[regr_voting]: "Green",
#     summeval_metrics_dict[regr_stacking]: "Fuchsia",
#     summeval_metrics_dict[regr_lin]: "Purple",
#     summeval_metrics_dict[regr_lin_svr]: "OliveDrab",
#     summeval_metrics_dict[regr_adaboost]: "Brown",
#     summeval_metrics_dict[regr_grad_boost]: "Olive",
#     summeval_metrics_dict[regr_lasso]: "Yellow",
#     summeval_metrics_dict[regr_elastic_net]: "Navy",
#     summeval_metrics_dict[regr_bagging]: "Aqua",
#     summeval_metrics_dict[regr_ridge]: "Teal"
# }
# # extra_cols = ['rouge_1_f_score', 'bert_recall_score', 'density', 'rouge_we_1_f']
# regr_bar_colors_dict = {
#     summeval_metrics_dict[regr_nn_keras]: "Green",
#     summeval_metrics_dict[regr_mlp]: "Lime",
#     summeval_metrics_dict[regr_forest]: "Blue",
#     summeval_metrics_dict["rouge_1_f_score"]: "Orange",
#     summeval_metrics_dict[regr_voting]: "Red",
#     summeval_metrics_dict[regr_stacking]: "Fuchsia",
#     summeval_metrics_dict[regr_lin]: "Purple",
#     summeval_metrics_dict[regr_lin_svr]: "OliveDrab",
#     summeval_metrics_dict["bert_recall_score"]: "Brown",
#     summeval_metrics_dict[regr_grad_boost]: "Olive",
#     summeval_metrics_dict["rouge_we_1_f"]: "Yellow",
#     summeval_metrics_dict["density"]: "Navy",
#     summeval_metrics_dict[regr_bagging]: "Aqua",
#     summeval_metrics_dict[regr_ridge]: "Teal"
# }
# regrs_list = [regr_dtree, regr_lin, regr_ridge, regr_lasso, regr_elastic_net, regr_lin_svr, regr_forest, regr_bagging,
#               regr_adaboost, regr_grad_boost, regr_mlp, regr_voting, regr_stacking,
#               regr_nn_keras]
#
# regr_names_for_combine_ablation = [regr_lin, regr_forest, regr_mlp, regr_grad_boost, regr_nn_keras,
#                                    regr_ridge, regr_lin_svr]
# regr_names_for_pca_features = [regr_dtree, regr_lin, regr_ridge, regr_forest, regr_bagging,
#                                regr_adaboost, regr_grad_boost, regr_mlp, regr_voting, regr_stacking,
#                                regr_nn_keras]
#
# regr_names_for_combine_ablation_lin = [regr_lin, regr_lasso,
#                                        regr_ridge, regr_lin_svr]
# regr_names_for_combine_ablation_nn = [
#     # regr_nn_keras,
#     regr_mlp, regr_ridge, regr_grad_boost,
#     regr_forest, regr_dtree]
# regr_remove_step_ablation = [regr_nn_keras, regr_mlp,
#                              regr_forest, regr_dtree, regr_ridge]
# regr_names_for_combine_ablation4 = [regr_lin, regr_forest, regr_mlp, regr_nn_keras]
# regr_names_four_quality_bar = ['rouge_1_f_score', 'bert_recall_score', 'density', 'rouge_we_1_f', regr_lin, regr_voting,
#                                # regr_lin_svr,
#                                # regr_mlp,
#                                regr_forest, regr_nn_keras, regr_mlp,
#                                # regr_voting,
#                                regr_grad_boost]
# regr_names_four_quality_bar_old = ['rouge_1_f_score', 'bert_recall_score', 'density', 'rouge_we_1_f', regr_lin,
#                                    regr_voting,
#                                    # regr_lin_svr,
#                                    # regr_mlp,
#                                    regr_forest, regr_nn_keras, regr_bagging,
#                                    # regr_voting,
#                                    regr_grad_boost]
# regr_names_four_quality_bar_all = ['rouge_1_f_score', 'bert_recall_score', 'density', 'rouge_we_1_f', regr_lin,
#                                    regr_ridge,
#                                    # regr_lin_svr,
#                                    # regr_stacking,
#                                    regr_nn_keras,
#                                    regr_mlp,
#                                    regr_forest, regr_grad_boost, regr_voting,
#                                    regr_bagging]
# summeval_quality_metric_dict = {
#     'rouge_1_f_score': 'ROUGE-1',
#     'rouge_2_f_score': 'ROUGE-2',
#     'rouge_3_f_score': 'ROUGE-3',
#     'rouge_4_f_score': 'ROUGE-4',
#     'rouge_l_f_score': 'ROUGE-L',
#     'rouge_s*_f_score': 'ROUGE-s*',
#     'rouge_su*_f_score': 'ROUGE-su*',
#     'rouge_w_1.2_f_score': 'ROUGE-w',
#     'rouge_we_1_f': 'ROUGE-we-1',
#     'rouge_we_2_f': 'ROUGE-we-2',
#     "rouge_we_3_f": 'ROUGE-we-3',
#     # 'bert_f_score': 'BERTScore-F1',
#     # 'bert_score_recall': 'BERTScore-R',
#     # 'bert_score_precision': 'BERTScore-P',
#     'bert_f_score': 'BERTScore-f',
#     'bert_recall_score': 'BERTScore-r',
#     'bert_precision_score': 'BERTScore-p',
#     'js-2': 'JS-2',
#     'glove_sms': 'SMS',
#     'bleu': 'BLEU',
#     'mover_score': 'MoverScore',
#     # 'sema_recall': 'Sema',
#     # 'smatch_recall': 'Smatch',
#     'cider': 'CIDEr',
#     'meteor': 'METEOR',
#     'summary_length': 'Length',
#     'compression': 'Stats-compression',
#     'coverage': 'Stats-coverage',
#     'density': 'Stats-density',
#     'percentage_novel_1-gram': 'Novel unigram',
#     'percentage_novel_2-gram': 'Novel bi-gram',
#     'percentage_novel_3-gram': 'Novel tri-gram',
#     'percentage_repeated_1-gram_in_summ': 'Repeated unigram',
#     'percentage_repeated_2-gram_in_summ': 'Repeated bi-gram',
#     'percentage_repeated_3-gram_in_summ': 'Repeated tri-gram',
# }
#
# fn_abbr = {
#     "summeval_abs_all_metrics": "abs",
#     "summeval_ext_all_metrics": "ext",
#     "summeval_mix_all_metrics": "mix"
# }
# idx_unnamed = "Unnamed: 0"
# # output_types = {"summeval_abs_all": "Abstractive", "summeval_ext_all": "Extractive", "summeval_all_scores": "Mix"}
# output_types = {"summeval_abs_all_metrics": "Abstractive", "summeval_ext_all_metrics": "Extractive",
#                 "summeval_mix_all_metrics": "Mix"}
#
# # output_types = {"abs_test_fnn08": "Abstractive", "ext_test_fnn08": "Extractive",
# #                 "mix_test_fnn08": "Mix"}
#
rouge_metrics_list = [*col_r1, *col_r2, *col_r3, *col_r4, *col_rw, *col_rs_star, *col_rsu, *col_rl]
# summeval_all_metrics = [*rouge_metrics_list, *col_syntac_stats,
#                         *col_js2, *col_summeval_bertscore, *col_rwe, *col_expert_annotate_stats]
#
# all_cols = [*col_r1, *col_r2, *col_r3, *col_r4, *col_rw, *col_rs_star, *col_rsu, *col_rl, *col_syntac_stats, *col_js2,
#             *col_summeval_bertscore, *col_smatch, *col_sema, *col_rwe, *col_expert_annotate_stats]
#
# summeval_metrics_r = [*rouge_metrics_list, *col_rwe]
# summeval_metrics_b_sm_rwe = [*col_summeval_bertscore, *col_bleu_cider_sms_meteor, *col_rwe]
#
# summeval_data_name = "summeval"
##### system info

ext_model_codes = ['M0', 'M1', 'M2', 'M5']

# import pandas as pd

model_names_list = ['M11', 'M13', 'M1', 'M14', 'M15', 'M12', 'M5', 'M17', 'M20', 'M23', 'M2', 'M0', 'M22',
                    'M8', 'M10', 'M9']
# model_names_dict = {
#     'M11': 'Improve-abs'
# }

#
# def get_summeval_models_data_as_df(data_type="mix", with_s3=False, new_js=True):
#     if with_s3:
#         df = pd.read_csv(summeval_data_all_with_s3_path)
#     else:
#         df = pd.read_csv(summeval_data_all_path)
#
#     if new_js:
#         df = pd.read_csv(summeval_data_all_new_JS)
#
#     if data_type == "abs":
#         return df[~df.model_id.isin(ext_model_codes)].copy()
#     elif data_type == "ext":
#         return df[df.model_id.isin(ext_model_codes)].copy()
#     elif data_type == "mix":
#         return df.copy()
#     elif data_type == "mix_minmax":
#         return pd.read_csv(summeval_data_mix_minmax)
#
#
# def get_summeval_pca_data(n_components=20):
#     path_xpca = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/summeval/models_with_docid/with_s3/train_xpca20.csv"
#     path_test_xpca = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/summeval/models_with_docid/with_s3/test_xpca20.csv"
#     if n_components == 20:
#         df_xpca20 = pd.read_csv(path_xpca)
#         df_test_xpca20 = pd.read_csv(path_test_xpca)
#         return df_xpca20, df_test_xpca20
#
#
# def get_summeval_data_as_df_split(df=None, data_type='abs', cols=None, label=None, with_s3=True, split_ratio=0.8):
#     """
#     return train_x, train_y, test_x,test_y splitted based on the split_ratio
#     """
#     if cols and label:
#         if df is None:
#             df = get_summeval_models_data_as_df(data_type=data_type, with_s3=with_s3)
#         split = split_ratio * len(df)
#
#         train_x = df.loc[:split, cols].copy()
#         train_y = df.loc[:split, [label]].copy()
#
#         test_x = df.loc[split:, cols].copy()
#         test_y = df.loc[split:, [label]].copy()
#
#         return train_x, train_y, test_x, test_y
#

# systems_code = {
#     "M0": "mosystem"
# }

######
corr_pearsonr = 'pearsonr'
corr_spearmanr = 'spearmanr'
corr_kendalltau = 'kendalltau'
# col_bertscore = ['bert_precision_score', 'bert_recall_score', 'bert_f_score']

# ######## features combination
# summeval_features_r12l_b = [*col_r1, *col_r2, *col_rl, *col_bertscore]
# summeval_features_r12l_b_m_bleu_meteor = [*col_r1, *col_r2, *col_rl, *col_bertscore,
#                                           *col_bleu_cider_sms_meteor]
# summeval_features_r12l = [*col_r1, *col_r2, *col_rl]
# summeval_features_b_bleu_meteor_sms_rsu = [*col_bertscore, *col_bleu_cider_sms_meteor, *col_rsu]
# summeval_features_b_rsu = [*col_bertscore, *col_rsu]
# summeval_features_b_rwe = [*col_bertscore, *col_rwe]
# summeval_features_r2 = [*col_r2]
# summeval_features_r1 = [*col_r1]
# summeval_features_rl = [*col_rl]
# summeval_features_r1b = [*col_r1, *col_bertscore]
# summeval_features_r1b_syntac = [*col_r1, *col_bertscore, *col_syntac_stats]

# summeval_features_r1lsu_b_rwe = [*col_r1, *col_rl, *col_rwe, *col_rsu]
#
# summeval_cols_lin5_c1 = ['rouge_l_recall', 'rouge_1_f_score', 'rouge_2_recall', 'bert_score_recall', 'rouge_1_recall']
# cols_lin12 = [*col_r1, *col_r2, *col_bertscore]
# cols_lin12_r_b = [*col_rsu, *col_r2, 'bert_recall_score', 'bert_f_score', 'glove_sms']
# cols_lin12_r1lb_syntac = [*col_rwe, *col_rsu, *col_r1, *col_bertscore]
# cols_lin12_systlevel_abs = ['rouge_w_1.2_precision', 'rouge_l_precision', 'rouge_su*_precision', 'rouge_s*_precision',
#                             'rouge_1_precision', 'rouge_we_3_p', 'glove_sms', 'rouge_2_precision', 'bert_recall_score']
# cols_lin12_summlevel_abs = ['rouge_w_1.2_precision', 'rouge_su*_precision', 'rouge_l_precision', 'rouge_1_precision',
#                             'density', 'rouge_s*_precision',
#                             'rouge_1_precision', 'rouge_we_3_p', 'glove_sms', 'rouge_2_precision', 'bert_recall_score']
# cols_lin12_summeval_coherence = ['rouge_w_1.2_precision', 'rouge_su*_precision', 'rouge_l_precision',
#                                  'rouge_1_precision',
#                                  'density', 'rouge_s*_precision',
#                                  'rouge_we_3_p', 'glove_sms', 'rouge_2_precision',
#                                  'bert_recall_score']
# summeval_features_wrapper_common = ['rouge_we_3_r', 'rouge_1_recall', 'rouge_we_3_p', 'rouge_1_f_score',
#                                     'rouge_2_f_score', 'rouge_3_recall', 'percentage_repeated_2-gram_in_summ',
#                                     'bert_recall_score', 'coverage', 'rouge_su*_recall', 'density',
#                                     'percentage_repeated_3-gram_in_summ']
# summeval_features_wrapper_top = ['rouge_1_recall', 'rouge_l_recall', 'rouge_we_3_p', 'rouge_1_f_score',
#                                  'percentage_repeated_2-gram_in_summ', 'bert_recall_score', 'rouge_su*_recall',
#                                  'density', 'percentage_repeated_3-gram_in_summ']
# summeval_features_wrapper_top5 = ['density', 'bert_recall_score', 'percentage_repeated_2-gram_in_summ',
#                                   'rouge_1_recall', 'rouge_we_3_r']
# summeval_features_wrapper_top4 = ['rouge_1_precision', 'rouge_su*_precision', 'rouge_l_precision', 'density']
# summeval_features_wrapper_abs_top7 = ['rouge_1_precision', 'rouge_2_f_score', 'rouge_l_recall', 'bert_precision_score',
#                                       'density', 'rouge_su*_precision', 'rouge_w_1.2_precision']
#
# summeval_features_repeated_grams = ['percentage_repeated_2-gram_in_summ', 'percentage_repeated_3-gram_in_summ',
#                                     'percentage_novel_1-gram', 'rouge_w_1.2_precision', 'rouge_su*_precision',
#                                     'rouge_l_precision',
#                                     'rouge_1_precision',
#                                     'density', 'rouge_s*_precision']
# summeval_features_selected8 = ['bert_recall_score', 'density', 'rouge_1_recall', 'rouge_l_recall', 'rouge_we_3_p',
#                                'percentage_repeated_2-gram_in_summ', 'percentage_novel_1-gram',
#                                'percentage_repeated_3-gram_in_summ']
#
# summeval_features_selected11 = ['bert_recall_score', 'density', 'rouge_1_recall', 'rouge_l_recall', 'rouge_we_3_p',
#                                 'percentage_repeated_2-gram_in_summ', 'percentage_novel_1-gram', 'coverage',
#                                 'rouge_su*_recall', 'rouge_2_f_score',
#                                 'percentage_repeated_3-gram_in_summ']
#
# summeval_features_selected10 = ['bert_recall_score', 'rouge_w_1.2_precision', 'rouge_1_recall', 'rouge_1_precision',
#                                 'rouge_l_recall', 'rouge_we_3_p',
#                                 'rouge_su*_recall', 'rouge_2_f_score', 'rouge_su*_precision', 'rouge_l_precision']
# summeval_features_selected13 = ['bert_recall_score', 'density', 'rouge_1_recall', 'rouge_l_recall', 'rouge_we_3_p',
#                                 'rouge_we_1_p', 'rouge_we_2_p',
#                                 'percentage_repeated_2-gram_in_summ', 'percentage_novel_1-gram', 'coverage',
#                                 'rouge_su*_recall',
#                                 'percentage_repeated_3-gram_in_summ']
# summeval_features_selected18 = ['bert_recall_score', 'density', 'rouge_1_recall', 'rouge_l_recall', 'rouge_we_3_r',
#                                 'rouge_we_1_r', 'rouge_we_2_r', 'rouge_we_3_p', 'rouge_we_1_f', 'rouge_we_2_f',
#                                 'rouge_we_3_f',
#                                 'rouge_we_1_p', 'rouge_we_2_p',
#                                 'percentage_repeated_2-gram_in_summ', 'percentage_novel_1-gram', 'coverage',
#                                 'rouge_su*_recall',
#                                 'percentage_repeated_3-gram_in_summ']
#
# summeval_features_linreg5 = ['density', 'coverage', 'percentage_repeated_2-gram_in_summ', 'percentage_novel_3-gram',
#                              'rouge_l_precision']
# summeval_features_linreg10 = ['percentage_novel_3-gram', 'density', 'percentage_repeated_3-gram_in_summ', 'bleu',
#                               'rouge_2_f_score', 'percentage_novel_1-gram', 'rouge_l_precision', 'bert_recall_score',
#                               'percentage_repeated_2-gram_in_summ', 'coverage']
# summeval_features_linreg20 = ['rouge_3_recall', 'bleu', 'percentage_novel_3-gram', 'rouge_2_recall',
#                               'percentage_novel_1-gram', 'rouge_l_recall', 'percentage_repeated_2-gram_in_summ',
#                               'rouge_l_f_score', 'rouge_2_f_score', 'percentage_repeated_1-gram_in_summ',
#                               'rouge_l_precision', 'rouge_4_recall', 'bert_recall_score', 'JS-2', 'bert_f_score',
#                               'density', 'rouge_2_precision', 'percentage_repeated_3-gram_in_summ',
#                               'bert_precision_score', 'coverage']
# summeval_features_all = [*col_r1, *col_syntac_stats, *col_r2, *col_bertscore, *col_rwe12, *col_rl, *col_rwe, *col_r3,
#                          *col_rs_star,
#                          *col_rsu,
#                          *col_bleu_cider_sms_meteor
#                          ]
#
# # summeval_features_relevance_selected14 = ['rouge_1_recall', 'rouge_we_1_f', 'rouge_su*_recall', 'bert_recall_score',
# #                                           'rouge_1_f_score', 'rouge_l_recall',
# #                                           'rouge_we_1_r', 'rouge_we_2_r',
# #                                           'rouge_2_recall', 'rouge_su*_f_score',
# #                                           'mover_score', 'glove_sms',
# #                                           'meteor', 'bert_f_score']
# summeval_features_relevance_selected14 = ['rouge_1_recall', 'rouge_we_1_f', 'rouge_su*_recall', 'bert_recall_score',
#                                           'rouge_1_f_score', 'rouge_l_recall',
#                                           'rouge_we_1_r', 'rouge_we_2_r',
#                                           'rouge_2_recall', 'rouge_su*_f_score',
#                                           'mover_score',
#                                           'meteor', *col_percent_novel_gram]
#
# summeval_features_consistency_selected14 = ['density', 'meteor', 'bert_recall_score', 'rouge_we_1_f', 'rouge_1_recall',
#                                             'rouge_su*_recall',
#                                             'rouge_1_f_score', 'rouge_l_recall',
#                                             'rouge_we_1_r', 'rouge_we_2_r',
#                                             'rouge_2_recall', 'rouge_su*_f_score',
#                                             'mover_score', 'glove_sms']
# summeval_features_fluency_selected14 = ['density', 'meteor', 'bert_recall_score', 'rouge_we_1_f', 'rouge_1_recall',
#                                         'rouge_su*_recall',
#                                         'rouge_1_f_score', 'rouge_l_recall',
#                                         'rouge_we_1_r', 'rouge_we_2_r',
#                                         'rouge_2_recall', 'rouge_su*_f_score',
#                                         'mover_score', 'glove_sms']
#
# summeval_feature_selected_no_ngram = ['bert_recall_score', 'density', 'rouge_1_recall', 'rouge_l_recall',
#                                       'rouge_we_3_r',
#                                       'rouge_we_1_r', 'rouge_we_2_r', 'rouge_we_3_p', 'rouge_we_1_f', 'rouge_we_2_f',
#                                       'rouge_we_3_f',
#                                       'rouge_we_1_p', 'rouge_we_2_p', 'coverage',
#                                       'rouge_su*_recall', *col_percent_novel_gram]
# # summeval_features_selected_rpf = [*col_bertscore, *col_rwe, *col_rwe12, *col_r1, *col_r2, *col_rl, *col_rsu,
# #                                   *col_percent_repeated_gram, 'density']
#
#
# summeval_features_selected_top10 = ['percentage_repeated_2-gram_in_summ', 'density', 'percentage_novel_2-gram',
#                                     'percentage_novel_3-gram', 'bert_recall_score', 'rouge_we_1_p',
#                                     'rouge_l_recall', 'rouge_su*_recall', 'percentage_novel_1-gram', 'coverage']
# summeval_features_selected_top15 = ['percentage_repeated_2-gram_in_summ', 'density', 'percentage_novel_2-gram',
#                                     'percentage_novel_3-gram', 'bert_recall_score', 'rouge_we_1_p',
#                                     'rouge_l_recall', 'rouge_su*_recall', 'percentage_novel_1-gram', 'coverage',
#                                     'rouge_we_3_f', 'rouge_we_3_r', 'rouge_we_2_r', 'rouge_we_1_f', 'rouge_we_3_p']
#
# summeval_features_selected21 = ['bert_recall_score', 'density', 'rouge_1_recall', 'rouge_l_recall', 'rouge_we_3_r',
#                                 'rouge_we_1_r', 'rouge_we_2_r', 'rouge_we_3_p', 'rouge_we_1_f', 'rouge_we_2_f',
#                                 'rouge_we_3_f',
#                                 'rouge_we_1_p', 'rouge_we_2_p',
#                                 'coverage',
#                                 # 'JS-2',
#                                 'rouge_su*_recall',
#                                 'percentage_repeated_1-gram_in_summ',
#                                 'percentage_repeated_2-gram_in_summ',
#                                 'percentage_repeated_3-gram_in_summ', *col_percent_novel_gram]
# summeval_features_selected21_trained = ['bert_recall_score', 'density', 'rouge_1_recall', 'rouge_l_recall',
#                                         'rouge_we_3_r',
#                                         'rouge_we_1_r', 'rouge_we_2_r', 'rouge_we_3_p', 'rouge_we_1_f', 'rouge_we_2_f',
#                                         'rouge_we_3_f',
#                                         'rouge_we_1_p', 'rouge_we_2_p',
#                                         'percentage_repeated_2-gram_in_summ', 'coverage',
#                                         'rouge_su*_recall',
#                                         'percentage_repeated_3-gram_in_summ', *col_percent_novel_gram]
#
# select_pos = 0
# summeval_features_selected_rpf_1 = [col_bertscore[select_pos], col_rwe[select_pos], col_rwe12[select_pos],
#                                     col_r1[select_pos], col_r2[select_pos], \
#                                     col_rl[select_pos], col_rsu[select_pos],
#                                     col_percent_repeated_gram[select_pos], 'density']
# weights_lin5_c3 = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# weights_lin5_c1 = [0.4, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# weights_lin5_c5_2 = [0.3, 0.3, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# weights_lin5_c0 = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# weights_lin5_c2 = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# weights_lin5_c6 = [0.1, 0.1, 0.0, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# weights_lin5_c10_1 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0]
# weights_lin5_c10_2 = [0.0, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
#
# summeval_example_combines = [
#     (summeval_cols_lin5_c1, weights_lin5_c3),
#     (summeval_cols_lin5_c1, weights_lin5_c1),
#     (summeval_cols_lin5_c1, weights_lin5_c5_2),
#     (summeval_cols_lin5_c1, weights_lin5_c0),
#     (summeval_cols_lin5_c1, weights_lin5_c2),
#     (summeval_cols_lin5_c1, weights_lin5_c6)
# ]
#
# summeval_example_combines_12 = [
#     (cols_lin12, weights_lin5_c3),
#     (cols_lin12, weights_lin5_c1),
#     (cols_lin12, weights_lin5_c5_2),
#     (cols_lin12, weights_lin5_c0),
#     (cols_lin12, weights_lin5_c2),
#     (cols_lin12, weights_lin5_c6),
#     (cols_lin12, weights_lin5_c10_1),
#     (cols_lin12, weights_lin5_c10_2)
# ]
# summeval_example_combines_12_rbs = [
#     (cols_lin12_r_b, weights_lin5_c3),
#     (cols_lin12_r_b, weights_lin5_c1),
#     (cols_lin12_r_b, weights_lin5_c5_2),
#     (cols_lin12_r_b, weights_lin5_c0),
#     (cols_lin12_r_b, weights_lin5_c2),
#     (cols_lin12_r_b, weights_lin5_c6),
#     (cols_lin12_r_b, weights_lin5_c10_1),
#     (cols_lin12_r_b, weights_lin5_c10_2)
# ]
#
# summeval_example_combines_12_rswb = [
#     (cols_lin12_r1lb_syntac, weights_lin5_c3),
#     (cols_lin12_r1lb_syntac, weights_lin5_c1),
#     (cols_lin12_r1lb_syntac, weights_lin5_c5_2),
#     (cols_lin12_r1lb_syntac, weights_lin5_c0),
#     (cols_lin12_r1lb_syntac, weights_lin5_c2),
#     (cols_lin12_r1lb_syntac, weights_lin5_c6),
#     (cols_lin12_r1lb_syntac, weights_lin5_c10_1),
#     (cols_lin12_r1lb_syntac, weights_lin5_c10_2)
# ]
#
# summeval_example_combines_systlevel_abs = [
#     (cols_lin12_systlevel_abs, weights_lin5_c3),
#     (cols_lin12_systlevel_abs, weights_lin5_c1),
#     (cols_lin12_systlevel_abs, weights_lin5_c5_2),
#     (cols_lin12_systlevel_abs, weights_lin5_c0),
#     (cols_lin12_systlevel_abs, weights_lin5_c2),
#     (cols_lin12_systlevel_abs, weights_lin5_c6),
#     (cols_lin12_systlevel_abs, weights_lin5_c10_1),
#     (cols_lin12_systlevel_abs, weights_lin5_c10_2)
# ]
#
# summeval_example_combines_summlevel_abs = [
#     (cols_lin12_summlevel_abs, weights_lin5_c3),
#     (cols_lin12_summlevel_abs, weights_lin5_c1),
#     (cols_lin12_summlevel_abs, weights_lin5_c5_2),
#     (cols_lin12_summlevel_abs, weights_lin5_c0),
#     (cols_lin12_summlevel_abs, weights_lin5_c2),
#     (cols_lin12_summlevel_abs, weights_lin5_c6),
#     (cols_lin12_summlevel_abs, weights_lin5_c10_1),
#     (cols_lin12_summlevel_abs, weights_lin5_c10_2)
# ]
# summeval_example_combines_summlevel_coherence = [
#     (cols_lin12_summeval_coherence, weights_lin5_c0),
#     (cols_lin12_summeval_coherence, weights_lin5_c1),
#     (cols_lin12_summeval_coherence, weights_lin5_c2),
#     (cols_lin12_summeval_coherence, weights_lin5_c3),
#     (cols_lin12_summeval_coherence, weights_lin5_c5_2),  # 4
#     (cols_lin12_summeval_coherence, weights_lin5_c6),  # 5
#     (cols_lin12_summeval_coherence, weights_lin5_c10_1),  # 6
#     # (cols_lin12_summeval_coherence, weights_lin5_c10_2),
#     (summeval_features_repeated_grams, weights_lin5_c0),  # 7
#     (summeval_features_repeated_grams, weights_lin5_c1),  # 8
#     (summeval_features_repeated_grams, weights_lin5_c2),  # 9
#     (summeval_features_selected21, weights_lin5_c0),  # 10
#     (summeval_features_selected21, weights_lin5_c1),  # 11
#     (summeval_features_selected21, weights_lin5_c2),  # 12
# ]
# summeval_example_combines_summlevel_relevance = [
#     (summeval_features_relevance_selected14, weights_lin5_c0),
#     (summeval_features_relevance_selected14, weights_lin5_c1),
#     (summeval_features_relevance_selected14, weights_lin5_c2),
#     (summeval_features_relevance_selected14, weights_lin5_c3),
#     (summeval_features_relevance_selected14, weights_lin5_c5_2),  # 4
#     (summeval_features_relevance_selected14, weights_lin5_c6),  # 5
#     (summeval_features_relevance_selected14, weights_lin5_c10_1),
#     (summeval_features_relevance_selected14, weights_lin5_c10_2)
#
# ]
# summeval_example_combines_summlevel_consistency = [
#     (summeval_features_consistency_selected14, weights_lin5_c0),
#     (summeval_features_consistency_selected14, weights_lin5_c1),
#     (summeval_features_consistency_selected14, weights_lin5_c2),
#     (summeval_features_consistency_selected14, weights_lin5_c3),
#     (summeval_features_consistency_selected14, weights_lin5_c5_2),  # 4
#     (summeval_features_consistency_selected14, weights_lin5_c6),  # 5
#     (summeval_features_consistency_selected14, weights_lin5_c10_1),
#     (summeval_features_consistency_selected14, weights_lin5_c10_2)
#
# ]
# summeval_example_combines_summlevel_fluency = [
#     (summeval_features_fluency_selected14, weights_lin5_c0),
#     (summeval_features_fluency_selected14, weights_lin5_c1),
#     (summeval_features_fluency_selected14, weights_lin5_c2),
#     (summeval_features_fluency_selected14, weights_lin5_c3),
#     (summeval_features_fluency_selected14, weights_lin5_c5_2),  # 4
#     (summeval_features_fluency_selected14, weights_lin5_c6),  # 5
#     (summeval_features_fluency_selected14, weights_lin5_c10_1),
#     (summeval_features_fluency_selected14, weights_lin5_c10_2)
#
# ]

# summeval_features_forest20 = ['rouge_we_3_p', 'rouge_1_f_score', 'rouge_2_recall', 'rouge_4_f_score',
#                               'rouge_l_precision', 'rouge_l_f_score',
#                               'rouge_w_1.2_recall', 'rouge_s*_recall', 'bert_recall_score', 'meteor', 'bleu',
#                               'coverage', 'density',
#                               'summary_length', 'percentage_novel_1-gram', 'percentage_repeated_2-gram_in_summ',
#                               'percentage_novel_3-gram',
#                               'percentage_repeated_3-gram_in_summ', 'JS-2', 'rouge_we_1_f']
# summeval_features_wrapper20_dict = {
#     "filter_20": set(features_coherence_filter_without_s3[:20]),
#     'forest': {'JS-2', 'coverage', 'summary_length', 'density', 'percentage_repeated_2-gram_in_summ',
#                'meteor', 'rouge_we_3_p', 'rouge_4_f_score', 'rouge_l_f_score', 'rouge_2_recall', 'bleu',
#                'rouge_we_1_f', 'rouge_s*_recall', 'rouge_l_precision', 'percentage_novel_3-gram',
#                'rouge_1_f_score', 'percentage_novel_1-gram', 'bert_recall_score', 'rouge_w_1.2_recall',
#                'percentage_repeated_3-gram_in_summ'},
#     'lin_regr': {'rouge_2_f_score', 'coverage',
#                  'rouge_4_recall', 'density',
#                  'bert_precision_score',
#                  'rouge_2_precision',
#                  'percentage_repeated_2-gram_in_summ',
#                  'percentage_repeated_1-gram_in_summ',
#                  'rouge_l_f_score', 'rouge_2_recall',
#                  'bert_f_score', 'bleu', 'rouge_3_recall',
#                  'rouge_l_recall', 'rouge_l_precision',
#                  'percentage_novel_3-gram',
#                  'percentage_novel_1-gram',
#                  'bert_recall_score', 'JS-2',
#                  'percentage_repeated_3-gram_in_summ'},
#     'lin_regr_without_JS': {'rouge_2_f_score', 'coverage',
#                             'rouge_4_recall', 'density',
#                             'bert_precision_score',
#                             'rouge_2_precision',
#                             'percentage_repeated_2-gram_in_summ',
#                             'percentage_repeated_1-gram_in_summ',
#                             'rouge_l_f_score', 'rouge_2_recall',
#                             'bert_f_score', 'bleu', 'rouge_3_recall',
#                             'rouge_l_recall', 'rouge_l_precision',
#                             'percentage_novel_3-gram',
#                             'percentage_novel_1-gram',
#                             'bert_recall_score',
#                             'percentage_repeated_3-gram_in_summ'},
#     'mlp': {'density', 'percentage_repeated_2-gram_in_summ', 'percentage_novel_2-gram',
#             'rouge_we_1_p', 'rouge_we_3_p', 'percentage_repeated_1-gram_in_summ', 'rouge_w_1.2_f_score',
#             'rouge_2_recall', 'JS-2', 'compression',
#             'bert_f_score', 'S3_Responsive', 'rouge_1_precision', 'rouge_we_3_r', 'rouge_l_precision',
#             'percentage_novel_3-gram', 'bert_recall_score', 'rouge_su*_recall', 'percentage_repeated_3-gram_in_summ',
#             'rouge_we_2_p'},
#     # 'svr_lin': {'rouge_2_f_score', 'density', 'percentage_repeated_2-gram_in_summ',
#     #             'rouge_w_1.2_precision', 'rouge_we_1_p', 'rouge_4_precision', 'cider',
#     #             'rouge_w_1.2_f_score', 'rouge_l_f_score', 'rouge_su*_precision',
#     #             'rouge_we_1_r',
#     #             'rouge_3_recall', 'mover_score', 'S3_Pyramid', 'rouge_l_precision',
#     #             'percentage_novel_3-gram', 'percentage_novel_1-gram', 'rouge_su*_recall',
#     #             'percentage_repeated_3-gram_in_summ', 'rouge_we_2_p'},
#     # 'ridge': {'JS-2',
#     #           'coverage',
#     #           'density',
#     #           'percentage_repeated_2-gram_in_summ',
#     #           'meteor',
#     #           'JS-1',
#     #           'rouge_we_2_r',
#     #           'rouge_we_2_f',
#     #           'rouge_4_precision',
#     #           'rouge_l_f_score',
#     #           'rouge_su*_precision',
#     #           'bert_f_score',
#     #           'S3_Responsive',
#     #           'bleu',
#     #           'S3_Pyramid',
#     #           'rouge_l_precision',
#     #           'percentage_novel_3-gram',
#     #           'percentage_novel_1-gram',
#     #           'bert_recall_score',
#     #           'percentage_repeated_3-gram_in_summ'},
#     # 'lasso': {
#     #     'rouge_2_f_score', 'compression', 'rouge_4_recall', 'density', 'rouge_3_f_score', 'rouge_2_precision',
#     #     'rouge_we_3_p', 'rouge_4_f_score', 'rouge_we_3_f', 'rouge_4_precision', 'rouge_2_recall', 'rouge_3_precision',
#     #     'rouge_1_precision', 'bleu', 'rouge_3_recall', 'rouge_1_recall', 'rouge_l_recall', 'rouge_we_3_r',
#     #     'rouge_1_f_score', 'glove_sms'},
#     # 'elastic_net': {'rouge_2_f_score', 'compression', 'rouge_4_recall', 'density',
#     #                 'rouge_3_f_score', 'rouge_2_precision', 'rouge_we_3_p',
#     #                 'rouge_we_3_f', 'cider', 'rouge_2_recall', 'rouge_3_precision',
#     #                 'rouge_1_precision', 'bleu', 'rouge_3_recall',
#     #                 'rouge_1_recall',
#     #                 'rouge_we_3_r', 'percentage_novel_3-gram', 'rouge_1_f_score',
#     #                 'percentage_novel_1-gram', 'glove_sms'},
#     'ada_boosting': {
#         'coverage',
#         'rouge_4_recall',
#         'summary_length',
#         'density',
#         'percentage_repeated_2-gram_in_summ',
#         'percentage_novel_2-gram',
#         'rouge_w_1.2_precision',
#         'rouge_4_precision',
#         'cider',
#         'JS-2',
#         'rouge_w_1.2_f_score',
#         'rouge_l_f_score',
#         'bert_f_score',
#         'rouge_3_recall',
#         'rouge_we_1_f',
#         'rouge_l_precision',
#         'percentage_novel_3-gram',
#         'rouge_1_f_score',
#         'bert_recall_score',
#         'rouge_we_2_p'},
#     'bagging': {
#         'JS-2', 'coverage', 'rouge_4_recall', 'summary_length', 'density', 'percentage_repeated_2-gram_in_summ',
#         'rouge_su*_f_score', 'percentage_novel_2-gram', 'rouge_s*_f_score', 'rouge_we_3_f', 'rouge_l_f_score',
#         'rouge_we_1_r', 'rouge_3_precision', 'bleu', 'rouge_l_recall', 'rouge_s*_recall', 'bert_recall_score',
#         'rouge_su*_recall', 'rouge_w_1.2_recall', 'percentage_repeated_3-gram_in_summ'}
# }
#
# trained_regrsor_coherence_cv_dir = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/wodeutil/nlp/metrics/experiments/trained_regressors/coherence_cv"

#
# def get_exp_path(top_combine=False, worst_combine=False, curr_remove="", four_quality_bar_path=False,
#                  is_best_remove=False,
#                  is_four_quality=False, four_quality_corr_type=None, path_name=None, is_regr_diff_feature_set=False,
#                  is_worst_remove=False, step=1, end=30):
#     top_combine_dir = "/Users/jackz/OneDrive/One Drive Sync/Research/Workspace/Multifacet_evaluation_metric/experiment_results/figs/top_feature_combine_ablation"
#     top_combine_name = "top_combine_abalation_step" + str(step) + ".csv"
#     top_combine_path = os.path.join(top_combine_dir, top_combine_name)
#     worst_combine_dir = "/Users/jackz/OneDrive/One Drive Sync/Research/Workspace/Multifacet_evaluation_metric/experiment_results/figs/worst_feature_combine_ablation"
#     worst_combine_name = "worst_combine_abalation_step" + str(step) + ".csv"
#     worst_combine_path = os.path.join(worst_combine_dir, worst_combine_name)
#
#     best_worst_dir = "/Users/jackz/OneDrive/One Drive Sync/Research/Workspace/Multifacet_evaluation_metric/experiment_results/figs/best_worst_feature_removal_ablation"
#     best_remove_name = "best_feature_remove_abalation_step" + str(step) + "_end" + str(end) + ".csv"
#     best_remove_path = os.path.join(best_worst_dir, best_remove_name)
#     worst_remove_name = "worst_feature_remove_abalation_step" + str(step) + "_end" + str(end) + ".csv"
#     worst_remove_path = os.path.join(best_worst_dir, worst_remove_name)
#
#     four_quality_path = "/Users/jackz/OneDrive/One Drive Sync/Research/Workspace/Multifacet_evaluation_metric/experiment_results/figs/four_quality_bar_chart/four_quality_bar_corrs.csv"
#     four_quality_dir = "/Users/jackz/OneDrive/One Drive Sync/Research/Workspace/Multifacet_evaluation_metric/experiment_results/figs/four_quality_bar_chart"
#
#     regr_pca_features_path = "/Users/jackz/OneDrive/One Drive Sync/Research/Workspace/Multifacet_evaluation_metric/experiment_results/csvs/regr_test_pca_features/regr_pca_featues_ablation.csv"
#     if not path_name is None:
#         if path_name == "regr_pca_features":
#             return regr_pca_features_path
#     if top_combine:
#         return top_combine_path
#     elif worst_combine:
#         return worst_combine_path
#     elif is_best_remove or curr_remove == "best":
#         return best_remove_path
#     elif is_worst_remove or curr_remove == "worst":
#         return worst_remove_path
#     elif four_quality_bar_path:
#         return four_quality_path
#     elif is_four_quality:
#         fn = f"mix_four_quality_bar_corrs_{four_quality_corr_type}.csv"
#         return os.path.join(four_quality_dir, fn)


random_state = 2021

## bagging
# defaultdict(<class 'list'>, {'mix_lin_regr_pearsonr': [0.4922230069385817], 'abs_lin_regr_pearsonr': [0.4598527246980132], 'ext_lin_regr_pearsonr': [0.5942955934837418], 'mix_lin_regr_spearmanr': [0.46871129579564313], 'abs_lin_regr_spearmanr': [0.438075987080514], 'ext_lin_regr_spearmanr': [0.5689722708639199], 'mix_lin_regr_kendalltau': [0.33732083496111986], 'abs_lin_regr_kendalltau': [0.31527407293665016], 'ext_lin_regr_kendalltau': [0.4159232317915916], 'mix_ridge_pearsonr': [0.4845087665305471], 'abs_ridge_pearsonr': [0.45055671150215326], 'ext_ridge_pearsonr': [0.5914443298561847], 'mix_ridge_spearmanr': [0.46412121768194375], 'abs_ridge_spearmanr': [0.43254105867559206], 'ext_ridge_spearmanr': [0.5706484614504778], 'mix_ridge_kendalltau': [0.3331154050074397], 'abs_ridge_kendalltau': [0.3102138470625797], 'ext_ridge_kendalltau': [0.4159232317915916], 'mix_lasso_pearsonr': [0.2582552548169802], 'abs_lasso_pearsonr': [0.2086824550088042], 'ext_lasso_pearsonr': [0.39384067580739673], 'mix_lasso_spearmanr': [0.23149729359863463], 'abs_lasso_spearmanr': [0.17645557770679088], 'ext_lasso_spearmanr': [0.3837009365847581], 'mix_lasso_kendalltau': [0.1646128247587572], 'abs_lasso_kendalltau': [0.12481890489373774], 'ext_lasso_kendalltau': [0.2725878675144484], 'mix_elastic_net_pearsonr': [0.2570958670039647], 'abs_elastic_net_pearsonr': [0.2060441543410912], 'ext_elastic_net_pearsonr': [0.3928708781713067], 'mix_elastic_net_spearmanr': [0.23109826034134887], 'abs_elastic_net_spearmanr': [0.1742897930884467], 'ext_elastic_net_spearmanr': [0.3834884579858187], 'mix_elastic_net_kendalltau': [0.16486020783917107], 'abs_elastic_net_kendalltau': [0.12349884597006719], 'ext_elastic_net_kendalltau': [0.2719205777530911], 'mix_svr_lin_pearsonr': [0.2550354821099697], 'abs_svr_lin_pearsonr': [0.1893377890651614], 'ext_svr_lin_pearsonr': [0.4133245067279029], 'mix_svr_lin_spearmanr': [0.2401562677341294], 'abs_svr_lin_spearmanr': [0.17478868301564687], 'ext_svr_lin_spearmanr': [0.413901033289195], 'mix_svr_lin_kendalltau': [0.16749567830221224], 'abs_svr_lin_kendalltau': [0.11902531295096143], 'ext_svr_lin_kendalltau': [0.30185286669718553], 'mix_mlp_pearsonr': [0.4146840595546787], 'abs_mlp_pearsonr': [0.37710217112472144], 'ext_mlp_pearsonr': [0.5369503269758908], 'mix_mlp_spearmanr': [0.4228618899462378], 'abs_mlp_spearmanr': [0.388076566406354], 'ext_mlp_spearmanr': [0.5087710596565592], 'mix_mlp_kendalltau': [0.30173959917655147], 'abs_mlp_kendalltau': [0.2769923641502041], 'ext_mlp_kendalltau': [0.3618899009573992], 'mix_dtree_pearsonr': [0.11037673479624956], 'abs_dtree_pearsonr': [0.10123918652031212], 'ext_dtree_pearsonr': [0.1403358440878793], 'mix_dtree_spearmanr': [0.10783997165036394], 'abs_dtree_spearmanr': [0.10548561155574909], 'ext_dtree_spearmanr': [0.10105871520856237], 'mix_dtree_kendalltau': [0.08071982144133479], 'abs_dtree_kendalltau': [0.0819470819467176], 'ext_dtree_kendalltau': [0.07567795805461253], 'mix_forest_pearsonr': [0.48312364936737523], 'abs_forest_pearsonr': [0.4708881343613588], 'ext_forest_pearsonr': [0.5205450154119289], 'mix_forest_spearmanr': [0.458134431480933], 'abs_forest_spearmanr': [0.45652705785263487], 'ext_forest_spearmanr': [0.4405666286063377], 'mix_forest_kendalltau': [0.3280853809451948], 'abs_forest_kendalltau': [0.3254678612916616], 'ext_forest_kendalltau': [0.3218652114505901], 'mix_bagging_pearsonr': [0.4815545920929268], 'abs_bagging_pearsonr': [0.4678728704659695], 'ext_bagging_pearsonr': [0.526268281336162], 'mix_bagging_spearmanr': [0.45509500317649515], 'abs_bagging_spearmanr': [0.45040681764336], 'ext_bagging_spearmanr': [0.45607729361152827], 'mix_bagging_kendalltau': [0.3289099750537596], 'abs_bagging_kendalltau': [0.32686125682220274], 'ext_bagging_kendalltau': [0.3298701493519519], 'mix_ada_boosting_pearsonr': [0.45912987089157675], 'abs_ada_boosting_pearsonr': [0.4369355359129808], 'ext_ada_boosting_pearsonr': [0.5336148096452036], 'mix_ada_boosting_spearmanr': [0.4348183702326345], 'abs_ada_boosting_spearmanr': [0.4222313715628728], 'ext_ada_boosting_spearmanr': [0.44609341336945096], 'mix_ada_boosting_kendalltau': [0.3312649594019812], 'abs_ada_boosting_kendalltau': [0.3208846716016444], 'ext_ada_boosting_kendalltau': [0.34690727286431533], 'mix_gradient_boosting_pearsonr': [0.4900800524743264], 'abs_gradient_boosting_pearsonr': [0.45086450754802], 'ext_gradient_boosting_pearsonr': [0.6118045232529543], 'mix_gradient_boosting_spearmanr': [0.46701314919298187], 'abs_gradient_boosting_spearmanr': [0.434614747253132], 'ext_gradient_boosting_spearmanr': [0.5488579838252253], 'mix_gradient_boosting_kendalltau': [0.3362076329145575], 'abs_gradient_boosting_kendalltau': [0.31116722295189736], 'ext_gradient_boosting_kendalltau': [0.4092524502071234], 'mix_nn_keras_pearsonr': [0.4436610037595011], 'abs_nn_keras_pearsonr': [0.42716352649256917], 'ext_nn_keras_pearsonr': [0.49951227029769407], 'mix_nn_keras_spearmanr': [0.43202952459816696], 'abs_nn_keras_spearmanr': [0.41419124990094314], 'ext_nn_keras_spearmanr': [0.48430103792462614], 'mix_nn_keras_kendalltau': [0.3100267699676271], 'abs_nn_keras_kendalltau': [0.2944464765854037], 'ext_nn_keras_kendalltau': [0.35521911937293105]})

##forest
# defaultdict(<class 'list'>, {'mix_lin_regr_pearsonr': [0.5059534529229717], 'abs_lin_regr_pearsonr': [0.4891111787603767], 'ext_lin_regr_pearsonr': [0.5600878752200107], 'mix_lin_regr_spearmanr': [0.48812986551844056], 'abs_lin_regr_spearmanr': [0.47721331262929845], 'ext_lin_regr_spearmanr': [0.5231721337100848], 'mix_lin_regr_kendalltau': [0.35241090714785456], 'abs_lin_regr_kendalltau': [0.34556209157420226], 'ext_lin_regr_kendalltau': [0.37856685491856973], 'mix_ridge_pearsonr': [0.5026496222456391], 'abs_ridge_pearsonr': [0.4806358751788285], 'ext_ridge_pearsonr': [0.5738361440468714], 'mix_ridge_spearmanr': [0.48673591954336504], 'abs_ridge_spearmanr': [0.46945847681428005], 'ext_ridge_spearmanr': [0.5496960791185042], 'mix_ridge_kendalltau': [0.35084417834158155], 'abs_ridge_kendalltau': [0.33778841124592013], 'ext_ridge_kendalltau': [0.4012475123057615], 'mix_lasso_pearsonr': [0.2582552548169802], 'abs_lasso_pearsonr': [0.2086824550088042], 'ext_lasso_pearsonr': [0.39384067580739673], 'mix_lasso_spearmanr': [0.23149729359863463], 'abs_lasso_spearmanr': [0.17645557770679088], 'ext_lasso_spearmanr': [0.3837009365847581], 'mix_lasso_kendalltau': [0.1646128247587572], 'abs_lasso_kendalltau': [0.12481890489373774], 'ext_lasso_kendalltau': [0.2725878675144484], 'mix_elastic_net_pearsonr': [0.25709586135168744], 'abs_elastic_net_pearsonr': [0.20604420056787043], 'ext_elastic_net_pearsonr': [0.392870982216947], 'mix_elastic_net_spearmanr': [0.23109826034134887], 'abs_elastic_net_spearmanr': [0.1742897930884467], 'ext_elastic_net_spearmanr': [0.3834884579858187], 'mix_elastic_net_kendalltau': [0.16486020783917107], 'abs_elastic_net_kendalltau': [0.12349884597006719], 'ext_elastic_net_kendalltau': [0.2719205777530911], 'mix_svr_lin_pearsonr': [0.3350845049368433], 'abs_svr_lin_pearsonr': [0.2967678524320981], 'ext_svr_lin_pearsonr': [0.4416557861921861], 'mix_svr_lin_spearmanr': [0.34544375092972857], 'abs_svr_lin_spearmanr': [0.3055083193211488], 'ext_svr_lin_spearmanr': [0.44561880868413184], 'mix_svr_lin_kendalltau': [0.24566719979414933], 'abs_svr_lin_kendalltau': [0.21311617956592355], 'ext_svr_lin_kendalltau': [0.32720183671816466], 'mix_mlp_pearsonr': [0.4443974297411014], 'abs_mlp_pearsonr': [0.4122948687771644], 'ext_mlp_pearsonr': [0.5568586031930838], 'mix_mlp_spearmanr': [0.429410121494409], 'abs_mlp_spearmanr': [0.40189446432000586], 'ext_mlp_spearmanr': [0.533359122979095], 'mix_mlp_kendalltau': [0.30854250057221055], 'abs_mlp_kendalltau': [0.2882128650014038], 'ext_mlp_kendalltau': [0.38523763650303794], 'mix_dtree_pearsonr': [0.23494281988895693], 'abs_dtree_pearsonr': [0.23090118090918474], 'ext_dtree_pearsonr': [0.22863319805323912], 'mix_dtree_spearmanr': [0.22894427548308063], 'abs_dtree_spearmanr': [0.2310071002730797], 'ext_dtree_spearmanr': [0.20367511933861593], 'mix_dtree_kendalltau': [0.16884101689726122], 'abs_dtree_kendalltau': [0.17008263367546025], 'ext_dtree_kendalltau': [0.1529801363062034], 'mix_forest_pearsonr': [0.47873588926519506], 'abs_forest_pearsonr': [0.4691152267631568], 'ext_forest_pearsonr': [0.5053504499690645], 'mix_forest_spearmanr': [0.46070761523994275], 'abs_forest_spearmanr': [0.4584850807865107], 'ext_forest_spearmanr': [0.4290811818406981], 'mix_forest_kendalltau': [0.32903366417004426], 'abs_forest_kendalltau': [0.3294280380626733], 'ext_forest_kendalltau': [0.30385410117252604], 'mix_bagging_pearsonr': [0.4986166840573728], 'abs_bagging_pearsonr': [0.492217549585085], 'ext_bagging_pearsonr': [0.5149148388327812], 'mix_bagging_spearmanr': [0.4845177592770826], 'abs_bagging_spearmanr': [0.48765726554097505], 'ext_bagging_spearmanr': [0.45881585851351014], 'mix_bagging_kendalltau': [0.35084417834158155], 'abs_bagging_kendalltau': [0.35502251386050787], 'ext_bagging_kendalltau': [0.3365409309364201], 'mix_ada_boosting_pearsonr': [0.4793497077603637], 'abs_ada_boosting_pearsonr': [0.46369485103684166], 'ext_ada_boosting_pearsonr': [0.5303733848213795], 'mix_ada_boosting_spearmanr': [0.46153674990636306], 'abs_ada_boosting_spearmanr': [0.45336835099251344], 'ext_ada_boosting_spearmanr': [0.4643475577221924], 'mix_ada_boosting_kendalltau': [0.35309581185208416], 'abs_ada_boosting_kendalltau': [0.3474323539299431], 'ext_ada_boosting_kendalltau': [0.360560803296121], 'mix_gradient_boosting_pearsonr': [0.48775527833716514], 'abs_gradient_boosting_pearsonr': [0.4770385002888317], 'ext_gradient_boosting_pearsonr': [0.5219492716201732], 'mix_gradient_boosting_spearmanr': [0.47346894205165674], 'abs_gradient_boosting_spearmanr': [0.4705422420717062], 'ext_gradient_boosting_spearmanr': [0.4650130420201502], 'mix_gradient_boosting_kendalltau': [0.3441649660622072], 'abs_gradient_boosting_kendalltau': [0.34028185587952003], 'ext_gradient_boosting_kendalltau': [0.3485483377884629], 'mix_nn_keras_pearsonr': [0.43277242976174934], 'abs_nn_keras_pearsonr': [0.4289281444901718], 'ext_nn_keras_pearsonr': [0.44914421505146124], 'mix_nn_keras_spearmanr': [0.43763108821267077], 'abs_nn_keras_spearmanr': [0.43658586441081887], 'ext_nn_keras_spearmanr': [0.41171726386304563], 'mix_nn_keras_kendalltau': [0.31225317406075187], 'abs_nn_keras_kendalltau': [0.3104338568831915], 'ext_nn_keras_kendalltau': [0.2998516322218451]})

##lin_reg
# defaultdict(<class 'list'>, {'mix_lin_regr_pearsonr': [0.5059534529229717], 'abs_lin_regr_pearsonr': [0.4891111787603767], 'ext_lin_regr_pearsonr': [0.5600878752200107], 'mix_lin_regr_spearmanr': [0.48812986551844056], 'abs_lin_regr_spearmanr': [0.47721331262929845], 'ext_lin_regr_spearmanr': [0.5231721337100848], 'mix_lin_regr_kendalltau': [0.35241090714785456], 'abs_lin_regr_kendalltau': [0.34556209157420226], 'ext_lin_regr_kendalltau': [0.37856685491856973], 'mix_ridge_pearsonr': [0.5026496222456391], 'abs_ridge_pearsonr': [0.4806358751788285], 'ext_ridge_pearsonr': [0.5738361440468714], 'mix_ridge_spearmanr': [0.48673591954336504], 'abs_ridge_spearmanr': [0.46945847681428005], 'ext_ridge_spearmanr': [0.5496960791185042], 'mix_ridge_kendalltau': [0.35084417834158155], 'abs_ridge_kendalltau': [0.33778841124592013], 'ext_ridge_kendalltau': [0.4012475123057615], 'mix_lasso_pearsonr': [0.2582552548169802], 'abs_lasso_pearsonr': [0.2086824550088042], 'ext_lasso_pearsonr': [0.39384067580739673], 'mix_lasso_spearmanr': [0.23149729359863463], 'abs_lasso_spearmanr': [0.17645557770679088], 'ext_lasso_spearmanr': [0.3837009365847581], 'mix_lasso_kendalltau': [0.1646128247587572], 'abs_lasso_kendalltau': [0.12481890489373774], 'ext_lasso_kendalltau': [0.2725878675144484], 'mix_elastic_net_pearsonr': [0.25709586135168744], 'abs_elastic_net_pearsonr': [0.20604420056787043], 'ext_elastic_net_pearsonr': [0.392870982216947], 'mix_elastic_net_spearmanr': [0.23109826034134887], 'abs_elastic_net_spearmanr': [0.1742897930884467], 'ext_elastic_net_spearmanr': [0.3834884579858187], 'mix_elastic_net_kendalltau': [0.16486020783917107], 'abs_elastic_net_kendalltau': [0.12349884597006719], 'ext_elastic_net_kendalltau': [0.2719205777530911], 'mix_svr_lin_pearsonr': [0.3350845049368433], 'abs_svr_lin_pearsonr': [0.2967678524320981], 'ext_svr_lin_pearsonr': [0.4416557861921861], 'mix_svr_lin_spearmanr': [0.34544375092972857], 'abs_svr_lin_spearmanr': [0.3055083193211488], 'ext_svr_lin_spearmanr': [0.44561880868413184], 'mix_svr_lin_kendalltau': [0.24566719979414933], 'abs_svr_lin_kendalltau': [0.21311617956592355], 'ext_svr_lin_kendalltau': [0.32720183671816466], 'mix_mlp_pearsonr': [0.4443974297411014], 'abs_mlp_pearsonr': [0.4122948687771644], 'ext_mlp_pearsonr': [0.5568586031930838], 'mix_mlp_spearmanr': [0.429410121494409], 'abs_mlp_spearmanr': [0.40189446432000586], 'ext_mlp_spearmanr': [0.533359122979095], 'mix_mlp_kendalltau': [0.30854250057221055], 'abs_mlp_kendalltau': [0.2882128650014038], 'ext_mlp_kendalltau': [0.38523763650303794], 'mix_dtree_pearsonr': [0.23494281988895693], 'abs_dtree_pearsonr': [0.23090118090918474], 'ext_dtree_pearsonr': [0.22863319805323912], 'mix_dtree_spearmanr': [0.22894427548308063], 'abs_dtree_spearmanr': [0.2310071002730797], 'ext_dtree_spearmanr': [0.20367511933861593], 'mix_dtree_kendalltau': [0.16884101689726122], 'abs_dtree_kendalltau': [0.17008263367546025], 'ext_dtree_kendalltau': [0.1529801363062034], 'mix_forest_pearsonr': [0.47873588926519506], 'abs_forest_pearsonr': [0.4691152267631568], 'ext_forest_pearsonr': [0.5053504499690645], 'mix_forest_spearmanr': [0.46070761523994275], 'abs_forest_spearmanr': [0.4584850807865107], 'ext_forest_spearmanr': [0.4290811818406981], 'mix_forest_kendalltau': [0.32903366417004426], 'abs_forest_kendalltau': [0.3294280380626733], 'ext_forest_kendalltau': [0.30385410117252604], 'mix_bagging_pearsonr': [0.4986166840573728], 'abs_bagging_pearsonr': [0.492217549585085], 'ext_bagging_pearsonr': [0.5149148388327812], 'mix_bagging_spearmanr': [0.4845177592770826], 'abs_bagging_spearmanr': [0.48765726554097505], 'ext_bagging_spearmanr': [0.45881585851351014], 'mix_bagging_kendalltau': [0.35084417834158155], 'abs_bagging_kendalltau': [0.35502251386050787], 'ext_bagging_kendalltau': [0.3365409309364201], 'mix_ada_boosting_pearsonr': [0.4793497077603637], 'abs_ada_boosting_pearsonr': [0.46369485103684166], 'ext_ada_boosting_pearsonr': [0.5303733848213795], 'mix_ada_boosting_spearmanr': [0.46153674990636306], 'abs_ada_boosting_spearmanr': [0.45336835099251344], 'ext_ada_boosting_spearmanr': [0.4643475577221924], 'mix_ada_boosting_kendalltau': [0.35309581185208416], 'abs_ada_boosting_kendalltau': [0.3474323539299431], 'ext_ada_boosting_kendalltau': [0.360560803296121], 'mix_gradient_boosting_pearsonr': [0.48775527833716514], 'abs_gradient_boosting_pearsonr': [0.4770385002888317], 'ext_gradient_boosting_pearsonr': [0.5219492716201732], 'mix_gradient_boosting_spearmanr': [0.47346894205165674], 'abs_gradient_boosting_spearmanr': [0.4705422420717062], 'ext_gradient_boosting_spearmanr': [0.4650130420201502], 'mix_gradient_boosting_kendalltau': [0.3441649660622072], 'abs_gradient_boosting_kendalltau': [0.34028185587952003], 'ext_gradient_boosting_kendalltau': [0.3485483377884629], 'mix_nn_keras_pearsonr': [0.43277242976174934], 'abs_nn_keras_pearsonr': [0.4289281444901718], 'ext_nn_keras_pearsonr': [0.44914421505146124], 'mix_nn_keras_spearmanr': [0.43763108821267077], 'abs_nn_keras_spearmanr': [0.43658586441081887], 'ext_nn_keras_spearmanr': [0.41171726386304563], 'mix_nn_keras_kendalltau': [0.31225317406075187], 'abs_nn_keras_kendalltau': [0.3104338568831915], 'ext_nn_keras_kendalltau': [0.2998516322218451]})

# defaultdict(<class 'list'>, {'mix_lin_regr_pearsonr': [0.47329343571766913], 'abs_lin_regr_pearsonr': [0.4573161282281985], 'ext_lin_regr_pearsonr': [0.5192162636865496], 'mix_lin_regr_spearmanr': [0.45547448436686244], 'abs_lin_regr_spearmanr': [0.44524027340165495], 'ext_lin_regr_spearmanr': [0.48092504843338985], 'mix_lin_regr_kendalltau': [0.3266771693543525], 'abs_lin_regr_kendalltau': [0.3207009818450735], 'ext_lin_regr_kendalltau': [0.34321171252088833], 'mix_ridge_pearsonr': [0.4815094367808788], 'abs_ridge_pearsonr': [0.4573880073501499], 'ext_ridge_pearsonr': [0.5527688302967825], 'mix_ridge_spearmanr': [0.46510320497674146], 'abs_ridge_spearmanr': [0.4468334039653302], 'ext_ridge_spearmanr': [0.5176950039061209], 'mix_ridge_kendalltau': [0.3329851906721055], 'abs_ridge_kendalltau': [0.31842754703208537], 'ext_ridge_kendalltau': [0.3778997767601229], 'mix_lasso_pearsonr': [0.24646611696127177], 'abs_lasso_pearsonr': [0.22858028834740154], 'ext_lasso_pearsonr': [0.3915141446464107], 'mix_lasso_spearmanr': [0.23074792477623912], 'abs_lasso_spearmanr': [0.2143409237379878], 'ext_lasso_spearmanr': [0.4015396879234923], 'mix_lasso_kendalltau': [0.16725813328834016], 'abs_lasso_kendalltau': [0.1550729589434347], 'ext_lasso_kendalltau': [0.2963236634287337], 'mix_elastic_net_pearsonr': [0.25042295362921496], 'abs_elastic_net_pearsonr': [0.2327986104879445], 'ext_elastic_net_pearsonr': [0.3938329024036327], 'mix_elastic_net_spearmanr': [0.2377028407931573], 'abs_elastic_net_spearmanr': [0.22435643077816994], 'ext_elastic_net_spearmanr': [0.4036503161518476], 'mix_elastic_net_kendalltau': [0.16943679482682844], 'abs_elastic_net_kendalltau': [0.15782037798550155], 'ext_elastic_net_kendalltau': [0.29861216820738223], 'mix_svr_lin_pearsonr': [0.41655311877132467], 'abs_svr_lin_pearsonr': [0.40632487028566117], 'ext_svr_lin_pearsonr': [0.5100202725974687], 'mix_svr_lin_spearmanr': [0.42084699762050315], 'abs_svr_lin_spearmanr': [0.4099251517833099], 'ext_svr_lin_spearmanr': [0.5080392017948228], 'mix_svr_lin_kendalltau': [0.2981467722832082], 'abs_svr_lin_kendalltau': [0.2879195185739215], 'ext_svr_lin_kendalltau': [0.3712289951756547], 'mix_mlp_pearsonr': [0.5162351895047337], 'abs_mlp_pearsonr': [0.5009845369343149], 'ext_mlp_pearsonr': [0.5643988833098317], 'mix_mlp_spearmanr': [0.5079813028465706], 'abs_mlp_spearmanr': [0.49659494624025646], 'ext_mlp_spearmanr': [0.53136422009791], 'mix_mlp_kendalltau': [0.3669165733159664], 'abs_mlp_kendalltau': [0.3580293147422019], 'ext_mlp_kendalltau': [0.38924010545371884], 'mix_dtree_pearsonr': [0.25802761712542843], 'abs_dtree_pearsonr': [0.25944447718456154], 'ext_dtree_pearsonr': [0.23799803976628892], 'mix_dtree_spearmanr': [0.2687806755577865], 'abs_dtree_spearmanr': [0.2699802414249923], 'ext_dtree_spearmanr': [0.24450906998973937], 'mix_dtree_kendalltau': [0.19935417658658244], 'abs_dtree_kendalltau': [0.1993411858471529], 'ext_dtree_kendalltau': [0.1839203546581207], 'mix_forest_pearsonr': [0.4646139823204022], 'abs_forest_pearsonr': [0.4457368965939843], 'ext_forest_pearsonr': [0.52265884806261], 'mix_forest_spearmanr': [0.45131253479286204], 'abs_forest_spearmanr': [0.4409187430698884], 'ext_forest_spearmanr': [0.49034476736348276], 'mix_forest_kendalltau': [0.32177093055165573], 'abs_forest_kendalltau': [0.3120472622343443], 'ext_forest_kendalltau': [0.3605557446405056], 'mix_bagging_pearsonr': [0.4710543883222546], 'abs_bagging_pearsonr': [0.4598647233840558], 'ext_bagging_pearsonr': [0.5019861593573169], 'mix_bagging_spearmanr': [0.46132025173387614], 'abs_bagging_spearmanr': [0.4585130151306409], 'ext_bagging_spearmanr': [0.45267769580217143], 'mix_bagging_kendalltau': [0.33257941883687264], 'abs_bagging_kendalltau': [0.3303080773451203], 'ext_bagging_kendalltau': [0.3285359930350583], 'mix_ada_boosting_pearsonr': [0.46563583542533177], 'abs_ada_boosting_pearsonr': [0.4473523699284606], 'ext_ada_boosting_pearsonr': [0.5308247318345011], 'mix_ada_boosting_spearmanr': [0.44750136574846544], 'abs_ada_boosting_spearmanr': [0.43329983507585346], 'ext_ada_boosting_spearmanr': [0.48659758610267867], 'mix_ada_boosting_kendalltau': [0.34404511352556205], 'abs_ada_boosting_kendalltau': [0.3308099704807145], 'ext_ada_boosting_kendalltau': [0.3793996931207225], 'mix_gradient_boosting_pearsonr': [0.48417890717344736], 'abs_gradient_boosting_pearsonr': [0.4453238338897443], 'ext_gradient_boosting_pearsonr': [0.6130548236689197], 'mix_gradient_boosting_spearmanr': [0.4843671320898156], 'abs_gradient_boosting_spearmanr': [0.45177560050573956], 'ext_gradient_boosting_spearmanr': [0.587646450497262], 'mix_gradient_boosting_kendalltau': [0.351091556574151], 'abs_gradient_boosting_kendalltau': [0.32319442647867347], 'ext_gradient_boosting_kendalltau': [0.4412722018125707], 'mix_nn_keras_pearsonr': [0.5141827068261563], 'abs_nn_keras_pearsonr': [0.5077175063657685], 'ext_nn_keras_pearsonr': [0.5461054228881279], 'mix_nn_keras_spearmanr': [0.507010512732681], 'abs_nn_keras_spearmanr': [0.500828308798362], 'ext_nn_keras_spearmanr': [0.5566369246459412], 'mix_nn_keras_kendalltau': [0.36386563490084406], 'abs_nn_keras_kendalltau': [0.3586893442040372], 'ext_nn_keras_kendalltau': [0.4025816686226552]})
filter_coherence_features=['percentage_repeated_2-gram_in_summ',
                                       'percentage_novel_3-gram', 'percentage_novel_2-gram',
                                       'percentage_novel_1-gram', 'density', 'percentage_repeated_3-gram_in_summ',
                                       'percentage_repeated_1-gram_in_summ', 'rouge_su*_precision',
                                       'rouge_s*_precision', 'rouge_1_precision', 'rouge_we_1_p', 'rouge_l_precision',
                                       'rouge_1_f_score', 'rouge_we_1_f', 'rouge_w_1.2_precision', 'rouge_s*_f_score',
                                       'rouge_s*_recall', 'rouge_su*_f_score', 'rouge_l_f_score', 'rouge_su*_recall',
                                       'bert_recall_score', 'rouge_1_recall', 'rouge_we_1_r', 'rouge_l_recall',
                                       'glove_sms', 'meteor', 'rouge_we_2_p', 'mover_score', 'rouge_2_precision',
                                       'bert_f_score', 'rouge_2_recall', 'rouge_w_1.2_f_score', 'rouge_we_3_p',
                                       'rouge_2_f_score', 'rouge_we_2_f', 'rouge_we_2_r', 'rouge_3_recall',
                                       'rouge_we_3_r', 'rouge_3_precision', 'rouge_we_3_f', 'rouge_3_f_score',
                                       'rouge_w_1.2_recall', 'bleu', 'JS-1', 'JS-2', 'rouge_4_recall',
                                       'rouge_4_precision', 'rouge_4_f_score', 'bert_precision_score', 'summary_length',
                                       'compression', 'coverage', 'cider']

# summeval_features_wrapper20_dict_newJS = {
#     "filter_20": set(features_coherence_filter_all_newjs[:20]),
#     'forest': {'rouge_su*_f_score', 'rouge_3_f_score', 'rouge_s*_recall', 'density',
#                'percentage_repeated_3-gram_in_summ',
#                'bert_recall_score', 'percentage_repeated_2-gram_in_summ', 'percentage_novel_1-gram',
#                'rouge_w_1.2_f_score',
#                'rouge_su*_recall', 'bleu', 'rouge_4_precision', 'rouge_l_f_score', 'rouge_w_1.2_recall', 'coverage',
#                'rouge_we_3_p', 'rouge_1_f_score', 'rouge_3_precision', 'summary_length', 'percentage_novel_3-gram'},
#     'dtree': {'rouge_4_recall', 'rouge_3_f_score', 'cider', 'rouge_s*_recall', 'rouge_3_recall',
#               'percentage_repeated_3-gram_in_summ', 'bert_recall_score', 'percentage_repeated_2-gram_in_summ',
#               'rouge_we_2_f', 'rouge_su*_recall', 'rouge_l_recall', 'JS-1', 'rouge_we_1_r', 'rouge_w_1.2_recall',
#               'coverage', 'rouge_1_f_score', 'percentage_novel_2-gram', 'rouge_1_recall', 'summary_length',
#               'rouge_we_1_f'},
#     'lin_regr': {'rouge_we_2_p', 'rouge_3_f_score', 'density', 'percentage_repeated_3-gram_in_summ',
#                  'bert_recall_score',
#                  'bert_f_score', 'bert_precision_score', 'percentage_repeated_2-gram_in_summ',
#                  'percentage_novel_1-gram',
#                  'rouge_l_precision', 'bleu', 'rouge_4_precision', 'rouge_l_recall', 'rouge_l_f_score', 'JS-1',
#                  'meteor',
#                  'coverage', 'percentage_repeated_1-gram_in_summ', 'rouge_1_recall', 'percentage_novel_3-gram'},
#     'mlp': {'rouge_s*_recall', 'density', 'rouge_3_recall', 'bert_recall_score', 'rouge_we_1_p',
#             'percentage_repeated_2-gram_in_summ', 'rouge_1_precision', 'rouge_we_2_r', 'rouge_l_precision', 'JS-2',
#             'rouge_4_f_score', 'JS-1', 'rouge_w_1.2_precision', 'coverage', 'rouge_we_3_p', 'rouge_1_f_score',
#             'rouge_3_precision', 'percentage_novel_2-gram', 'rouge_1_recall', 'percentage_novel_3-gram'},
#     'svr_lin': {'density', 'percentage_repeated_3-gram_in_summ', 'rouge_we_3_r', 'rouge_2_recall', 'rouge_we_1_p',
#                 'percentage_repeated_2-gram_in_summ', 'percentage_novel_1-gram', 'rouge_l_precision', 'JS-2',
#                 'rouge_4_precision', 'rouge_4_f_score', 'rouge_l_f_score', 'rouge_w_1.2_precision', 'meteor',
#                 'rouge_we_3_p', 'percentage_repeated_1-gram_in_summ', 'rouge_2_f_score', 'rouge_su*_precision',
#                 'rouge_1_recall', 'percentage_novel_3-gram'},
#     'ridge': {'rouge_3_f_score', 'density', 'rouge_3_recall', 'percentage_repeated_3-gram_in_summ',
#               'bert_recall_score',
#               'percentage_repeated_2-gram_in_summ', 'rouge_we_2_r', 'percentage_novel_1-gram', 'rouge_we_2_f',
#               'rouge_l_precision', 'bleu', 'JS-2', 'rouge_4_precision', 'rouge_4_f_score', 'rouge_l_f_score', 'JS-1',
#               'meteor', 'coverage', 'rouge_su*_precision', 'percentage_novel_3-gram'},
#     # 'lasso': {'glove_sms', 'rouge_4_recall', 'rouge_3_f_score', 'cider', 'density', 'rouge_3_recall', 'rouge_we_3_r',
#     #           'rouge_2_recall', 'rouge_1_precision', 'compression', 'percentage_novel_1-gram', 'bleu', 'rouge_we_3_f',
#     #           'rouge_we_3_p', 'rouge_1_f_score', 'percentage_repeated_1-gram_in_summ', 'rouge_2_f_score',
#     #           'rouge_3_precision', 'rouge_2_precision', 'rouge_1_recall'},
#     # 'elastic_net': {'glove_sms', 'rouge_4_recall', 'rouge_3_f_score', 'cider', 'density', 'rouge_3_recall',
#     #                 'rouge_we_3_r',
#     #                 'rouge_2_recall', 'rouge_1_precision', 'compression', 'percentage_novel_1-gram', 'bleu',
#     #                 'rouge_we_3_f', 'rouge_we_3_p', 'rouge_1_f_score', 'rouge_2_f_score', 'rouge_3_precision',
#     #                 'rouge_2_precision', 'rouge_1_recall', 'percentage_novel_3-gram'},
#     'ada_boosting': {'rouge_3_f_score', 'rouge_s*_recall', 'density', 'rouge_3_recall', 'bert_recall_score',
#                      'percentage_repeated_2-gram_in_summ', 'percentage_novel_1-gram', 'rouge_l_precision', 'bleu',
#                      'JS-2',
#                      'rouge_s*_f_score', 'rouge_4_f_score', 'rouge_l_f_score', 'rouge_w_1.2_precision',
#                      'rouge_w_1.2_recall', 'coverage', 'rouge_we_3_f', 'percentage_repeated_1-gram_in_summ',
#                      'rouge_2_f_score', 'summary_length'},
#     'bagging': {'rouge_su*_f_score', 'cider', 'rouge_s*_recall', 'density', 'rouge_3_recall',
#                 'percentage_repeated_3-gram_in_summ', 'bert_recall_score', 'rouge_we_3_r',
#                 'percentage_repeated_2-gram_in_summ', 'percentage_novel_1-gram', 'rouge_w_1.2_f_score',
#                 'rouge_s*_f_score',
#                 'rouge_4_f_score', 'rouge_l_f_score', 'meteor', 'rouge_w_1.2_recall', 'coverage', 'rouge_1_f_score',
#                 'rouge_3_precision', 'percentage_novel_2-gram'},
#     'voting': {'rouge_su*_f_score', 'rouge_3_f_score', 'density', 'bert_recall_score', 'rouge_we_3_r',
#                'rouge_2_recall',
#                'percentage_repeated_2-gram_in_summ', 'rouge_we_2_r', 'rouge_we_2_f', 'rouge_l_precision',
#                'rouge_4_precision', 'rouge_s*_f_score', 'rouge_l_recall', 'rouge_w_1.2_precision', 'coverage',
#                'rouge_we_3_f', 'rouge_s*_precision', 'rouge_3_precision', 'percentage_novel_2-gram',
#                'percentage_novel_3-gram'},
#     'gradient_boosting': {'rouge_we_1_f', 'density', 'percentage_repeated_3-gram_in_summ', 'bert_precision_score',
#                           'rouge_2_recall', 'rouge_we_1_p', 'percentage_repeated_2-gram_in_summ',
#                           'percentage_novel_1-gram', 'rouge_4_precision', 'rouge_s*_f_score', 'rouge_4_f_score',
#                           'rouge_l_f_score', 'rouge_w_1.2_recall', 'coverage', 'rouge_we_3_f', 'rouge_we_3_p',
#                           'rouge_s*_precision', 'rouge_su*_precision', 'rouge_1_recall', 'percentage_novel_3-gram'},
#     'stacking': {'mover_score', 'rouge_su*_f_score', 'rouge_we_2_p', 'rouge_3_f_score', 'density', 'rouge_3_recall',
#                  'percentage_repeated_3-gram_in_summ', 'rouge_we_3_r', 'percentage_repeated_2-gram_in_summ', 'JS-2',
#                  'rouge_s*_f_score', 'rouge_l_recall', 'rouge_l_f_score', 'JS-1', 'rouge_we_3_f', 'rouge_s*_precision',
#                  'percentage_repeated_1-gram_in_summ', 'rouge_2_f_score', 'percentage_novel_2-gram',
#                  'percentage_novel_3-gram'}}
#
# # these are using the new js
# #  regr name: forest
# # features: rouge_1_f_score, rouge_3_precision, percentage_repeated_2-gram_in_summ, density, bleu, rouge_w_1.2_recall, rouge_4_precision, percentage_repeated_3-gram_in_summ, rouge_l_f_score, rouge_su*_f_score, rouge_3_f_score, rouge_s*_recall, percentage_novel_1-gram, coverage, bert_recall_score, rouge_su*_recall, rouge_we_3_p, percentage_novel_3-gram, summary_length, rouge_w_1.2_f_score
# #  regr name: lin_regr
# # features: percentage_repeated_2-gram_in_summ, rouge_l_recall, density, bleu, rouge_l_precision, rouge_4_precision, bert_precision_score, percentage_repeated_3-gram_in_summ, rouge_l_f_score, bert_f_score, rouge_1_recall, rouge_3_f_score, percentage_novel_1-gram, coverage, bert_recall_score, percentage_repeated_1-gram_in_summ, percentage_novel_3-gram, rouge_we_2_p, JS-1, meteor
# #  regr name: bagging
# # features: rouge_1_f_score, percentage_repeated_2-gram_in_summ, percentage_novel_2-gram, density, rouge_w_1.2_recall, cider, bert_precision_score, rouge_l_f_score, mover_score, rouge_s*_recall, rouge_s*_f_score, rouge_we_2_r, rouge_we_1_p, percentage_novel_1-gram, rouge_s*_precision, bert_recall_score, percentage_novel_3-gram, rouge_w_1.2_f_score, JS-1, meteor
# #  regr name: mlp
# # features: rouge_w_1.2_precision, rouge_4_f_score, rouge_1_f_score, rouge_3_precision, percentage_repeated_2-gram_in_summ, percentage_novel_2-gram, density, rouge_l_precision, rouge_1_recall, rouge_3_recall, rouge_s*_recall, rouge_we_2_r, rouge_we_1_p, coverage, rouge_1_precision, JS-2, bert_recall_score, rouge_we_3_p, percentage_novel_3-gram, JS-1
# #  regr name: ada_boosting
# # features: rouge_w_1.2_precision, rouge_4_f_score, percentage_repeated_2-gram_in_summ, density, bleu, rouge_l_precision, rouge_w_1.2_recall, rouge_2_f_score, rouge_l_f_score, rouge_we_3_f, rouge_3_recall, rouge_3_f_score, rouge_s*_recall, rouge_s*_f_score, percentage_novel_1-gram, coverage, JS-2, bert_recall_score, percentage_repeated_1-gram_in_summ, summary_length
# #  regr name: voting
# # features: rouge_3_precision, percentage_repeated_2-gram_in_summ, percentage_novel_2-gram, rouge_we_1_r, density, rouge_4_precision, rouge_l_f_score, rouge_we_3_f, rouge_s*_recall, rouge_we_2_r, coverage, rouge_s*_precision, JS-2, bert_recall_score, percentage_repeated_1-gram_in_summ, rouge_su*_recall, percentage_novel_3-gram, rouge_w_1.2_f_score, JS-1, rouge_su*_precision
# #  regr name: gradient_boosting
# # features: rouge_4_f_score, percentage_repeated_2-gram_in_summ, density, rouge_w_1.2_recall, rouge_4_precision, bert_precision_score, percentage_repeated_3-gram_in_summ, rouge_l_f_score, rouge_we_3_f, rouge_1_recall, rouge_we_1_f, rouge_s*_f_score, rouge_we_1_p, percentage_novel_1-gram, coverage, rouge_s*_precision, rouge_2_recall, rouge_we_3_p, percentage_novel_3-gram, rouge_su*_precision
# curr_feature_sets = {
#     'forest': {'rouge_1_f_score', 'rouge_3_precision', 'percentage_repeated_2-gram_in_summ', 'density', 'bleu',
#                'rouge_w_1.2_recall', 'rouge_4_precision', 'percentage_repeated_3-gram_in_summ', 'rouge_l_f_score',
#                'rouge_su*_f_score', 'rouge_3_f_score', 'rouge_s*_recall', 'percentage_novel_1-gram', 'coverage',
#                'bert_recall_score', 'rouge_su*_recall', 'rouge_we_3_p', 'percentage_novel_3-gram', 'summary_length',
#                'rouge_w_1.2_f_score'},
#     'lin_regr': {'percentage_repeated_2-gram_in_summ', 'rouge_l_recall', 'density', 'bleu', 'rouge_l_precision',
#                  'rouge_4_precision', 'bert_precision_score', 'percentage_repeated_3-gram_in_summ', 'rouge_l_f_score',
#                  'bert_f_score', 'rouge_1_recall', 'rouge_3_f_score', 'percentage_novel_1-gram', 'coverage',
#                  'bert_recall_score', 'percentage_repeated_1-gram_in_summ', 'percentage_novel_3-gram', 'rouge_we_2_p',
#                  'JS-1', 'meteor'},
#     'bagging': {'rouge_1_f_score', 'percentage_repeated_2-gram_in_summ', 'percentage_novel_2-gram', 'density',
#                 'rouge_w_1.2_recall', 'cider', 'bert_precision_score', 'rouge_l_f_score', 'mover_score',
#                 'rouge_s*_recall',
#                 'rouge_s*_f_score', 'rouge_we_2_r', 'rouge_we_1_p', 'percentage_novel_1-gram', 'rouge_s*_precision',
#                 'bert_recall_score', 'percentage_novel_3-gram', 'rouge_w_1.2_f_score', 'JS-1', 'meteor'},
#     'mlp': {'rouge_w_1.2_precision', 'rouge_4_f_score', 'rouge_1_f_score', 'rouge_3_precision',
#             'percentage_repeated_2-gram_in_summ', 'percentage_novel_2-gram', 'density', 'rouge_l_precision',
#             'rouge_1_recall', 'rouge_3_recall', 'rouge_s*_recall', 'rouge_we_2_r', 'rouge_we_1_p', 'coverage',
#             'rouge_1_precision', 'JS-2', 'bert_recall_score', 'rouge_we_3_p', 'percentage_novel_3-gram', 'JS-1'},
#     'ada_boosting': {'rouge_w_1.2_precision', 'rouge_4_f_score', 'percentage_repeated_2-gram_in_summ', 'density',
#                      'bleu',
#                      'rouge_l_precision', 'rouge_w_1.2_recall', 'rouge_2_f_score', 'rouge_l_f_score', 'rouge_we_3_f',
#                      'rouge_3_recall', 'rouge_3_f_score', 'rouge_s*_recall', 'rouge_s*_f_score',
#                      'percentage_novel_1-gram',
#                      'coverage', 'JS-2', 'bert_recall_score', 'percentage_repeated_1-gram_in_summ', 'summary_length'},
#     'voting': {'rouge_3_precision', 'percentage_repeated_2-gram_in_summ', 'percentage_novel_2-gram', 'rouge_we_1_r',
#                'density', 'rouge_4_precision', 'rouge_l_f_score', 'rouge_we_3_f', 'rouge_s*_recall', 'rouge_we_2_r',
#                'coverage', 'rouge_s*_precision', 'JS-2', 'bert_recall_score', 'percentage_repeated_1-gram_in_summ',
#                'rouge_su*_recall', 'percentage_novel_3-gram', 'rouge_w_1.2_f_score', 'JS-1', 'rouge_su*_precision'},
#     'gradient_boosting': {'rouge_4_f_score', 'percentage_repeated_2-gram_in_summ', 'density', 'rouge_w_1.2_recall',
#                           'rouge_4_precision', 'bert_precision_score', 'percentage_repeated_3-gram_in_summ',
#                           'rouge_l_f_score', 'rouge_we_3_f', 'rouge_1_recall', 'rouge_we_1_f', 'rouge_s*_f_score',
#                           'rouge_we_1_p', 'percentage_novel_1-gram', 'coverage', 'rouge_s*_precision', 'rouge_2_recall',
#                           'rouge_we_3_p', 'percentage_novel_3-gram', 'rouge_su*_precision'}}
# feature_format_map = {
#     'percentage_repeated_2-gram_in_summ': 'repeated-bi-gram',
#     'percentage_repeated_3-gram_in_summ': 'repeated-tri-gram',
#     'percentage_repeated_1-gram_in_summ': 'repeated-uni-gram',
#     'percentage_novel_1-gram': 'novel-uni-gram',
#     'percentage_novel_2-gram': 'novel-bi-gram',
#     'percentage_novel_3-gram': 'novel-tri-gram'
# }
#
#
# def format_feature_name(feature_name):
#     import re
#     feature_name = re.sub("s\*", "s", feature_name)
#     feature_name = re.sub("su\*", "su", feature_name)
#     if feature_name in feature_format_map:
#         # feature_name = re.sub("_", "-", feature_name)
#         # return feature_format_map[feature_name]
#         return re.sub("_", "-", feature_format_map[feature_name])
#
#     else:
#         feature_name = re.sub("_", "-", feature_name)
#         return feature_name
#
#
# def format_feature_sets():
#     for regr_name, values in curr_feature_sets.items():
#         curr_vals = list(values)
#         feature_sets_str = ", ".join(format_feature_name(val) for val in curr_vals if val)
#         print(f"{regr_name}: feature_sets")
#         print(feature_sets_str)
#         print()
