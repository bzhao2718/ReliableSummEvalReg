# ReliableSummEvalReg
This repo contains scripts for text summarization evaluation using predictive models. There are 12 regressors provided (see `models.py` file for an example). 

## Requirements, Installation, and Usage

The score calculation is done using the [summ_eval package](https://github.com/Yale-LILY/SummEval), so you need to download it first. You can set up summ_eval either via pip:

```
pip install summ_eval
```

Or clone the repo into `cal_scores/` directory and install it manually:

```
git clone https://github.com/Yale-LILY/SummEval.git
cd evaluation
pip install -e .
```

For setup and usage details, please see the original summ_eval repo [here](https://github.com/Yale-LILY/SummEval). (Note that the original repo might be updated so make sure to follow the updated setup instructions in that case). In addition, the human annotated dataset can also be obtained from the [link](https://github.com/Yale-LILY/SummEval) provided by summ_eval. Once the package is setup, copy the `JS_metric` file into the directory `cal_scores/SummEval/evaluation/summ_eval` in case JS metrics are needed.

There are two ways to calculate the metric scores. For small samples of summaries, you can import the `MetricScorer` class we provided and calculate a pair of summaries like the following example:

```python
example_cand = "cats and dogs have the advantage over marine pets in that they can interact with humans through the sense of touch."
example_ref = "cats and dogs can interact with humans through the sense of touch, therefore they have the advantage over marine pets."

scorer = MetricScorer()
# ROUGE-WE metrics
rwe_1 = scorer.get_rouge_we(example_cand, example_ref, n_gram=1)
rwe_2 = scorer.get_rouge_we(example_cand, example_ref, n_gram=2)
rwe_3 = scorer.get_rouge_we(example_cand, example_ref, n_gram=3)
print("rouge_we n_gram=1: ")
print(rwe_1)

# BLEU
bleu_scores = scorer.get_bleu(example_cand, example_ref)

# METEOR
meteor_scores = scorer.get_meteor(example_cand, example_ref)
```

However, for large samples of summaries, we recommend following the instructions from [summ_eval](https://github.com/Yale-LILY/SummEval) and use the command line interface to calculate metric scores, as they will be much faster. For example, to calculate ROUGE and BertScore of candidate and reference summaries in a json file and write the results to `output.jsonl`, the following command-line `calc-scores` is provided by the package:

```bash
calc-scores --config-file=examples/basic.config --metrics "rouge, bert_score" --jsonl-file data.jsonl --output-file rouge_bertscore.jsonl

```

## Training and Evaluating

We recommend fine tune the model or re-train the model if you have a larger human annotated dataset available. To extract features, use the `get_features_as_df` function in the file `feature-extract.py`. This function runs the command-line interface `calc-scores` and converts the results to a pandas dataframe object. It’s also possible to run the `calc-scores` command manually and call the function `scores_json_to_df` in the `df_util.py` file. 

Once the features are ready, you can simply use the regressors for training (see a set of regressors `regressive_dict` in the `models.py` file). There are 12 regressors and most of them are using `scikit-learn`:

```python
regr_dict_regularized = {
    regr_forest: RandomForestRegressor(n_jobs=-1, random_state=random_state),
    regr_dtree: DecisionTreeRegressor(random_state=random_state, max_depth=3),
    regr_lin: LinearRegression(n_jobs=-1),
    regr_mlp: MLPRegressor(random_state=random_state),
    regr_lin_svr: LinearSVR(epsilon=1.5, random_state=random_state),
    regr_ridge: Ridge(alpha=1, solver='cholesky'),
    regr_adaboost: AdaBoostRegressor(DecisionTreeRegressor(random_state=random_state, max_depth=3), n_estimators=200,learning_rate=0.5, random_state=random_state),
    regr_bagging: BaggingRegressor(DecisionTreeRegressor(random_state=random_state, max_depth=3), n_estimators=100,max_samples=1.0, bootstrap=True,n_jobs=-1),
    regr_voting: VotingRegressor(estimators=[(regr_ridge, Ridge(alpha=1, solver='cholesky')), (regr_forest, RandomForestRegressor(n_jobs=-1)),
                    (regr_mlp, MLPRegressor())], n_jobs=-1),
    regr_grad_boost: GradientBoostingRegressor(n_estimators=150,random_state=random_state),
    regr_stacking: StackingRegressor(estimators=[(regr_ridge, Ridge(alpha=1, solver='cholesky')), (regr_lin_svr, LinearSVR(epsilon=1.5)),(regr_mlp, MLPRegressor())],
        final_estimator=RandomForestRegressor(n_jobs=-1, n_estimators=10, 		 random_state=11), n_jobs=-1)
}
```

To predict the similarity score of a collection of candidate and reference summaries, use the `evaluate_from_path` function, pass the path of the file containing pairs of candidate and reference summaries.

```python
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
```

The above code calculates the scores for the summary file specified by the `summary_path` variable, then extracts features using these scores. The default size of the feature set is 20 and the default regressor is `NNReg`. The predicted scores will be saved to the path specified by the `save_to` variable.

## Acknowledgements

This work will not be possible without some awesome works as a foundation. The score calculations are mostly done by [SummEval](https://github.com/Yale-LILY/SummEval) and  we recommend cite it as well if you use this repo. Also, the summ_eval package in turn uses many tools from other projects like the scripts for score calculations from original papers.

