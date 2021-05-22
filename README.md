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

```
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

```
calc-scores --config-file=examples/basic.config --metrics "rouge, bert_score" --jsonl-file data.jsonl --output-file rouge_bertscore.jsonl

```

## Training and Evaluating

We recommend fine tune the model or re-train the model if you have a larger human annotated dataset available. To extract features, use the `get_features_as_df` function in the file `feature-extract.py`. This function runs the command-line interface `calc-scores` and converts the results to a pandas dataframe object. It’s also possible to run the `calc-scores` command manually and call the function `scores_json_to_df` in the `df_util.py` file. 

Once the features are ready, you can simply use the regressors for training (see a set of regressors `regressive_dict` in the `models.py` file).

To predict the similarity score of a collection of candidate and reference summaries, use the `evaluate_from_path` function, pass the path of the file containing pairs of candidate and reference summaries.

## Acknowledgements

This work will not be possible without some awesome works as a foundation. The score calculations are mostly done by [SummEval](https://github.com/Yale-LILY/SummEval) and  we recommend cite it as well if you use this repo. Also, the summ_eval package in turn uses many tools from other projects like the scripts for score calculations from original papers.

