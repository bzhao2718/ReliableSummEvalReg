import csv
import json
import os
import re
import sys
from collections import defaultdict
import pandas as pd

# from scipy.stats import pearsonr, spearmanr, kendalltau
from scoring.wodeutil.nlp.metrics.file_util import load_df, get_files_in_dir
from scoring.wodeutil.nlp.metrics.eval_constants import *

def clean_summary(seq, clean_sep=False):
    if seq:
        seq = re.sub(r'\n', '', seq)  # remove newline character
        seq = re.sub(r'\t', '', seq)
        if clean_sep:
            seq = re.sub('<t>', '', seq)
            seq = re.sub('</t>', '', seq)
        seq = seq.strip()
        return seq


def get_expert_annoation_stats(annotations):
    """take the average of the 3 calculations from each expert annotator"""
    if annotations:
        coherence = []
        consistency = []
        fluency = []
        relevance = []
        for annotate in annotations:
            coherence.append(annotate['coherence'])
            consistency.append(annotate['consistency'])
            fluency.append(annotate['fluency'])
            relevance.append(annotate['relevance'])
        if coherence and consistency and fluency and relevance:
            return [sum(coherence) / len(coherence), sum(consistency) / len(consistency), sum(fluency) / len(
                fluency), sum(relevance) / len(relevance)]
        else:
            return -1


def jsonl_to_csv(jsonl_file, csv_path=""):
    """ convert paired human annotations jsonl to a csv"""
    print("Reading the input")
    bad_lines = 0
    if jsonl_file is not None:
        try:
            with open(jsonl_file) as inputf:
                with open(csv_path, 'w') as result:
                    print("-------------- start writing to csv -----------------")
                    print(f"write to {csv_path}")
                    data_writer = csv.writer(result, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    headers = ['line_id', 'model_id', 'decoded', 'reference', 'coherence', 'consistency',
                               'fluency',
                               'relevance', 'filepath', 'id', 'text']
                    data_writer.writerow(headers)
                    row_id = 0
                    for count, line in enumerate(inputf):
                        try:
                            data = json.loads(line)
                            if len(data['decoded']) == 0:
                                print(f"bad line: {data['id']}")
                                bad_lines += 1
                                raise ValueError("data id error!")
                            if data.get("reference", None):
                                reference = data['reference']
                            else:  # there are 10 additional references added, the first is the orginal
                                reference = data["references"][0]
                            # article = data['text']
                            annotation_stats = get_expert_annoation_stats(data['expert_annotations'])
                            if not annotation_stats or annotation_stats == -1:
                                raise ValueError("avg_expert is -1!!!")
                            row_item = [row_id, data['model_id'], data['decoded'], reference, *annotation_stats,
                                        data['filepath'], data['id'], data['text']]
                            data_writer.writerow(row_item)
                            row_id += 1
                        except:
                            bad_lines += 1
                            raise ValueError("error when reading inputf!")
                    print(f"write total rows: {row_id}")
                    print("-------------- finish writing to csv -----------------")
        except Exception as e:
            print("Input did not match required format")
            print(e)
            sys.exit()
        print(f"This many bad lines encountered during loading: {bad_lines}")


def rename_column(name: str):
    sent_mover_prefix = "sentence_movers_"
    if name.startswith(sent_mover_prefix):
        name = re.sub(sent_mover_prefix, '', name)
    return name


def add_scores_to_df_with_lineid(df, fp="", metric_list=[]):
    if fp:
        try:
            with open(fp) as inputf:
                dlist = defaultdict(list)
                ids = df['line_id'].values.tolist()
                for count, line in enumerate(inputf):
                    data = json.loads(line)
                    if count != ids[count]:
                        raise ValueError("id doesn't match")
                    for key, value in data.items():
                        if key == "id":
                            continue
                        if key == "rouge":
                            dlist = add_rouge_scores(dlist=dlist, rouge_values=value)
                        else:
                            dlist[key].append(value)
                for k, v in dlist.items():
                    col = rename_column(k)
                    metric_list.append(col)
                    df[col] = v
                    print(f"add {k} with len {len(v)} to df with col {col}")
            return df.copy()
        except Exception as e:
            print(e)
            sys.exit()


def add_scores_to_df(df, score_path="", metric_list=[]):
    if score_path:
        try:
            with open(score_path) as inputf:
                dlist = defaultdict(list)
                # ids = df['line_id'].values.tolist()
                for count, line in enumerate(inputf):
                    data = json.loads(line)
                    if count != data['id']:
                        raise ValueError("id doesn't match")
                    for key, value in data.items():
                        if key == "id":
                            continue
                        if key == "rouge":
                            dlist = add_rouge_scores(dlist=dlist, rouge_values=value)
                        else:
                            dlist[key].append(value)
                for k, v in dlist.items():
                    col = rename_column(k)
                    metric_list.append(col)
                    df[col] = v
                    print(f"add {k} with len {len(v)} to df with col {col}")
            return df.copy()
        except Exception as e:
            print(e)
            sys.exit()


def add_rouge_scores(dlist: defaultdict, rouge_values: dict):
    if rouge_values:
        for k, v in rouge_values.items():
            if k == "id":
                continue
            if not k in rouge_metrics_list:
                continue
            dlist[k].append(v)
        return dlist


def collect_scores(df, scores_dir=None, save_to=""):
    if scores_dir and save_to:
        files = get_files_in_dir(scores_dir)
        if files:
            metric_list = []
            for file in files:
                fp = os.path.join(scores_dir, file)
                df = add_scores_to_df(df, fp, metric_list=metric_list)
            df.to_csv(save_to, index=False)
            print(f"collected {len(files)} files with the following columns: ")
            print(metric_list)


def split_by_model(from_path="", save_dir=""):
    if save_dir and from_path:
        df = load_df(from_path)
        models = set(df['model_id'].tolist())
        if models:
            for idx, model in enumerate(models):
                model_path = os.path.join(save_dir, model + ".csv")
                df_model = df[df['model_id'] == model]
                df_model.to_csv(model_path, index=False)
            print(f"split these {len(models)} models:")
            print(models)

def do_add_scores_to_df(df_path, score_path=""):
    # score_path = "/evaluation/summ_eval/output_JS12.jsonl"
    df = load_df(df_path)
    df = add_scores_to_df(df, score_path=score_path)
    df.to_csv(df_path, index=False)


def scores_json_to_df(score_path=None, save_to=None):
    """
    convert calculated scores to a dataframe object
    """
    if score_path:
        df_scores = pd.DataFrame()
        df_scores = add_scores_to_df(df_scores, score_path=score_path)
        if save_to:
            df_scores.to_csv(save_to, index=False)

if __name__ == '__main__':
    do_add_scores_to_df()
