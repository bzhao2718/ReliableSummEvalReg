import os

import spacy
from cal_scores.SummEval.evaluation.summ_eval.rouge_metric import RougeMetric
from cal_scores.SummEval.evaluation.summ_eval.rouge_we_metric import RougeWeMetric
from cal_scores.SummEval.evaluation.summ_eval.bert_score_metric import BertScoreMetric
from cal_scores.SummEval.evaluation.summ_eval.blanc_metric import BlancMetric
from cal_scores.SummEval.evaluation.summ_eval.chrfpp_metric import ChrfppMetric
from cal_scores.SummEval.evaluation.summ_eval.cider_metric import CiderMetric
from cal_scores.SummEval.evaluation.summ_eval.mover_score_metric import MoverScoreMetric
from cal_scores.SummEval.evaluation.summ_eval.sentence_movers_metric import SentenceMoversMetric
from cal_scores.SummEval.evaluation.summ_eval.summa_qa_metric import SummaQAMetric
from cal_scores.SummEval.evaluation.summ_eval.meteor_metric import MeteorMetric
from cal_scores.SummEval.evaluation.summ_eval.bleu_metric import BleuMetric
from cal_scores.SummEval.evaluation.summ_eval.data_stats_metric import DataStatsMetric
from cal_scores.SummEval.evaluation.summ_eval.s3_metric import S3Metric
from cal_scores.wodeutil.nlp.metrics.file_util import load_df, clean_cnn_text
from collections import defaultdict

from cal_scores.SummEval.evaluation.summ_eval.JS_metric import JSMetric
import pandas as pd

# from set_path import set_path


ROUGE_HOME = os.environ['ROUGE_HOME'] if 'ROUGE_HOME' in os.environ else None


class MetricScorer():
    def __init__(self, ROUGE_HOME=None):
        if ROUGE_HOME:
            self.ROUGE_HOME = ROUGE_HOME
        else:
            self.ROUGE_HOME = ROUGE_HOME
        self.rouge = None
        self.bertscorer = None
        self.chrfpp = None
        self.cider = None
        self.moverscore = None
        self.sent_mover = None
        self.summaqa = None
        self.meteor = None
        self.bleu = None
        self.blanc = None
        self.txt_stats = None
        self.s3 = None
        self.js = None

    def get_rouge(self, cand, refs, batch_mode=False):
        """
        cand: str
        refs: str or list of strings
        """
        if not self.rouge:
            self.rouge = RougeMetric(self.ROUGE_HOME)
        if cand and refs:
            if batch_mode:
                score_dict = self.rouge.evaluate_batch(cand, refs)["rouge"]
            else:
                score_dict = self.rouge.evaluate_example(cand, refs)['rouge']
            return score_dict

    def get_rouge_we(self, cands, refs, n_gram=1, batch_mode=False, aggregate=True):
        if cands and refs:
            self.rouge_we = RougeWeMetric(n_gram=n_gram)
            if batch_mode:
                score = self.rouge_we.evaluate_batch(cands, refs, aggregate=aggregate)
            else:
                score = self.rouge_we.evaluate_example(cands, refs)
            return score

    def get_bert_score(self, cands, refs, batch_mode=False, aggregate=True, use_default_params=True, lang='en',
                       model_type='roberta-large', num_layers=17,
                       verbose=False, idf=False, batch_size=3, rescale_with_baseline=False):
        """
        cands: str
        refs: list of strings
        """
        if cands and refs:
            bertscorer = self.bertscorer
            if use_default_params:
                if not bertscorer:
                    self.bertscorer = BertScoreMetric(lang='en', model_type='roberta-large', num_layers=17,
                                                      verbose=False, idf=False, \
                                                      batch_size=3, rescale_with_baseline=False)
                    bertscorer = self.bertscorer
            else:
                bertscorer = BertScoreMetric(lang=lang, model_type=model_type, num_layers=num_layers,
                                             verbose=verbose, idf=idf, \
                                             batch_size=batch_size, rescale_with_baseline=rescale_with_baseline)
            if batch_mode:
                score_dict = bertscorer.evaluate_batch(cands, refs, aggregate=aggregate)
            else:
                score_dict = bertscorer.evaluate_example(cands, refs)
            return score_dict

    def get_chrfpp(self, cands, refs, batch_mode=False, aggregate=True):
        if not self.chrfpp:
            self.chrfpp = ChrfppMetric()
        if cands and refs:
            if batch_mode:
                cands, refs = self.str_to_list(cands, refs)
                score = self.chrfpp.evaluate_batch(cands, refs,
                                                   aggregate=aggregate)  # not working due to ziplongest method, need to fix later
            else:
                score = self.chrfpp.evaluate_example(cands, refs)
            return score

    def get_cider(self, cands, refs, tokenize=False, batch_mode=True, aggregate=True):
        if cands and refs:
            if not self.cider:
                self.cider = CiderMetric(tokenize=tokenize)
            if batch_mode:
                score = self.cider.evaluate_batch(cands, refs, aggregate=aggregate)
            # else:
            #     score = self.cider.evaluate_example(cands, refs)
            return score

    def str_to_list(self, cands, refs, exclude_cands=False, exclude_refs=False):
        """
        if cands or refs is of type str, turn them into list
        """
        if cands and refs:
            if isinstance(cands, str) and not exclude_cands:
                cands = [cands]
            if isinstance(refs, str) and not exclude_refs:
                refs = [refs]
            return cands, refs

    def get_moverscore(self, cands, refs, version=2, use_default=True, stop_wordsf=None, n_gram=1, remove_subwords=True,
                       aggregate=False, batch_mode=False):
        """
        default using version 2
        """
        if cands and refs:
            if use_default:
                if not self.moverscore:
                    self.moverscore = MoverScoreMetric(version=version, stop_wordsf=stop_wordsf, n_gram=n_gram,
                                                       remove_subwords=remove_subwords)
                moverscore = self.moverscore
            else:
                moverscore = MoverScoreMetric(version=version, stop_wordsf=stop_wordsf, n_gram=n_gram,
                                              remove_subwords=remove_subwords)

            if batch_mode:
                cands, refs = self.str_to_list(cands, refs)
                scores = moverscore.evaluate_batch(cands, refs, aggregate=aggregate)
            else:
                scores = moverscore.evaluate_example(cands, refs)
            return scores

    def get_txt_stats(self, cand, ref):
        if cand and ref:
            if not self.txt_stats:
                self.txt_stats = DataStatsMetric()
            stats = self.txt_stats.evaluate_example(cand, ref)
            return stats

    def get_s3(self, cand, ref, batch_mode=False):
        if cand and ref:
            if not self.s3:
                self.s3 = S3Metric()
            if batch_mode:
                score = self.s3.evaluate_batch(cand, ref)
            else:
                score = self.s3.evaluate_example(cand, ref)
            return score
    def get_JS(self, cand, ref, batch_mode=False):
        if cand and ref:
            if not self.js:
                self.js = JSMetric()
            if batch_mode:
                score = self.js.evaluate_JS_batch(cand, ref)
            else:
                score = self.js.get_JS(cand, ref)
            return score

    def get_sent_mover(self, cands, refs, wordrep='glove', metric_type='sms', batch_mode=False):
        """
        metric_type: sms, wms, s+wms
        """
        if cands and refs:
            if not self.sent_mover:
                self.sent_mover = SentenceMoversMetric(wordrep=wordrep, metric=metric_type)
            if batch_mode:
                cands, refs = self.str_to_list(cands, refs)
                score = self.sent_mover.evaluate_batch(cands, refs)
            else:
                score = self.sent_mover.evaluate_example(cands, refs)
            return score

    def get_summaqa(self, cands, src_article, use_default_batch=True, batch_size=8, batch_mode=False):
        if cands and src_article:
            if use_default_batch:
                if not self.summaqa:
                    self.summaqa = SummaQAMetric(batch_size=batch_size)
                    summaqa = self.summaqa
                else:
                    summaqa = SummaQAMetric(batch_size=batch_size)
            if batch_mode:
                cands, refs = self.str_to_list(cands, src_article)
                score = summaqa.evaluate_batch(cands, src_article)
            else:
                score = summaqa.evaluate_example(cands, src_article)
            return score

    def get_meteor(self, cands, refs, batch_mode=False, aggregate=True):
        if cands and refs:
            if not self.meteor:
                self.meteor = MeteorMetric()
            if batch_mode:
                cands, refs = self.str_to_list(cands, refs)
                score = self.meteor.evaluate_batch(cands, refs, aggregate=aggregate)
            else:
                score = self.meteor.evaluate_example(cands, refs)
            return score

    def get_bleu(self, cands, refs, batch_mode=False, aggregate=True):
        if cands and refs:
            if not self.bleu:
                self.bleu = BleuMetric()
            if batch_mode:
                score = self.bleu.evaluate_batch(cands, refs, aggregate=aggregate)
            else:
                score = self.bleu.evaluate_example(cands, refs)
            return score

    def get_blanc(self, cands, src_texts, use_tune=True, use_default=True, batch_mode=False, aggregate=True,
                  device="cpu"):
        if cands and src_texts:
            if use_default:
                if not self.blanc:
                    blanc = self.blanc = BlancMetric(use_tune=use_tune, device=device)
            else:
                blanc = BlancMetric(use_tune=use_tune, device=device)
            if batch_mode:
                score = blanc.evaluate_batch(cands, src_texts, aggregate=aggregate)
            else:
                score = blanc.evaluate_example(cands, src_texts)
            return score

    def add_metric_score(self, score_dict, results: defaultdict, is_batch_mode=False):
        if score_dict:
            if isinstance(score_dict, dict):
                for k, v in score_dict.items():
                    results[k].append(v)
            elif isinstance(score_dict, list):
                for score in score_dict:
                    for k, v in score.items():
                        results[k].append(v)

    def add_JS_to_results(self, cand: str, ref: str, results: defaultdict, scorer=None):
        if cand and ref:
            js_dict = scorer.get_JS(cand, ref, batch_mode=True)
            self.add_metric_score(js_dict, results=results)

    def add_scores_to_results(self, cand: str, ref: str, results: defaultdict, include_rouge=False,
                              include_bertscore=True, include_sent_mover=False, include_rwe=True, \
                              include_moverscore=False, include_bleu=True, include_meteor=True, include_txt_stats=True):
        if cand and ref:
            if include_rouge:
                rouge_dict = self.get_rouge(cand, ref)
                self.add_metric_score(rouge_dict, results=results)
            # rouge_we
            if include_rwe:
                rwe1 = self.get_rouge_we(cand, ref, n_gram=1)
                rwe2 = self.get_rouge_we(cand, ref, n_gram=2)
                self.add_metric_score([rwe1, rwe2], results=results)
            # bertscore
            if include_bertscore:
                bertscore = self.get_bert_score(cand, ref)
                self.add_metric_score(bertscore, results=results)
            # moverscore
            if include_moverscore:
                moverscore = self.get_moverscore(cand, ref)  # defalut version 2
                self.add_metric_score(moverscore, results=results)
            # sent_mover
            if include_sent_mover:
                sent_mover = self.get_sent_mover(cand, ref)
                self.add_metric_score(sent_mover, results=results)
            # bleu
            if include_bleu:
                bleu_score = self.get_bleu(cand, ref)
                self.add_metric_score(bleu_score, results=results)
            # meteor
            if include_meteor:
                meteor_score = self.get_meteor(cand, ref)
                self.add_metric_score(meteor_score, results=results)
            if include_txt_stats:
                txt_stats = self.get_txt_stats(cand, ref)
                self.add_metric_score(txt_stats, results=results)

    def add_batch_scores_to_results(self, cand: str, ref: str, results: defaultdict, include_rouge=True,
                                    include_bertscore=True, include_sent_mover=True, include_rwe=True, \
                                    include_moverscore=True, include_bleu=True, include_meteor=True,
                                    include_txt_stats=True):
        if cand and ref:
            if include_rouge:
                rouge_dict = self.get_rouge(cand, ref, batch_mode=True)
                self.add_metric_score(rouge_dict, results=results)
            # rouge_we
            if include_rwe:
                rwe1 = self.get_rouge_we(cand, ref, n_gram=1, batch_mode=True, aggregate=False)
                rwe2 = self.get_rouge_we(cand, ref, n_gram=2, batch_mode=True, aggregate=False)
                self.add_metric_score(rwe1, results=results)
                self.add_metric_score(rwe2, results=results)

            # bertscore
            if include_bertscore:
                bertscore = self.get_bert_score(cand, ref, batch_mode=True, aggregate=False)
                self.add_metric_score(bertscore, results=results)
            # moverscore
            if include_moverscore:
                moverscore = self.get_moverscore(cand, ref, batch_mode=True, aggregate=False)  # defalut version 2
                self.add_metric_score(moverscore, results=results)
            # sent_mover
            if include_sent_mover:
                sent_mover = self.get_sent_mover(cand, ref, batch_mode=True, aggregate=False)
                self.add_metric_score(sent_mover, results=results)
            # bleu
            if include_bleu:
                bleu_score = self.get_bleu(cand, ref, batch_mode=True, aggregate=False)
                self.add_metric_score(bleu_score, results=results)
            # meteor
            if include_meteor:
                meteor_score = self.get_meteor(cand, ref, batch_mode=True, aggregate=False)
                self.add_metric_score(meteor_score, results=results)
            # if include_txt_stats:
            #     txt_stats = self.get_txt_stats(cand, ref)
            #     self.add_metric_score(txt_stats, results=results)

    def cal_batch_scores_from_txt(self, cands, refs, save_to=""):
        if cands and refs:
            results = defaultdict(list)
            assert len(cands) == len(refs)
            "cands and refs should be of the same length"
            print(f"................ start calculating batch scores for {len(cands)} cand ref pair(s) ................")
            self.add_batch_scores_to_results(cands, refs, results, include_rouge=False, include_moverscore=False,
                                             include_bertscore=False,
                                             include_sent_mover=False, include_rwe=False)
            df = pd.DataFrame(results)
            print(
                f"................ finish calculating  batch scores for {len(cands)} cand ref pair(s) ................")
            if save_to:
                df.to_csv(save_to, index=False)
                print(
                    f"................ append batch results and save it to {save_to} ...............")

    def cal_scores_from_csv(self, src_path="", save_to="", clean_text=False, cand_col="cand", ref_col="ref"):
        """
        return a dataframe contains the calculated scores
        save to the save_to path if it is specified
        assume the cand summary is under column 'cand' and reference summary is under column 'ref'
        """
        if src_path:
            df = load_df(src_path)[:20].copy()
            cands = df[cand_col].values.tolist()
            refs = df[ref_col].values.tolist()
            if cands and refs:
                results = defaultdict(list)
                assert len(cands) == len(refs)
                "cands and refs should be of the same length"
                print(f"................ start calculating scores for {len(cands)} cand ref pair(s) ................")
                for cand, ref in zip(cands, refs):
                    if clean_text:
                        cand = clean_cnn_text(cand, remove_newline=True, clean_sep=True)
                        ref = clean_cnn_text(ref, remove_newline=True, clean_sep=True)
                    self.add_scores_to_results(cand, ref, results)
                    self.add_JS_to_results(cand, ref, results=results)
                for k, v in results.items():
                    df[k] = v
                print(f"................ finish calculating scores for {len(cands)} cand ref pair(s) ................")
                if save_to:
                    df.to_csv(save_to)
                    print(
                        f"................ append results and save it to {save_to} ...............")

def example_rouge_we():
    example_cand = "cats and dogs have the advantage over marine pets in that they can interact with humans through the sense of touch."
    example_ref = "cats and dogs can interact with humans through the sense of touch, therefore they have the advantage over marine pets."
    scorer = MetricScorer()
    score1 = scorer.get_rouge_we(example_cand, example_ref, n_gram=1)
    score2 = scorer.get_rouge_we(example_cand, example_ref, n_gram=2)
    score3 = scorer.get_rouge_we(example_cand, example_ref, n_gram=3)
    print("rouge_we n_gram=1: ")
    print(score1)
    print("rouge_we n_gram=2: ")
    print(score2)
    print("rouge_we n_gram=3: ")
    print(score3)


def example_meteor():
    example_cand = "cats and dogs have the advantage over marine pets in that they can interact with humans through the sense of touch."
    example_ref = "cats and dogs can interact with humans through the sense of touch, therefore they have the advantage over marine pets."
    scorer = MetricScorer()
    score = scorer.get_meteor(example_cand, example_ref)
    print(score)


def example_bleu():
    example_cand = "cats and dogs have the advantage over marine pets in that they can interact with humans through the sense of touch."
    example_ref = "cats and dogs can interact with humans through the sense of touch, therefore they have the advantage over marine pets."
    scorer = MetricScorer()
    score = scorer.get_bleu(example_cand, example_ref)
    print(score)

if __name__ == '__main__':
    # scorer = MetricScorer()
    example_bleu()
