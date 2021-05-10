import pandas as pd
import re
import os


def load_df(csv_path=None, ignore_idx=False):
    if csv_path:
        if ignore_idx:
            df = pd.read_csv(csv_path, index_col=[0])
        else:
            df = pd.read_csv(csv_path)
        return df


def clean_cnn_text(seq, clean_sep=False, sep_start='<t>', sep_end='</t>', remove_newline=False):
    if seq:
        if remove_newline:
            seq = re.sub(r'\n', '', seq)  # remove newline character
        seq = re.sub(r'\t', '', seq)
        if clean_sep:
            seq = re.sub(sep_start, '', seq)
            seq = re.sub(sep_end, '', seq)
        seq = seq.strip()
        return seq


def get_filename_from_path(filepath=""):
    if filepath:
        basename = os.path.basename(filepath)
        return basename


def get_file_names(file_dir=None, ext='.csv'):
    if file_dir:
        files = os.listdir(file_dir)
        if files:
            filenames = []
            for filename in files:
                if filename:
                    # TODO: can use os function to split the ext and filename
                    # but use re might be more flexible if need other modifications in the future
                    filename = re.sub(ext, '', filename)
                    filenames.append(filename)
            return files, filenames


def get_files_in_dir(file_dir=None, sort_by_name=False, excludes=None, chk_endswith=None):
    if file_dir:
        files = os.listdir(file_dir)
        if files:
            if sort_by_name:
                files = sorted(files)
                if excludes:
                    files = [file for file in files if file and file not in excludes]
                if chk_endswith:
                    files = [file for file in files if file.endswith(chk_endswith)]
            return files


def clean_summary(seq, clean_sep=False, sep_start='<t>', sep_end='</t>'):
    if seq:
        seq = re.sub(r'\n', '', seq)  # remove newline character
        seq = re.sub(r'\t', '', seq)
        if clean_sep:
            seq = re.sub(sep_start, '', seq)
            seq = re.sub(sep_end, '', seq)
        seq = seq.strip()
        return seq


def clean_cand_ref(candidate, ref, cand_as_str=False, new_sent_sep=" ", sep_start='<t>', sep_end='</t>'):
    """
    clean separator for multiple sentences summary
    """
    if candidate and ref:
        cand_list = []
        ref_list = []
        cand_sents = candidate.split(sep_end)
        ref_sents = ref.split(sep_end)
        for cand_sent, ref_sent in zip(cand_sents, ref_sents):
            cand_sent = clean_summary(cand_sent, clean_sep=True)
            ref_sent = clean_summary(ref_sent, clean_sep=True)
            if cand_sent and ref_sent:
                cand_list.append(cand_sent)
                ref_list.append(ref_sent)
        if cand_as_str:
            cand_str = new_sent_sep.join(sent for sent in cand_list)
            ref_str = new_sent_sep.join(sent for sent in ref_list)
            return cand_str, ref_str
        else:
            return cand_list, ref_list


def get_summary_as_list_from_df(df, col_cand='cand', col_ref='ref', clean_summ=False, cand_as_str=False):
    cands = df[col_cand].to_list()
    refs = df[col_ref].to_list()
    if cands and refs:
        cand_list = []
        ref_list = []
        for cand, ref in zip(cands, refs):
            if clean_summ:
                if clean_summ:
                    cand, ref = clean_cand_ref(cand, ref, cand_as_str=cand_as_str)
                if cand and ref:
                    cand_list.append(cand)
                    ref_list.append(ref)
        return cand_list, ref_list


def get_docId_summary_as_list_from_df(df, col_cand='cand', col_ref='ref', col_docId='doc_id', clean_summ=False,
                                      cand_as_str=True):
    cands = df[col_cand].to_list()
    refs = df[col_ref].to_list()
    docIds = df[col_docId].to_list()
    if cands and refs:
        cand_list = []
        ref_list = []
        docId_list = []
        for doc_id, cand, ref in zip(docIds, cands, refs):
            if clean_summ:
                if clean_summ:
                    cand, ref = clean_cand_ref(cand, ref, cand_as_str=cand_as_str)
                if cand and ref:
                    docId_list.append(doc_id)
                    cand_list.append(cand)
                    ref_list.append(ref)
        return docId_list, cand_list, ref_list
