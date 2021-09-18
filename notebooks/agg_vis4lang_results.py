import json
import pickle
import numpy as np
import pandas as pd


results_dir = "/science/image/nlp-datasets/emanuele/results/xm-influence/flickr30kentities_vis4lang/"


def get_rows(df, basedir, name):
    
    # none ablation
    mlm = pickle.load(open(basedir + 'val_none_mlm.pkl', 'rb'))
    mlm = np.mean(mlm)
    df = df.append({'Model': name, 'Mask': "None", "MLM": mlm}, ignore_index=True)
    
    # object ablation (with different thresholds)
    for i in range(9, -1, -1):
        mlm = pickle.load(open(basedir + f'val_object0.{i}_mlm.pkl', 'rb'))
        mlm = np.mean(mlm)
        df = df.append({'Model': "UNITER", 'Mask': i/10, "MLM": mlm}, ignore_index=True)

    # all ablation
    mlm = pickle.load(open(basedir + 'val_all_mlm.pkl', 'rb'))
    mlm = np.mean(mlm)
    df = df.append({'Model': "UNITER", 'Mask': "All", "MLM": mlm}, ignore_index=True)

    return df


df = pd.DataFrame(columns=["Model", "Mask", "MLM"])

# BERT
basedir = results_dir + "bert/"
mlm = pickle.load(open(basedir + 'val_mlm.pkl', 'rb'))
mlm = np.mean(bert_mlm)
df = df.append({'Model': "BERT", 'Mask': None, "MLM": mlm}, ignore_index=True)

# BERT_CC
basedir = results_dir + "bert-cc_pre5/"
mlm = pickle.load(open(basedir + 'val_mlm.pkl', 'rb'))
mlm = np.mean(bert_mlm)
df = df.append({'Model': "BERT-CC", 'Mask': None, "MLM": mlm}, ignore_index=True)


name2dir = {
    "UNITER": "ctrl_uniter/",
    "VL-BERT": "ctrl_vl-bert/",
    "VisualBERT": "ctrl_visualbert/",
    "ViLBERT": "ctrl_vilbert/",
    "LXMERT": "ctrl_lxmert/",
    
    "UNITER_rnd-vl": "ctrl_uniter_rnd-vl",
    "ViLBERT_rnd-vl": "ctrl_vilbert_rnd-vl",
    
    "UNITER_rnd-v-vl": "ctrl_uniter_rnd-v-vl",
    "UNITER_bert-v-vl": "ctrl_uniter_bert-v-vl",
    
    "UNITER_thr02": "ctrl_uniter_thr02/",
    "UNITER_thr04": "ctrl_uniter_thr04/",
    "UNITER_thr06": "ctrl_uniter_thr06/",
    "UNITER_thr10": "ctrl_uniter_thr10/",
    "UNITER_thr02iot": "ctrl_uniter_thr02iot/",
    "UNITER_thr04iot": "ctrl_uniter_thr04iot/",
    "UNITER_thr06iot": "ctrl_uniter_thr06iot/",
    
    "UNITER-xent": "ctrl_uniter_xent/"
}
for name, d in name2dir.items():
    basedir = results_dir + d
    for abl in ['all', 'object', 'none']:
        res = get_row(basedir, abl, name)
        # nats to bits
        res['MLM'] /= np.log(2)
        df = df.append(res, ignore_index=True)

# Results by seed
models = ['lxmert', 'vilbert', 'vl-bert', 'visualbert', 'uniter']
names = ['LXMERT_s%s', 'ViLBERT_s%s', 'VL-BERT_s%s', 'VisualBERT_s%s', 'UNITER_s%s']
seeds = ['0', '1234', '27', '33', '42', '54', '69', '73', '89', '93']
basedir = results_dir + "ctrl_%s_s%s/"
for abl in ['all', 'phrase', 'none']:
    for s in seeds:
        for im, m in enumerate(models):
            res = get_row(basedir % (m, s), abl, names[im] % s)
            res['MLM'] /= np.log(2)
            df = df.append(res, ignore_index=True)

df.to_csv("val_%.1f_vis4lang.tsv" % thr, sep='\t')
