import json
import pickle
import numpy as np
import pandas as pd

# Overlap threshold
thr = 0.5

results_dir = "/science/image/nlp-datasets/emanuele/results/xm-influence/flickr30kentities_lang4vis/"


def get_row(basedir, mask_level, name):
    mrc = np.array(pickle.load(open(basedir + 'val_%.1f_%s_kld.pkl' % (thr, mask_level), 'rb')))
    try:
        mrc = np.mean(mrc)
    except:
        mrc = None
    
    mxe = np.array(pickle.load(open(basedir + 'val_%.1f_%s_xe.pkl' % (thr, mask_level), 'rb')))
    mxe = np.mean(mxe)
        
    res = {'Model': name, 'Mask': mask_level.capitalize(), "MRC-KL": mrc, "MRC-XE": mxe}
    return res


df = pd.DataFrame(columns=["Model", "Mask", "MRC-KL", "MRC-XE"])

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
    for abl in ['all', 'phrase', 'none']:
        res = get_row(basedir, abl, name)
        # nats to bits
        res['MRC-KL'] /= np.log(2)
        res['MRC-XE'] /= np.log(2)
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
            res['MRC-KL'] /= np.log(2)
            res['MRC-XE'] /= np.log(2)
            df = df.append(res, ignore_index=True)

df.to_csv("val_%.1f_lang4vis.tsv" % thr, sep='\t')
