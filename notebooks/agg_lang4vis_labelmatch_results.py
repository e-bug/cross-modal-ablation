import json
import pickle
import numpy as np
import pandas as pd

# Overlap threshold
thr = 0.5

results_dir = "/science/image/nlp-datasets/emanuele/results/xm-influence/flickr30kentities_lang4vis/"


vg_ix2label = pickle.load(open("flickr30k_entities_vg/ix2label.pkl", "rb"))
vg_ixs = list(vg_ix2label.keys())


def xent_1601(filename):
    score = torch.tensor(pickle.load(open(filename, "rb")))[vg_ixs].reshape(-1, 1601)
    targt = torch.tensor(list(vg_ix2label.values()), dtype=torch.long)
    label = torch.ones(len(vg_ix2label), dtype=torch.long).view(-1,)
    
    loss = nn.CrossEntropyLoss(reduction='none')(score, targt)
    return (torch.sum(loss * (label.view(-1) == 1)) / max(torch.sum((label == 1)), 1)).item()


def get_row(basedir, mask_level, name):
    mrc = np.array(pickle.load(open(basedir + 'val_%.1f_%s_kld.pkl' % (thr, mask_level), 'rb')))
    mrc = np.mean(mrc[vg_ixs])
    
    mxe = np.array(pickle.load(open(basedir + 'val_%.1f_%s_xe.pkl' % (thr, mask_level), 'rb')))
    mxe = np.mean(mxe[vg_ixs])
    
    mxe_vg = xent_1601(basedir + 'val_%.1f_%s_score.pkl' % (thr, mask_level))
        
    res = {'Model': name, 'Mask': mask_level.capitalize(), "MRC-KL": mrc, "MRC-XE": mxe, "MRC-XE-VG": mxe_vg}
    return res


df = pd.DataFrame(columns=["Model", "Mask", "MRC-KL", "MRC-XE", "MRC-XE-VG"])

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
        res['MRC-XE-VG'] /= np.log(2)
        df = df.append(res, ignore_index=True)

df.to_csv("val_labelmatch_%.1f_lang4vis.tsv" % thr, sep='\t')
