# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import json
from volta.volta.datasets.flickr30ke_ablation_dataset import FlickrVis4LangDataset
import yaml
import random
import logging
import argparse
from io import open
from tqdm import tqdm
import _pickle as cPickle
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from volta.config import BertConfig
from volta.datasets import FlickrVis4LangDataset
from volta.datasets._all_image_features_reader import ImageFeaturesH5Reader
from volta.train_utils import print_and_log, freeze_layers, tbLogger, summary_parameters, save, resume

from transformers import AutoTokenizer, BertForMaskedLM

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--from_pretrained", default="", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    # Output
    parser.add_argument("--output_dir", default="results", type=str,
                        help="The output directory where the model checkpoints will be written.")
    # Task
    parser.add_argument("--tasks_config_file", default="config_tasks/vilbert_trainval_tasks.yml", type=str,
                        help="The config file which specified the tasks details.")
    parser.add_argument("--task", default="", type=str,
                        help="training task number")
    # Text
    parser.add_argument("--do_lower_case", default=True, type=bool,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    # Evaluation
    parser.add_argument("--split", default="", type=str,
                        help="which split to use.")
    parser.add_argument("--batch_size", default=30, type=int,
                        help="batch size.")
    parser.add_argument("--drop_last", action="store_true",
                        help="whether to drop last incomplete batch")
    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    return parser.parse_args()


def main():
    args = parse_args()

    # Devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    default_gpu = True

    # Load config
    config = BertConfig.from_json_file("config/ctrl_uniter_base.json")
    
    # Load task config
    with open(args.tasks_config_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))
    task_id = args.task.strip()
    task = "TASK" + task_id
    task_name = task_cfg[task]["name"]
    
    # Output dirs
    savePath = args.output_dir  # os.path.join(args.output_dir, timeStamp)
    if default_gpu and not os.path.exists(savePath):
        os.makedirs(savePath)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset
    feats_h5path = task_cfg[task]["features_h5path1"]
    features_reader = ImageFeaturesH5Reader(feats_h5path, config)
    batch_size = task_cfg[task]["batch_size"]
    num_workers = 0
    logger.info("Loading %s Dataset with batch size %d" % (task_name, batch_size))
    eval_split = args.split or task_cfg[task]["val_split"]
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    dset = FlickrVis4LangDataset(
        task, task_cfg[task]["dataroot"], "all", eval_split, features_reader, None,
        tokenizer, args.bert_model, max_seq_length=task_cfg[task]["max_seq_length"],
        max_region_num=task_cfg[task]["max_region_num"], num_locs=config.num_locs,
        add_global_imgfeat=config.add_global_imgfeat
    )
    dl = DataLoader(dset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Model
    model = BertForMaskedLM.from_pretrained(args.bert_model)

    # Load checkpoint
    if args.from_pretrained:
        logger.info("Loading weights from %s" % args.from_pretrained)
        state_dict = torch.load(args.from_pretrained, map_location='cpu')
        model.load_state_dict(state_dict)

    # Move to GPU(s)
    model.to(device)

    # Print summary
    if default_gpu:
        print("***** Running evaluation *****")
        print("  Num Iters: ", len(dl))
        print("  Batch size: ", batch_size)
        
    # Evaluate
    model.eval()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
    phrase_ids, image_ids, pred_tokens, true_tokens, pred_scores, lm_losses = [], [], [], [], [], []
    for i, batch in tqdm(enumerate(dl), total=len(dl)):
    #     batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch[:-1])
    #     phrase_id, caption, input_mask, segment_ids, lm_label_ids, features, spatials, image_mask, image_id = batch
        phrase_id, caption, input_mask, segment_ids, lm_label_ids, features, spatials, image_cls, \
        obj_labels, obj_confs, attr_labels, attr_confs, image_attrs, image_mask, image_labels, image_id = batch
        with torch.no_grad():
            outputs = model.forward(caption, attention_mask=input_mask, token_type_ids=segment_ids)
            predictions_t = outputs[0]
            
            # loss = masked_loss_t + masked_loss_v + pair_match_loss
            target_ixs = [[] for _ in range(predictions_t.size(0))]
            xs, ys = torch.where(lm_label_ids != -1)
            for x, y in zip(xs, ys):
                target_ixs[x].append(y.item())
            for bix in range(predictions_t.size(0)):
                pred_bix_tokens, true_bix_tokens, bix_predictions = [], [], []
                for masked_ix in target_ixs[bix]:
                    predicted_index = torch.argmax(predictions_t[bix, masked_ix]).item()
                    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
                    label_token = tokenizer.convert_ids_to_tokens([lm_label_ids[bix, masked_ix].item()])[0]
                    pred_bix_tokens.append(predicted_token)
                    true_bix_tokens.append(label_token)
                    bix_predictions.append(predictions_t[bix, masked_ix].numpy())
                masked_lm_loss = loss_fct(predictions_t[bix].view(-1, config.vocab_size), lm_label_ids[bix].view(-1),).unsqueeze(0).item()

                pred_tokens.append(pred_bix_tokens)
                true_tokens.append(true_bix_tokens)
                pred_scores.append(bix_predictions)
                image_ids.append(image_id[bix].item())
                phrase_ids.append(phrase_id[bix].item())
                lm_losses.append(masked_lm_loss)

    eval_path = os.path.join(savePath, eval_split)
    cPickle.dump(pred_tokens, open(eval_path + "_preds.pkl", "wb"))
    cPickle.dump(true_tokens, open(eval_path + "_truth.pkl", "wb"))
    cPickle.dump(pred_scores, open(eval_path + "_score.pkl", "wb"))
    cPickle.dump(image_ids, open(eval_path + "_imgids.pkl", "wb"))
    cPickle.dump(phrase_ids, open(eval_path + "_phrids.pkl", "wb"))
    cPickle.dump(lm_losses, open(eval_path + "_mlm.pkl", "wb"))


if __name__ == "__main__":
    main()
