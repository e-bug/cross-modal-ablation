# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import random
import logging
import _pickle as cPickle

import numpy as np

import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


class ConceptualCaptionsTextDataset(Dataset):
    def __init__(
        self,
        dataroot,
        annotations_jsonpath,
        tokenizer,
        bert_model,
        padding_index=0,
        max_seq_length=38,
        max_region_num=0,
        num_locs=5,
        add_global_imgfeat=None,
    ):
        super().__init__()
        self.num_locs = num_locs
        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self._add_global_imgfeat = add_global_imgfeat

#        if "roberta" in bert_model:
#            cache_path = os.path.join(
#                dataroot,
#                "cache",
#                "roberta"
#                + "_"
#                + str(max_seq_length)
#                + ".pkl",
#            )
#        else:
#            cache_path = os.path.join(
#                dataroot,
#                "cache",
#                "bert"
#                + "_"
#                + str(max_seq_length)
#                + ".pkl",
#            )
#        if not os.path.exists(cache_path):
        self.captions = list(json.load(open(annotations_jsonpath, "r")).values())
#            self.entries = []
#            self.tokenize(max_seq_length)
#            self.tensorize()
#            cPickle.dump(self.entries, open(cache_path, "wb"))
#        else:
#            logger.info("Loading from %s" % cache_path)
#            self.entries = cPickle.load(open(cache_path, "rb"))

    def tokenize(self, ix): #, min_seq_len=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
#        for ix, line in enumerate(self.corpus):
        line = self.captions[ix]
        
        # tokenize
        tokens = self._tokenizer.encode(line)

        # truncate
        tokens = tokens[:self._max_seq_length - 2]
        tokens, tokens_label = self.random_word(tokens, self._tokenizer)
        tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)
        lm_label_ids = [-1] + tokens_label + [-1]

        segment_ids = [0] * len(tokens)
        input_mask = [1] * len(tokens)

        # Zero-pad up to the sequence length.
        while len(tokens) < self._max_seq_length:
            tokens.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        assert_eq(len(tokens), self._max_seq_length)
        entry = {
            "q_token": tokens,
            "q_input_mask": input_mask,
            "q_segment_ids": segment_ids,
            "q_lm_labels": lm_label_ids,
        }
#            self.entries.append(entry)
        return entry

    def random_word(self, tokens, tokenizer):
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability

            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = np.random.randint(len(tokenizer))

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return tokens, output_label

    def tensorize(self, entry):
#        for entry in self.entries:
        question = torch.from_numpy(np.array(entry["q_token"]))
        entry["q_token"] = question

        q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
        entry["q_input_mask"] = q_input_mask

        q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
        entry["q_segment_ids"] = q_segment_ids

        q_lm_ids = torch.from_numpy(np.array(entry["q_lm_labels"]))
        entry["q_lm_labels"] = q_lm_ids

    def __getitem__(self, index):
        max_region_num = self._max_region_num
        features = torch.zeros((max_region_num, 2048), dtype=torch.float)
        spatials = torch.zeros((max_region_num, self.num_locs), dtype=torch.float)
        image_mask = torch.zeros(max_region_num, dtype=torch.long)

        entry = self.tokenize(index)
        self.tensorize(entry)
        question = entry["q_token"]
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]
        lm_label_ids = entry["q_lm_labels"]

        # return question, input_mask, segment_ids, lm_label_ids, None, features, spatials, None, None, None, None, None, None, None, image_mask
        return question, input_mask, segment_ids, lm_label_ids, features, spatials, image_mask

    def __len__(self):
        return len(self.captions)

