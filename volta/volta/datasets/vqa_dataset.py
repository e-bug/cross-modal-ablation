# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import _pickle as cPickle
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _create_entry(question, answer):
    answer.pop("image_id")
    answer.pop("question_id")
    entry = {
        "question_id": question["question_id"],
        "image_id": question["image_id"],
        "question": question["question"],
        "answer": answer,
    }
    return entry


def _load_dataset(dataroot, name):
    """Load entries

    dataroot: root path of dataset
    name: 'train', 'val', 'trainval', 'minsval'
    """
    if name == "train" or name == "val":
        question_path = os.path.join(dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % name)
        questions = sorted(json.load(open(question_path))["questions"], key=lambda x: x["question_id"])
        answer_path = os.path.join(dataroot, "cache", "%s_target.pkl" % name)
        answers = cPickle.load(open(answer_path, "rb"))
        answers = sorted(answers, key=lambda x: x["question_id"])

    elif name == "trainval":
        question_path_train = os.path.join(dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "train")
        questions_train = sorted(json.load(open(question_path_train))["questions"], key=lambda x: x["question_id"])
        answer_path_train = os.path.join(dataroot, "cache", "%s_target.pkl" % "train")
        answers_train = cPickle.load(open(answer_path_train, "rb"))
        answers_train = sorted(answers_train, key=lambda x: x["question_id"])

        question_path_val = os.path.join(dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "val")
        questions_val = sorted(json.load(open(question_path_val))["questions"], key=lambda x: x["question_id"])
        answer_path_val = os.path.join(dataroot, "cache", "%s_target.pkl" % "val")
        answers_val = cPickle.load(open(answer_path_val, "rb"))
        answers_val = sorted(answers_val, key=lambda x: x["question_id"])
        questions = questions_train + questions_val[:-3000]
        answers = answers_train + answers_val[:-3000]

    elif name == "minval":
        question_path_val = os.path.join(dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "val")
        questions_val = sorted(json.load(open(question_path_val))["questions"], key=lambda x: x["question_id"])
        answer_path_val = os.path.join(dataroot, "cache", "%s_target.pkl" % "val")
        answers_val = cPickle.load(open(answer_path_val, "rb"))
        answers_val = sorted(answers_val, key=lambda x: x["question_id"])
        questions = questions_val[-3000:]
        answers = answers_val[-3000:]

    elif name == "test":
        question_path_test = os.path.join(dataroot, "v2_OpenEnded_mscoco_%s2015_questions.json" % "test")
        questions_test = sorted(json.load(open(question_path_test))["questions"], key=lambda x: x["question_id"])
        questions = questions_test

    elif name == "mteval":
        question_path_train = os.path.join(dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "train")
        questions_train = sorted(json.load(open(question_path_train))["questions"], key=lambda x: x["question_id"])
        answer_path_train = os.path.join(dataroot, "cache", "%s_target.pkl" % "train")
        answers_train = cPickle.load(open(answer_path_train, "rb"))
        answers_train = sorted(answers_train, key=lambda x: x["question_id"])

        question_path_val = os.path.join(dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "val")
        questions_val = sorted(json.load(open(question_path_val))["questions"], key=lambda x: x["question_id"])
        answer_path_val = os.path.join(dataroot, "cache", "%s_target.pkl" % "val")
        answers_val = cPickle.load(open(answer_path_val, "rb"))
        answers_val = sorted(answers_val, key=lambda x: x["question_id"])

        questions = questions_train
        answers = answers_train
    else:
        assert False, "data split is not recognized."

    if "test" in name:
        entries = []
        for question in questions:
            entries.append(question)
    elif name == "mteval":
        entries = []
        remove_ids = np.load(os.path.join(dataroot, "cache", "coco_test_ids.npy"))
        remove_ids = [int(x) for x in remove_ids]

        for question, answer in zip(questions, answers):
            if int(question["image_id"]) in remove_ids:
                entries.append(_create_entry(question, answer))
    else:
        assert_eq(len(questions), len(answers))
        entries = []
        remove_ids = []
        for question, answer in zip(questions, answers):
            if "train" in name and int(question["image_id"]) in remove_ids:
                continue
            assert_eq(question["question_id"], answer["question_id"])
            assert_eq(question["image_id"], answer["image_id"])
            entries.append(_create_entry(question, answer))

    return entries


class VQAClassificationDataset(Dataset):
    def __init__(
        self,
        task,
        dataroot,
        annotations_jsonpath,
        split,
        image_features_reader,
        gt_image_features_reader,
        tokenizer,
        bert_model,
        padding_index=0,
        max_seq_length=16,
        max_region_num=101,
        num_locs=5,
        add_global_imgfeat=None,
        append_mask_sep=False,
    ):
        super().__init__()
        self.split = split
        ans2label_path = os.path.join(dataroot, "trainval_ans2label.pkl")
        label2ans_path = os.path.join(dataroot, "trainval_label2ans.pkl")
        self.ans2label = cPickle.load(open(ans2label_path, "rb"))
        self.label2ans = cPickle.load(open(label2ans_path, "rb"))
        self.num_labels = len(self.ans2label)
        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self._num_locs = num_locs
        self._add_global_imgfeat = add_global_imgfeat
        self._append_mask_sep = append_mask_sep

        if "roberta" in bert_model:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task
                + "_"
                + split
                + "_"
                + "roberta"
                + "_"
                + str(max_seq_length)
                + ".pkl",
            )
        else:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task
                + "_"
                + split
                + "_"
                + str(max_seq_length) +
                ".pkl",
            )
        if not os.path.exists(cache_path):
            self.entries = _load_dataset(dataroot, split)
            self.tokenize(max_seq_length)
            self.tensorize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))

        self.qid2imgid = {e["question_id"]: e["image_id"] for e in self.entries}

    def tokenize(self, max_length=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.entries:
            tokens = self._tokenizer.encode(entry["question"])
            tokens = tokens[: max_length - 2]
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), max_length)
            entry["q_token"] = tokens
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids

            if "test" not in self.split:
                answer = entry["answer"]
                labels = np.array(answer["labels"])
                scores = np.array(answer["scores"], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry["answer"]["labels"] = labels
                    entry["answer"]["scores"] = scores
                else:
                    entry["answer"]["labels"] = None
                    entry["answer"]["scores"] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        image_id = entry["image_id"]
        question_id = entry["question_id"]
        features, num_boxes, boxes, _ = self._image_features_reader[image_id]

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        question = entry["q_token"]
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]

        if self._append_mask_sep:
            mask_id = self._tokenizer.convert_tokens_to_ids(['[MASK]', '[CLS]'])
            mask_pos = input_mask.sum()
            question = torch.cat([question[:mask_pos], torch.tensor(mask_id), question[mask_pos:]])
            input_mask = torch.cat([input_mask[:mask_pos], torch.tensor([1, 1]), input_mask[mask_pos:]])
            segment_ids = torch.cat([segment_ids[:mask_pos], torch.tensor([1, 1]), segment_ids[mask_pos:]])

        target = torch.zeros(self.num_labels)

        if "test" not in self.split:
            answer = entry["answer"]
            labels = answer["labels"]
            scores = answer["scores"]
            if labels is not None:
                target.scatter_(0, labels, scores)

        return features, spatials, image_mask, question, target, input_mask, segment_ids, question_id

    def __len__(self):
        return len(self.entries)
