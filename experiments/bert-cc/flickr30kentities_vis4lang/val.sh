#!/bin/bash

TASK=18
MODEL=bert-cc
TASKS_CONFIG=xm-influence_test_tasks
CHECKPOINT=/science/image/nlp-datasets/emanuele/checkpoints/xm-influence/bert/conceptual_captions/pytorch_model_4.bin
OUTPUT_DIR=/science/image/nlp-datasets/emanuele/results/xm-influence/flickr30kentities_vis4lang/${MODEL}

source activate /science/image/nlp-datasets/emanuele/envs/xm-influence

cd ../../../../volta
python ablate_textonly_lang.py \
        --bert_model bert-base-uncased --from_pretrained $CHECKPOINT \
        --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --split val \
        --output_dir ${OUTPUT_DIR}

conda deactivate
