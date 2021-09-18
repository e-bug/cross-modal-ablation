#!/bin/bash

TASK=18
MODEL=bert
TASKS_CONFIG=xm-influence_test_tasks
OUTPUT_DIR=/science/image/nlp-datasets/emanuele/results/xm-influence/flickr30kentities_vis4lang/${MODEL}

source activate /science/image/nlp-datasets/emanuele/envs/xm-influence

cd ../../../volta
python ablate_textonly_lang.py \
        --bert_model bert-base-uncased \
        --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --split val \
        --output_dir ${OUTPUT_DIR}

conda deactivate
