#!/bin/bash

TASK=18
MODEL=ctrl_lxmert
MODEL_CONFIG=ctrl_lxmert
TASKS_CONFIG=xm-influence_test_tasks
PRETRAINED=/science/image/nlp-datasets/emanuele/checkpoints/mpre-unmasked/conceptual_captions_s89/volta/ctrl_lxmert/ctrl_lxmert/pytorch_model_9.bin
OUTPUT_DIR=/science/image/nlp-datasets/emanuele/results/xm-influence/flickr30kentities_vis4lang/${MODEL}_s89

source activate /science/image/nlp-datasets/emanuele/envs/xm-influence

cd ../../../../volta
python ablate_vis4lang.py \
        --bert_model bert-base-uncased --config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
        --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --split val \
        --output_dir ${OUTPUT_DIR} --dump_results --masking none

conda deactivate