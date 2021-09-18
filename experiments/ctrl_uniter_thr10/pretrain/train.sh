#!/bin/bash

MODEL=ctrl_uniter
MODEL_CONFIG=ctrl_uniter_base
DATA=/science/image/nlp-datasets/emanuele/data
ANNOS=$DATA/conceptual_captions/annotations
FEATS=$DATA/conceptual_captions/resnet101_faster_rcnn_genome_imgfeats/volta
OUTPUT_DIR=/science/image/nlp-datasets/emanuele/checkpoints/xm-influence/ctrl_uniter_thr10/conceptual_captions
LOGGING_DIR=$HOME/projects/cross-modal-influence/logs/ctrl_uniter_thr10/conceptual_captions

source activate /science/image/nlp-datasets/emanuele/envs/xm-influence

cd ../../../volta
python train_concap.py \
  --bert_model bert-base-uncased --config_file config/${MODEL_CONFIG}.json \
  --train_batch_size 256 --gradient_accumulation_steps 4 --max_seq_length 38 \
  --learning_rate 1e-4 --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --weight_decay 0.01 --warmup_proportion 0.1 --clip_grad_norm 5.0 \
  --objective 1 --threshold 2.0 --num_workers 10 \
  --annotations_path $ANNOS --features_path $FEATS \
  --output_dir ${OUTPUT_DIR} \
  --logdir ${LOGGING_DIR} \
  --num_train_epochs 10 \
  --resume_file ${OUTPUT_DIR}/${MODEL_CONFIG}/pytorch_ckpt_latest.tar

conda deactivate
