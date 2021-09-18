#!/bin/bash

DATA=/gs/hs0/tgb-deepmt/bugliarello.e/data/conceptual_captions/
ANNOS=$DATA/annotations
FEATS=$DATA/resnet101_faster_rcnn_genome_imgfeats/volta
MODEL=ctrl_vilbert
MODEL_CONFIG=ctrl_vilbert_base
OUTPUT_DIR=/gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/xm-influence/conceptual_captions_scratch/${MODEL}
LOGGING_DIR=../../../logs/ctrl_uniter_scratch/conceptual_captions

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/xm-influence

cd ../../../volta
python train_concap.py \
  --bert_model bert-base-uncased --from_pretrained "" --config_file config/${MODEL_CONFIG}.json \
  --train_batch_size 256 --gradient_accumulation_steps 1 --max_seq_length 38 \
  --learning_rate 1e-4 --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --weight_decay 0.01 --warmup_proportion 0.1 --clip_grad_norm 5.0 \
  --objective 1 \
  --annotations_path $ANNOS --features_path $FEATS \
  --output_dir ${OUTPUT_DIR} \
  --logdir ${LOGGING_DIR} \
  --num_train_epochs 10 \
  --resume_file ${OUTPUT_DIR}/${MODEL_CONFIG}/pytorch_ckpt_latest.tar

conda deactivate
