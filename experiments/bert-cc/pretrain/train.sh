#!/bin/bash

DATA=/science/image/nlp-datasets/emanuele/data
ANNOS=$DATA/conceptual_captions/annotations
FEATS=$DATA/conceptual_captions/resnet101_faster_rcnn_genome_imgfeats/volta
OUTPUT_DIR=/science/image/nlp-datasets/emanuele/checkpoints/xm-influence/bert/conceptual_captions
LOGGING_DIR=$HOME/projects/cross-modal-influence/logs/bert/conceptual_captions

source activate /science/image/nlp-datasets/emanuele/envs/xm-influence

cd ../../../volta
python train_bert_concap.py \
  --bert_model bert-base-uncased --from_pretrained  bert-base-uncased \
  --train_batch_size 256 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --weight_decay 0.01 --warmup_proportion 0.1 --clip_grad_norm 5.0 \
  --objective 2 \
  --annotations_path $ANNOS --features_path $FEATS \
  --output_dir ${OUTPUT_DIR} \
  --logdir ${LOGGING_DIR} \
  --num_train_epochs 10 \
#  --resume_file ${OUTPUT_DIR}/${MODEL_CONFIG}/pytorch_ckpt_latest.tar

conda deactivate
