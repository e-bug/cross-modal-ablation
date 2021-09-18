#!/bin/bash

DATA=/science/image/nlp-datasets/emanuele/data/conceptual_captions/
ANNOS=$DATA/annotations
FEATS=$DATA/resnet101_faster_rcnn_genome_imgfeats/volta
MODEL=ctrl_uniter
MODEL_CONFIG=ctrl_uniter_base
PRETRAINED=/science/image/nlp-datasets/emanuele/checkpoints/xm-influence/conceptual_captions_viz/${MODEL}_scratch/${MODEL_CONFIG}/pytorch_model_9.bin
OUTPUT_DIR=/science/image/nlp-datasets/emanuele/checkpoints/xm-influence/conceptual_captions_fromviz/${MODEL}_scratch
LOGGING_DIR=../../../logs/ctrl_uniter_scratch/conceptual_captions_fromviz-scratch

source activate /science/image/nlp-datasets/emanuele/envs/xm-influence

cd ../../../volta
python train_concap.py \
  --bert_model bert-base-uncased --from_pretrained $PRETRAINED --config_file config/${MODEL_CONFIG}.json \
  --train_batch_size 256 --gradient_accumulation_steps 2 --max_seq_length 38 \
  --learning_rate 1e-4 --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --weight_decay 0.01 --warmup_proportion 0.1 --clip_grad_norm 5.0 \
  --objective 1 \
  --annotations_path $ANNOS --features_path $FEATS \
  --output_dir ${OUTPUT_DIR} \
  --logdir ${LOGGING_DIR} \
  --num_train_epochs 10 \
#  --resume_file ${OUTPUT_DIR}/${MODEL_CONFIG}/pytorch_ckpt_latest.tar

conda deactivate
