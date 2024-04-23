#!/bin/bash
model_path=/mnt/f/down/m2m100_418M/
#train_file=datasets/train2.jsonl
#val_file=datasets/dev2.jsonl
dataset_name=datasets/ikcest2022
root=/mnt/f/workspace/nmt/ckpt
exp=ikcest_mft_v2
outdir=${root}/${exp}
logdir=${outdir}/logs
mkdir -p $outdir $logdir

export WANDB_PROJECT=m2m_ft
# export WANDB_LOG_MODEL=checkpoint
# --lang_pairs zh-fr,zh-th


python scripts/run_translation.py \
  --model_name_or_path $model_path \
  --per_device_train_batch_size 16 --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2  --warmup_steps 4000 \
  --learning_rate 1e-4  --lr_scheduler_type inverse_sqrt  \
  --dataset_name $dataset_name \
  --output_dir $outdir \
  --overwrite_output_dir --fp16  \
  --num_train_epochs 3 \
  --do_train \
  --max_source_length 128 \
  --save_steps 10000  --logging_steps 20 --report_to wandb --run_name $exp



#  > $outdir/train.log
# python predict.py -d datasets/ikcest2022 -o ikcest_m2m.jsonl -m facebook/m2m100 -n m2m -b 8 -l 400 -lp zh-fr