# m2m ft
# 测试中英

#!/bin/bash
# m2m_ft
model_path=/mnt/f/down/m2m100_418M/
train_file=datasets/test.jsonl
val_file=datasets/test.jsonl
root=/mnt/f/workspace/nmt/ckpt
exp=m2m_ft_ikcest_zhen
outdir=${root}/${exp}
logdir=${outdir}/logs
mkdir -p $outdir $logdir

export WANDB_PROJECT=m2m_ft
export WANDB_LOG_MODEL=checkpoint
# m2m ft

#!/bin/bash
# m2m_ft
model_path=/mnt/f/down/m2m100_418M/
train_file=datasets/train2.jsonl
# train_file=datasets/dev2.jsonl

val_file=datasets/dev2.jsonl
root=/mnt/f/workspace/nmt/ckpt
exp=m2m_ft_ikcest_go
outdir=${root}/${exp}
logdir=${outdir}/logs
mkdir -p $outdir $logdir

export WANDB_PROJECT=m2m_ft
# export WANDB_LOG_MODEL=checkpoint


#export WANDB_LOG_MODEL=false


python  scripts/run_translation_ikcest.py \
--model_name_or_path $model_path \
--per_device_train_batch_size 16 --per_device_eval_batch_size 4 --gradient_accumulation_steps 4 \
--train_file $train_file \
--validation_file $val_file \
--output_dir $outdir \
--overwrite_output_dir \
--num_train_epochs 3 \
--do_train \
--max_source_length 512 \
--evaluation_strategy  steps --save_strategy steps  --save_steps 2000 --logging_steps 20 --logging_dir $logdir \
--report_to wandb --run_name $exp  --fp16  > $outdir/train.log

#  > $outdir/train.log  --fp16  --learning_rate 1e-4  --lr_scheduler_type cosine

# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \ --stage sft \ --do_train \ --dataset tianma_dataset \ --finetuning_type lora 
# \ --output_dir retrain/8 \ --overwrite_cache \ --per_device_train_batch_size 8 \ --per_device_eval_batch_size 8 
# \ --gradient_accumulation_steps 2 \ --lr_scheduler_type cosine \ --evaluation_strategy steps \ --save_strategy steps 
# \ --logging_steps 20 \ --save_steps 20 \ --learning_rate 5e-4 \ --num_train_epochs 8.0 \ --dev_ratio 0.04
#  \ --lora_rank 4 \ --plot_loss \ --fp16CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \ --stage sft 
#  \ --do_train \ --dataset tianma_dataset \ --finetuning_type lora \ --output_dir retrain/8 \
#   --overwrite_cache \ --per_device_train_batch_size 8 \ --per_device_eval_batch_size 8 \ --gradient_accumulation_steps 2 
#   \ --lr_scheduler_type cosine \ --evaluation_strategy steps \ --save_strategy steps \ --logging_steps 20 
#   \ --save_steps 20 \ --learning_rate 5e-4 \ --num_train_epochs 8.0 \ --dev_ratio 0.04
#  \ --lora_rank 4 \ --plot_loss \ --fp16
model_path=/mnt/f/down/m2m100_418M/
outdir=/mnt/f/workspace/nmt/ckpt/m2m_ft_ikcest_zhen/checkpoint-500/
python scripts/predict.py -i  test_zhen.jsonl  -o pred_zhen_pt.jsonl -m  $model_path -n m2m_pt_zhen -b 8 -l 400
python scripts/eval.py pred_zhen_pt.jsonl  score_m2m_pt_zhen.csv
cat score_m2m_pt_zhen.csv


python scripts/predict.py -i  test_zhen.jsonl  -o pred_zhen_ft.jsonl -m  $outdir -n m2m_ft_zhen -b 8 -l 400
python scripts/eval.py pred_zhen_ft.jsonl  score_m2m_ft_zhen.csv
cat score_m2m_ft_zhen.csv