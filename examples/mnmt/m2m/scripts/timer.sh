#!/bin/bash

# 设置需要等待的小时数
hours=${1:-5}

# 将小时数转换为秒数
seconds=$((hours * 1))

# 等待指定的秒数
sleep $seconds

# 执行命令
echo $seconds


## 1 eval script 
# ckpt_dir=/mnt/f/workspace/nmt/ckpt/ikcest_mft_lora16_full
# ckpt_dir=/mnt/f/workspace/nmt/ckpt/ikcest_zhen_lora8_full_whead2
ckpt_dir=/mnt/f/workspace/nmt/ckpt/ikcest_mft_lora8_full_whead/
eval_dir=${ckpt_dir}/eval
mkdir -p $eval_dir
for ckpt in  30000 60000 90000
# for ckpt in  5000 10000 15000 
do
    python scripts/predict_lora.py -i  datasets/test2.jsonl  -o ${eval_dir}/pred_${ckpt}.jsonl -m ${ckpt_dir}/checkpoint-${ckpt}  -n step_${ckpt} -b 8 -l 400 --use_lora
    # python scripts/predict_lora.py -i  test_zhen2.jsonl  -o ${eval_dir}/pred_${ckpt}.jsonl -m ${ckpt_dir}/checkpoint-${ckpt}  -n step_${ckpt} -b 8 -l 400 --use_lora
    python scripts/eval.py  ${eval_dir}/pred_${ckpt}.jsonl   ${eval_dir}/pred_${ckpt}.csv 
done


# # 2 train script
# model_path=/mnt/f/down/m2m100_418M/
# train_file=datasets/train2.jsonl
# # train_file=datasets/dev2.jsonl

# val_file=datasets/dev2.jsonl
# root=/mnt/f/workspace/nmt/ckpt
# # 数据集-语言方向-微调方式 
# # exp=ikcest_mft_lora16
# # exp=ikcest_zhen_lora16_lr

# # exp=ikcest_mft_lora16_cosine
# exp=ikcest_mft_dora16_full

# outdir=${root}/${exp}
# logdir=${outdir}/logs
# mkdir -p $outdir $logdir

# export WANDB_PROJECT=m2m_ft

# python scripts/run_translation_lora.py \
# --model_name_or_path $model_path \
# --per_device_train_batch_size 32 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 \
# --learning_rate 1e-4  --lr_scheduler_type cosine --warmup_steps 500 \
# --train_file $train_file \
# --validation_file $val_file \
# --output_dir $outdir \
# --overwrite_output_dir --fp16  \
# --num_train_epochs 3 \
# --do_train \
# --max_source_length 128 \
# --save_steps 5000  --logging_steps 20 --report_to wandb --run_name $exp  \
# --lora_training --lora_rank 16  --lora_alpha 32 \
# --lora_target q_proj,k_proj,fc1,fc2,embed_tokens,out_proj --use_dora > $outdir/train.log


# # eval

# ckpt_dir=/mnt/f/workspace/nmt/ckpt/ikcest_mft_dora16_full
# eval_dir=${ckpt_dir}/eval
# mkdir -p $eval_dir
# for ckpt in  30000 60000 90000
# do
#     python scripts/predict_lora.py -i  datasets/test2.jsonl  -o ${eval_dir}/pred_${ckpt}.jsonl -m ${ckpt_dir}/checkpoint-${ckpt}  -n step_${ckpt} -b 8 -l 400 --use_lora
#     python scripts/eval.py  ${eval_dir}/pred_${ckpt}.jsonl   ${eval_dir}/pred_${ckpt}.csv 
# done
