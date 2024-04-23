# m2m ft

#!/bin/bash
# m2m_ft
model_path=/mnt/f/down/m2m100_418M/
train_file=datasets/train2.jsonl
# train_file=datasets/dev2.jsonl

val_file=datasets/dev2.jsonl
root=/mnt/f/workspace/nmt/ckpt
# 数据集-语言方向-微调方式 
# exp=ikcest_mft_lora16
# exp=ikcest_zhen_lora16_lr

# exp=ikcest_mft_lora16_cosine
# exp=ikcest_mft_lora8_full_whead
exp=ikcest_zhen_lora8_full_whead
exp=ikcest_zhen_lora8_full_whead2
exp=ikcest_mft_lora8_full_whead


outdir=${root}/${exp}
logdir=${outdir}/logs
mkdir -p $outdir $logdir

export WANDB_PROJECT=m2m_ft
# export WANDB_LOG_MODEL=checkpoint

# lora16_v2: --learning_rate 1e-4  --lr_scheduler_type inverse_sqrt --warmup_steps 4000
# cosine:  -learning_rate 1e-4  --lr_scheduler_type cosine --warmup_steps 500
# 修改qk矩阵 https://www.reddit.com/r/LocalLLaMA/comments/15l5auj/why_cant_lora_fine_tune_add_knowledge/?rdt=32915
# 他们说qk 只会修改关注的位置，得用于所有的层，才能起到作用。
# ikcest_mft_lora16_full: -learning_rate 1e-4  --lr_scheduler_type cosine --warmup_steps 500
#         target_modules = ["q_proj", "k_proj","fc1","fc2","embed_tokens","out_proj"]
# ?如何全量微调embedding： modules_to_save = ["embed_tokens"]

#ikcest_zhen_lora8_full_whead:  全量微调embed和lm_head，参考chinese-llama, warmup修改为5%  17-30.4。但是提升很大，应该不是加了个k_proj达到的；应该是cosine的scheduler加上5%的warmup。
# ikcest_zhen_lora8_full_whead2: 上一个写错了，写了llm_head，而不是lm_head

python scripts/run_translation_lora.py \
--model_name_or_path $model_path \
--per_device_train_batch_size 32 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1  --learning_rate 1e-4  --lr_scheduler_type cosine --warmup_ratio 0.05 \
--train_file $train_file \
--validation_file $val_file \
--output_dir $outdir \
--overwrite_output_dir --fp16  \
--num_train_epochs 3 \
--do_train \
--max_source_length 128 \
--save_steps 10000  --logging_steps 20 --report_to wandb --run_name $exp  \
--lora_training --lora_rank 8  --lora_alpha 16 \
--lora_target q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
--modules_to_save embed_tokens,lm_head  > $outdir/train.log

# --lora_target 'q_proj,k_proj,fc1,fc2,embed_tokens,out_proj' --use_dora


# deepspeed  scripts/run_translation_ikcest.py \
# --deepspeed deepspeed/ds_z3.json \
# --model_name_or_path $model_path \
# --per_device_train_batch_size 24 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 \
# --train_file $train_file \
# --validation_file $val_file \
# --output_dir $outdir \
# --overwrite_output_dir --fp16  \
# --num_train_epochs 3 \
# --do_train \
# --max_source_length 128 \
# --save_steps 200  --logging_steps 20

# 存在问题 wandb？ logging step？ save stra？
# --evaluation_strategy  steps --save_strategy steps  --save_steps 2000 --logging_steps 20 --logging_dir $logdir \
# --report_to wandb --run_name $exp  --fp16  > $outdir/train.log



#  2000 --logging_steps 20 --logging_dir $logdir \
# --report_to tensorboard --run_name $exp  --fp16 
#  > $outdir/train.log
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