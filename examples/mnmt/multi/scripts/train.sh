#!/bin/bash
if [ $# -lt 4 ];then
  echo "usage: bash $0 <src> <tgt>  <data> <save_dir>"
  exit
fi

src=$1
tgt=$2
data=$3
save=$4
arch=transformer
lang_pair=${src}-${tgt}
wandb_project=${5:-"ikcest22"}
wandb_run_name=mTransformer

export WANDB_NAME=$wandb_run_name


epochs=150
mkdir -p $save
CUDA_VISIBLE_DEVICES=0 fairseq-train --fp16 -s $src -t $tgt \
    $data  --save-dir $save --max-epoch $epochs \
    --arch $arch --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4  --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8192 --update-freq 1 --max-update 100000 \
    --validate-interval 1 --no-epoch-checkpoints  \
    --wandb-project $wandb_project \
    --tensorboard-logdir $save/vislogs  >> $save/train.log 2>&1



# 
# --keep-best-checkpoints 1  --wandb-run-name $wandb_run_name  