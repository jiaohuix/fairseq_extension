#!/bin/bash
if [ $# -lt 4 ];then
  echo "usage: bash $0 <src> <tgt>  <data> <save_dir> <wandb_project=ikcest22> <pretrained_ckpt=ckpt/6.pt> <epoch=50>"
  exit
fi
SRC=$1
TGT=$2
DATA=$3
SAVE=$4
wandb_project=${5:-"ikcest22"}
CKPT=${6:-"ckpt/6e6d_no_mono.pt"}
epoch=${7:-"10"}

mkdir -p $SAVE

export WANDB_NAME=mRASP2_ft_${SRC}-${TGT}


fairseq-train \
    $DATA \
    --user-dir mcolt \
    -s src -t tgt \
    --dropout 0.3 --weight-decay 0.0001 \
    --arch transformer_vaswani_wmt_en_de_big \
    --share-all-embeddings  \
    --max-tokens 4096 --update-freq 2  --max-epoch $epoch --max-update 10000  \
    --lr 3e-5 --lr-scheduler inverse_sqrt --warmup-updates 4000 --fp16 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --save-dir $SAVE \
    --encoder-learned-pos  --decoder-learned-pos \
    --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler --restore-file $CKPT \
    --no-epoch-checkpoints  --validate-interval 1 --keep-best-checkpoints 3 \
    --wandb-project $wandb_project --tensorboard-logdir $SAVE/vislogs \
    --patience 20 --seed 42 >> $SAVE/train.log 2>&1


