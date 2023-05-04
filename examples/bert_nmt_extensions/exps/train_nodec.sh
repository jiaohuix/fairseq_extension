#!/usr/bin/env bash
echo "<data> <save>"
src=de
tgt=en
bedropout=0.5
ARCH=bert_fused_iwslt_de_en
# data bin
DATAPATH=$1
SAVEDIR=$2
epochs=70
updates=150000
mkdir -p $SAVEDIR
# checkpoint_nmt.pt是初始的nmt启动文件
#if [ ! -f $SAVEDIR/checkpoint_nmt.pt ]
#then
#    cp /your_pretrained_nmt_model $SAVEDIR/checkpoint_nmt.pt
#fi
# 如果没有last，从nmt开始训练；有说明已经是bert-fused从头训练了
if [ ! -f "$SAVEDIR/checkpoint_last.pt" ]
then
warmup="--warmup-from-nmt --reset-lr-scheduler --reset-optimizer --warmup-nmt-file $SAVEDIR/checkpoint_nmt.pt"
else
warmup=""
fi
#exp_args="--decoder-no-bert --bert-gates [0,0,0,0,0,0] --linear-bert-gate-rate 0 "
exp_args="--decoder-no-bert "

python  bert_nmt_extensions/train.py $DATAPATH  --user-dir  bert_nmt_extensions --task bert_nmt --fp16  $exp_args \
    -a $ARCH --optimizer adam --lr 0.0005 -s $src -t $tgt --label-smoothing 0.1 \
    --dropout 0.3 --max-tokens 4096 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy  --max-update $updates  --max-epoch $epochs \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --adam-betas '(0.9,0.98)' --save-dir $SAVEDIR --share-all-embeddings $warmup \
    --encoder-bert-dropout --encoder-bert-dropout-ratio $bedropout \
    --bert-model-name bert-base-german-dbmdz-uncased --tensorboard-logdir $save/vislogs/   --no-epoch-checkpoints >> $SAVEDIR/training.log
