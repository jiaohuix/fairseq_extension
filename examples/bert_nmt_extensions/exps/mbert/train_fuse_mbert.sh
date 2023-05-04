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
ptm=bert-base-multilingual-cased
mkdir -p $SAVEDIR
if [ ! -f "$SAVEDIR/checkpoint_last.pt" ]
then
warmup="--warmup-from-nmt --reset-lr-scheduler --reset-optimizer --warmup-nmt-file $SAVEDIR/checkpoint_nmt.pt"
else
warmup=""
fi

python  bert_nmt_extensions/train.py $DATAPATH  --user-dir  bert_nmt_extensions --task bert_nmt --fp16  $exp_args \
    -a $ARCH --optimizer adam --lr 0.0005 -s $src -t $tgt --label-smoothing 0.1 \
    --dropout 0.3 --max-tokens 4096 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy  --max-update $updates  --max-epoch $epochs \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --adam-betas '(0.9,0.98)' --save-dir $SAVEDIR --share-all-embeddings $warmup \
    --encoder-bert-dropout --encoder-bert-dropout-ratio $bedropout \
    --bert-model-name $ptm --tensorboard-logdir $save/vislogs/   --no-epoch-checkpoints >> $SAVEDIR/training.log