if [ $# -lt 4 ];then
  echo "usage: bash $0 <src> <tgt> <data> <ckpt>"
  exit
fi
SRC=$1
TGT=$2
DATA=$3
SAVE=$4
tokens=4096
epoch=50
optim_cmds="--optimizer adam  --lr 5e-4  --weight-decay 0.0001"

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    $DATA --save-dir $SAVE --tensorboard-logdir $SAVE/visual --no-epoch-checkpoints \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed  \
    $optim_cmds --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt \
    --dropout 0.3 --warmup-updates 4000 --warmup-init-lr 1e-07  \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens $tokens --max-epoch $epoch \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric  --fp16 --update-freq 1 > $SAVE/train.log

