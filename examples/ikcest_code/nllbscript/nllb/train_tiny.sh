echo "bash train.sh <src> <tgt> <data> <save>"
SRC=$1
TGT=$2
DATA=$3
SAVE=$4
epoch=50

SRC_Code=zho_Hans
TGT_Code=arb_Arab
if [ "$SRC"x == "ar"x ];then
    tmp=$SRC_Code
    SRC_Code=$TGT_Code
    TGT_Code=$tmp
fi

fairseq-train \
    $DATA \
    --user-dir extension  -s $SRC -t $TGT \
    --task nllb_translation --src-lang-code $SRC_Code --tgt-lang-code  $TGT_Code \
    --dropout 0.1 --weight-decay 0.0001 \
    --arch transformer_12_12 \
    --share-all-embeddings  \
    --max-tokens 2048 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --save-dir $SAVE \
    --reset-optimizer --reset-dataloader --fp16 --update-freq 4 --max-epoch $epoch \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric  --no-epoch-checkpoints
