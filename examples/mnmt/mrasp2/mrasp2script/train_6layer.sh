echo "bash train_tiny.sh <src> <tgt> <data> <save>"
SRC=$1
TGT=$2
DATA=$3
SAVE=$4

fairseq-train \
    $DATA \
    --user-dir mcolt \
    -s $SRC -t $TGT \
    --dropout 0.2 --weight-decay 0.0001\
    --arch transformer_vaswani_wmt_en_de_big \
    --share-all-embeddings  \
    --max-tokens 4096 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --save-dir $SAVE \
    --encoder-learned-pos  --decoder-learned-pos \
    --reset-optimizer --reset-dataloader --fp16 --update-freq 4 --max-epoch 50 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric  --no-epoch-checkpoints
