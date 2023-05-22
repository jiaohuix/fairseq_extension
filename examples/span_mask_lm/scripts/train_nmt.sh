echo "<data> <save> <ptm>(opt) <src=de>(opt) <tgt=en>(opt)  <tokens=4096>(opt) <updates=60000>(opt)"
DATA=$1
SAVE=$2
PTM=${3:-"checkpoint_last.pt"}
src=${4:-"de"}
tgt=${5:-"en"}
tokens=${6:-"4096"}
updates=${7:-"60000"}
mkdir -p $SAVE
if [ ! -f "$PTM" ]
then
finetune_args="--reset-lr-scheduler --reset-optimizer --reset-dataloader --restore-file $PTM"
else
finetune_args=""
fi


fairseq-train \
    data-bin/iwslt14.tokenized.de-en --save-dir checkpoints/nmt_ft -s $src -t $tgt $finetune_args \
    --arch transformer_iwslt_de_en --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens $tokens --no-epoch-checkpoints --max-update $updates \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --fp16 \
    --tensorboard-logdir $SAVE/vislogs/   >> $SAVE/train.log
