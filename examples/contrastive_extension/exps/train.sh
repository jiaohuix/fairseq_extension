DATA=$1
SAVE=$2
epochs=40
updates=100000
shift 2 # Remove the first four parameters
exp_args=$@ # Accepts arguments of any length
echo "exp_args: $exp_args"

mkdir -p $SAVE
# --criterion label_smoothed_cross_entropy
python fairseq_cli/train.py   $DATA \
    --arch transformer_iwslt_de_en --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 --max-epoch $epochs --max-update $updates \
    $exp_args --label-smoothing 0.1 \
    --max-tokens 4096 --fp16 --no-epoch-checkpoints --save-dir $SAVE \
    --tensorboard-logdir $SAVE/vislogs/   --no-epoch-checkpoints >> $SAVE/training.log
#    --eval-bleu \
#    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#    --eval-bleu-detok moses \
#    --eval-bleu-remove-bpe \
#    --eval-bleu-print-samples \
#    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
