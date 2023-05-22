# span
DATA=$1
SAVE=$2
tokens=${3:-"2048"}
updates=${4:-"50000"}
fairseq-train --task span_masked_lm \
  $DATA  --save-dir $SAVE \
  --arch transformer_iwslt_de_en --share-all-embeddings \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens $tokens --update-freq 16 \
  --fp16 --max-update $updates --no-epoch-checkpoints \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --tensorboard-logdir $SAVE/vislogs/   >> $SAVE/train.log
