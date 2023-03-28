DATA=./download_prepare/data/
SAVE=./models/one-way/
tokens=4096
freq=8
mkdir -p $SAVE

fairseq-train ${DATA}de-en-databin/ --arch transformer_iwslt_de_en --ddp-backend no_c10d --optimizer adam --adam-betas '(0.9, 0.98)' \
--clip-norm 1.0 --lr 0.0004 --lr-scheduler inverse_sqrt --warmup-updates 1000 --warmup-init-lr 1e-07 --dropout 0.3 \
--weight-decay 0.00002 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens $tokens --update-freq $freq \
--attention-dropout 0.1 --activation-dropout 0.1 --max-epoch 75 --save-dir ${SAVE}  --encoder-embed-dim 768 --decoder-embed-dim 768 \
--no-epoch-checkpoints --save-interval 5 --pretrained_model jhu-clsp/bibert-ende --use_drop_embedding 1  --tensorboard-logdir  $SAVE/visuallog > $SAVE/train.log
