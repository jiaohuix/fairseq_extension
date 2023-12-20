data=$1
save=$2
ptm=$3
cuda=${4:-'0'}
epochs=20
arch=transformer_iwslt_de_en
mkdir -p $save
CUDA_VISIBLE_DEVICES=$cuda fairseq-train --fp16  \
    $data  --save-dir $save --max-epoch $epochs \
    --arch $arch --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4  --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler --restore-file $ptm \
    --max-tokens 4096 --tensorboard-logdir $save/vislogs/ \
    --validate-interval 1 --no-epoch-checkpoints  --keep-best-checkpoints 3 >> $save/train.log