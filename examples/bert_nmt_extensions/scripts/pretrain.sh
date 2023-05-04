data=$1
save=$2
mkdir -p $save
CUDA_VISIBLE_DEVICES=0 python train.py --fp16  \
    $data  --save-dir $save --max-epoch 50 \
    --arch transformer_iwslt_de_en --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4  --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --tensorboard-logdir $save/vislogs/ --no-epoch-checkpoints   >> $save/train.log
