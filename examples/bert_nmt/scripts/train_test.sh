src=$1
tgt=$2
data=$3
save=$4
arch=transformer
epochs=70
mkdir -p $save
CUDA_VISIBLE_DEVICES=0 fairseq-train --fp16 -s $src -t $tgt \
    $data  --save-dir $save --max-epoch $epochs \
    --arch $arch --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4  --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --tensorboard-logdir $save/vislogs/ \
    --validate-interval 1 --no-epoch-checkpoints  --keep-best-checkpoints 10  >> $save/train.log


bash scripts/eval.sh $src $tgt $data $save/checkpoint_best.pt > $save/gen_best_${src}${tgt}.txt
bash scripts/avg10.sh $save/ 10
bash scripts/eval.sh $src $tgt $data $save/avg10.pt > $save/gen_avg10_${src}${tgt}.txt
