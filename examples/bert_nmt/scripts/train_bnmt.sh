data=$1
save=$2
ptm=${3:-"bert-base-german-uncased/"}
#arch=${4:-"bnmt_qb_iwslt_de_en"}
arch=bnmt_qb_iwslt_de_en
epochs=${4:-"75"}
nq=${5:-"8"}
freq=2
mkdir -p $save

CUDA_VISIBLE_DEVICES=0 python train_bnmt.py --fp16  \
    $data    --user-dir bnmt --task bert_nmt \
    --arch $arch --bert-model-name $ptm --n-query $nq  \
    --save-dir $save  --restore-file $save/checkpoint_warmup.pt --reset-optimizer --reset-lr-scheduler  \
    --share-all-embeddings --max-epoch $epochs \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4  --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8192 --update-freq $freq --tensorboard-logdir $save/vislogs/ \
     --validate-interval 1 --no-epoch-checkpoints  --keep-best-checkpoints 10   >> $save/train.log 2>&1
#  2>&1
# --keep-best-checkpoints 3
#  --restore-file $save/checkpoint_warmup.pt # 文档说的是save-dir下面的，但是实践发现需要指定save-dir
# --reset-optimizer --reset-lr-scheduler
