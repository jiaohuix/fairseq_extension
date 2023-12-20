data=$1
save=$2
ptm=$3
rank=${4:-"8"}
alpha=${5:-"16"}

epochs=60
mkdir -p $save
arch=bnmt_qb_iwslt_de_en
# 关于lora：默认放到save-dir下面的loras目录，然后每个文件夹对应save-dir的ckpt.pt：
# /save-dir
#   - ckpt_best.pt
#   - ckpt_last.pt
#   /loras
#      - /ckpt_best
#      - /ckpt_last

CUDA_VISIBLE_DEVICES=0 python train_bnmt.py --fp16  \
    $data    --user-dir bnmt --task bert_nmt --use-lora --lora-rank $rank --lora-alpha $alpha \
    --arch $arch --bert-model-name $ptm --n-query 8  \
    --save-dir $save  --restore-file $save/checkpoint_warmup.pt  \
    --share-all-embeddings --max-epoch $epochs \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4  --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8192 --tensorboard-logdir $save/vislogs/ \
    --validate-interval 1 --no-epoch-checkpoints  --keep-best-checkpoints 10  >> $save/train.log

#
#  --restore-file $save/checkpoint_warmup.pt # 文档说的是save-dir下面的，但是实践发现需要指定save-dir
# --reset-optimizer --reset-lr-scheduler