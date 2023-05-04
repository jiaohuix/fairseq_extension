#!/bin/bash
if [ $# -lt 2 ];then
  echo "usage: bash $0 <indir> <outdir> <ptm>(opt) <bsz>(opt)"
  echo "ptm: [xlm-roberta-base,bert-base-multilingual-cased,jhu-clsp/bibert-ende,uklfr/gottbert-base...]"
  exit
fi

indir=$1
outdir=$2
ptm=${3:-"xlm-roberta-base"}
bsz=${4:-"8"}
epochs=1

python run_mlm.py \
    --model_name_or_path $ptm \
    --train_file $indir/train.txt \
    --validation_file $indir/valid.txt \
    --per_device_train_batch_size $bsz \
    --per_device_eval_batch_size $bsz \
    --do_train \
    --do_eval \
    --output_dir $outdir/$ptm --num_train_epochs=$epochs  \
    --fp16 --gradient_accumulation_steps  2  --save_steps 2000 --logging_dir $outdir/visual  --overwrite_output_dir --line_by_line

