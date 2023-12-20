#!/bin/bash
# Download and prepare the unidirectional data
if [ $# -lt 4 ];then
  echo "usage: bash $0 <src=de> <tgt=en> <infolder> <outfolder> <model>"
  exit
fi

src=${1:-"de"}
tgt=${2:-"en"}
infolder=$3
outfolder=$4
model=${5:-"bert/bert-base-german-uncased"}

TEXT=$infolder
python preprocess.py --source-lang $src --target-lang $tgt \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir $outfolder \
    --joined-dictionary --workers 20 --bert-model-name $model

# dbmdz
# git lfs clone https://huggingface.co/dbmdz/bert-base-german-uncased


