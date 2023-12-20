#!/bin/bash
# Download and prepare the unidirectional data
if [ $# -lt 4 ];then
  echo "usage: bash $0 <src=de> <tgt=en> <infolder> <outfolder> "
  exit
fi

src=${1:-"de"}
tgt=${2:-"en"}
infolder=$3
outfolder=$4

TEXT=$infolder
fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir $outfolder \
    --joined-dictionary --workers 20