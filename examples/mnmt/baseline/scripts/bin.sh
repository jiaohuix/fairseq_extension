#!/bin/bash
# fairseq binarize scripts

if [ $# -lt 2 ];then
  echo "usage: bash $0 <infolder> <outfolder> <src=de> <tgt=en>  <joint_dict=0>(0/1) <dict_path=0>(0 or /path/to/dict.lang.txt)"
  exit
fi

infolder=$1
outfolder=$2
src=${3:-"de"}
tgt=${4:-"en"}
joint_dict=${5:-"0"}
dict_path=${6:-"0"}

args=""

if [[ $joint_dict == "1" ]]; then
  args="--joined-dictionary"
fi

if [[ -e $dict_path ]]; then
  args="$args --srcdict $dict_path"
fi

echo "extra args: $args"

TEXT=$infolder
fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir $outfolder \
    --workers 20 $args



