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

echo "dict_path: $dict_path"
args=""

if [[ $joint_dict == "1" ]]; then
  args="--joined-dictionary"
fi

# if [[ -e $dict_path ]]; then
#   args="$args --srcdict $dict_path"
# fi

if [[ -f $dict_path ]]; then  # 更改为 -f 表示检查是否是文件
  args="$args --srcdict $dict_path"
elif [[ $dict_path != "0" ]]; then  # 如果路径不是 "0" 且不是文件，可能是无效路径
  echo "Invalid dictionary path: $dict_path"
fi

echo "extra args: $args"

TEXT=$infolder
fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir $outfolder \
    --workers 20 $args



