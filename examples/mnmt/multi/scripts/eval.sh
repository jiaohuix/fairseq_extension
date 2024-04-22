#!/bin/bash
if [ $# -lt 4 ];then
  echo "usage: bash $0 <src=de> <tgt=en>  <data> <ckpt(.pt)>"
  exit
fi

src=${1:-"de"}
tgt=${2:-"en"}
data=$3
ckpt=$4

# 语言列表
langs=("en" "fr" "de" "ru" "nl" "it" "ro" "tgt")

# 检查tgt_lang是否在语言列表中
is_in_list=false
for lang in "${langs[@]}"; do
    if [ "$tgt" = "$lang" ]; then
        is_in_list=true
        break
    fi
done
if [ "$is_in_list" = true ]; then
  args="$args --tokenizer moses"
fi

echo "args: $args"



fairseq-generate  $data  -s src -t tgt \
    --path $ckpt \
    --beam 5 --batch-size 128 --remove-bpe  --sacrebleu --lenpen 1 $args

# --tokenizer moses
