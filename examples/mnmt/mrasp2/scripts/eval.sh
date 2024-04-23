#!/bin/bash
# fairseq binarize scripts

if [ $# -lt 2 ];then
  echo "usage: bash $0 <src> <tgt> <data> <ckpt>"
  exit
fi
SRC=$1
TGT=$2
DATA=$3
CKPT=$4

# moses语言列表
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


lang_token="LANG_TOK_"`echo "${TGT} " | tr '[a-z]' '[A-Z]'`
echo "lang_token: $lang_token"
fairseq-generate $DATA --user-dir mcolt \
    --source-lang src --target-lang tgt \
    --path $CKPT \
    --task translation_w_langtok \
    --lang-prefix-tok ${lang_token} \
    --beam 5 --remove-bpe --gen-subset test  --sacrebleu --lenpen 1 $args
