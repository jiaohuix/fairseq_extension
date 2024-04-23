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

lang_token="LANG_TOK_"`echo "${TGT} " | tr '[a-z]' '[A-Z]'`
echo "lang_token: $lang_token"
fairseq-generate $DATA --user-dir mcolt \
    --source-lang src --target-lang tgt \
    --path $CKPT \
    --task translation_w_langtok \
    --lang-prefix-tok ${lang_token} \
    --beam 5 --remove-bpe