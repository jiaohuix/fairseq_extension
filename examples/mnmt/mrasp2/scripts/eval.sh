echo "bash generate.sh <src> <tgt> <data> <ckpt>"
SRC=$1
TGT=$2
DATA=$3
CKPT=$4

lang_token="LANG_TOK_"`echo "${TGT} " | tr '[a-z]' '[A-Z]'`
fairseq-generate $DATA --user-dir mcolt \
    --source-lang $SRC --target-lang $TGT \
    --path $CKPT \
    --task translation_w_langtok \
    --lang-prefix-tok ${lang_token} \
    --beam 5 --remove-bpe
