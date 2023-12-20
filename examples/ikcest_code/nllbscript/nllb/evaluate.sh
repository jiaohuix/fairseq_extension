echo "bash eval.sh <src> <tgt> <data> <ckpt>"
SRC=$1
TGT=$2
DATA=$3
CKPT=$4

SRC_Code=zho_Hans
TGT_Code=arb_Arab
if [ "$SRC"x == "ar"x ];then
    tmp=$SRC_Code
    SRC_Code=$TGT_Code
    TGT_Code=$tmp
fi


fairseq-generate $DATA \
    --source-lang $SRC --target-lang $TGT \
    --user-dir extension  --task nllb_translation \
    --src-lang-code $SRC_Code --tgt-lang-code  $TGT_Code \
    --path $CKPT \
    --beam 5 --remove-bpe --post-process  sentencepiece  --gen-subset valid


