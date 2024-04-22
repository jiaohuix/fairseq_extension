echo "bash evaluate.sh <src> <tgt> <data> <ckpt>"
SRC=$1
TGT=$2
DATA=$3
CKPT=$4
fairseq-generate $DATA \
    --source-lang $SRC --target-lang $TGT \
    --path $CKPT \
    --beam 5 --remove-bpe --gen-subset valid


