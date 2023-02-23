if [ $# -lt 4 ];then
  echo "usage: bash $0 <src> <tgt> <data> <ckpt> <tag>"
  exit
fi
SRC=$1
TGT=$2
DATA=$3
CKPT=$4
tag=${5:-"test"}
fairseq-generate $DATA \
    --source-lang $SRC --target-lang $TGT \
    --path $CKPT \
    --beam 5 --remove-bpe > gen.$tag.txt
grep "BLEU" gen.$tag.txt > RES.$tag.txt
cat RES.$tag.txt