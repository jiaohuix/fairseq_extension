if [ $# -lt 4 ];then
  echo "usage: bash $0 <src> <tgt> <infolder> <outfolder> <joined(n/y)>(opt)"
  exit
fi
SRC=$1
TGT=$2
infolder=$3
outfolder=$4
joint=${5:-"n"}
dict_cmds="--srcdict $infolder/dict.${SRC}.txt --tgtdict $infolder/dict.${TGT}.txt"
if [ "$joint"x == "y"x ];then
  echo "build joined dictionary"
  dict_cmds="--joined-dictionary"
fi

fairseq-preprocess \
    --source-lang $SRC --target-lang $TGT $dict_cmds \
    --trainpref $infolder/train --validpref $infolder/valid --testpref $infolder/test \
    --destdir $outfolder --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20