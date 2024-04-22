echo "bash bin.sh <src> <tgt> <infolder> <outfolder>"
SRC=$1
TGT=$2
TEXT=$3
SAVR=$4
fairseq-preprocess \
    --source-lang $SRC --target-lang $TGT \
    --srcdict $TEXT/bpe_vocab --tgtdict $TEXT/bpe_vocab\
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir $SAVR --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20
