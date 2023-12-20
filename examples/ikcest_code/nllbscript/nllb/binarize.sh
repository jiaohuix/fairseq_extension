echo "<src> <tgt> <dict> <infolder> <outfolder>"
SRC=$1
TRG=$2
DICT=$3
INDIR=$4
OUTDIR=$5
fairseq-preprocess \
  -s ${SRC} \
  -t ${TRG} \
  --trainpref $INDIR/train.spm \
  --validpref $INDIR/valid.spm \
  --testpref $INDIR/test.spm \
  --destdir $OUTDIR \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict $DICT \
  --tgtdict $DICT \
  --workers 20
