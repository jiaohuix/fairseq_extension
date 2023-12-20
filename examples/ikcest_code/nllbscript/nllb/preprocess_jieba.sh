echo "<src> <tgt> <spm_model> <dict> <infolder> <outfolder>  "
SRC=$1
TRG=$2
SPM_MODEL=$3
DICT=$4
INDIR=$5
OUTDIR=$6

echo  "step1: encode with spm. "
for prefix in train valid test
  do
      for lang in $SRC $TRG
          do

                 use_jieba="--use-jieba"
                 dict_path="--dict-path $DICT"
                 if [ "$lang"x == "ar"x ];then
                     use_jieba=""
                     dict_path=""
                     echo "use_jieba= ${use_jieba}"
                 fi
                 python scripts/spm_encode.py \
                     --model $SPM_MODEL \
                     --output_format=piece \
                     --inputs=$INDIR/$prefix.$lang \
                     --outputs=$INDIR/$prefix.spm.$lang $use_jieba $dict_path
          done

  done


echo "step2: binarize spm. "
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

echo "all done!"