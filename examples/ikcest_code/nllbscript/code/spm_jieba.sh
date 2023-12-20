echo "<src> <tgt> <spm_model> <infolder>  "
SRC=$1
TRG=$2
SPM_MODEL=$3
INDIR=$4

echo  "step1: encode with spm. "
for prefix in train valid test
  do
      for lang in $SRC $TRG
          do
                 use_jieba="--use-jieba"
                 if [ "$lang"x == "ar"x ];then
                     use_jieba=""
                     echo "use_jieba= ${use_jieba}"
                 fi
                 python spm_encode.py \
                     --model $SPM_MODEL \
                     --output_format=piece \
                     --inputs=$INDIR/$prefix.$lang \
                     --outputs=$INDIR/$prefix.spm.$lang $use_jieba
          done

  done