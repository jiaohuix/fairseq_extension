echo "<src> <tgt> <spm_model> <dict> <infolder>  "
SRC=$1
TRG=$2
SPM_MODEL=$3
DICT=$4
INDIR=$5
workers=10
echo  "step1: encode with spm. "
for prefix in train valid test
  do
      for lang in $SRC $TRG
          do
            bash  apply_spm_paral_jieba.sh $workers $lang $INDIR/$prefix $SPM_MODEL $DICT
          done

  done
