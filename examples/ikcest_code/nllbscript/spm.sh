echo "<src> <tgt> <spm_model>  <infolder> "
SRC=$1
TRG=$2
SPM_MODEL=$3
INDIR=$4

for prefix in train valid test
  do
      for lang in $SRC $TRG
          do

                 cat $INDIR/$prefix.$lang | perl normalize-punctuation.perl  -l en > $INDIR/$prefix.tmp.$lang
                 python scripts/spm_encode.py \
                     --model $SPM_MODEL \
                     --output_format=piece \
                     --inputs=$INDIR/$prefix.tmp.$lang \
                     --outputs=$INDIR/$prefix.spm.$lang
                 rm $INDIR/$prefix.tmp.$lang

          done

  done