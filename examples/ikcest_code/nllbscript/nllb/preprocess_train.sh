#echo "<src> <tgt> <spm_model> <dict> <infolder> <outfolder>  <prefix> "
SRC=$1
TRG=$2
SPM_MODEL=$3
DICT=$4
INDIR=$5
OUTDIR=$6
prefix=$7

echo  "step1: encode with spm. "
for lang in $SRC $TRG
    do
        python scripts/spm_encode.py \
               --model $SPM_MODEL \
               --output_format=piece \
               --inputs=$INDIR/$prefix.$lang \
               --outputs=$INDIR/$prefix.spm.$lang
    done
