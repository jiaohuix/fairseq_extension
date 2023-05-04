#!/bin/bash
if [ $# -lt 2 ];then
  echo "usage: bash $0 <infolder> <outfolder> <seed=1>(opt)"
  exit
fi

src=src
tgt=tgt
infolder=$1
outfolder=$2
seed=${3:-"1"}

if [ ! -d $outfolder ];then
  mkdir -p $outfolder
fi
# train.src/tgt
#seed=1
echo "shuffle..."
shuf --random-source=<(yes $seed) $infolder/train.$src >  $outfolder/train.shuf.$src
shuf --random-source=<(yes $seed) $infolder/train.$tgt >  $outfolder/train.shuf.$tgt
echo "merge res2"
# merge res2
#paste -d"[SEP]" $infolder/train.$src  $outfolder/train.shuf.$src >  $outfolder/train.res2.$src
#paste -d"[SEP]" $infolder/train.$tgt  $outfolder/train.shuf.$tgt >  $outfolder/train.res2.$tgt
#paste -d abc file1 /dev/null /dev/null file2
srcfile2=$outfolder/train.shuf.$src
tgtfile2=$outfolder/train.shuf.$tgt
awk 'BEGIN{FS="\n";OFS=" [SEP] "} {getline f2 < "'"$srcfile2"'"; print $0,f2}' $infolder/train.$src >  $outfolder/train.res2.$src
awk 'BEGIN{FS="\n";OFS=" [SEP] "} {getline f2 < "'"$tgtfile2"'"; print $0,f2}' $infolder/train.$tgt >  $outfolder/train.res2.$tgt

# merge res1+2
cat $infolder/train.$src  $infolder/train.$src $outfolder/train.res2.$src > $outfolder/train.$src
cat $infolder/train.$tgt  $infolder/train.$tgt $outfolder/train.res2.$tgt > $outfolder/train.$tgt

paste $outfolder/train.$src $outfolder/train.$tgt > $outfolder/train
cp  $infolder/valid $outfolder
cp  $infolder/test  $outfolder



