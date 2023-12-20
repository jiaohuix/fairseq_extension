#!/bin/bash
# Download and prepare the unidirectional data
if [ $# -lt 4 ];then
  echo "usage: bash $0 <src=de> <tgt=en> <infolder> <outfolder> <seed=42>(opt)"
  exit
fi

src=${1:-"de"}
tgt=${2:-"en"}
infolder=$3
outfolder=$4
seed=${5:-"42"}

if [ ! -d $outfolder ];then
  mkdir -p $outfolder
fi

# Prepare the bidirectional data
for prefix in train valid test
do
    echo "process $prefix..."
    srcfile=${infolder}/${prefix}.${src}
    tgtfile=${infolder}/${prefix}.${tgt}
    srcfile_dual=${outfolder}/${prefix}.src
    tgtfile_dual=${outfolder}/${prefix}.tgt

    cat $srcfile $tgtfile > $srcfile_dual
    cat $tgtfile $srcfile  > $tgtfile_dual
    wc $srcfile_dual
done


for prefix in train.bert valid.bert test.bert
do
    echo "process $prefix..."
    srcfile=${infolder}/${prefix}.${src}
    tgtfile=${infolder}/${prefix}.${tgt}
    srcfile_dual=${outfolder}/${prefix}.src
    tgtfile_dual=${outfolder}/${prefix}.tgt

    cat $srcfile $tgtfile > $srcfile_dual
    cat $tgtfile $srcfile  > $tgtfile_dual
    wc $srcfile_dual
done
echo "done"