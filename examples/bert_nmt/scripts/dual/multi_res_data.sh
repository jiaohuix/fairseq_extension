#!/bin/bash
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


function make_multi_res_data() {
    # params
    local src=$1
    local tgt=$2
    local prefix=$3
    local infolder=$4 # all exp logs
    local outfolder=$5   # all exp logs
    local seed=$6 # all exp logs
#    shift 4 # Remove the first four parameters
#    local exp_args="$@" # Accepts arguments of any length
    srcfile=${infolder}/${prefix}.${src}
    tgtfile=${infolder}/${prefix}.${tgt}

    echo "srcfile $srcfile"
    srcfile_shuf=${srcfile}.shuf
    tgtfile_shuf=${tgtfile}.shuf

    echo "shuffle..."
    shuf --random-source=<(yes $seed) $srcfile >  ${srcfile}.shuf
    shuf --random-source=<(yes $seed) $tgtfile >  ${tgtfile}.shuf

    echo "merge res2"
    srcfile_res2=$outfolder/${prefix}.res2.$src
    tgtfile_res2=$outfolder/${prefix}.res2.$tgt
    awk 'BEGIN{FS="\n";OFS=" [SEP] "} {getline f2 < "'"$srcfile_shuf"'"; print $0,f2}' $srcfile >  $srcfile_res2
    awk 'BEGIN{FS="\n";OFS=" [SEP] "} {getline f2 < "'"$tgtfile_shuf"'"; print $0,f2}' $tgtfile >  $tgtfile_res2

    # merge res1+2
    srcfile_mres=$outfolder/${prefix}.$src
    tgtfile_mres=$outfolder/${prefix}.$tgt
    cat $srcfile  $srcfile_res2 > $srcfile_mres
    cat $tgtfile  $tgtfile_res2 > $tgtfile_mres

    rm $srcfile_shuf $tgtfile_shuf $srcfile_res2 $tgtfile_res2
}


for prefix in train valid test
do
    echo "process $prefix..."
    make_multi_res_data  $src $tgt $prefix $infolder $outfolder $seed
    make_multi_res_data  $src $tgt ${prefix}.bert $infolder $outfolder $seed
    wc $outfolder/${prefix}.$src

done

echo "done"
