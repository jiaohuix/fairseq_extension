#!/usr/bin/env bash
lng=$1
infolder=$2
outfolder=$3
echo "src lng $lng"
scripts=nmt_data_tools/mosesdecoder/scripts/
for sub  in train valid test
do
    sed -r 's/(@@ )|(@@ ?$)//g' $infolder/${sub}.${lng} > $outfolder/${sub}.bert.${lng}.tok
    perl $scripts/tokenizer/detokenizer.perl -l $lng < $infolder/${sub}.bert.${lng}.tok > $outfolder/${sub}.bert.${lng}
    rm $outfolder/${sub}.bert.${lng}.tok
done
