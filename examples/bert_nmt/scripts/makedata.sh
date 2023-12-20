#!/usr/bin/env bash
lng=$1
folder=$2
echo "src lng $lng"
scripts=nmt_data_tools/mosesdecoder/scripts/
for sub  in train valid test
do
    sed -r 's/(@@ )|(@@ ?$)//g' $folder/${sub}.${lng} > $folder/${sub}.bert.${lng}.tok
    perl $scripts/tokenizer/detokenizer.perl -l $lng < $folder/${sub}.bert.${lng}.tok > $folder/${sub}.bert.${lng}
    rm $folder/${sub}.bert.${lng}.tok
done