data=$1
ckpt=$2
src=${4:-"de"}
tgt=${4:-"en"}
fairseq-generate  $data -s $src -t $tgt \
    --path $ckpt \
    --beam 5 --batch-size 128 --remove-bpe
