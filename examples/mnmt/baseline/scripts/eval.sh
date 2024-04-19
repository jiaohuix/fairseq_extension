src=${1:-"de"}
tgt=${2:-"en"}
data=$3
ckpt=$4
fairseq-generate  $data  -s $src -t $tgt \
    --path $ckpt \
    --beam 5 --batch-size 128 --remove-bpe
