src=${1:-"de"}
tgt=${2:-"en"}
data=$3
ckpt=$4
ptm=${5:-"bert/bert-base-15lang-cased/"}


python generate_bnmt.py  $data --user-dir bnmt --task bert_nmt -s $src -t $tgt \
    --path $ckpt \
    --beam 5 --batch-size 64 --remove-bpe --bert-model-name $ptm
