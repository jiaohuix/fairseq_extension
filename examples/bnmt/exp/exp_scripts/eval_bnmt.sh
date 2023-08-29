data=$1
ckpt=$2
fairseq-generate  $data --user-dir bert_nmt \
    --path $ckpt \
    --beam 5 --batch-size 128 --remove-bpe
