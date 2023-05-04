data=$1
ckpt=$2
ptm=${3:-"bert-base-german-dbmdz-uncased"}
python bert_nmt_extensions/generate.py  $data \
    --path $ckpt  --user-dir  bert_nmt_extensions --task bert_nmt \
    --batch-size 128 --beam 5 --remove-bpe --bert-model-name $ptm
