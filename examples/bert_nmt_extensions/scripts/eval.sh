data=$1
ckpt=$2
python fairseq_cli/generate.py  $data \
    --path $ckpt \
    --beam 5 --batch-size 128 --remove-bpe
