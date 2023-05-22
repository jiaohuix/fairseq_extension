data=$1
ckpt=$2
src=${3:-"de"}
tgt=${4:-"en"}
subset=${5:-"test"}
python fairseq_cli/generate.py  $data \
    --path $ckpt \
    --beam 5 --batch-size 128 --remove-bpe --gen-subset $subset
