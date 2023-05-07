DATA=$1
CKPT=$2
python fairseq_cli/generate.py  $DATA --path $CKPT --user-dir contrastive_extension \
    --gen-subset test --beam 5 --lenpen 1 --max-tokens 8192 --remove-bpe
