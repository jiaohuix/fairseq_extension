ckpt_dir=$1
n_ckpt=$2
use_lora=${3:-"0"}

lora_args=""
if [ "$use_lora"x = "1"x ]; then
  lora_args="--use-lora"
fi

python scripts/avg_loras.py \
      --inputs $ckpt_dir/checkpoint.best*.pt  \
      --output $ckpt_dir/avg${n_ckpt}.pt  \
      --num-best-checkpoints $n_ckpt  $lora_args
