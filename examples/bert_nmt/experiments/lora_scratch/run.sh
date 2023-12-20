# # bert15 单向训练
base_dir=ckpt_lora
model=bert/bert-base-15lang-cased/
data=data-bin/iwslt14_deen_mbert15/
src=de
tgt=en
# avg_ckpt_nums=( 3 5 10 )
avg_ckpt_nums=( 10 )

########################### EXPERIMENT1 rank=8 ###########################
# experiments params
rank=8
alpha=16
tag=scratch
save_dir=$base_dir/${tag}_rank${rank}
mkdir -p $save_dir # >> $save_dir/train.log
# train bert nmt model
bash scripts/train_lora.sh $data $save_dir  $model $rank $alpha
# average ckpts (1=use_lora)
for num in ${avg_ckpt_nums[@]}
do
    echo "average ${num}"
    bash scripts/avg.sh $save_dir $num 1
done
# rm $save_dir/checkpoint.best*

# evaluate results
bash scripts/eval_lora.sh $src $tgt  $data $save_dir/checkpoint_best.pt $model >  $save_dir/gen_best.txt 2>&1
for num in ${avg_ckpt_nums[@]}
do
    bash scripts/eval_lora.sh $src $tgt $data $save_dir/avg${num}.pt $model  >  $save_dir/gen_avg${num}.txt 2>&1
done
###################################################################


########################### EXPERIMENT2 rank=16 ###########################
# experiments params
rank=16
alpha=32
tag=scratch
save_dir=$base_dir/${tag}_rank${rank}
mkdir -p $save_dir # >> $save_dir/train.log
# train bert nmt model
bash scripts/train_lora.sh $data $save_dir  $model $rank $alpha
# average ckpts (1=use_lora)
for num in ${avg_ckpt_nums[@]}
do
    echo "average ${num}"
    bash scripts/avg.sh $save_dir $num 1
done
# rm $save_dir/checkpoint.best*

# evaluate results
bash scripts/eval_lora.sh $src $tgt  $data $save_dir/checkpoint_best.pt $model >  $save_dir/gen_best.txt 2>&1
for num in ${avg_ckpt_nums[@]}
do
    bash scripts/eval_lora.sh $src $tgt $data $save_dir/avg${num}.pt $model  >  $save_dir/gen_avg${num}.txt 2>&1
done
###################################################################


########################### EXPERIMENT3 rank=32 ###########################
# experiments params
rank=32
alpha=64
tag=scratch
save_dir=$base_dir/${tag}_rank${rank}
mkdir -p $save_dir # >> $save_dir/train.log
# train bert nmt model
bash scripts/train_lora.sh $data $save_dir  $model $rank $alpha
# average ckpts (1=use_lora)
for num in ${avg_ckpt_nums[@]}
do
    echo "average ${num}"
    bash scripts/avg.sh $save_dir $num 1
done
# rm $save_dir/checkpoint.best*

# evaluate results
bash scripts/eval_lora.sh $src $tgt  $data $save_dir/checkpoint_best.pt $model >  $save_dir/gen_best.txt 2>&1
for num in ${avg_ckpt_nums[@]}
do
    bash scripts/eval_lora.sh $src $tgt $data $save_dir/avg${num}.pt $model  >  $save_dir/gen_avg${num}.txt 2>&1
done
###################################################################


########################### EXPERIMENT4 rank=64 ###########################
# experiments params
rank=64
alpha=128
tag=scratch
save_dir=$base_dir/${tag}_rank${rank}
mkdir -p $save_dir # >> $save_dir/train.log
# train bert nmt model
bash scripts/train_lora.sh $data $save_dir  $model $rank $alpha
# average ckpts (1=use_lora)
for num in ${avg_ckpt_nums[@]}
do
    echo "average ${num}"
    bash scripts/avg.sh $save_dir $num 1
done
# rm $save_dir/checkpoint.best*

# evaluate results
bash scripts/eval_lora.sh $src $tgt  $data $save_dir/checkpoint_best.pt $model >  $save_dir/gen_best.txt 2>&1
for num in ${avg_ckpt_nums[@]}
do
    bash scripts/eval_lora.sh $src $tgt $data $save_dir/avg${num}.pt $model  >  $save_dir/gen_avg${num}.txt 2>&1
done
###################################################################