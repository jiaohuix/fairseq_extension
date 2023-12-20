# 学习率实验：1.reset 5e-4  2.not reset 1e-4 3. reset 1e-4  4. 3e-4?
scripts=scripts
model=bert/bert-base-15lang-cased/
warm_ckpt=ckpt_dual/bert15_mres_dual/avg10.pt
data=data-bin/iwslt14_mres_dual_bert15/
eval_data=data-bin-uni/bert15/
src=src
tgt=tgt
eval_src=de
eval_tgt=en
use_lora=1

exp_name="exp_dual_lora_ft_lr"
base_dir=ckpt_lora_ft/$exp_name
log_dir=$base_dir/logs/
mkdir -p $base_dir
mkdir -p $log_dir

avg_ckpt_nums=( 5 10 )

function experiment_pipe() {
    # params
    local expname=$1
    local data=$2
    shift 2 # Remove the first four parameters
    local exp_args="$@" # Accepts arguments of any length

    lang=lang${eval_src}_${eval_tgt}

    echo "--------EXP: $expname --------"
    echo "--------EXP: $expname --------" >> $log_dir/exps.log

    # make ckpt
    local save_dir=$base_dir/$expname
    mkdir -p $save_dir # >> $save_dir/train.log
    echo "copy warm ckpt ${warm_ckpt} to ${save_dir}"
    cp $warm_ckpt $save_dir/checkpoint_last.pt

    # train
    bash $scripts/train_lora_ft.sh  $data  $save_dir  $exp_args

    # average ckpts (1=use_lora)
    for num in ${avg_ckpt_nums[@]}
    do
        echo "average ${num}"
        bash $scripts/avg.sh $save_dir $num $use_lora
    done

    # rm $save_dir/checkpoint.best*

    # evaluate results
    bash $scripts/eval_lora.sh $eval_src $eval_tgt  $eval_data $save_dir/checkpoint_best.pt $model >  $save_dir/gen_best_${lang}.txt 2>&1
#    echo "------------[BEST BLEU]------------" >> $log_dir/exps.log
#    grep "BLEU4" $save_dir/gen_best_${lang}.txt  >> $log_dir/exps.log

    for num in ${avg_ckpt_nums[@]}
    do
        bash $scripts/eval_lora.sh $eval_src $eval_tgt  $eval_data $save_dir/avg${num}.pt $model  >  $save_dir/gen_avg${num}_${lang}.txt 2>&1
#        echo "------------[AVG${num} BLEU]------------" >> $log_dir/exps.log
#        grep "BLEU4" $save_dir/gen_avg${num}_${lang}.txt  >> $log_dir/exps.log
    done
    # extract ppl
    grep "valid on"  $save_dir/train.log >  $log_dir/eval_metric_${$expname}.log
    grep "BLEU4" $save_dir/gen_*  >> $log_dir/exps.log

}


###################################################################
# resetlr lr=5e-4
exp_name=r8_lr5e-4_resetlr
lr=5e-4
rank=8
alpha=16
extra_args="--reset-lr-scheduler --max-epoch 60"
exp_args="$model $rank $alpha $lr $extra_args"
experiment_pipe $exp_name  $data  $exp_args

###################################################################
# 不reset，更小学习率
exp_name=r8_lr1e-4_noresetlr
lr=1e-4
rank=8
alpha=16
extra_args=" --max-epoch 60"
exp_args="$model $rank $alpha $lr $extra_args"
experiment_pipe $exp_name  $data  $exp_args

###################################################################
# resetlr lr=1e-4
exp_name=r8_lr1e-4_resetlr
lr=1e-4
rank=8
alpha=16
extra_args="--reset-lr-scheduler --max-epoch 20"
exp_args="$model $rank $alpha $lr $extra_args"
experiment_pipe $exp_name  $data  $exp_args
