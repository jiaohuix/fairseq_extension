#ptm: bert-base-german
# exp: bert attention
# x1. baseline: dropnet             | --bert-gates [1,1,1,1,1,1]
# 2. dynamic gate:                 |  --dynamic-gate --gate-type  dynamic
# 3. dynamic lr:                   |  --dynamic-gate --gate-type  dynamiclr16/32/64/128
# 4. glu gate:                     |  --dynamic-gate --gate-type  glu
# 5. glu norm:                     |  --dynamic-gate --gate-type  glunorm
# 6. linear gate:                  |  --linear-bert-gate-rate 0

data=iwslt_de_en_baseg
warm_nmt_ckpt=ckpt/deen_nmt/checkpoint_best.pt
scripts=bert_nmt_extensions/exps_bertattn/
ptm=bert-base-german-cased
exp_name="exp_gate"
save_base=ckpt/$exp_name
log_dir=ckpt/$exp_name/logs/
if [ ! -d $log_dir ];then
    mkdir -p $log_dir
fi

# train: data ckpt ptm exp_args

function warm_from_nmt() {
    local warm_nmt_ckpt=$1
    local save_dir=$2

    if [ ! -d $save_dir ];then
        mkdir -p $save_dir
    fi
    cp $warm_nmt_ckpt $save_dir/checkpoint_nmt.pt
    echo "copy nmt_ckpt to $save_dir for warmup bert-fuse."
}

function experiment_pipe() {
    # params
    local expname=$1
    local data=$2
    local save_base=$3
    local log_dir=$4  # all exp logs
    shift 4 # Remove the first four parameters
    local exp_args="$@" # Accepts arguments of any length


    echo "--------EXP: $expname --------"
    echo "--------EXP: $expname --------" >> $log_dir/exps.log

    # make ckpt
    local save_dir=$save_base/$expname
    warm_from_nmt $warm_nmt_ckpt $save_dir
    # train
    bash $scripts/train_fuse.sh  $data  $save_dir $ptm $exp_args
    # eval
    bash $scripts/eval_fuse.sh $data $save_dir/checkpoint_best.pt $ptm > $save_dir/gen_${expname}.txt
    # extract ppl
    bash $scripts/extract_ppl.sh $save_dir/training.log $save_dir/ppl_${expname}.txt
    # delete ckpt
    rm $save_dir/checkpoint_nmt.pt
    # save log
    cp $save_dir/ppl_${expname}.txt $log_dir/ppl_${expname}.txt
    grep "BLEU4" $save_dir/gen_${expname}.txt  >> $log_dir/exps.log
}


subexp="dynamic"
exp_args="--dynamic-gate --gate-type  dynamic"
experiment_pipe $subexp $data $save_base $log_dir $exp_args


subexp="dynamic_lr128"
exp_args="--dynamic-gate --gate-type  dynamiclr128"
experiment_pipe $subexp $data $save_base $log_dir $exp_args

subexp="dynamic_lr64"
exp_args="--dynamic-gate --gate-type  dynamiclr64"
experiment_pipe $subexp $data $save_base $log_dir $exp_args

subexp="glu"
exp_args=" --dynamic-gate --gate-type  glu"
experiment_pipe $subexp $data $save_base $log_dir $exp_args

subexp="glunorm"
exp_args="--dynamic-gate --gate-type  glunorm"
experiment_pipe $subexp $data $save_base $log_dir $exp_args

subexp="linar_gate"
exp_args="--linear-bert-gate-rate 0"
experiment_pipe $subexp $data $save_base $log_dir $exp_args



