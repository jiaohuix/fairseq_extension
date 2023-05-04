#ptm: bert-base-german
# exp: bert attention
# 1. baseline: lr same  | --fuse-lr-multiply 1  没必要
# 2. different lr:  x5  | --fuse-lr-multiply 5
# 3. different lr:  x10 | --fuse-lr-multiply 10
# 4. freeze nmt xx      | --freeze-nmt  没必要
# 此外还有调小nmt的lr，然后把bert-attn的任性变成10倍


data=iwslt_de_en_baseg
warm_nmt_ckpt=ckpt/deen_nmt/checkpoint_best.pt
scripts=bert_nmt_extensions/exps_bertattn/
ptm=bert-base-german-cased
exp_name="exp_lr_group"
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


subexp="lr_x5"
exp_args="--fuse-lr-multiply 5"
experiment_pipe $subexp $data $save_base $log_dir $exp_args


subexp="lr_x10"
exp_args="--fuse-lr-multiply 10"
experiment_pipe $subexp $data $save_base $log_dir $exp_args


