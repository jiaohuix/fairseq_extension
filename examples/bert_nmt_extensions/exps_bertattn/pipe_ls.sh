#ptm: bert-base-german
# exp: bert attention

# --ls-type selk --bert-output-layer -1  =  --ls-type sto1

data=iwslt_de_en_baseg
warm_nmt_ckpt=ckpt/deen_nmt/checkpoint_best.pt
scripts=bert_nmt_extensions/exps_bertattn/
ptm=bert-base-german-cased
exp_name="exp_layer_select"
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


#subexp="base"
#exp_args="--ls-type sto1"
#experiment_pipe $subexp $data $save_base $log_dir $exp_args

subexp="sto2"
exp_args="--ls-type sto2"
experiment_pipe $subexp $data $save_base $log_dir $exp_args

subexp="sto4"
exp_args="--ls-type sto4"
experiment_pipe $subexp $data $save_base $log_dir $exp_args

subexp="tok_moe_k2"
exp_args="--ls-type tok_moe_k2"
experiment_pipe $subexp $data $save_base $log_dir $exp_args

subexp="seq_moe_k2"
exp_args="--ls-type seq_moe_k2"
experiment_pipe $subexp $data $save_base $log_dir $exp_args


subexp="sto6"
exp_args="--ls-type sto6"
experiment_pipe $subexp $data $save_base $log_dir $exp_args

subexp="sto8"
exp_args="--ls-type sto8"
experiment_pipe $subexp $data $save_base $log_dir $exp_args


subexp="tok_moe_k1"
exp_args="--ls-type tok_moe_k1"
experiment_pipe $subexp $data $save_base $log_dir $exp_args

subexp="seq_moe_k1"
exp_args="--ls-type seq_moe_k1"
experiment_pipe $subexp $data $save_base $log_dir $exp_args
