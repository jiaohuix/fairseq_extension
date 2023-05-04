#ptm: bert-base-german
# exp: bert attention
# 1. baseline: 37.17                      | --bert-gates [1,1,1,1,1,1]
# 2. remove encoder bertattn: 36.41?      | --encoder-no-bert
# 3. remove decoder bertattn: 36.44?      | --decoder-no-bert
# 4. interleaved bert attn: 0 1 0 1 0 1   | --bert-gates [0,1,0,1,0,1]
# 5. interleaved bert attn: 1 0 1 0 1 0   | --bert-gates [1,0,1,0,1,0]
# 6. share encoder bertattn               | --share-enc-bertattn
# 7. share decoder bertattn               | --share-dec-bertattn
# 8. share enc,dec bertattn               | --share-all-bertattn

data=iwslt_de_en_baseg
warm_nmt_ckpt=ckpt/deen_nmt/checkpoint_best.pt
scripts=bert_nmt_extensions/exps_bertattn/
ptm=bert-base-german-cased
exp_name="exp_bertattn"
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


subexp="base"
# 不能空，否则把log传为expargs
exp_args="--bert-gates 1,1,1,1,1,1"
experiment_pipe $subexp $data $save_base $log_dir $exp_args

subexp="noenc"
exp_args="--encoder-no-bert"
experiment_pipe $subexp $data $save_base $log_dir $exp_args

subexp="nodec"
exp_args="--decoder-no-bert"
experiment_pipe $subexp $data $save_base $log_dir $exp_args

subexp="01"
exp_args="--bert-gates 0,1,0,1,0,1"
experiment_pipe $subexp $data $save_base $log_dir $exp_args

subexp="10"
exp_args="--bert-gates 1,0,1,0,1,0"
experiment_pipe $subexp $data $save_base $log_dir $exp_args

subexp="share_enc"
exp_args="--share-enc-bertattn"
save_dir=$save_base/$subexp
experiment_pipe $subexp $data $save_base $log_dir $exp_args

subexp="share_dec"
exp_args="--share-dec-bertattn"
experiment_pipe $subexp $data $save_base $log_dir $exp_args

subexp="share_all"
exp_args=" --share-all-bertattn"
experiment_pipe $subexp $data $save_base $log_dir $exp_args


