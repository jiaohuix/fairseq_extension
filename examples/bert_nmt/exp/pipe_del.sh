
data=data-bin/iwslt14_mbert
#warm_nmt_ckpt=ckpt/deen_nmt/checkpoint_best.pt
scripts=exp/exp_scripts/
#ptm=bert-base-german-cased
ptm=miugod/mbert_trim_ende
exp_name="exp_bnmt_qattn_deltop"

save_base=ckpt/$exp_name
log_dir=ckpt/$exp_name/logs/
if [ ! -d $log_dir ];then
    mkdir -p $log_dir
fi

# train: data ckpt ptm exp_args

#function warm_from_nmt() {
#    local warm_nmt_ckpt=$1
#    local save_dir=$2
#
#    if [ ! -d $save_dir ];then
#        mkdir -p $save_dir
#    fi
#    cp $warm_nmt_ckpt $save_dir/checkpoint_nmt.pt
#    echo "copy nmt_ckpt to $save_dir for warmup bert-fuse."
#}

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
#    warm_from_nmt $warm_nmt_ckpt $save_dir
    # train
    bash $scripts/train_bnmt.sh  $data  $save_dir  $exp_args
    # eval
    bash $scripts/eval_bnmt.sh $data $save_dir/checkpoint_best.pt > $save_dir/gen_${expname}.txt
    # extract ppl
    bash $scripts/extract_ppl.sh $save_dir/training.log $save_dir/ppl_${expname}.txt
    # delete ckpt
#    rm $save_dir/checkpoint_nmt.pt
    # save log
    cp $save_dir/ppl_${expname}.txt $log_dir/ppl_${expname}.txt
    grep "BLEU4" $save_dir/gen_${expname}.txt  >> $log_dir/exps.log
}


## qblock
# 1.删除top的query
subexp="qb1_del"
exp_args="--del-top-qtok --q-layers 1 --n-query 8  --q-drop 0.1  --arch bnmt_iwslt_de_en --bert-model-name  $ptm "
experiment_pipe $subexp $data $save_base $log_dir  $exp_args
## 2. 不删除（之前的pad token写错了）


subexp="qb1_no_del"
exp_args="--q-layers 1 --n-query 8  --q-drop 0.1  --arch bnmt_iwslt_de_en --bert-model-name  $ptm "
experiment_pipe $subexp $data $save_base $log_dir  $exp_args



# drop 0
subexp="qb1_del_dp0"
exp_args="--del-top-qtok --q-layers 1 --n-query 8  --q-drop 0.  --arch bnmt_iwslt_de_en --bert-model-name  $ptm "
experiment_pipe $subexp $data $save_base $log_dir  $exp_args

subexp="qb1_no_del_dp0"
exp_args="--q-layers 1 --n-query 8  --q-drop 0.  --arch bnmt_iwslt_de_en --bert-model-name  $ptm "
experiment_pipe $subexp $data $save_base $log_dir  $exp_args


# drop0 + layernorm
subexp="qb1_del_dp0_lnorm"
exp_args="--del-top-qtok --q-layers 1 --n-query 8 --layernorm-embedding  --q-drop 0.  --arch bnmt_iwslt_de_en --bert-model-name  $ptm "
experiment_pipe $subexp $data $save_base $log_dir  $exp_args

subexp="qb1_no_del_dp0_lnorm"
exp_args="--q-layers 1 --n-query 8  --layernorm-embedding  --q-drop 0.  --arch bnmt_iwslt_de_en --bert-model-name  $ptm "
experiment_pipe $subexp $data $save_base $log_dir  $exp_args

