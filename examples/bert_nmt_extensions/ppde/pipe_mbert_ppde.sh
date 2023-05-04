#ptm: bert-base-german
# exp: bert attention
# 1.base: trim+mlm+lora+share
# 2.pdde: bert_aug_biconcat
# 3.pdde: bert_aug_concat
# 4.pdde: bert_aug_replace
# 5.pdde: bert_aug_insert
# 6.pdde: bert_aug_inserte


data=iwslt_de_en_mbert
warm_nmt_ckpt=ckpt/deen_nmt/checkpoint_best.pt
scripts=bert_nmt_extensions/exps_bertattn/
#ptm=bert-base-multilingual-cased
ptm=miugod/mbert_trim_ende
exp_name="exp_mbert_ppde"
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
    local ptm=$5  # all exp logs
    shift 5 # Remove the first four parameters
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


# 1.base: trim+mlm+lora+share
subexp="mbert_trim_mlm_lora_share"
exp_args="--bert-gates 1,1,1,1,1,1  --use-lora  --share-all-bertattn"
data=iwslt_de_en_mbert_trim
ptm=miugod/mbert_trim_ende_mlm
experiment_pipe $subexp $data $save_base $log_dir $ptm $exp_args

# 2.pdde: bert_aug_biconcat
subexp="mbert_trim_mlm_lora_share_biconcat"
exp_args="--bert-gates 1,1,1,1,1,1  --use-lora  --share-all-bertattn"
data=data-bin/bert_aug_biconcat/
ptm=miugod/mbert_trim_ende_mlm
experiment_pipe $subexp $data $save_base $log_dir $ptm $exp_args

# 3.pdde: bert_aug_concat
subexp="mbert_trim_mlm_lora_share_concat"
exp_args="--bert-gates 1,1,1,1,1,1  --use-lora  --share-all-bertattn"
data=data-bin/bert_aug_concat/
ptm=miugod/mbert_trim_ende_mlm
experiment_pipe $subexp $data $save_base $log_dir $ptm $exp_args

# 4.pdde: bert_aug_replace
subexp="mbert_trim_mlm_lora_share_replace"
exp_args="--bert-gates 1,1,1,1,1,1  --use-lora  --share-all-bertattn"
data=data-bin/bert_aug_replace/
ptm=miugod/mbert_trim_ende_mlm
experiment_pipe $subexp $data $save_base $log_dir $ptm $exp_args

# 5.pdde: bert_aug_insert
subexp="mbert_trim_mlm_lora_share_insert"
exp_args="--bert-gates 1,1,1,1,1,1  --use-lora  --share-all-bertattn"
data=data-bin/bert_aug_insert/
ptm=miugod/mbert_trim_ende_mlm
experiment_pipe $subexp $data $save_base $log_dir $ptm $exp_args


# 6.pdde: bert_aug_inserte
subexp="mbert_trim_mlm_lora_share_inserte"
exp_args="--bert-gates 1,1,1,1,1,1  --use-lora  --share-all-bertattn"
data=data-bin/bert_aug_inserte/
ptm=miugod/mbert_trim_ende_mlm
experiment_pipe $subexp $data $save_base $log_dir $ptm $exp_args

