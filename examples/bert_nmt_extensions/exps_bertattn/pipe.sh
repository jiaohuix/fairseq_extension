# 此pipe要包括，模型训练，评估，日志ppl抽取等
# 先pretrain

data=iwslt_de_en
warm_nmt_ckpt=ckpt/deen_nmt/checkpoint_best.pt
scripts=bert_nmt_extensions/exps_bertattn/
#ptm=bert-base-german-dbmdz-uncased
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
    local save_dir=$3
    local exp_args=$4

    # make ckpt
    warm_from_nmt $warm_nmt_ckpt $save_dir
    # train
    bash $scripts/train_fuse.sh  $data  $save_dir $ptm $exp_args
    # eval 
    bash $scripts/eval_fuse.sh $data $save_dir/checkpoint_best.pt $ptm >> $save_dir/gen_${expname}.txt
    # extract ppl
    bash $scripts/extract_ppl.sh $save_dir/training.log $save_dir/ppl_${expname}.txt
    # delete ckpt
    rm $save_dir/checkpoint_nmt.pt
}



subexp="base"
save_dir=$save_base/$subexp
echo "--------stage0: $subexp --------"
echo "--------stage0: $subexp --------" >> $save_base/exp.log
exp_args=""
experiment_pipe $subexp $data $save_dir $exp_args
# extract logs
cp $save_dir/ppl_${subexp}.txt $log_dir/ppl_${subexp}.txt
grep "BLEU4" $save_dir/gen_${subexp}.txt  >> $save_base/exp.log

