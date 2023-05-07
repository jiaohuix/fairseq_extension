# exp: bert attention
data=data-bin/iwslt14.tokenized.bidirection.de-en
scripts=contrastive_extension/exps/
exp_name="exp_loss"
save_base=ckpt/$exp_name
log_dir=ckpt/$exp_name/logs/
if [ ! -d $log_dir ];then
    mkdir -p $log_dir
fi

# train: data ckpt ptm exp_args

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
    # train
    bash $scripts/train.sh  $data  $save_dir $exp_args
    # eval
    bash $scripts/eval.sh $data $save_dir/checkpoint_best.pt  > $save_dir/gen_${expname}.txt
    # extract ppl
    bash $scripts/extract_ppl.sh $save_dir/training.log $save_dir/ppl_${expname}.txt
    # save log
    cp $save_dir/ppl_${expname}.txt $log_dir/ppl_${expname}.txt
    grep "BLEU4" $save_dir/gen_${expname}.txt  >> $log_dir/exps.log
}


subexp="base"
exp_args="--criterion label_smoothed_cross_entropy"
experiment_pipe $subexp $data $save_base $log_dir $exp_args


subexp="rdrop"
exp_args="--criterion label_smoothed_cross_entropy_with_rdrop --rdrop-alpha 5."
experiment_pipe $subexp $data $save_base $log_dir $exp_args


subexp="sent_ct"
exp_args="--user-dir contrastive_extension  --criterion label_smoothed_cross_entropy_with_contrastive --contrastive-lambda 5  --temperature 0.1"
experiment_pipe $subexp $data $save_base $log_dir $exp_args

subexp="entity_ct"
exp_args="--user-dir contrastive_extension  --task entity_ct_translation --criterion label_smoothed_ce_with_entity_contrastive --contrastive-lambda 1.  --temperature 0.1 --use-entity-ct  --entity-dict dict.de-en.bpe.txt --topk -1"
experiment_pipe $subexp $data $save_base $log_dir $exp_args

subexp="sent_ent_ct"
exp_args=" --user-dir contrastive_extension --task entity_ct_translation --criterion label_smoothed_ce_with_multi_granularity_contrastive --contrastive-lambda 1.  --temperature 0.1 --use-entity-ct  --entity-dict dict.de-en.bpe.txt --topk -1"
experiment_pipe $subexp $data $save_base $log_dir $exp_args