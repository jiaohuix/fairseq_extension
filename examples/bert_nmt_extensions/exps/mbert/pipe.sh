data=iwslt_de_en_mbert
warm_nmt_ckpt=ckpt/deen_nmt/checkpoint_best.pt
scripts=bert_nmt_extensions/exps/
exp_name="exp3_mbert"
ptm=bert-base-multilingual-cased

# base
echo "stage0: base"
echo "stage0: base" > ${exp_name}.txt
exp="base"
save=ckpt/abl/$exp
mkdir -p $save
cp $warm_nmt_ckpt $save/checkpoint_nmt.pt
bash train_mbert.sh  ${data} $save
# eval
bash $scripts/eval_fuse.sh ${data}_base $save/checkpoint_best.pt  $ptm | grep "BLEU4" >> ${exp_name}.txt
# extract ppl
bash $scripts/extract_ppl.sh $save/training.log $save/../$exp.txt
