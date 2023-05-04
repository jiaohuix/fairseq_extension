data=iwslt_de_en
warm_nmt_ckpt=ckpt/deen_nmt/checkpoint_best.pt
scripts=bert_nmt_extensions/exps/
exp_name="exp1_fuse_abl"

# base
echo "stage0: base"
echo "stage0: base" > ${exp_name}.txt
exp="base"
save=ckpt/abl/$exp
mkdir -p $save
cp $warm_nmt_ckpt $save/checkpoint_nmt.pt
bash $scripts/train_base.sh  ${data}_base $save
# eval
bash $scripts/eval_fuse.sh ${data}_base $save/checkpoint_best.pt   bert-base-german-cased | grep "BLEU4" >> ${exp_name}.txt
# extract ppl
bash $scripts/extract_ppl.sh $save/training.log $save/../$exp.txt


# enc-dec
echo "stage1: nodec"
echo "stage1: nodec" >> ${exp_name}.txt

exp="nodec"
save=ckpt/abl/$exp
mkdir -p $save
cp $warm_nmt_ckpt $save/checkpoint_nmt.pt
bash $scripts/train_nodec.sh  $data $save
bash $scripts/eval_fuse.sh ${data} $save/checkpoint_best.pt | grep "BLEU4"  >> ${exp_name}.txt
bash $scripts/extract_ppl.sh $save/training.log $save/../$exp.txt


echo "stage2: noenc"
echo "stage2: noenc"  >> ${exp_name}.txt
exp="noenc"
save=ckpt/abl/$exp
mkdir -p $save
cp $warm_nmt_ckpt $save/checkpoint_nmt.pt
bash $scripts/train_noenc.sh  $data $save
bash $scripts/eval_fuse.sh ${data} $save/checkpoint_best.pt | grep "BLEU4"  >> ${exp_name}.txt
bash $scripts/extract_ppl.sh $save/training.log $save/../$exp.txt

echo "stage3: enctop1" # 36.4
echo "stage3: enctop1"  >> ${exp_name}.txt
exp="enctop1"
save=ckpt/abl/$exp
mkdir -p $save
cp $warm_nmt_ckpt $save/checkpoint_nmt.pt
bash $scripts/train_enctop1.sh  $data $save
bash $scripts/eval_fuse.sh ${data} $save/checkpoint_best.pt | grep "BLEU4"  >> ${exp_name}.txt
bash $scripts/extract_ppl.sh $save/training.log $save/../$exp.txt

# linear gate
echo "stage4: gate0"
echo "stage4: gate0" >> ${exp_name}.txt
exp="gate0"
save=ckpt/abl/$exp
mkdir -p $save
cp $warm_nmt_ckpt $save/checkpoint_nmt.pt
bash $scripts/train_fuse_gate0.sh  $data $save
bash $scripts/eval_fuse.sh ${data} $save/checkpoint_best.pt | grep "BLEU4"  >> ${exp_name}.txt
bash $scripts/extract_ppl.sh $save/training.log $save/../$exp.txt

echo "stage5: gate0.5"
echo "stage5: gate0.5" >> ${exp_name}.txt
exp="gate0.5"
save=ckpt/abl/$exp
mkdir -p $save
cp $warm_nmt_ckpt $save/checkpoint_nmt.pt
bash $scripts/train_fuse_gate0.5.sh  $data $save
bash $scripts/eval_fuse.sh ${data} $save/checkpoint_best.pt | grep "BLEU4"  >> ${exp_name}.txt
bash $scripts/extract_ppl.sh $save/training.log $save/../$exp.txt

echo "stage6: gate1"
echo "stage6: gate1" >> ${exp_name}.txt
exp="gate1"
save=ckpt/abl/$exp
mkdir -p $save
cp $warm_nmt_ckpt $save/checkpoint_nmt.pt
bash $scripts/train_fuse_gate1.sh  $data $save
bash $scripts/eval_fuse.sh ${data} $save/checkpoint_best.pt | grep "BLEU4"  >> ${exp_name}.txt
bash $scripts/extract_ppl.sh $save/training.log $save/../$exp.txt

