exp_name="exp_mbert"
save_base=ckpt/$exp_name

subexp="mbert_trim_lora"
data=iwslt_de_en_mbert_trim
ptm=miugod/mbert_trim_ende
save_dir=$save_base/$subexp
bash eval_fuse.sh $data $save_dir/checkpoint_best.pt  $ptm > $save_dir/gen.txt

subexp="mbert_trim"
data=iwslt_de_en_mbert_trim
ptm=miugod/mbert_trim_ende
save_dir=$save_base/$subexp
bash eval_fuse.sh $data $save_dir/checkpoint_best.pt  $ptm >  $save_dir/gen.txt

subexp="mbert_trim_mlm"
data=iwslt_de_en_mbert_trim
ptm=miugod/mbert_trim_ende_mlm
save_dir=$save_base/$subexp
bash eval_fuse.sh $data $save_dir/checkpoint_best.pt  $ptm >  $save_dir/gen.txt


subexp="mbert_trim_wwm"
data=iwslt_de_en_mbert_trim
ptm=miugod/mbert_trim_ende_wwm
save_dir=$save_base/$subexp
bash eval_fuse.sh $data $save_dir/checkpoint_best.pt  $ptm >  $save_dir/gen.txt

subexp="mbert_raw"
data=iwslt_de_en_mbert_raw
ptm=bert-base-multilingual-cased
save_dir=$save_base/$subexp
bash eval_fuse.sh $data $save_dir/checkpoint_best.pt  $ptm >  $save_dir/gen.txt


subexp="mbert_trim_lora"
data=iwslt_de_en_mbert_trim
ptm=miugod/mbert_trim_ende
save_dir=$save_base/$subexp
bash eval_fuse.sh $data $save_dir/checkpoint_best.pt  $ptm >  $save_dir/gen.txt
