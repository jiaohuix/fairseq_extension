# eval size
src=de
tgt=en
save=ckpt_abl/1w/
#bash scripts/eval_bnmt.sh de en  data-bin-uni/deen_bcased  $save/avg10.pt bert/bert-base-15lang-cased/ > $save/gen_avg10_deen.txt
bash scripts/eval_bnmt.sh en de data-bin-uni/ende_bcased  $save/avg10.pt bert/bert-base-15lang-cased/ > $save/gen_avg10_ende.txt


save=ckpt_abl/5w/
bash scripts/eval_bnmt.sh de en  data-bin-uni/deen_bcased  $save/avg10.pt bert/bert-base-15lang-cased/ > $save/gen_avg10_deen.txt
bash scripts/eval_bnmt.sh en de data-bin-uni/ende_bcased  $save/avg10.pt bert/bert-base-15lang-cased/ > $save/gen_avg10_ende.txt


save=ckpt_abl/10w/
bash scripts/eval_bnmt.sh de en  data-bin-uni/deen_bcased  $save/avg10.pt bert/bert-base-15lang-cased/ > $save/gen_avg10_deen.txt
bash scripts/eval_bnmt.sh en de data-bin-uni/ende_bcased  $save/avg10.pt bert/bert-base-15lang-cased/ > $save/gen_avg10_ende.txt
