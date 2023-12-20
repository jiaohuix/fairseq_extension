bash dual/train.sh data-bin/iwslt14_mres/  ckpt3/mres
bash dual/train.sh data-bin/iwslt14_dual/  ckpt3/dual
bash dual/train_ft.sh data-bin/iwslt14_deen_uni/  ckpt3/dual_ft ckpt3/dual/checkpoint_best.pt


# bsz=8192
bash dual/train.sh data-bin/iwslt14_mres_dual/  ckpt3/mres_dual
bash dual/train_ft.sh data-bin/iwslt14_deen_uni/  ckpt3/mres_dual_ft ckpt3/mres_dual/checkpoint_best.pt