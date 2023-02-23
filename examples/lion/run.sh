bash train_lion_exp1.sh de en data-bin/iwslt14_deen/ ckpt/lion_exp1

bash train_lion_exp3.sh de en data-bin/iwslt14_deen/ ckpt/lion_exp3
bash eval.sh de en data-bin/iwslt14_deen/  ckpt/lion_exp3/checkpoint_best.pt  lion3

bash train_lion_exp2.sh de en data-bin/iwslt14_deen/ ckpt/lion_exp2
bash eval.sh de en data-bin/iwslt14_deen/  ckpt/lion_exp2/checkpoint_best.pt   lion2

bash train.sh  de en data-bin/iwslt14_deen/ ckpt/base
bash eval.sh de en data-bin/iwslt14_deen/  ckpt/base/checkpoint_best.pt  base
