TEXT=iwslt14_parafuse
src=de
tgt=en
destdir=iwslt_${src}_${tgt}_parafuse_mbert
#ptm=bert-base-german-cased
ptm=bert-base-multilingual-cased

python bert_nmt_extensions/preprocess.py --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir $destdir  --joined-dictionary --bert-model-name $ptm \
  --workers 20  --user-dir  bert_nmt_extensions --task bert_nmt