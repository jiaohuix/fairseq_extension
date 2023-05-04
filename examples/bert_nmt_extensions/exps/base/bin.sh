TEXT=iwslt14.tokenized.de-en
src=de
tgt=en
destdir=iwslt_${src}_${tgt}_base
python bert_nmt_extensions/preprocess.py --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir $destdir  --joined-dictionary --bert-model-name bert-base-german-cased \
  --workers 20  --user-dir  bert_nmt_extensions --task bert_nmt