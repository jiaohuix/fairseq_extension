ptm=${1:-"bert-base-multilingual-cased"}
TEXT=iwslt14.tokenized.de-en
src=de
tgt=en
destdir=iwslt_${src}_${tgt}_mbert
python bert_nmt_extensions/preprocess.py --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir $destdir  --joined-dictionary --bert-model-name $ptm \
  --workers 20  --user-dir  bert_nmt_extensions --task bert_nmt