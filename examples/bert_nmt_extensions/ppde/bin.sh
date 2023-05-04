TEXT=$1
SAVR=$2
ptm=${3:-"bert-base-multilingual-cased"}
src=de
tgt=en
destdir=$SAVR
python bert_nmt_extensions/preprocess.py --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir $destdir  --joined-dictionary --bert-model-name $ptm \
  --workers 20  --user-dir  bert_nmt_extensions --task bert_nmt