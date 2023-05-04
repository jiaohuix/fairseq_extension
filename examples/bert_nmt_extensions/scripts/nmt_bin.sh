TEXT=iwslt14.tokenized.de-en
src=de
tgt=en
destdir=iwslt14_deen_nmt
python fairseq_cli/preprocess.py --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir $destdir  --joined-dictionary  --workers 20
