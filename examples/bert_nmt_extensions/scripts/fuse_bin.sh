TEXT=iwslt14.tokenized.de-en
src=de
tgt=en
destdir=iwslt_${src}_${tgt}
python preprocess.py --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir $destdir  --joined-dictionary --bert-model-name bert-base-german-dbmdz-uncased  --workers 20
  
