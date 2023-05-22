# translation data
bash nmt_data_tools/examples/data_scripts/prepare-iwslt14.sh
TEXT=iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en --joined-dictionary \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20

# span corruption data
mkdir span_corrup_data
cat iwslt14.tokenized.de-en/train.* > span_corrup_data/train
cat iwslt14.tokenized.de-en/valid.* > span_corrup_data/valid
cat iwslt14.tokenized.de-en/test.* > span_corrup_data/test

TEXT=span_corrup_data
fairseq-preprocess \
   --only-source --srcdict data-bin/iwslt14.tokenized.de-en/dict.de.txt \
   --trainpref $TEXT/train \
   --validpref $TEXT/valid \
   --testpref $TEXT/test \
   --destdir data-bin/span_corrup \
   --workers 20

# translation span corruption data
mkdir trans_span_corrup_data
paste -d" " iwslt14.tokenized.de-en/train.de iwslt14.tokenized.de-en/train.en > trans_span_corrup_data/train
paste -d" " iwslt14.tokenized.de-en/valid.de iwslt14.tokenized.de-en/valid.en  > trans_span_corrup_data/valid
paste -d" " iwslt14.tokenized.de-en/test.de iwslt14.tokenized.de-en/test.en  > trans_span_corrup_data/test

cat span_corrup_data/train >> trans_span_corrup_data/train
cat span_corrup_data/valid >> trans_span_corrup_data/valid
cat span_corrup_data/test >> trans_span_corrup_data/test

TEXT=trans_span_corrup_data
fairseq-preprocess \
    --only-source --srcdict data-bin/iwslt14.tokenized.de-en/dict.de.txt \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --destdir data-bin/trans_span_corrup \
    --workers 20
