# translation data
bash nmt_data_tools/examples/data_scripts/prepare-iwslt14.sh
TEXT=iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en --joined-dictionary \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20

# denoise data
mkdir denoise_data
cat iwslt14.tokenized.de-en/train.* > denoise_data/train
cat iwslt14.tokenized.de-en/valid.* > denoise_data/valid
cat iwslt14.tokenized.de-en/test.* > denoise_data/test

TEXT=denoise_data
fairseq-preprocess \
   --only-source --srcdict data-bin/iwslt14.tokenized.de-en/dict.de.txt \
   --trainpref $TEXT/train \
   --validpref $TEXT/valid \
   --testpref $TEXT/test \
   --destdir data-bin/denoise_data \
   --workers 20

