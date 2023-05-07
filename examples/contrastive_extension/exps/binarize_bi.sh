# Download and prepare the unidirectional data
#bash prepare-iwslt14.sh

# Preprocess/binarize the unidirectional data
TEXT=iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --joined-dictionary --workers 20

# Prepare the bidirectional data
cd iwslt14.tokenized.de-en
cat train.en train.de > train.src
cat train.de train.en > train.tgt
cat valid.en valid.de > valid.src
cat valid.de valid.en > valid.tgt
cat test.en test.de > test.src
cat test.de test.en > test.tgt
cd ..

# Preprocess/binarize the bidirectional data
TEXT=iwslt14.tokenized.de-en
fairseq-preprocess --source-lang src --target-lang tgt \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.bidirection.de-en \
    --srcdict data-bin/iwslt14.tokenized.de-en/dict.en.txt --tgtdict data-bin/iwslt14.tokenized.de-en/dict.de.txt --workers 20