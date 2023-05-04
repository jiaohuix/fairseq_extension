mosesdecoder=nmt_data_tools/mosesdecoder
## de-subnmt data
mkdir data_desubnmt
sed -r 's/(@@ )|(@@ ?$)//g' iwslt14.tokenized.de-en/train.en > data_desubnmt/train.en
sed -r 's/(@@ )|(@@ ?$)//g' iwslt14.tokenized.de-en/train.de > data_desubnmt/train.de
sed -r 's/(@@ )|(@@ ?$)//g' iwslt14.tokenized.de-en/valid.en > data_desubnmt/valid.en
sed -r 's/(@@ )|(@@ ?$)//g' iwslt14.tokenized.de-en/valid.de > data_desubnmt/valid.de
sed -r 's/(@@ )|(@@ ?$)//g' iwslt14.tokenized.de-en/test.en > data_desubnmt/test.en
sed -r 's/(@@ )|(@@ ?$)//g' iwslt14.tokenized.de-en/test.de > data_desubnmt/test.de

## de-mose data
mkdir data_demose
perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < data_desubnmt/train.en > data_demose/train.en
perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < data_desubnmt/train.de > data_demose/train.de
perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < data_desubnmt/valid.en > data_demose/valid.en
perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < data_desubnmt/valid.de > data_demose/valid.de
perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < data_desubnmt/test.en > data_demose/test.en
perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < data_desubnmt/test.de > data_demose/test.de
