## download IWSLT'14 dataset from fairseq
#wget https://raw.githubusercontent.com/pytorch/fairseq/master/examples/translation/prepare-iwslt14.sh
echo "step1: preprocess iwslt14..."
seed=1
git clone https://gitee.com/miugod/nmt_data_tools.git
bash prepare-iwslt14.sh
mosesdecoder=nmt_data_tools/mosesdecoder
pip install  subword-nmt  tokenizers

echo "step2: de-subnmt data..."
mkdir data_desubnmt
sed -r 's/(@@ )|(@@ ?$)//g' iwslt14.tokenized.de-en/train.en > data_desubnmt/train.en
sed -r 's/(@@ )|(@@ ?$)//g' iwslt14.tokenized.de-en/train.de > data_desubnmt/train.de
sed -r 's/(@@ )|(@@ ?$)//g' iwslt14.tokenized.de-en/valid.en > data_desubnmt/valid.en
sed -r 's/(@@ )|(@@ ?$)//g' iwslt14.tokenized.de-en/valid.de > data_desubnmt/valid.de
sed -r 's/(@@ )|(@@ ?$)//g' iwslt14.tokenized.de-en/test.en > data_desubnmt/test.en
sed -r 's/(@@ )|(@@ ?$)//g' iwslt14.tokenized.de-en/test.de > data_desubnmt/test.de

echo "step3: de-mose data..."
mkdir data_demose
perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < data_desubnmt/train.en > data_demose/train.en
perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < data_desubnmt/train.de > data_demose/train.de
perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < data_desubnmt/valid.en > data_demose/valid.en
perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < data_desubnmt/valid.de > data_demose/valid.de
perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < data_desubnmt/test.en > data_demose/test.en
perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < data_desubnmt/test.de > data_demose/test.de

echo "step3: concat data..."
# train:  src, tgt, src[SEP]tgt, tgt[SEP]src
mkdir mlm_dataset
## train & shuffle
cat  data_demose/train.* > mlm_dataset/train.txt
awk 'BEGIN{FS="\n";OFS=" [SEP] "} {getline f2 < "'"data_demose/train.en"'"; print $0,f2}'  data_demose/train.de >> mlm_dataset/train.txt
awk 'BEGIN{FS="\n";OFS=" [SEP] "} {getline f2 < "'"data_demose/train.de"'"; print $0,f2}'  data_demose/train.en >> mlm_dataset/train.txt

echo "shuffle train..."
shuf --random-source=<(yes $seed) mlm_dataset/train.txt > mlm_dataset/train.shuf
mv mlm_dataset/train.shuf mlm_dataset/train.txt

## valid & test
cat  data_demose/valid.* > mlm_dataset/valid.txt
cat  data_demose/test.*  > mlm_dataset/test.txt



#rm -rf data_desubnmt
#rm -rf data_demose
#rm -rf iwslt14.tokenized.de-en
#rm -rf orig
#rm -rf prepare-iwslt14.sh
echo "all done!"
