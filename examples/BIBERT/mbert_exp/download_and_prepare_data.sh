## download IWSLT'14 dataset from fairseq
#wget https://raw.githubusercontent.com/pytorch/fairseq/master/examples/translation/prepare-iwslt14.sh
git clone https://gitee.com/miugod/nmt_data_tools.git
bash prepare-iwslt14.sh
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

## train 8K tokenizer for ordinary translation:
cat data_demose/train.en data_demose/valid.en data_demose/test.en | shuf > data_demose/train.all
mkdir 8k-vocab-models
python vocab_trainer.py --data data_demose/train.all --size 8000 --output 8k-vocab-models

## train 12K tokenizer for dual-directional translation
cat data_demose/train.en data_demose/valid.en data_demose/test.en data_demose/train.de data_demose/valid.de data_demose/test.de | shuf > data_demose/train.all.dual
mkdir 12k-vocab-models
python vocab_trainer.py --data data_demose/train.all.dual --size 12000 --output 12k-vocab-models



## tokenize translation data
mkdir mbert_tok
mkdir mbert_8k_tok
mkdir mbert_12k_tok

for prefix in "valid" "test" "train" ;
do
    for lang in "en" "de" ;
    do
        python transform_tokenize.py --input data_demose/${prefix}.${lang} --output mbert_tok/${prefix}.${lang} --pretrained_model bert-base-multilingual-cased
    done
done

for prefix in "valid" "test" "train" ;
do
    python transform_tokenize.py --input data_demose/${prefix}.en --output mbert_8k_tok/${prefix}.en --pretrained_model 8k-vocab-models
done


for prefix in "valid" "test" "train" ;
do
    for lang in "en" "de";
    do
    python transform_tokenize.py --input data_demose/${prefix}.${lang} --output mbert_12k_tok/${prefix}.${lang} --pretrained_model 12k-vocab-models
    done
done


mkdir data   # for one-way translation data
cp mbert_tok/*.de data/ 
cp mbert_8k_tok/*.en data/

mkdir data_mixed_ft # for dual-directional fine-tuning data. we first preprocess this because it will be easier to finish
cp mbert_tok/*.de data_mixed_ft/
cp mbert_12k_tok/*.en data_mixed_ft/

mkdir data_mixed # preprocess dual-directional data

cd data_mixed
cat ../mbert_tok/train.en ../mbert_tok/train.de > train.all.en
cat ../mbert_12k_tok/train.de ../mbert_12k_tok/train.en > train.all.de
paste -d '@@@' train.all.en /dev/null /dev/null train.all.de | shuf > train.all
cat train.all | awk -F'@@@' '{print $1}' > train.de
cat train.all | awk -F'@@@' '{print $2}' > train.en
rm train.all*

cat ../mbert_tok/valid.en ../mbert_tok/valid.de > valid.all.en
cat ../mbert_12k_tok/valid.de ../mbert_12k_tok/valid.en > valid.all.de
paste -d '@@@' valid.all.en /dev/null /dev/null valid.all.de | shuf > valid.all
cat valid.all | awk -F'@@@' '{print $1}' > valid.de
cat valid.all | awk -F'@@@' '{print $2}' > valid.en
rm valid.all*

cp ../mbert_tok/test.de .
cp ../mbert_12k_tok/test.en .
cd ..



## get src and tgt vocabulary
python get_vocab.py --tokenizer bert-base-multilingual-cased --output data/src_vocab.txt
python get_vocab.py --tokenizer bert-base-multilingual-cased --output data_mixed/src_vocab.txt
python get_vocab.py --tokenizer bert-base-multilingual-cased --output data_mixed_ft/src_vocab.txt
python get_vocab.py --tokenizer 8k-vocab-models --output data/tgt_vocab.txt
python get_vocab.py --tokenizer 12k-vocab-models --output data_mixed/tgt_vocab.txt
python get_vocab.py --tokenizer 12k-vocab-models --output data_mixed_ft/tgt_vocab.txt



function swap2line() {
    file=$1
    i=$2
    j=$3
    tempfile=tmp
    # 从文件中获取第i行和第j行
    word_i=$(sed -n "${i}p" "$file")
    word_j=$(sed -n "${j}p" "$file")
    # 交换第i行和第j行
    sed "${i}s/.*/${word_j}/" "$file" > "$tempfile"
    sed "${j}s/.*/${word_i}/" "$tempfile" > "$file"
    # 删除临时文件
    rm "$tempfile"

}


## fairseq preprocess
# s pad /s unk ->  PAD unk s /s
TEXT=data

sed -i "s|<s>|[PAD]|g"  $TEXT/src_vocab.txt
sed -i "s|<pad>|[UNK]|g"  $TEXT/src_vocab.txt
sed -i "s|</s>|<s>|g"  $TEXT/src_vocab.txt
sed -i "s|<unk>|</s>|g"  $TEXT/src_vocab.txt

srcvocab=$TEXT/src_vocab.txt
tgtvocab=$TEXT/tgt_vocab.txt
#for sp_token in [UNK] [CLS]  [SEP] [MASK]  # 不能用[]，在正则中是匹配括号内任意单个字符
for sp_token in UNK CLS  SEP  MASK
do
   srcidx=$(grep -n $sp_token $srcvocab | cut -f1 -d":")
   tgtidx=$(grep -n $sp_token $tgtvocab | cut -f1 -d":")
   echo "$TEXT token=$sp_token, i= $srcidx, j=$tgtidx"
   # 把目标词表的特殊符号交换到源的位置
   swap2line $tgtvocab $srcidx $tgtidx
done


fairseq-preprocess --source-lang de --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
--testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25

TEXT=data_mixed
srcvocab=$TEXT/src_vocab.txt
tgtvocab=$TEXT/tgt_vocab.txt
for sp_token in UNK CLS  SEP  MASK
do
   srcidx=$(grep -n $sp_token $srcvocab | cut -f1 -d":")
   tgtidx=$(grep -n $sp_token $tgtvocab | cut -f1 -d":")
   echo "$TEXT token=$sp_token i= $srcidx, j=$tgtidx"
   # 把目标词表的特殊符号交换到源的位置
   swap2line $tgtvocab $srcidx $tgtidx
done



fairseq-preprocess --source-lang de --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
--testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25

TEXT=data_mixed_ft
srcvocab=$TEXT/src_vocab.txt
tgtvocab=$TEXT/tgt_vocab.txt
for sp_token in UNK CLS  SEP  MASK
do
   srcidx=$(grep -n $sp_token $srcvocab | cut -f1 -d":")
   tgtidx=$(grep -n $sp_token $tgtvocab | cut -f1 -d":")
   echo "$TEXT token=$sp_token i= $srcidx, j=$tgtidx"
   # 把目标词表的特殊符号交换到源的位置
   swap2line $tgtvocab $srcidx $tgtidx
done

fairseq-preprocess --source-lang de --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
--testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25

# remove useless files
#rm -rf data_desubnmt
#rm -rf data_demose
#rm -rf iwslt14.tokenized.de-en
#rm -rf orig
#rm -rf subword-nmt
#rm -rf mosesdecoder
#rm -rf prepare-iwslt14.sh
#rm -rf mbert_tok
#rm -rf mbert_8k_tok
#rm -rf mbert_12k_tok









