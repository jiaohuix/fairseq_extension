## download IWSLT'14 dataset from fairseq
#wget https://raw.githubusercontent.com/pytorch/fairseq/master/examples/translation/prepare-iwslt14.sh
src=th
tgt=zh
git clone https://gitee.com/miugod/nmt_data_tools.git
bash prepare-zhth.sh
mosesdecoder=nmt_data_tools/mosesdecoder
ptm=monsoon-nlp/bert-base-thai
## de-subnmt data
#mkdir data_desubnmt
mkdir data_demose
#bpe_folder=iwslt14.tokenized.$src-$tgt
bpe_folder=datasets/bpe_8000/zh_th/
# 只去了bpe，似乎不需要分词。。
sed -r 's/(@@ )|(@@ ?$)//g' $bpe_folder/train.$tgt > data_demose/train.$tgt
sed -r 's/(@@ )|(@@ ?$)//g' $bpe_folder/train.$src > data_demose/train.$src
sed -r 's/(@@ )|(@@ ?$)//g' $bpe_folder/valid.$tgt > data_demose/valid.$tgt
sed -r 's/(@@ )|(@@ ?$)//g' $bpe_folder/valid.$src > data_demose/valid.$src
sed -r 's/(@@ )|(@@ ?$)//g' $bpe_folder/test.$src_$tgt.$src > data_demose/test.$src
sed -r 's/(@@ )|(@@ ?$)//g' $bpe_folder/test.$tgt_$src.$tgt > data_demose/test.$tgt

## de-mose data
#mkdir data_demose
#perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l $tgt -q < data_desubnmt/train.$tgt > data_demose/train.$tgt
#perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l $tgt -q < data_desubnmt/train.$src > data_demose/train.$src
#perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l $tgt -q < data_desubnmt/valid.$tgt > data_demose/valid.$tgt
#perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l $tgt -q < data_desubnmt/valid.$src > data_demose/valid.$src
#perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l $tgt -q < data_desubnmt/test.$tgt > data_demose/test.$tgt
#perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l $tgt -q < data_desubnmt/test.$src > data_demose/test.$src

## train 8K tokenizer for ordinary translation: (tgt)
#cat data_demose/train.$tgt data_demose/valid.$tgt data_demose/test.$tgt | shuf > data_demose/train.all
mkdir 8k-vocab-models
#python vocab_trainer.py --data data_demose/train.all --size 8000 --output 8k-vocab-models
cp $bpe_folder/code.$tgt  8k-vocab-models/

### train 12K tokenizer for dual-directional translation
#cat data_demose/train.$tgt data_demose/valid.$tgt data_demose/test.$tgt data_demose/train.$src data_demose/valid.$src data_demose/test.$src | shuf > data_demose/train.all.dual
#mkdir 12k-vocab-models
#python vocab_trainer.py --data data_demose/train.all.dual --size 12000 --output 12k-vocab-models



## tokenize translation data
mkdir thai_bert_tok
mkdir thai_bert_8k_tok
#mkdir thai_bert_12k_tok
# bert编码src和tgt
for prefix in "valid" "test" "train" ;
do
    for lang in $tgt $src ;
    do
        python transform_tokenize.py --input data_demose/${prefix}.${lang} --output thai_bert_tok/${prefix}.${lang} --pretrained_model $ptm
    done
done
# tgt用bpe编码
for prefix in "valid" "test" "train" ;
do
    subword-nmt apply-bpe -c $bpe_folder/code.$tgt  < data_demose/${prefix}.$tgt  >  thai_bert_8k_tok/${prefix}.$tgt
done


#for prefix in "valid" "test" "train" ;
#do
#    for lang in $tgt $src ;
#    do
#    python transform_tokenize.py --input data_demose/${prefix}.${lang} --output thai_bert_12k_tok/${prefix}.${lang} --pretrained_model 12k-vocab-models
#    done
#done


mkdir data_bpe   # for one-way translation data
cp thai_bert_tok/*.$src data_bpe/
cp thai_bert_8k_tok/*.$tgt data_bpe/

#mkdir data_mixed_ft # for dual-directional fine-tuning data. we first preprocess this because it will be easier to finish
#cp thai_bert_tok/*.$src data_mixed_ft/
#cp thai_bert_12k_tok/*.$tgt data_mixed_ft/
#
#mkdir data_mixed # preprocess dual-directional data
#
#cd data_mixed
#cat ../thai_bert_tok/train.$tgt ../thai_bert_tok/train.$src > train.all.$tgt
#cat ../thai_bert_12k_tok/train.$src ../thai_bert_12k_tok/train.$tgt > train.all.$src
#paste -d '@@@' train.all.$tgt /dev/null /dev/null train.all.$src | shuf > train.all
#cat train.all | awk -F'@@@' '{print $1}' > train.$src
#cat train.all | awk -F'@@@' '{print $2}' > train.$tgt
#rm train.all*

#cat ../thai_bert_tok/valid.$tgt ../thai_bert_tok/valid.$src > valid.all.$tgt
#cat ../thai_bert_12k_tok/valid.$src ../thai_bert_12k_tok/valid.$tgt > valid.all.$src
#paste -d '@@@' valid.all.$tgt /dev/null /dev/null valid.all.$src | shuf > valid.all
#cat valid.all | awk -F'@@@' '{print $1}' > valid.$src
#cat valid.all | awk -F'@@@' '{print $2}' > valid.$tgt
#rm valid.all*
#
#cp ../thai_bert_tok/test.$src .
#cp ../thai_bert_12k_tok/test.$tgt .
#cd ..



## get src and tgt vocabulary
python get_vocab.py --tokenizer $ptm --output data_bpe/src_vocab.txt
#python get_vocab.py --tokenizer $ptm --output data_mixed/src_vocab.txt
#python get_vocab.py --tokenizer $ptm --output data_mixed_ft/src_vocab.txt

# 用tools获取dict词表，然后添加特殊符号
#python get_vocab.py --tokenizer 8k-vocab-models --output data/tgt_vocab.txt
head -n 7 data_bpe/src_vocab.txt > data_bpe/tgt_vocab.txt
cut -f1 -d" "  $bpe_folder/dict.$tgt.txt >> data_bpe/tgt_vocab.txt


## fairseq preprocess
# s pad /s unk ->  PAD unk s /s
TEXT=data_bpe



### fairseq preprocess
srcvocab=$TEXT/src_vocab.txt
tgtvocab=$TEXT/tgt_vocab.txt


fairseq-preprocess --source-lang $src --target-lang $tgt  --trainpref $TEXT/train --validpref $TEXT/valid \
--testpref $TEXT/test --destdir ${TEXT}/$src-$tgt-databin --srcdict $TEXT/src_vocab.txt \
--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25









