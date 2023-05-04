### 不合理!!!目标语言中文不能用char!!!序列过长
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
sed -r 's/(@@ )|(@@ ?$)//g' $bpe_folder/train.$tgt > data_demose/train.$tgt
sed -r 's/(@@ )|(@@ ?$)//g' $bpe_folder/train.$src > data_demose/train.$src
sed -r 's/(@@ )|(@@ ?$)//g' $bpe_folder/valid.$tgt > data_demose/valid.$tgt
sed -r 's/(@@ )|(@@ ?$)//g' $bpe_folder/valid.$src > data_demose/valid.$src
sed -r 's/(@@ )|(@@ ?$)//g' $bpe_folder/test.${src}_${tgt}.$src > data_demose/test.$src
sed -r 's/(@@ )|(@@ ?$)//g' $bpe_folder/test.${tgt}_${src}.$tgt > data_demose/test.$tgt

## train 8K tokenizer for ordinary translation:
cat data_demose/train.$tgt data_demose/valid.$tgt data_demose/test.$tgt | shuf > data_demose/train.all
mkdir 8k-vocab-models
python vocab_trainer.py --data data_demose/train.all --size 8000 --output 8k-vocab-models



## tokenize translation data
mkdir thai_bert_tok
mkdir thai_bert_8k_tok
#mkdir thai_bert_12k_tok

for prefix in "valid" "test" "train" ;
do
    for lang in $tgt $src ;
    do
        python transform_tokenize.py --input data_demose/${prefix}.${lang} --output thai_bert_tok/${prefix}.${lang} --pretrained_model $ptm
    done
done

for prefix in "valid" "test" "train" ;
do
    python transform_tokenize.py --input data_demose/${prefix}.$tgt --output thai_bert_8k_tok/${prefix}.$tgt --pretrained_model 8k-vocab-models
done


mkdir data   # for one-way translation data
cp thai_bert_tok/*.$src data/
cp thai_bert_8k_tok/*.$tgt data/


## get src and tgt vocabulary
python get_vocab.py --tokenizer $ptm --output data/src_vocab.txt
python get_vocab.py --tokenizer 8k-vocab-models --output data/tgt_vocab.txt


### fairseq preprocess
# s pad /s unk ->  PAD unk s /s
TEXT=data
srcvocab=$TEXT/src_vocab.txt
tgtvocab=$TEXT/tgt_vocab.txt
#echo "<unk>" >> $tgtvocab
echo "<s>" >> $tgtvocab
echo "</s>" >> $tgtvocab


#for sp_token in [UNK] [CLS]  [SEP] [MASK]  # 不能用[]，在正则中是匹配括号内任意单个字符
python swap.py  $tgtvocab 2 3
python swap.py  $tgtvocab 3 4
python swap.py  $tgtvocab 4 5
# []代表匹配括号内任意一个字符,需要加\转义
sed -i "s|\[UNK\]|<unk>|g" $tgtvocab
#for sp_token in "<unk>"  "<s>" "</s>"
for sp_token in  "<s>" "</s>"
do
   srcidx=$(grep -n $sp_token $srcvocab | cut -f1 -d":")
   tgtidx=$(grep -n $sp_token $tgtvocab | cut -f1 -d":")
   echo "$TEXT token=$sp_token, i= $srcidx, j=$tgtidx"
   # 把目标词表的特殊符号交换到源的位置
#   swap2line $tgtvocab $srcidx $tgtidx
   python swap.py  $tgtvocab $srcidx $tgtidx
done

fairseq-preprocess --source-lang $src --target-lang $tgt  --trainpref $TEXT/train --validpref $TEXT/valid \
--testpref $TEXT/test --destdir ${TEXT}/$src-$tgt-databin --srcdict $TEXT/src_vocab.txt \
--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25






