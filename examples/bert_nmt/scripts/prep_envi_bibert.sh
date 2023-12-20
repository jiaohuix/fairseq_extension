# 下载iwslt15 vi-en数据，并且转为train valid test格式
# 单向训练数据，目标语言不共享
# 参数
src=${1:-"en"}
tgt=${2:-"vi"}
bpe_ops=${3:-"10000"} # 目标语言
ptm=${4:-"bert-base-multilingual-cased"}
data_dir=iwslt15_envi

tmp_dir=./tmp/
de_bpe_dir=$tmp_dir/data_desubnmt
de_mose_dir=$tmp_dir/data_demose

lang=${src}${tgt}
suffix=${lang}_${bpe_ops}
bpe_dir=$tmp_dir/bpe_$suffix
mbert_data=$tmp_dir/mbert_$suffix
destdir=data-bin/mbert_$suffix
echo "destdir: $destdir"

#git clone https://gitee.com/miugod/nmt_data_tools.git
mosesdecoder=nmt_data_tools/mosesdecoder


## de-subnmt data
# mkdir -p $de_bpe_dir
# for prefix in train valid test
#  do
#    sed -r 's/(@@ )|(@@ ?$)//g' $data_dir/$prefix.$src > $de_bpe_dir/$prefix.$src
#    sed -r 's/(@@ )|(@@ ?$)//g' $data_dir/$prefix.$tgt > $de_bpe_dir/$prefix.$tgt
#  done


# ## de-mose data
# mkdir -p $de_mose_dir
# for prefix in train valid test
#  do
#    perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l $src -q < $de_bpe_dir/$prefix.$src > $de_mose_dir/$prefix.$src
#    perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l $tgt -q < $de_bpe_dir/$prefix.$tgt > $de_mose_dir/$prefix.$tgt
#  done

#训练词表 1w op，结果是1w词表
mkdir -p $bpe_dir
rm $de_mose_dir/train.all
for prefix in train valid test
 do
#   cat $de_mose_dir/$prefix.$src >> $de_mose_dir/train.all # 不共享，只单向
   cat $de_mose_dir/$prefix.$tgt >> $de_mose_dir/train.all
 done
python vocab_trainer.py --data $de_mose_dir/train.all --size $bpe_ops --output $bpe_dir


# 分词 不共享词表,目标语言用自己学的词表
# tokenize translation data
mkdir -p $mbert_data
for prefix in train valid test
do
   python transform_tokenize.py --input $de_mose_dir/$prefix.$src --output $mbert_data/$prefix.$src --pretrained_model $ptm
#   python transform_tokenize.py --input $de_mose_dir/$prefix.$tgt --output $mbert_data/$prefix.$tgt --pretrained_model $ptm
   python transform_tokenize.py --input $de_mose_dir/$prefix.$tgt --output $mbert_data/$prefix.$tgt --pretrained_model $bpe_dir # 目标语言使用自己学习的
done


# ## get src and tgt vocabulary
# # 复用bert的词表dat
python get_vocab.py --tokenizer $ptm --output $mbert_data/src_vocab.txt
python get_vocab.py --tokenizer $bpe_dir --output $mbert_data/tgt_vocab0.txt
# 更新tgt_vocab的special tokens ，与src对齐
# src tgt tgt_out
python align_bert_dict.py $mbert_data/src_vocab.txt  $mbert_data/tgt_vocab0.txt $mbert_data/tgt_vocab.txt
rm $mbert_data/tgt_vocab0.txt
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

# 不用處理特殊符號，代碼處改過了

python fairseq_cli/preprocess.py --source-lang $src --target-lang $tgt  --trainpref $mbert_data/train --validpref $mbert_data/valid \
--testpref $mbert_data/test --destdir $destdir --srcdict $mbert_data/src_vocab.txt \
--tgtdict $mbert_data/tgt_vocab.txt --vocab_file $mbert_data/src_vocab.txt --workers 20

#TEXT=data_mixed
#srcvocab=$TEXT/src_vocab.txt
#tgtvocab=$TEXT/tgt_vocab.txt
#for sp_token in UNK CLS  SEP  MASK
#do
#   srcidx=$(grep -n $sp_token $srcvocab | cut -f1 -d":")
#   tgtidx=$(grep -n $sp_token $tgtvocab | cut -f1 -d":")
#   echo "$TEXT token=$sp_token i= $srcidx, j=$tgtidx"
#   # 把目标词表的特殊符号交换到源的位置
#   swap2line $tgtvocab $srcidx $tgtidx
#done
#
#
#
#fairseq-preprocess --source-lang de --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
#--testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
#--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25
#
#TEXT=data_mixed_ft
#srcvocab=$TEXT/src_vocab.txt
#tgtvocab=$TEXT/tgt_vocab.txt
#for sp_token in UNK CLS  SEP  MASK
#do
#   srcidx=$(grep -n $sp_token $srcvocab | cut -f1 -d":")
#   tgtidx=$(grep -n $sp_token $tgtvocab | cut -f1 -d":")
#   echo "$TEXT token=$sp_token i= $srcidx, j=$tgtidx"
#   # 把目标词表的特殊符号交换到源的位置
#   swap2line $tgtvocab $srcidx $tgtidx
#done
#
#fairseq-preprocess --source-lang de --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
#--testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
#--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25

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