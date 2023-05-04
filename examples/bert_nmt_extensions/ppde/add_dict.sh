#!/bin/bash
if [ $# -lt 3 ];then
  echo "usage: bash $0 <infolder> <outfolder> <dictpath> <k_repeat=1>(opt)"
  exit
fi

# 输入： iwslt
# 将原始文件夹的train变成train.bert，然后再添加约束

# 对train.bert.en de 添加约束
src=de
tgt=en
infolder=$1
outfolder=$2
dict=$3
k_repeat=$4

mode=${5:-"biconcat"}
#mode=replace
#mode=insert
#mode=concat
dict_prob=0.8
max_pairs=5

TRAIN=train
VALID=valid
TEST=test
root=bert_nmt_extensions/ppde/
scripts=nmt_data_tools/mosesdecoder/scripts/
tmp_dir=$outfolder/tmp/
if [ ! -d $tmp_dir ];then
  mkdir -p $tmp_dir
fi

# 1. detok
echo "stage1: detokenize..."
for lang in $src $tgt
do
  for prefix  in $TRAIN $VALID $TEST
  do
      if [ ! -e $infolder/${prefix}.bert.${lang} ];then
        sed -r 's/(@@ )|(@@ ?$)//g' $infolder/${prefix}.${lang} > $infolder/${prefix}.bert.${lang}.tok
        perl $scripts/tokenizer/detokenizer.perl -l $lang < $infolder/${prefix}.bert.${lang}.tok > $infolder/${prefix}.bert.${lang}
        rm $infolder/${prefix}.bert.${lang}.tok
      fi

  done
done


# 2. 添加词典
echo "stage2: augment..."
## train增强k份
prefix=$TRAIN
rm -r $outfolder/$prefix.$src && rm $outfolder/$prefix.bert.$src
rm -r $outfolder/$prefix.$tgt && rm $outfolder/$prefix.bert.$tgt
for k in $(seq 1 $k_repeat)
do
  echo "train k=$k"
  python $root/add_dict.py -i  $infolder/$prefix.bert.$src -o $tmp_dir/bert.aug.$k.$src \
          -d  $dict  --max-pairs $max_pairs --dict-prob  $dict_prob --seed $k  --mode $mode
  python $root/add_dict.py -i  $infolder/$prefix.bert.$tgt -o $tmp_dir/bert.aug.$k.$tgt  \
          -d  $dict --is-rev --max-pairs $max_pairs --dict-prob  $dict_prob --seed $k  --mode $mode

  # aug text
  cat $tmp_dir/bert.aug.$k.$src >> $outfolder/$prefix.bert.$src
  cat $tmp_dir/bert.aug.$k.$tgt >> $outfolder/$prefix.bert.$tgt
  # raw text
  cat $infolder/$prefix.$src >> $outfolder/$prefix.$src
  cat $infolder/$prefix.$tgt >> $outfolder/$prefix.$tgt

done

## valid和test各一份
for prefix in  $VALID $TEST
do
  python $root/add_dict.py -i $infolder/$prefix.bert.$src -o   $outfolder/$prefix.bert.$src  -d  $dict --test  --mode $mode
  python $root/add_dict.py -i $infolder/$prefix.bert.$tgt -o   $outfolder/$prefix.bert.$tgt  -d  $dict --test --is-rev  --mode $mode
  cat $infolder/$prefix.$src > $outfolder/$prefix.$src
  cat $infolder/$prefix.$tgt > $outfolder/$prefix.$tgt
done

# 3.创建双向数据集
echo "stage3: build bidirect data..."
#rm -rf $tmp_dir
bifolder=${outfolder}/bi
if [ ! -d $bifolder ];then
  mkdir -p $bifolder
fi

for prefix in $TRAIN $VALID $TEST
do
  # raw
  cat $outfolder/$prefix.$src $outfolder/$prefix.$tgt >  $bifolder/$prefix.src
  cat $outfolder/$prefix.$tgt $outfolder/$prefix.$src  >  $bifolder/$prefix.tgt
  # bert
  cat $outfolder/$prefix.bert.$src $outfolder/$prefix.bert.$tgt > $bifolder/$prefix.bert.src
  cat $outfolder/$prefix.bert.$tgt $outfolder/$prefix.bert.$src  > $bifolder/$prefix.bert.tgt
done
