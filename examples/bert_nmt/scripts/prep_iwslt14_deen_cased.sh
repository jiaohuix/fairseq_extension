#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh


#echo 'Cloning Subword nmt_data_tools repository ...'
#git clone https://github.com/MiuGod0126/nmt_data_tools.git
pip install subword-nmt

SCRIPTS=nmt_data_tools/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000

URL="http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz"
GZ=de-en.tgz

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=de
tgt=en
lang=de-en
prep=iwslt14.tokenized.de-en
tmp=$prep/tmp
orig=orig
cased_folder=$prep/cased

mkdir -p $orig $tmp $prep $cased_folder

echo "Downloading data from ${URL}..."
cd $orig
if [ ! -e $GZ ];then
    wget "$URL"
fi

if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf $GZ
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat $orig/$lang/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done
# tmp/tok是大写分词

# 下面是长度过滤 clean
perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175
# # 保存cased
# mv $tmp/train.tags.$lang.clean.$src $cased_folder/train.$src
# mv $tmp/train.tags.$lang.clean.$tgt $cased_folder/train.$tgt

# train小写
for l in $src $tgt; do
    perl $LC < $tmp/train.tags.$lang.clean.$l > $tmp/train.tags.$lang.$l
done

# 抽取valid test，全小写了
echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT14.TED*.$l.xml`; do
    fname=${o##*/}
    f=$tmp/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -l $l > $f # cased
    # perl $TOKENIZER -threads 8 -l $l | \
    # perl $LC > $f
    echo ""
    done
done

# 创建train valid和test
echo "creating train, valid, test..."
for l in $src $tgt; do
    # 划分训练集和验证集，1/23
    awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/valid.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/train.$l
    # 保存cased的训练和验证集
    awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.$lang.clean.$l > $cased_folder/valid.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.$lang.clean.$l > $cased_folder/train.$l

    # cased test
    cat $tmp/IWSLT14.TED.dev2010.de-en.$l \
        $tmp/IWSLT14.TEDX.dev2012.de-en.$l \
        $tmp/IWSLT14.TED.tst2010.de-en.$l \
        $tmp/IWSLT14.TED.tst2011.de-en.$l \
        $tmp/IWSLT14.TED.tst2012.de-en.$l \
        > $cased_folder/test.$l
        # > $tmp/test.$l

    # uncased test
    perl $LC < $cased_folder/test.$l > $tmp/test.$l

done

TRAIN=$tmp/train.en-de
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
subword-nmt learn-bpe -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        subword-nmt apply-bpe -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done
