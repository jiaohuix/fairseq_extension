#!/usr/bin/env bash
#pip install subword-nmt
BPE_TOKENS=10000
URL="https://github.com/stefan-it/nmt-en-vi/raw/master/data/"
FILES=( "train-en-vi.tgz" "dev-2012-en-vi.tgz" "test-2013-en-vi.tgz" )
src=vi
tgt=en
lang=${src}-${tgt}
prep=iwslt15.tokenized.vi-en
tmp=$prep/tmp
orig=orig
joint_bpe=${1:-"1"}

mkdir -p $orig $tmp $prep

echo "Downloading data from ${URL}..."
cd $orig
for file in ${FILES[@]}
do
  if [ ! -e $file ];then
      wget "${URL}/${file}"  && tar -xzvf $file
  fi
done

# rename
for lang in $src $tgt
do
  mv tst2012.${lang} valid.${lang}
  mv tst2013.${lang} test.${lang}
done
cd ..


# learn bpe
SRC_CODES=$prep/codes.${BPE_TOKENS}.${src}
TGT_CODES=$prep/codes.${BPE_TOKENS}.${tgt}
if [ $joint_bpe == 0 ];then
  echo "learn seperate bpe..."
  subword-nmt learn-bpe -s $BPE_TOKENS < ${orig}/train.${src}  > $SRC_CODES
  subword-nmt learn-bpe -s $BPE_TOKENS < ${orig}/train.${tgt}  > $TGT_CODES
elif [ $joint_bpe == 1 ]; then
  echo "learn joint bpe..."
  cat ${orig}/train.${src} ${orig}/train.${tgt} > ${tmp}/train
  subword-nmt learn-bpe -s $BPE_TOKENS < ${tmp}/train  > $SRC_CODES
  cp $SRC_CODES $TGT_CODES
fi



for prefix in train valid test
do
  echo "apply bpe to ${prefix}..."
  subword-nmt apply-bpe -c  $SRC_CODES < ${orig}/${prefix}.${src} >  ${prep}/${prefix}.${src}
  subword-nmt apply-bpe -c  $SRC_CODES < ${orig}/${prefix}.${tgt} >  ${prep}/${prefix}.${tgt}
done

# build vocab (no need)


rm -r $tmp