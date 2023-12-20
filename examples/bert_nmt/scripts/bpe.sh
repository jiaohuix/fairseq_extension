#!/usr/bin/env bash
# params: src tgt infolder outfolder bpe_ops joint_bpe
# 还要指定分词脚本，如moses、jieba、thai（应该写个pkg专门处理分词，都用py）
# 本脚本不再额外使用moses，仅仅做bpe

if [ $# -lt 2 ];then
  echo "usage: bash $0 <infolder> <outfolder> <src=zh> <tgt=en> <bpe_ops=10000> <joint_bpe=1>(0/1)  "
  exit
fi


infolder=$1
outfolder=$2
src=${3:-"zh"}
tgt=${4:-"en"}
bpe_ops=${5:-"10000"}
joint_bpe=${6:-"1"}

lang=${src}-${tgt}
tmp=$outfolder/tmp

mkdir -p  $tmp $outfolder

# learn bpe
SRC_CODES=$outfolder/codes.${bpe_ops}.${src}
TGT_CODES=$outfolder/codes.${bpe_ops}.${tgt}
if [ $joint_bpe == 0 ];then
  echo "learn seperate bpe..."
  subword-nmt learn-bpe -s $bpe_ops < ${infolder}/train.${src}  > $SRC_CODES
  subword-nmt learn-bpe -s $bpe_ops < ${infolder}/train.${tgt}  > $TGT_CODES
elif [ $joint_bpe == 1 ]; then
  echo "learn joint bpe..."
  cat ${infolder}/train.${src} ${infolder}/train.${tgt} > ${tmp}/train
  subword-nmt learn-bpe -s $bpe_ops < ${tmp}/train  > $SRC_CODES
  cp $SRC_CODES $TGT_CODES
fi


for prefix in train valid test
do
  echo "apply bpe to ${prefix}..."
  subword-nmt apply-bpe -c  $SRC_CODES < ${infolder}/${prefix}.${src} >  ${outfolder}/${prefix}.${src}
  subword-nmt apply-bpe -c  $SRC_CODES < ${infolder}/${prefix}.${tgt} >  ${outfolder}/${prefix}.${tgt}
done

# build vocab (no need)


rm -r $tmp