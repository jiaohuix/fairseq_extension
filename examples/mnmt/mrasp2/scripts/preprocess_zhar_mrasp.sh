# just process train  [1.split 2.dedup 3.clean(len, langid) 4. learn joint bpe 5.apply to train 6.prepend tag]
#!/bin/sh
#folder=data
folder=$1
outfolder=$2
SRC=zh
TRG=ar
workers=6
domains=("bible" "ccmatrix" "ikcest" "qed" "ted" "tico" "opsub" "un")
tmp_folder=$folder/tmp
if [ ! -d $tmp_folder ];then
    mkdir -p $tmp_folder
fi
if [ ! -d $outfolder ];then
    mkdir -p $outfolder
fi
# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
#src_bpe_operations=18000
#tgt_bpe_operations=18000
bpe_ops=18000
# length filter
lower=2
upper=256
lengRatio=2.5
eng_k=2
valid_num=1000
# lang id ratio
#threshold=0.3

mosesdecoder=./mosesdecoder
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl


# tokenize
echo "--------------1 tokenize --------------"
for mid in  ${domains[@]}
  do
     bash my_tools/cut.sh $workers $folder/train.${mid}.$SRC   $tmp_folder/train.${mid}.$SRC
     cp  $folder/train.${mid}.$TRG   $tmp_folder/train.${mid}.$TRG
  done


# deduplicate
echo  "-------------- 2 deduplicate --------------"
for mid in  ${domains[@]}
  do
     in_lines=$(cat $tmp_folder/train.${mid}.$SRC | wc -l )
     wc $tmp_folder/train.${mid}.$SRC
     python my_tools/deduplicate_pairs.py  $tmp_folder/train.${mid} $SRC $TRG 6
     out_lines=$(cat $tmp_folder/train.${mid}.dedup.$SRC | wc -l )
     rm $tmp_folder/train.${mid}.$SRC && rm $tmp_folder/train.${mid}.$TRG
     echo "[Deduplicate ${mid}]: [$out_lines/$in_lines]. "
  done


echo  "-------------- 3 Length filter --------------"
for mid in  ${domains[@]}
  do
#     python my_tools/check_pair.py $tmp_folder/train.${mid}.clean $SRC $TRG  $upper $lengRatio 0
     perl $CLEAN -ratio $lengRatio $tmp_folder/train.${mid}.dedup $SRC $TRG $tmp_folder/train.${mid}.clean $lower $upper
     out_lines=$(cat $tmp_folder/train.${mid}.clean.$SRC | wc -l )
     rm $tmp_folder/train.${mid}.dedup.$SRC && rm $tmp_folder/train.${mid}.dedup.$TRG
  done

echo  "-------------- 4 Lang ID filter --------------"
# lang id filter，只过滤train
# remove both src and tgt langid error
if [ ! -e lid.176.bin ];then
 wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
fi

for mid in  ${domains[@]}
  do
     # Delete several consecutive English words first
     python my_tools/remove_eng.py $SRC $TRG  $tmp_folder/train.${mid}.clean  $tmp_folder/train.${mid}.clean_eng  $eng_k
     # lang id filter
     in_lines=$(cat $tmp_folder/train.${mid}.clean.$SRC | wc -l )
     echo "---${mid}---"
     python my_tools/data_filter.py --src-lang $SRC --tgt-lang $TRG \
            --in-prefix $tmp_folder/train.${mid}.clean_eng  \
            --out-prefix $tmp_folder/train.${mid}.id --wt

     out_lines=$(cat $tmp_folder/train.${mid}.id.$SRC | wc -l )
     rm $tmp_folder/train.${mid}.clean.$SRC && rm $tmp_folder/train.${mid}.clean.$TRG
     rm $tmp_folder/train.${mid}.clean_eng.$SRC && rm $tmp_folder/train.${mid}.clean_eng.$TRG
     echo "[Lang ID filter ${mid}]: [$out_lines/$in_lines]. "
  done


echo  "-------------- 5 Learn joint bpe --------------"
#files=()
#for mid in  ${domains[@]}
#  do
#    files+=("$tmp_folder/train.${mid}.id.$SRC")
#    files+=("$tmp_folder/train.${mid}.id.$TRG")
#  done
#
#echo ${files[@]}
#cat ${files[@]} | subword-nmt learn-bpe -s $bpe_ops > $outfolder/code.$SRC$TRG
wget https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/pretrain/dataset/codes.bpe.32000 -P $outfolder
mv $outfolder/codes.bpe.32000  $outfolder/code.$SRC$TRG
echo  "-------------- 6 Apply  bpe --------------"
# FILE.lang -> FILE.lang.bpe
for mid in  ${domains[@]}
  do
    bash my_tools/apply_bpe_paral.sh $workers $tmp_folder/train.${mid}.id.$SRC     $outfolder/train.${mid}.$SRC    $outfolder/code.$SRC$TRG
    bash my_tools/apply_bpe_paral.sh $workers $tmp_folder/train.${mid}.id.$TRG     $outfolder/train.${mid}.$TRG    $outfolder/code.$SRC$TRG
    rm $tmp_folder/train.${mid}.id.$SRC && rm $tmp_folder/train.${mid}.id.$TRG
  done


echo  "-------------- 6 Build vocab --------------"
files=()
for mid in  ${domains[@]}
  do
    files+=("$outfolder/train.${mid}.$SRC")
    files+=("$outfolder/train.${mid}.$TRG")
  done
cat ${files[@]} > $tmp_folder/tmp.all
# $tmp_folder/tmp.all -> $tmp_folder/tmp.all.json
bash my_tools/build_dictionary_paral.sh $workers $tmp_folder/tmp.all
python my_tools/json2vocab.py $tmp_folder/tmp.all.json  $outfolder/vocab.$SRC
cp $outfolder/vocab.$SRC $outfolder/vocab.$TRG


echo "Done!"
