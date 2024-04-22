echo "bash prep.sh <infolder>"
src=zh
tgt=ar
valid_num=1000
infolder=$1
outfolder=datasets/bpe/${src}_${tgt}
if [ ! -d $outfolder ];then
    mkdir -p $outfolder
fi
wget https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2021/mrasp2/bpe_vocab -P $outfolder
wget https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/pretrain/dataset/codes.bpe.32000 -P $outfolder

# train dev split
cp $infolder/train.$src $infolder/train.tmp.$src && cp $infolder/train.$tgt $infolder/train.tmp.$tgt
python nmt_data_tools/my_tools/train_dev_split.py $src $tgt $infolder/train.tmp $infolder/ $valid_num # train.lang/ dev.lang
mv $infolder/dev.$src  $infolder/valid.$src && mv $infolder/dev.$tgt  $infolder/valid.$tgt


# tokenize
for prefix in train valid test.${src}_${tgt}
    do
        python nmt_data_tools/my_tools/cut_multi.py  $infolder/$prefix.$src  $infolder/$prefix.tok.$src 4 zh
    done

# learn bpe

# apply bpe
for prefix in train valid test.${src}_${tgt} test.${tgt}_${src}
    do
        if [ -e $infolder/$prefix.tok.$src ];then
            subword-nmt apply-bpe -c $outfolder/codes.bpe.32000 < $infolder/$prefix.tok.$src >  $outfolder/$prefix.$src
            sed -i "s/^/LANG_TOK_ZH /g" $outfolder/$prefix.$src

        fi

        if [ -e $infolder/$prefix.$tgt ];then
          subword-nmt apply-bpe -c $outfolder/codes.bpe.32000 < $infolder/$prefix.$tgt >  $outfolder/$prefix.$tgt
          sed -i "s/^/LANG_TOK_AR /g" $outfolder/$prefix.$tgt
        fi
    done

mv $outfolder/test.${src}_${tgt}.$src $outfolder/test.$src
mv $outfolder/test.${tgt}_${src}.$tgt $outfolder/test.$tgt

echo "all done!"

