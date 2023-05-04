folder=data_demose/
src=de
tgt=en
# joint embed
paste  $folder/train.$src $folder/train.$tgt > train.all
paste  $folder/valid.$src $folder/valid.$tgt > valid.all
paste  $folder/test.$src $folder/test.$tgt > test.all

bash tasks/embed/embed.sh  train.all  train.bin
bash tasks/embed/embed.sh  valid.all  valid.bin
bash tasks/embed/embed.sh  test.all  test.bin

# search
python ../search/laser/laser_search.py -d   train.bin -q  train.bin -o sim  -k 2 -b 512  --index IVF --nlist 100


# extract from train by idx
## train
cut -f2 sim.idx > sim.top2.idx
python ../search/laser/extract_text_by_idx.py $folder/train.$src  sim.top2.idx  $folder/train.top1.$src
python ../search/laser/extract_text_by_idx.py $folder/train.$tgt  sim.top2.idx  $folder/train.top1.$tgt

# valid
python ../search/laser/laser_search.py -d   train.bin -q  valid.bin -o sim  -k 1 -b 512  --index IVF --nlist 100
cut -f1 sim.idx > sim.top1.idx
python ../search/laser/extract_text_by_idx.py $folder/train.$src   sim.top1.idx  $folder/valid.top1.$src
python ../search/laser/extract_text_by_idx.py $folder/train.$tgt  sim.top1.idx  $folder/valid.top1.$tgt
# test
python ../search/laser/laser_search.py -d   train.bin -q  test.bin -o sim  -k 1 -b 512  --index IVF --nlist 100
cut -f1 sim.idx > sim.top1.idx
python ../search/laser/extract_text_by_idx.py $folder/train.$src   sim.top1.idx  $folder/test.top1.$src
python ../search/laser/extract_text_by_idx.py $folder/train.$tgt  sim.top1.idx  $folder/test.top1.$tgt

# merge for parafuse  (src1,src2,tgt2) -> prefix.bert.de
#ls data_demose/
#test.de  test.en  test.top1.de  test.top1.en  train.de  train.en  train.top1.de  train.top1.en  valid.de  valid.en  valid.top1.de  valid.top1.en
# 合并3
for prefix in train valid test
  do
    src1=$folder/$prefix.$src
    src2=$folder/$prefix.top1.$src
    tgt2=$folder/$prefix.top1.$tgt
    awk 'BEGIN{FS="\n";OFS=" [SEP] "} {getline f2 < "'"$src2"'"; print $0,f2}' $src1 > $folder/tmp
    awk 'BEGIN{FS="\n";OFS=" [SEP] "} {getline f2 < "'"$tgt2"'"; print $0,f2}' $folder/tmp >  $folder/$prefix.bert.$src
    rm  $folder/tmp
  done
# 合并2
folder=data_demose/
src=de
tgt=en
for prefix in train valid test
  do
    src1=$folder/$prefix.$src
    src2=$folder/$prefix.top1.$src
    awk 'BEGIN{FS="\n";OFS=" [SEP] "} {getline f2 < "'"$src2"'"; print $0,f2}' $src1 > $folder/$prefix.bert.$src
  done


# merge3 tgt ： tgt tgt2 src2
  folder=data_demose/
  src=de
  tgt=en
  for prefix in train valid test
    do
      tgt1=$folder/$prefix.$tgt
      tgt2=$folder/$prefix.top1.$tgt
      src2=$folder/$prefix.top1.$src
      awk 'BEGIN{FS="\n";OFS=" [SEP] "} {getline f2 < "'"$tgt2"'"; print $0,f2}' $tgt1> $folder/tmp
      awk 'BEGIN{FS="\n";OFS=" [SEP] "} {getline f2 < "'"$src2"'"; print $0,f2}' $folder/tmp >  $folder/$prefix.bert.$tgt
      rm  $folder/tmp
    done
