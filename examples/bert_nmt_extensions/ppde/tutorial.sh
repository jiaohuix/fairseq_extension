# install
git clone https://github.com/clab/fast_align.git
cd fast_align
mkdir build
cd build
cmake ..
make
pip install fasttext wget

# train
src=$1
tgt=$2
folder=$3
prefix=${4:-"train"}

paste $folder/$prefix.$src $folder/$prefix.$tgt  | sed 's/ *\t */ ||| /g' > $folder/$prefix.${src}-${tgt}
echo "train forward model..."
./fast_align/build/fast_align -i $folder/$prefix.${src}-${tgt} -d -v -o -p $folder/$prefix.fwd_params > $folder/$prefix.fwd_align
echo "train backward model..."
./fast_align/build/fast_align -i $folder/$prefix.${src}-${tgt}  -r -d -v -o -p $folder/$prefix.rev_params > $folder/$prefix.rev_align
echo "generates asymmetric alignments..."
./fast_align/build/atools  -i $folder/$prefix.fwd_align -j $folder/$prefix.rev_align -c grow-diag-final-and > $folder/$prefix.sym_align
echo "done"

# 抽取词典
python nmt_data_tools/my_tools/extract_dict.py align_ende/train.de-en align_ende/train.sym_align dict.de-en.txt 50000

# 过滤词典
git clone https://github.com/stopwords-iso/stopwords-en.git
git clone https://github.com/stopwords-iso/stopwords-de.git
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
#wget https://dl.fbaipublicfiles.com/arrival/dictionaries/de-en.txt

# 对源端添加词典约束
python add_dict.py -i iwslt14.tokenized.de-en/train.bert.de -o iwslt14.tokenized.de-en/train.bert.aug.de -d dict.de-en.txt

# 批量添加

# 添加n份

# 双向训练

# 二值化
ptm=miugod/mbert_trim_ende_mlm
#bash bin.sh bert_aug/ iwslt14_deen_trim_augk2 miugod/mbert_trim_ende_mlm

cp bert_nmt_extensions/ppde/add_dict.sh add.sh
cp bert_nmt_extensions/ppde/bin.sh bin.sh
ptm=miugod/mbert_trim_ende_mlm
bash add.sh iwslt14.tokenized.de-en/ data_aug/bert_aug_replace dict.de-en.txt 2 replace
bash add.sh iwslt14.tokenized.de-en/ data_aug/bert_aug_insert  dict.de-en.txt 2 insert
bash add.sh iwslt14.tokenized.de-en/ data_aug/bert_aug_inserte  dict.de-en.txt 2 inserte
bash add.sh iwslt14.tokenized.de-en/ data_aug/bert_aug_concat  dict.de-en.txt 2 concat
bash add.sh iwslt14.tokenized.de-en/ data_aug/bert_aug_biconcat  dict.de-en.txt 2 biconcat

bash bin.sh data_aug/bert_aug_replace/ data-bin/bert_aug_replace miugod/mbert_trim_ende_mlm
bash bin.sh data_aug/bert_aug_insert/ data-bin/bert_aug_insert miugod/mbert_trim_ende_mlm
bash bin.sh data_aug/bert_aug_inserte/ data-bin/bert_aug_inserte miugod/mbert_trim_ende_mlm
bash bin.sh data_aug/bert_aug_concat/ data-bin/bert_aug_concat miugod/mbert_trim_ende_mlm
bash bin.sh data_aug/bert_aug_biconcat/ data-bin/bert_aug_biconcat miugod/mbert_trim_ende_mlm
