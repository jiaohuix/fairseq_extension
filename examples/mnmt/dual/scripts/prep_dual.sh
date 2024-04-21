dataset=datasets/iwslt2017
wandb_proj=iwslt2017
data_outdir=train_data
lang_pairs=( "it-en" "en-it" "ro-en"  "en-ro" "nl-en" "en-nl"  "it-ro" "ro-it")
dual_lang_pairs=( "it-en" "ro-en" "nl-en" "it-ro" )

# lang_pairs=("it-en" "en-it" )
# dual_lang_pairs=( "it-en" )


# dataset=datasets/ikcest2022
# wandb_proj=ikcest2022
# data_outdir=train_data
# lang_pairs=("zh-th" "th-zh" "zh-fr" "fr-zh" "zh-ru" "ru-zh" "zh-ar" "ar-zh")
# dual_lang_pairs=( "zh-th" "zh-fr" "zh-ru" "zh-ar" )


# 1 单向数据处理，添加src的tag，并且bin
echo "------------step1: process uni-diretional data------------"
for lang_pair in "${lang_pairs[@]}"
do
    src_lang=$(echo "$lang_pair" | cut -d'-' -f1)
    tgt_lang=$(echo "$lang_pair" | cut -d'-' -f2)

    train_data_dir=$data_outdir/$wandb_proj/$lang_pair
    data_bin_dir=data-bin/${wandb_proj}/${lang_pair} # 默认单向

    # 首先处理好单向数据
    echo "prep ${lang_pair}"
    python scripts/prep_data.py -i $dataset -o $data_outdir -v 10000 -l $lang_pair

    # 源端添加tag
    echo "add tag ${lang_pair}"
    tag="<to_${tgt_lang}>"
    for prefix in train valid test
    do
        awk '{print "'"$tag"' " $0}' $train_data_dir/$prefix.$src_lang  > $train_data_dir/$prefix.src
        cp $train_data_dir/$prefix.$tgt_lang   $train_data_dir/$prefix.tgt
    done


done

#2  合并双向，并且bin。
echo "------------step2: process dual-diretional data------------"
for lang_pair in "${dual_lang_pairs[@]}"
do
    echo "prep dual ${lang_pair}"
    src_lang=$(echo "$lang_pair" | cut -d'-' -f1)
    tgt_lang=$(echo "$lang_pair" | cut -d'-' -f2)
    reverse_pair=${tgt_lang}-${src_lang}
    # 训练集
    train_fwd_dir=$data_outdir/$wandb_proj/$lang_pair
    train_bwd_dir=$data_outdir/$wandb_proj/$reverse_pair
    train_dual_dir=$data_outdir/${wandb_proj}_dual/$lang_pair
    mkdir -p $train_dual_dir
    # 二值目录
    data_bin_dual_dir=data-bin/${wandb_proj}_dual/${lang_pair} 

    # 合并数据
    for prefix in train valid test
    do
        cat $train_fwd_dir/$prefix.src $train_bwd_dir/$prefix.src > $train_dual_dir/$prefix.src
        cat $train_fwd_dir/$prefix.tgt $train_bwd_dir/$prefix.tgt > $train_dual_dir/$prefix.tgt
    done

    # todo：sample valid和test/ 或者减少eval的频率 不要采样了，没多少，倒不如减少eval频率

    # 二值化双向
    bash scripts/bin.sh  $train_dual_dir  $data_bin_dual_dir  src tgt 1

    # 二值化单向(复用dual的字典)
    echo "dic :$data_bin_dual_dir/dict.src.txt"
    wc $data_bin_dual_dir/dict.src.txt
    train_data_dir=$data_outdir/$wandb_proj/$lang_pair
    data_bin_dir=data-bin/${wandb_proj}/${lang_pair} # 默认单向
    bash scripts/bin.sh  $train_data_dir  $data_bin_dir  src tgt 1  $data_bin_dual_dir/dict.src.txt

    train_data_dir=$data_outdir/$wandb_proj/$reverse_pair
    data_bin_dir=data-bin/${wandb_proj}/${reverse_pair} # 默认单向
    bash scripts/bin.sh  $train_data_dir  $data_bin_dir  src tgt 1  $data_bin_dual_dir/dict.src.txt


done