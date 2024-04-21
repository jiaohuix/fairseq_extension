# 1 处理好单向py 2 双向添加tag，并且合并双向 data_name_dual 3 合并多向  data_name_multi + bin数据 4 bin 双向和多向的数据，指定字典
data_outdir=train_data

dataset=datasets/ikcest2022
wandb_proj=ikcest2022
# lang_pairs=("zh-ar" "ar-zh" )
# dual_lang_pairs=( "zh-ar" )

lang_pairs=("zh-th" "th-zh" "zh-fr" "fr-zh" "zh-ru" "ru-zh" "zh-ar" "ar-zh")
dual_lang_pairs=( "zh-th" "zh-fr" "zh-ru" "zh-ar" )


# dataset=datasets/iwslt2017
# wandb_proj=iwslt2017
# lang_pairs=( "it-en" "en-it" "ro-en"  "en-ro" "nl-en" "en-nl"  "it-ro" "ro-it")
# lang_pairs=("it-en" "en-it"  "ro-en"  "en-ro" )
# dual_lang_pairs=( "it-en" "ro-en")

# lang_pairs=("it-en" "en-it" )
# dual_lang_pairs=( "it-en" )

# 1 处理好单向py + langid
echo "------------step1: process uni-diretional data with tag------------"

python scripts/prep_data.py -i $dataset -o $data_outdir -v 32000 # merge bpe ops
for lang_pair in "${dual_lang_pairs[@]}"
do
    src_lang=$(echo "$lang_pair" | cut -d'-' -f1)
    tgt_lang=$(echo "$lang_pair" | cut -d'-' -f2)
    reverse_pair=${tgt_lang}-${src_lang}

    # 前向
    train_data_dir=$data_outdir/$wandb_proj/$lang_pair
    echo "add tag ${lang_pair}"
    tag="<to_${tgt_lang}>"
    for prefix in train valid test
    do
        awk '{print "'"$tag"' " $0}' $train_data_dir/$prefix.$src_lang  > $train_data_dir/$prefix.src
        cp $train_data_dir/$prefix.$tgt_lang   $train_data_dir/$prefix.tgt
    done

    # 反向
    train_data_dir=$data_outdir/$wandb_proj/$reverse_pair
    echo "add tag ${reverse_pair}"
    tag="<to_${src_lang}>"
    for prefix in train valid test
    do
        awk '{print "'"$tag"' " $0}' $train_data_dir/$prefix.$tgt_lang  > $train_data_dir/$prefix.src
        cp $train_data_dir/$prefix.$src_lang   $train_data_dir/$prefix.tgt
    done

done

# 2 合并双向/多向
echo "------------step2: process merge data------------"
train_multi_dir=$data_outdir/${wandb_proj}_multi/
mkdir -p $train_multi_dir

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



    # 合并双向数据
    for prefix in train valid test
    do
        cat $train_fwd_dir/$prefix.src $train_bwd_dir/$prefix.src > $train_dual_dir/$prefix.src
        cat $train_fwd_dir/$prefix.tgt $train_bwd_dir/$prefix.tgt > $train_dual_dir/$prefix.tgt

        # 多向
        cat $train_dual_dir/$prefix.src >>  $train_multi_dir/$prefix.src
        cat $train_dual_dir/$prefix.tgt >>  $train_multi_dir/$prefix.tgt
    done
done


# 3 二值化 m21

echo "------------step3: process bin data------------"
data_bin_multi_dir=data-bin/${wandb_proj}_multi
bash scripts/bin.sh  $train_multi_dir  $data_bin_multi_dir  src tgt 1

for lang_pair in "${dual_lang_pairs[@]}"
do

    # 双向
    train_dual_dir=$data_outdir/${wandb_proj}_dual/$lang_pair
    data_bin_dual_dir=data-bin/${wandb_proj}_dual/${lang_pair} 
    echo "dic:  $data_bin_multi_dir/dict.src.txt"
    wc  $data_bin_multi_dir/dict.src.txt
    bash scripts/bin.sh  $train_dual_dir  $data_bin_dual_dir  src tgt 1 $data_bin_multi_dir/dict.src.txt
    
    # 单向
    train_data_dir=$data_outdir/$wandb_proj/$lang_pair
    data_bin_dir=data-bin/${wandb_proj}/${lang_pair} 
    bash scripts/bin.sh  $train_data_dir  $data_bin_dir  src tgt 1  $data_bin_multi_dir/dict.src.txt

    train_data_dir=$data_outdir/$wandb_proj/$reverse_pair
    data_bin_dir=data-bin/${wandb_proj}/${reverse_pair} 
    bash scripts/bin.sh  $train_data_dir  $data_bin_dir  src tgt 1  $data_bin_multi_dir/dict.src.txt
done



