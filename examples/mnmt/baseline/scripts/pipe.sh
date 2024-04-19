# # 1 完成一种语言数据、模型训练、评估

# dataset=miugod/ikcest2022 
# data_outdir=train_data
# src_lang=zh
# tgt_lang=fr
# lang_pair=${src_lang}-${tgt_lang}
# data_bin_dir=data-bin/ikcest22-${lang_pair}
# ckpt_dir=ckpt/ikcest22/${lang_pair}

# export HF_ENDPOINT=https://hf-mirror.com
# python scripts/prep_data.py -i $dataset -o $data_outdir -v 10000 -l $lang_pair
# bash scripts/bin.sh  $data_outdir/ikcest2022/$lang_pair $data_bin_dir $src_lang $tgt_lang 1
# bash scripts/train.sh $src_lang $tgt_lang $data_bin_dir $ckpt_dir
# bash scripts/eval.sh $src_lang $tgt_lang $data_bin_dir $ckpt_dir/checkpoint_best.pt >  $ckpt_dir/gen.txt


# #eg:
# # python scripts/prep_data.py -i miugod/ikcest2022 -o train_data/ -v 10000 -l zh-th
# bash scripts/bin.sh train_data/ikcest2022/zh-th data-bin/ikcest22-zh-th zh th 1
# bash scripts/train.sh zh th data-bin/ikcest22-zh-th  ckpt/ikcest22/zh-th ikcest22
# bash scripts/eval.sh zh th data-bin/ikcest22-zh-th/ ckpt/ikcest22/zh-th/checkpoint_best.pt > ckpt/ikcest22/zh-th/gen.txt

# # todo: decode+eval


export HF_ENDPOINT=https://hf-mirror.com
dataset=miugod/ikcest2022
wandb_proj=ikcest22
data_outdir=train_data

lang_pairs=("zh-th" "th-zh" "zh-fr" "fr-zh" "zh-ru" "ru-zh" "zh-ar" "ar-zh")

for lang_pair in "${lang_pairs[@]}"
do
    src_lang=$(echo "$lang_pair" | cut -d'-' -f1)
    tgt_lang=$(echo "$lang_pair" | cut -d'-' -f2)

    echo "Source language: $src_lang"
    echo "Target language: $tgt_lang"

    data_bin_dir=data-bin/ikcest22-${lang_pair}
    ckpt_dir=ckpt/ikcest22/${lang_pair}

    python scripts/prep_data.py -i $dataset -o $data_outdir -v 10000 -l $lang_pair
    bash scripts/bin.sh  $data_outdir/ikcest2022/$lang_pair $data_bin_dir $src_lang $tgt_lang 1
    bash scripts/train.sh $src_lang $tgt_lang $data_bin_dir $ckpt_dir $wandb_proj # wandb_proj=ikcest22
    bash scripts/eval.sh $src_lang $tgt_lang $data_bin_dir $ckpt_dir/checkpoint_best.pt >  $ckpt_dir/gen.txt
done




# 3 两个数据集

dataset=iwslt17
wandb_proj=iwslt17
data_outdir=train_data

lang_pairs=("en-it" "it-en" "en-ro" "ro-en" "en-nl" "nl-en" "it-ro" "ro-it")

for lang_pair in "${lang_pairs[@]}"
do
    src_lang=$(echo "$lang_pair" | cut -d'-' -f1)
    tgt_lang=$(echo "$lang_pair" | cut -d'-' -f2)

    echo "Source language: $src_lang"
    echo "Target language: $tgt_lang"

    data_bin_dir=data-bin/ikcest22-${lang_pair}
    ckpt_dir=ckpt/ikcest22/${lang_pair}

    python scripts/prep_data.py -i $dataset -o $data_outdir -v 10000 -l $lang_pair
    bash scripts/bin.sh  $data_outdir/ikcest2022/$lang_pair $data_bin_dir $src_lang $tgt_lang 1
    bash scripts/train.sh $src_lang $tgt_lang $data_bin_dir $ckpt_dir $wandb_proj # wandb_proj=ikcest22
    bash scripts/eval.sh $src_lang $tgt_lang $data_bin_dir $ckpt_dir/checkpoint_best.pt >  $ckpt_dir/gen.txt
done



