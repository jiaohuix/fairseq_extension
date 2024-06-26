# 训练一种语言
# dataset=datasets/iwslt2017
# wandb_proj=iwslt2017
# data_outdir=train_data
# lang_pairs=( "it-en" "en-it" "ro-en"  "en-ro" "nl-en" "en-nl"  "it-ro" "ro-it")
# dual_lang_pairs=( "it-en" "ro-en" "nl-en"  "it-ro" )

dataset=datasets/ikcest2022
data_outdir=train_data
wandb_proj=ikcest2022
lang_pairs=("fr-zh" "zh-fr" "ru-zh" "zh-ru"  "th-zh" "zh-th" "ar-zh"  "zh-ar" )
dual_lang_pairs=("zh-fr" "zh-ru" "zh-th" "zh-ar" )



echo "-------step1.1: train multi---------"
report_dir=ckpt/${wandb_proj}_multi/report
data_bin_dir=data-bin/${wandb_proj}_multi/
ckpt_dir=ckpt/${wandb_proj}_multi/
mkdir -p $ckpt_dir $report_dir
echo "lang_pair,bleu,ref_len,hypo_len" > $report_dir/bleu.txt
# bash scripts/train_6layer.sh src tgt $data_bin_dir $ckpt_dir $wandb_proj

echo "-------step1.2: eval multi---------"
# eval each pair
# for lang_pair in "${lang_pairs[@]}"
# do
#     src_lang=$(echo "$lang_pair" | cut -d'-' -f1)
#     tgt_lang=$(echo "$lang_pair" | cut -d'-' -f2)
#     tgt_lang_id="LANG_TOK_"`echo "${tgt_lang} " | tr '[a-z]' '[A-Z]'`

#     echo "Source language: $src_lang"
#     echo "Target language: $tgt_lang"

#     bash scripts/eval.sh $src_lang $tgt_lang data-bin/${wandb_proj}/${lang_pair}  $ckpt_dir/checkpoint_best.pt >  $ckpt_dir/gen_${lang_pair}.txt

#     # report
#     hypo_file=$report_dir/${src_lang}_${tgt_lang}.rst
#     cat $ckpt_dir/gen_${lang_pair}.txt | grep -P "^D" | sort -V | cut -f 3-  | sed "s/$tgt_lang_id //g"   > $hypo_file
#     sed -i "s/<unk>//g" $hypo_file
#     score=$(tail "$ckpt_dir/gen_${lang_pair}.txt" | sed "s/,//" | grep "BLEU" | cut -d' ' -f7)
#     # 获取长度
#     ref_len=$(wc -l < $data_outdir/${wandb_proj}/$lang_pair/test.${tgt_lang})
#     hypo_len=$(wc -l < $hypo_file)
#     echo "$lang_pair,$score,$ref_len,$hypo_len" >> $report_dir/bleu.txt

# done


echo "-------step2: dual finetune---------"
ckpt=ckpt/${wandb_proj}_multi/checkpoint_best.pt
report_dir=ckpt/${wandb_proj}_dual/report
mkdir -p  $report_dir

for lang_pair in "${dual_lang_pairs[@]}"
do
    src_lang=$(echo "$lang_pair" | cut -d'-' -f1)
    tgt_lang=$(echo "$lang_pair" | cut -d'-' -f2)
    src_lang_id="LANG_TOK_"`echo "${src_lang} " | tr '[a-z]' '[A-Z]'`
    tgt_lang_id="LANG_TOK_"`echo "${tgt_lang} " | tr '[a-z]' '[A-Z]'`

    reverse_pair=${tgt_lang}-${src_lang}

    echo "Source language: $src_lang"
    echo "Target language: $tgt_lang"

    data_bin_dir=data-bin/${wandb_proj}_dual/${lang_pair}
    ckpt_dir=ckpt/${wandb_proj}_dual/${lang_pair}
    mkdir -p $ckpt_dir

    bash scripts/train_6layer_ft.sh $src_lang $tgt_lang $data_bin_dir $ckpt_dir $wandb_proj $ckpt

    # todo : 评估两个语向

    bash scripts/eval.sh $src_lang $tgt_lang data-bin/${wandb_proj}/${lang_pair}  $ckpt_dir/checkpoint_best.pt >  $ckpt_dir/gen_${lang_pair}.txt
    bash scripts/eval.sh $tgt_lang $src_lang data-bin/${wandb_proj}/${reverse_pair}  $ckpt_dir/checkpoint_best.pt >  $ckpt_dir/gen_${reverse_pair}.txt

    if [[ -e $ckpt_dir/checkpoint_best.pt ]]; then
        # fwd
        hypo_file=$report_dir/${src_lang}_${tgt_lang}.rst
        cat $ckpt_dir/gen_${lang_pair}.txt | grep -P "^D" | sort -V | cut -f 3- | sed "s/$tgt_lang_id //g" > $hypo_file
        sed -i "s/<unk>//g" $hypo_file

        score=$(tail "$ckpt_dir/gen_${lang_pair}.txt" | sed "s/,//" | grep "BLEU" | cut -d' ' -f7)
        # 获取长度
        ref_len=$(wc -l < $data_outdir/${wandb_proj}/$lang_pair/test.${tgt_lang})
        hypo_len=$(wc -l < $hypo_file)
        echo "$lang_pair,$score,$ref_len,$hypo_len" >> $report_dir/bleu.txt

        # bwd
        hypo_file=$report_dir/${tgt_lang}_${src_lang}.rst
        cat $ckpt_dir/gen_${reverse_pair}.txt | grep -P "^D" | sort -V | cut -f 3- | sed "s/$src_lang_id //g" > $hypo_file
        sed -i "s/<unk>//g" $hypo_file

        score=$(tail "$ckpt_dir/gen_${reverse_pair}.txt" | sed "s/,//" | grep "BLEU" | cut -d' ' -f7)
        ref_len=$(wc -l < $data_outdir/${wandb_proj}/$lang_pair/test.${tgt_lang})
        hypo_len=$(wc -l < $hypo_file)
        echo "$reverse_pair,$score,$ref_len,$hypo_len" >> $report_dir/bleu.txt

    fi
done


#
#echo "-------step3: uni finetune---------"
#
#report_dir=ckpt/${wandb_proj}/report
#mkdir -p $report_dir # 添加bleu结果， 预测文本，ref
#echo "lang_pair,bleu,ref_len,hypo_len" > $report_dir/bleu.txt
#
#for lang_pair in "${dual_lang_pairs[@]}"
#do
#    src_lang=$(echo "$lang_pair" | cut -d'-' -f1)
#    tgt_lang=$(echo "$lang_pair" | cut -d'-' -f2)
#    reverse_pair=${tgt_lang}-${src_lang}
#
#    echo "Source language: $src_lang"
#    echo "Target language: $tgt_lang"
#
#    ckpt=ckpt/${wandb_proj}_multi/checkpoint_best.pt
#    data_bin_fwd=data-bin/${wandb_proj}/${lang_pair}
#    data_bin_bwd=data-bin/${wandb_proj}/${reverse_pair}
#    ckpt_dir_fwd=ckpt/${wandb_proj}/${lang_pair}
#    ckpt_dir_bwd=ckpt/${wandb_proj}/${reverse_pair}
#    mkdir -p $ckpt_dir_fwd $ckpt_dir_bwd
#
#
#
#    bash scripts/train_ft.sh uni.$src_lang $tgt_lang $data_bin_fwd $ckpt_dir_fwd  $ckpt $wandb_proj
#    bash scripts/train_ft.sh uni.$tgt_lang $src_lang  $data_bin_bwd $ckpt_dir_bwd  $ckpt $wandb_proj
#
#
#    bash scripts/eval.sh $src_lang $tgt_lang $data_bin_fwd $ckpt_dir_fwd/checkpoint_best.pt >  $ckpt_dir_fwd/gen_${lang_pair}.txt
#    bash scripts/eval.sh $tgt_lang $src_lang $data_bin_bwd $ckpt_dir_bwd/checkpoint_best.pt >  $ckpt_dir_bwd/gen_${reverse_pair}.txt
#
#    if [[ -e $ckpt_dir_fwd/checkpoint_best.pt ]]; then
#        # fwd
#        hypo_file=$report_dir/${src_lang}_${tgt_lang}.rst
#        cat $ckpt_dir_fwd/gen_${lang_pair}.txt | grep -P "^D" | sort -V | cut -f 3- > $hypo_file
#        score=$(tail "$ckpt_dir_fwd/gen_${lang_pair}.txt" | sed "s/,//" | grep "BLEU" | cut -d' ' -f7)
#        # 获取长度
#        ref_len=$(wc -l < $data_outdir/${wandb_proj}/$lang_pair/test.${tgt_lang})
#        hypo_len=$(wc -l < $hypo_file)
#        echo "$lang_pair,$score,$ref_len,$hypo_len" >> $report_dir/bleu.txt
#    fi
#
#    if [[ -e $ckpt_dir_bwd/checkpoint_best.pt ]]; then
#        # bwd
#        hypo_file=$report_dir/${tgt_lang}_${src_lang}.rst
#        cat $ckpt_dir_bwd/gen_${reverse_pair}.txt | grep -P "^D" | sort -V | cut -f 3- > $hypo_file
#        score=$(tail "$ckpt_dir_bwd/gen_${reverse_pair}.txt" | sed "s/,//" | grep "BLEU" | cut -d' ' -f7)
#        ref_len=$(wc -l < $data_outdir/${wandb_proj}/$reverse_pair/test.${tgt_lang})
#        hypo_len=$(wc -l < $hypo_file)
#        echo "$reverse_pair,$score,$ref_len,$hypo_len" >> $report_dir/bleu.txt
#    fi
#
#done
