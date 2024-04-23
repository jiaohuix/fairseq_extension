dataset=datasets/ikcest2022
wandb_proj=ikcest2022
data_outdir=train_data
lang_pairs=("fr-zh" "zh-fr" "ru-zh" "zh-ru"  "th-zh" "zh-th" "ar-zh"  "zh-ar" )

ckpt=ckpt/${wandb_proj}_multi/
report_dir=ckpt/${wandb_proj}_multi/report
mkdir -p $report_dir

# eval each pair
for lang_pair in "${lang_pairs[@]}"
do
    src_lang=$(echo "$lang_pair" | cut -d'-' -f1)
    tgt_lang=$(echo "$lang_pair" | cut -d'-' -f2)
    tgt_lang_id="LANG_TOK_"`echo "${tgt_lang} " | tr '[a-z]' '[A-Z]'`
    echo "Source language: $src_lang"
    echo "Target language: $tgt_lang"

    bash scripts/eval.sh $src_lang $tgt_lang data-bin/${wandb_proj}/${lang_pair} $ckpt/checkpoint_best.pt > $ckpt_dir/gen_${lang_pair}.txt

    # report
    hypo_file=$report_dir/${src_lang}_${tgt_lang}.rst
    cat $ckpt_dir/gen_${lang_pair}.txt | grep -P "^D" | sort -V | cut -f 3-  | sed "s/$tgt_lang_id //g"   > $hypo_file
    sed -i "s/<unk>//g" $hypo_file
    score=$(tail "$ckpt_dir/gen_${lang_pair}.txt" | sed "s/,//" | grep "BLEU" | cut -d' ' -f7)
    # 获取长度
    ref_len=$(wc -l < $data_outdir/${wandb_proj}/$lang_pair/test.${tgt_lang})
    hypo_len=$(wc -l < $hypo_file)
    echo "$lang_pair,$score,$ref_len,$hypo_len" >> $report_dir/bleu.txt

done

