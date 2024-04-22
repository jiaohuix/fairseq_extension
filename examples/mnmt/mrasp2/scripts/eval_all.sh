ckpt=${1:-"ckpt/6e6d_no_mono.pt"}
ckpt_dir=${2:-"ckpt/"}
dataset=datasets/ikcest2022
wandb_proj=ikcest2022
lang_pairs=("zh-th" "th-zh" "zh-fr" "fr-zh" "zh-ru" "ru-zh" "zh-ar" "ar-zh")
#dual_lang_pairs=( "zh-th" "zh-fr" "zh-ru" "zh-ar" )

report_dir=ckpt/${wandb_proj}_multi/report
#ckpt_dir=ckpt/${wandb_proj}_multi/
mkdir -p $report_dir

# eval each pair
for lang_pair in "${lang_pairs[@]}"
do
    src_lang=$(echo "$lang_pair" | cut -d'-' -f1)
    tgt_lang=$(echo "$lang_pair" | cut -d'-' -f2)
    echo "Source language: $src_lang"
    echo "Target language: $tgt_lang"

    bash scripts/eval.sh $src_lang $tgt_lang data-bin/${wandb_proj}/${lang_pair} $ckpt > $ckpt_dir/gen_${lang_pair}.txt

    # report
    hypo_file=$report_dir/${src_lang}_${tgt_lang}.rst
    cat $ckpt_dir/gen_${lang_pair}.txt | grep -P "^D" | sort -V | cut -f 3- > $hypo_file
    score=$(tail "$ckpt_dir/gen_${lang_pair}.txt" | sed "s/,//" | grep "BLEU" | cut -d' ' -f7)
    # 获取长度
    ref_len=$(wc -l < $data_outdir/${wandb_proj}/$lang_pair/test.${tgt_lang})
    hypo_len=$(wc -l < $hypo_file)
    echo "$lang_pair,$score,$ref_len,$hypo_len" >> $report_dir/bleu.txt

done
