# dataset=miugod/ikcest2022
# wandb_proj=ikcest22
# lang_pairs=("zh-th" "th-zh" "zh-fr" "fr-zh" "zh-ru" "ru-zh" "zh-ar" "ar-zh")


dataset=iwslt2017
wandb_proj=iwslt2017
lang_pairs=( "it-en" "en-it" "ro-en"  "en-ro" "nl-en" "en-nl"  "it-ro" "ro-it")


export HF_ENDPOINT=https://hf-mirror.com
data_outdir=train_data
report_dir=ckpt/${wandb_proj}/report
mkdir -p $report_dir # 添加bleu结果， 预测文本，ref
echo "lang_pair,bleu,ref_len,hypo_len" > $report_dir/bleu.txt


for lang_pair in "${lang_pairs[@]}"
do
    src_lang=$(echo "$lang_pair" | cut -d'-' -f1)
    tgt_lang=$(echo "$lang_pair" | cut -d'-' -f2)

    echo "Source language: $src_lang"
    echo "Target language: $tgt_lang"

    data_bin_dir=data-bin/${wandb_proj}/${lang_pair}
    ckpt_dir=ckpt/${wandb_proj}/${lang_pair}

    # 检查数据目录是否存在
    if [ ! -d $data_bin_dir ]; then
        echo "Directory $data_bin_dir not exists."
        python scripts/prep_data.py -i $dataset -o $data_outdir -v 10000 -l $lang_pair
        bash scripts/bin.sh  $data_outdir/${wandb_proj}/$lang_pair $data_bin_dir $src_lang $tgt_lang 1
    fi

    bash scripts/train.sh $src_lang $tgt_lang $data_bin_dir $ckpt_dir $wandb_proj 
    bash scripts/eval.sh $src_lang $tgt_lang $data_bin_dir $ckpt_dir/checkpoint_best.pt >  $ckpt_dir/gen.txt


    if [[ -e $ckpt_dir/checkpoint_best.pt ]]; then
        hypo_file=$report_dir/${src_lang}_${tgt_lang}.rst
        cat $ckpt_dir/gen.txt | grep -P "^D" | sort -V | cut -f 3- > $hypo_file
        score=$(tail "$ckpt_dir/gen.txt" | sed "s/,//" | grep "BLEU" | cut -d' ' -f7)

        ref_len=$(wc -l < $data_outdir/${wandb_proj}/$lang_pair/test.${tgt_lang})
        hypo_len=$(wc -l < $hypo_file)
        echo "ref_len: $ref_len"
        echo "hypo_len: $hypo_len"
        echo "$lang_pair,$score,$ref_len,$hypo_len" >> $report_dir/bleu.txt


    fi
done

