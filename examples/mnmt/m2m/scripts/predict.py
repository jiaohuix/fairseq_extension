import time
import argparse
import logging
import json
import os

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

logger = logging.getLogger("traslation")

def read_lines(infile):
    with open(infile, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def main(args):

    print("Loading data...")
    data_name_suffix = args.dataset_name.rstrip("/").split("/")[-1]
    data_langs_map = {
        "ikcest2022": ["zh-th", "th-zh", "zh-fr", "fr-zh", "zh-ru", "ru-zh", "zh-ar", "ar-zh"],
        "iwslt2017": ["en-it", "it-en", "en-ro", "ro-en", "en-nl", "nl-en", "it-ro", "ro-it"]
    }
    # infer language pair
    if args.lang_pairs is not None:
        cfg_pairs = args.lang_pairs.strip().split(",")
        valid_pairs = []
        for pair in cfg_pairs:
            langs = pair.split("-")
            if len(langs) == 2 and pair in data_langs_map[data_name_suffix]:
                valid_pairs.append(pair)
        if len(valid_pairs) > 0:
            lang_pairs = valid_pairs
        else:
            # Handle case when no valid pairs found
            raise ValueError("No valid language pairs found in provided config.")
    else:
        lang_pairs = data_langs_map[data_name_suffix]




    df_data = []
    for lang_pair in lang_pairs:
        source_lang = lang_pair.split('-')[0]
        target_lang = lang_pair.split('-')[1]

        dataset_config_name = data_name_suffix + "-" + lang_pair

        tst_dataset = load_dataset(args.dataset_name, dataset_config_name, cache_dir="./datasets/",
                               verification_mode="no_checks")["test"]

        print(tst_dataset)
        for data in tst_dataset:
            source_text = data["translation"][source_lang]
            target_text = data["translation"][target_lang]
            new_data = {"src_lang": source_lang, "tgt_lang": target_lang, "src": source_text, "ref": target_text}
            df_data.append(new_data)

    df = pd.DataFrame(df_data)
    df['lang_dir'] = df['src_lang'] + '-' + df['tgt_lang']

    # tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("Loading model...")
    start = time.time()
    translation_pipeline = pipeline('translation', args.model_path, src_lang='zh', tgt_lang="en", device=0)
    load_time = time.time() - start
    print(f"Load {load_time} secs.")

    df_ls = []
    # todo 1 在上一步完成推理 2 指定lang_pairs
    for lang_dir, df_lang in df.groupby('lang_dir'):
        print(f"Translating {lang_dir}...")
        start = time.time()
        src_lang, tgt_lang = lang_dir.split("-")
        src_texts = df_lang['src'].tolist()
        predicted_texts = translation_pipeline(src_texts, src_lang=src_lang, tgt_lang=tgt_lang,batch_size=args.batch_size, max_length=args.max_length)

        inference_time = time.time() - start
        print(f"Infer {inference_time} secs.")

        predicted_texts = [pred["translation_text"] for pred in predicted_texts]
        df_lang["pred"] = predicted_texts
        df_lang["model"] = args.model_name
        df_ls.append(df_lang)

    df_output = pd.concat(df_ls, axis=0)

    # 保存到jsonl文件
    df_output.to_json(args.output_file, orient='records', lines=True, force_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate texts using m2m100 model and save results to a JSON file")
    parser.add_argument("-d","--dataset_name", type=str, default="miugod/ikcest2022", help="test dataset name")
    parser.add_argument("-o","--output_file", type=str, default="results_m2m100.jsonl", help="Output JSON file name")
    parser.add_argument("-m","--model_path", type=str, default="facebook/m2m100", help="Model name or path")
    parser.add_argument("-n","--model_name", type=str, default="m2m100", help="Model name")
    parser.add_argument("-b","--batch_size", type=int, default=4, help="batch_size")
    parser.add_argument("-l","--max_length", type=int, default=200, help="max_length")
    parser.add_argument("-lp","--lang_pairs", type=str, default=None, help="Language pairs,such as zh-ru,fr-zh")


    args = parser.parse_args()

    main(args)
# python predict.py -d datasets/ikcest2022 -o ikcest_m2m.jsonl -m facebook/m2m100 -n m2m -b 8 -l 400