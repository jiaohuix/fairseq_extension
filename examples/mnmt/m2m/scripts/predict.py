import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import argparse
import logging
import time
logger = logging.getLogger("traslation")

def read_lines(infile):
    with open(infile, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def main(args):
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("Loading model...")
    start = time.time()
    translation_pipeline = pipeline('translation', args.model_path, src_lang='zh', tgt_lang="en", device=0)
    load_time = time.time() - start
    print(f"Load {load_time} secs.")

    print("Loading data...")
    data_ls = [json.loads(line.strip()) for line in read_lines(args.infile)]

    print("Preprocessing data...")
    df_data = []
    for data in data_ls:
        # 解析数据
        data = data["translation"]
        source_lang, source_text = list(data.items())[0]
        target_lang, target_text = list(data.items())[1]
        new_data = {"src_lang": source_lang, "tgt_lang": target_lang, "src": source_text, "ref": target_text}
        df_data.append(new_data)
    df = pd.DataFrame(df_data)
    df['lang_dir'] = df['src_lang'] + '-' + df['tgt_lang']

    df_ls = []
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
    parser.add_argument("-i","--infile", type=str, default="test.jsonl", help="test jsonl file")
    parser.add_argument("-o","--output_file", type=str, default="results_m2m100.jsonl", help="Output JSON file name")
    parser.add_argument("-m","--model_path", type=str, default="facebook/m2m100", help="Model name or path")
    parser.add_argument("-n","--model_name", type=str, default="m2m100", help="Model name")
    parser.add_argument("-b","--batch_size", type=int, default=4, help="batch_size")
    parser.add_argument("-l","--max_length", type=int, default=200, help="max_length")
    args = parser.parse_args()

    main(args)
