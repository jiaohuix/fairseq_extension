# -*- coding: utf-8 -*-
'''
目的：根据dataset，来进行多语言的分词
修改点：1 输入一个data，把他所有的语言都处理好 2 统一的词表

1 迭代所有语言moses处理，保存data到本地
2 训练bpe
3 迭代所有语言，bpe处理

'''
import os
import random
import shutil
import argparse
import subprocess
from statistics import mean
from functools import partial
from multiprocessing import Pool

import jieba
from subword_nmt import subword_nmt
from pythainlp.tokenize import word_tokenize
from datasets import concatenate_datasets, load_dataset
from sacremoses import MosesTokenizer, MosesDetokenizer, MosesTruecaser, MosesDetruecaser

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# 参数

def parse_arguments():
    parser = argparse.ArgumentParser(description="Your script description here.")
    parser.add_argument('-i', '--data_name', type=str, default="miugod/ikcest2022", help="Data name")
    parser.add_argument('-o', '--out_root', type=str, default="train_data", help="Output root directory")
    parser.add_argument('-v', '--bpe_codes', type=str, default="dict/codes.bpe.32000.txt", help="bpe code")
    parser.add_argument('-n', '--num_procs', type=int, default=8, help="num process")
    return parser.parse_args()


###################### change: 函数放这

def write_file(res, file):
    with open(file, 'w', encoding='utf-8') as f:
        f.writelines(res)
    print(f'write to {file} success, total {len(res)} lines.')


def write_file_append(text_list, outfile):
    try:
        # 以追加模式打开文件，如果文件不存在则创建
        with open(outfile, 'a') as file:
            # 逐行写入文本列表中的内容
            for line in text_list:
                file.write(line.strip() + '\n')  # 写入一行文本并添加换行符
        print(f'write to {outfile} success, total {len(text_list)} lines.')

    except IOError:
        print(f'write to {outfile} error.')


## 保存到本地
def save_dataset(dataset,lang_pair, dir):
    # 创建目录
    os.makedirs(dir, exist_ok=True)

    # 保存数据集
    print("save dataset",dataset)
    for split in list(dataset.keys()):
        for lang in lang_pair.split("-"):
            split_fairseq = split.replace("validation", "valid")
            filename = os.path.join(dir, f"{split_fairseq}.{lang}")
            with open(filename, "w", encoding="utf-8") as f:
                print("for dataset",dataset,"lang",lang, "split",split)
                for text in dataset[split][lang]:
                    f.write(text.strip() + "\n")


## 预分词
class MultilingualTokenizer:
    def __init__(self):
        self.moses_tokenizer = MosesTokenizer()
        self.moses_detokenizer = MosesDetokenizer()

    def tokenize(self, text, lang="en", return_str=False):
        if lang == 'zh':
            tokens = jieba.lcut(text)
        elif lang == 'th':
            tokens = word_tokenize(text)
        else:
            self.moses_tokenizer.lang = lang
            tokens = self.moses_tokenizer.tokenize(text)
        if return_str:
            tokens = " ".join(tokens)
        return tokens

    def detokenize(self, tokens, lang="en"):
        if lang in ["zh", "th"]:
            detokenized_text = ''.join(tokens)
        else:
            self.moses_detokenizer.lang = lang
            detokenized_text = self.moses_detokenizer.detokenize(tokens)
        return detokenized_text


class BPETokenizer:

    def __init__(self, codes_file):
        bpe_parser = subword_nmt.create_apply_bpe_parser()
        bpe_args = bpe_parser.parse_args(args=['-c', codes_file])
        self.bpe = subword_nmt.BPE(bpe_args.codes, bpe_args.merges,
                                   bpe_args.separator, None,
                                   bpe_args.glossaries)

    def tokenize(self, raw_string):
        """
        Tokenize string(BPE)
        """
        raw_string = raw_string.strip('\n')
        if not raw_string:
            return raw_string
        bpe_str = self.bpe.process_line(raw_string)
        return " ".join(bpe_str.split())  # ?


def tokenize_example(example, tokenizer: MultilingualTokenizer):
    langs = list(example.keys())
    for lang in langs:
        example[lang] = tokenizer.tokenize(example[lang], lang=lang, return_str=True)
    return example


def truecase_example(example, truecaser: MosesTruecaser):
    truecase_langs = ["en", "fr", "ru", "it", "ro", "nl", "de"]
    # langs = list(example.keys())
    langs = lang_pair.split("-")
    for lang in langs:
        if lang in truecase_langs:
            example[lang] = truecaser.truecase(example[lang], return_str=True)
    return example


def bpe_example(example, tokenizer: BPETokenizer):
    # langs = list(example.keys())
    # langs = lang_pair.split("-")
    langs = list(example.keys())
    for lang in langs:
        example[lang] = tokenizer.tokenize(example[lang])

    return example


# def calculate_statics(dataset):
#     def _map_fn(example):
#         # 添加src_lang tgt_lang, ratio; 以及打印平均值
#         example["src_len"] = len(example[src_lang].split())
#         example["tgt_len"] = len(example[tgt_lang].split())
#         example["ratio"] = max(example["src_len"], example["tgt_len"]) / min(example["src_len"], example["tgt_len"])
#         return example
#
#     for split in splits:
#         dataset[split] = dataset[split].map(lambda example: _map_fn(example))
#         avg_src_len = mean(dataset[split]["src_len"])
#         avg_tgt_len = mean(dataset[split]["tgt_len"])
#         avg_ratio = mean(dataset[split]["ratio"])
#         print(f"{split} | avg_src_len :{avg_src_len} |avg_tgt_len: {avg_tgt_len} | avg_ratio：{avg_ratio} ")


######################




# # 2 moses
# def process_lang_pair_moses(lang_pair):
#     mtokenizer = MultilingualTokenizer()
#     print("moses tokenize")
#     for split in splits:
#         all_datasets[lang_pair][split] = all_datasets[lang_pair][split].map(
#             lambda example: tokenize_example(example["translation"], tokenizer=mtokenizer),
#             remove_columns=["translation"])
#     print(all_datasets[lang_pair])
#
#
# # 3 对所有语言apply bpe
#
#
# def process_lang_pair_bpe(lang_pair):
#     outdir = os.path.join(out_root, data_name_suffix, lang_pair)  # train_data/ikcest2022/zh-fr
#     if not os.path.exists(outdir):
#         os.makedirs(outdir)
#
#     dataset = all_datasets[lang_pair]
#
#     print(f"apply bpe {lang_pair}")
#     for split in splits:
#         dataset[split] = dataset[split].map(lambda example: bpe_example(example, bpe_tokenizer))
#
#     print(dataset)
#
#     print(f"save data to {outdir}...")
#     save_dataset(dataset, outdir)


def process_lang_pair(lang_pair):
    # Moses tokenize
    mtokenizer = MultilingualTokenizer()
    for split in splits:
        all_datasets[lang_pair][split] = all_datasets[lang_pair][split].map(
            lambda example: tokenize_example(example["translation"], tokenizer=mtokenizer),
            remove_columns=["translation"])

    # Apply BPE
    outdir = os.path.join(out_root, data_name_suffix, lang_pair)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    dataset = all_datasets[lang_pair]

    print(f"apply bpe {lang_pair}")
    for split in splits:
        dataset[split] = dataset[split].map(lambda example: bpe_example(example, bpe_tokenizer))

    print(dataset)

    print(f"save data to {outdir}...")
    save_dataset(dataset,lang_pair, outdir)


if __name__ == "__main__":
    args = parse_arguments()
    data_name = args.data_name
    # lang_pair = args.lang_pair # change
    out_root = args.out_root
    splits = ["train", "validation", "test"]

    if not os.path.exists(out_root):
        os.makedirs(out_root)

    data_langs_map = {
        # "ikcest2022": ["zh-th", "th-zh", "zh-fr", "fr-zh", "zh-ru", "ru-zh", "zh-ar", "ar-zh"],
        "ikcest2022": ["zh-ar" ,"ar-zh","zh-th", "th-zh"],
        "iwslt2017": ["en-it", "it-en", "en-ro", "ro-en", "en-nl", "nl-en", "it-ro", "ro-it"]
    }
    print(f"process: {data_name} ")

    data_name_suffix = data_name.rstrip("/").split("/")[-1]
    print("data_name_suffix", data_name_suffix)
    assert data_name_suffix in data_langs_map.keys()
    lang_pairs = data_langs_map[data_name_suffix]

    # 1 迭代所有的语言(先moses分词) 追加写入文件
    all_datasets = {}  # lang_pair: data={train valid test}

    # 1 load data
    for lang_pair in lang_pairs:
        cfg_name = data_name_suffix + "-" + lang_pair
        all_datasets[lang_pair] = load_dataset(data_name, cfg_name, cache_dir="./datasets/",
                                               verification_mode="no_checks")  # todo: 保存到字典
        print(all_datasets[lang_pair]["train"], len(all_datasets[lang_pair]["train"]))
    bpe_tokenizer = BPETokenizer(args.bpe_codes)


    # Define the number of processes you want to use
    # num_processes = 8  # Change this number according to your system's capabilities

    # Moses tokenize and apply BPE concurrently
    with Pool(len(lang_pairs)) as pool:
        pool.map(process_lang_pair, lang_pairs)