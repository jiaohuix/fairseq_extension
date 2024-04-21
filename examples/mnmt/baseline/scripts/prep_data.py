# !pip install datasets sacremoses subword-nmt jieba pythainlp tokenizers
'''
export HF_ENDPOINT=https://hf-mirror.com
python scripts/prep_data.py -i miugod/ikcest2022 -o train_data/ -v 10000 -l zh-th
'''
import os
import random
import argparse
import subprocess
from statistics import mean
from functools import partial

import jieba
from subword_nmt import subword_nmt
from pythainlp.tokenize import word_tokenize
from datasets import concatenate_datasets, load_dataset
from sacremoses import MosesTokenizer, MosesDetokenizer, MosesTruecaser, MosesDetruecaser
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 参数

def parse_arguments():
    parser = argparse.ArgumentParser(description="Your script description here.")
    parser.add_argument('-i','--data_name', type=str, default="miugod/ikcest2022", help="Data name")
    parser.add_argument('-o','--out_root', type=str, default="train_data", help="Output root directory")
    parser.add_argument('-v','--vocab_size', type=int, default=10000, help="Vocabulary size")
    parser.add_argument('-l','--lang_pair', type=str, default="zh-fr", help="Language pair")
    return parser.parse_args()

args = parse_arguments()

vocab_size = args.vocab_size # merge operations 
data_name = args.data_name
lang_pair = args.lang_pair
out_root = args.out_root

print(f"process: {data_name} {lang_pair}")



src_lang, tgt_lang = lang_pair.split("-")
splits = ["train", "validation", "test"]

outdir = os.path.join(out_root, data_name.split("/")[-1], lang_pair)
if not os.path.exists(outdir):
    os.makedirs(outdir)

# 加载数据(先用命令行下到本地)
cfg_name = os.path.basename(data_name) + "-" + lang_pair
dataset = load_dataset(data_name, cfg_name, cache_dir="./datasets/", verification_mode="no_checks")
# dataset["train"] = dataset["train"].filter(lambda x: random.random() < 0.05)
print(dataset["train"], len(dataset["train"]))


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




def calculate_statics(dataset):
    def _map_fn(example):
        # 添加src_lang tgt_lang, ratio; 以及打印平均值
        example["src_len"] = len(example[src_lang].split())
        example["tgt_len"] = len(example[tgt_lang].split())
        example["ratio"] = max(example["src_len"],  example["tgt_len"]) / min(example["src_len"],  example["tgt_len"])
        return example

    for split in splits:
        dataset[split] = dataset[split].map(lambda example: _map_fn(example))
        avg_src_len = mean(dataset[split]["src_len"])
        avg_tgt_len = mean(dataset[split]["tgt_len"])
        avg_ratio = mean(dataset[split]["ratio"])
        print(f"{split} | avg_src_len :{avg_src_len} |avg_tgt_len: {avg_tgt_len} | avg_ratio：{avg_ratio} ")


mtokenizer = MultilingualTokenizer()
tokenize_fn = partial(tokenize_example, tokenizer=mtokenizer)

print("moses tokenize")
for split in splits:
    dataset[split] = dataset[split].map(lambda example: tokenize_fn(example["translation"]),
                                        remove_columns=["translation"])
print(f"-----preprocess moses statics------")
calculate_statics(dataset)
print(dataset)

## 2 学习truecase
# tokenized_corpus = []
# truecase_langs = ["en", "fr", "ru", "it", "ro", "nl", "de"]

# for split in splits:
#     dataset_split = dataset[split]
#     if src_lang in truecase_langs:
#         tokenized_corpus.extend(dataset_split[src_lang])
#     if tgt_lang in truecase_langs:
#         tokenized_corpus.extend(dataset_split[tgt_lang])

# print("train truecase")
# truecaser = MosesTruecaser()
# truecaser_path = os.path.join(outdir, f"{lang_pair}.truecasemodel")
# truecaser.train(tokenized_corpus, save_to=truecaser_path, processes=4)

# print("apply truecase")
# truecase_fn = partial(truecase_example, truecaser=truecaser)
# for split in splits:
#     dataset[split] = dataset[split].map(lambda example: truecase_fn(example))

# print(f"-----preprocess truecase------")
# calculate_statics(dataset)
# print(dataset)


# 学习bpe
def batch_iterator(dataset, batch_size=1000):
    corpus = concatenate_datasets([dataset[split] for split in splits])
    for i in range(0, len(corpus), batch_size):
        yield corpus[i: i + batch_size][src_lang] + corpus[i: i + batch_size][tgt_lang]


# print("train bpe(tokenizers)")
# tokenizer = Tokenizer(BPE(unk_token="<unk>"))
# # tokenizer = Tokenizer(BPE(unk_token="<unk>", end_of_word_suffix="@@"))

# # if lang_pair.find("ru")==-1:
# #     tokenizer.pre_tokenizer = Whitespace()
# # https://pypi.org/project/tokenizers/
# tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
# tokenizer.decoder = decoders.ByteLevel()
# tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

# # trainer = BpeTrainer(special_tokens=["<unk>"], end_of_word_suffix="@@", vocab_size=vocab_size, min_frequency = 2 ,initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
# trainer = BpeTrainer(vocab_size=vocab_size, min_frequency = 2 ,initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

# tokenizer.train_from_iterator(batch_iterator(dataset=dataset), trainer=trainer)

# # lang_pair
# bpe_path = os.path.join(outdir, "tokenizer.json")
# tokenizer.save(bpe_path)
# tokenizer.model.save(os.path.dirname(bpe_path))


print("train bpe(subword-nmt)")

# 先不要写类
corpus = concatenate_datasets([dataset[split] for split in splits])
text_ls = corpus[src_lang] + corpus[tgt_lang]
text_ls = [t.strip()+"\n" for t in text_ls]
def write_file(res,file):
    with open(file,'w',encoding='utf-8') as f:
        f.writelines(res)
    print(f'write to {file} success, total {len(res)} lines.')


# 1 保存文件到tmp
tmp_file=f"./corpus_{lang_pair}.tmp"
codes_file = os.path.join(outdir, "codes.txt")
write_file(text_ls, tmp_file)

# 2 训练bpe
subprocess.run(['subword-nmt', 'learn-bpe',
                '-s',  str(vocab_size),
                '--input' ,tmp_file,
                '-o',  codes_file]) # codes file
# 3 apply函数
class BPETokenizer:

    def __init__(self, codes_file):
        bpe_parser = subword_nmt.create_apply_bpe_parser()
        bpe_args = bpe_parser.parse_args(args=['-c', codes_file])
        self.bpe = subword_nmt.BPE(bpe_args.codes, bpe_args.merges,
                                   bpe_args.separator, None,
                                   bpe_args.glossaries)

    def tokenize(self, raw_string):
        """
        Tokenize string(BPE/jieba+BPE)
        """
        raw_string = raw_string.strip('\n')
        if not raw_string:
            return raw_string
        bpe_str = self.bpe.process_line(raw_string)
        return " ".join(bpe_str.split()) # ?


# def bpe_example(example, tokenizer: BPETokenizer):
#     # langs = list(example.keys())
#     # langs = lang_pair.split("-")
#     example[src_lang] = tokenizer.decode(tokenizer.encode(example[src_lang]).ids, skip_special_tokens=False)
#     example[tgt_lang] = tokenizer.decode(tokenizer.encode(example[tgt_lang]).ids, skip_special_tokens=False)
    
#     return example

def bpe_example(example, tokenizer: BPETokenizer):
    # langs = list(example.keys())
    # langs = lang_pair.split("-")
    example[src_lang] = tokenizer.tokenize(example[src_lang])
    example[tgt_lang] = tokenizer.tokenize(example[tgt_lang])
    
    return example



print("apply bpe")
bpe_tokenizer = BPETokenizer(codes_file)
for split in splits:
    dataset[split] = dataset[split].map(lambda example: bpe_example(example, bpe_tokenizer))

print(f"-----preprocess bpe statics------")
calculate_statics(dataset)
print(dataset)



## 保存到本地
def save_dataset(dataset, dir):
    # 创建目录
    os.makedirs(dir, exist_ok=True)

    # 保存数据集
    for split in list(dataset.keys()):
        for lang in lang_pair.split("-"):
            split_fairseq = split.replace("validation", "valid")
            filename = os.path.join(dir, f"{split_fairseq}.{lang}")
            with open(filename, "w", encoding="utf-8") as f:
                for text in dataset[split][lang]:
                    f.write(text.strip() + "\n")


print(f"save data to {outdir}...")
save_dataset(dataset, outdir)
