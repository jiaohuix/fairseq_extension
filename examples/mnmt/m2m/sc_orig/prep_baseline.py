# !pip install datasets sacremoses subword-nmt jieba pythainlp tokenizers
import os
import random
from functools import partial
import jieba
from pythainlp.tokenize import word_tokenize
from datasets import concatenate_datasets, load_dataset
from sacremoses import MosesTokenizer, MosesDetokenizer, MosesTruecaser, MosesDetruecaser

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 参数
vocab_size=10000
# data_name = "iwslt2017"  # miugod/ikcest2022
data_name = "miugod/ikcest2022" # miugod/ikcest2022
# data_name = "datasets/ikcest2022" # miugod/ikcest2022
lang_pair = "zh-fr"
# lang_pair = "en-fr"
out_root = "train_data"

src_lang, tgt_lang = lang_pair.split("-")

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
    langs = list(example.keys())
    for lang in langs:
        if lang in truecase_langs:
            example[lang] = truecaser.truecase(example[lang], return_str=True)
    return example


splits = ["train", "validation", "test"]
# splits = [ "validation", "test"]
mtokenizer = MultilingualTokenizer()
tokenize_fn = partial(tokenize_example, tokenizer=mtokenizer)

print("moses tokenize")
for split in splits:
    dataset[split] = dataset[split].map(lambda example: tokenize_fn(example["translation"]),
                                        remove_columns=["translation"])

## 2 学习truecase
tokenized_corpus = []
truecase_langs = ["en", "fr", "ru", "it", "ro", "nl", "de"]
for split in splits:
    dataset_split = dataset[split]
    if src_lang in truecase_langs:
        tokenized_corpus.extend(dataset_split[src_lang])
    if tgt_lang in truecase_langs:
        tokenized_corpus.extend(dataset_split[tgt_lang])

print("train truecase")
truecaser = MosesTruecaser()
truecaser_path = os.path.join(outdir, f"{lang_pair}.truecasemodel")
truecaser.train(tokenized_corpus, save_to=truecaser_path, processes=4)

print("apply truecase")
truecase_fn = partial(truecase_example, truecaser=truecaser)
for split in splits:
    dataset[split] = dataset[split].map(lambda example: truecase_fn(example))

# 学习bpe
tokenized_corpus = []
truecase_langs = ["en", "fr", "ru", "it", "ro", "nl", "de"]  # 这就多此一举了


def batch_iterator(dataset, batch_size=1000):
    corpus = concatenate_datasets([dataset[split] for split in splits])
    for i in range(0, len(corpus), batch_size):
        yield corpus[i: i + batch_size][src_lang] + corpus[i: i + batch_size][tgt_lang]


print("train bpe")
tokenizer = Tokenizer(BPE(unk_token="<unk>", end_of_word_suffix="@@"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(special_tokens=["<unk>"], end_of_word_suffix="@@", vocab_size=vocab_size, min_frequency = 2 )
tokenizer.train_from_iterator(batch_iterator(dataset=dataset), trainer=trainer)

# lang_pair
bpe_path = os.path.join(outdir, "tokenizer.json")
tokenizer.save(bpe_path)
tokenizer.model.save(os.path.dirname(bpe_path))


def bpe_example(example, tokenizer: Tokenizer):
    langs = list(example.keys())
    for lang in langs:
        example[lang] = tokenizer.decode(tokenizer.encode(example[lang]).ids, skip_special_tokens=False)
    return example


print("apply bpe")
bpe_fn = partial(bpe_example, tokenizer=tokenizer)
for split in splits:
    dataset[split] = dataset[split].map(lambda example: bpe_fn(example))


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


print("save data...")
save_dataset(dataset, outdir)
## 参数

'''
!TEXT=train_data/iwslt2017/en-fr && fairseq-preprocess --source-lang en --target-lang fr \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt17.tokenized.en-fr --joined-dictionary  \
    --workers 4

bash scripts/bin.sh train_data/ikcest2022/zh-fr/ data-bin/ikcest22-zhfr zh fr 1

train:

'''