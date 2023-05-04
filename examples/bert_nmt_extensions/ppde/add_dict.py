'''
功能：根据词典，从原句中找出词典词，然后拼接到后面
dict={"a":"A"} 词典包含两列，要去重保证一对一，然后有个参数决定前向还是反向映射
1.处理好假数据 √
2.读取真的词典 √
3.构造词典fastalign √
4.测试效果 √ （比微调的mlm下降了0.3分，猜测是没有加入语境信息）
5.使用多种模式（1.替换 2.插入 3.拼接句尾 4.拼接=(双语concat biconcat)）
'''
#coding:utf-8
import argparse
from tqdm import tqdm
import ahocorasick
import numpy as np
import random
from functools import partial
from multiprocessing import Pool

total_lines = 0
total_aug_lines = 0

def same_seeds(seed=1):
    random.seed(seed)
    np.random.seed(seed)


def make_AC(word_set):
    automaton  = ahocorasick.Automaton()
    for word in word_set:
        automaton.add_word(word,word)
    automaton.make_automaton()
    return automaton

def search(sent,automaton):
    word_list = set()
    for item in automaton.iter(sent): # item=(pos, word)
        word_list.add(item[1])
    word_list = list(word_list)
    return word_list

def load_dict(filepath, is_rev=False):
    bi_dict = {}
    with open(filepath,'r',encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split("\t")
            if len(line)!=2: continue
            src, tgt = line
            if not is_rev:
                bi_dict[src] = tgt
            else:
                bi_dict[tgt] = src
    return bi_dict

def random_sample(args,word_list):
    new_list = []
    if len(word_list) > 0:
        for idx, word in enumerate(word_list):
            if (np.random.uniform(0,1) < args.dict_prob) and idx<args.max_pairs:
                new_list.append(word)
    return new_list

def aug_line(line,word_list,word_dict,args=None):
    # tgt_words = [word_dict[name] for name in word_list]
    new_line = line
    if len(word_list)==0: return new_line+"[SEP]\n"
    if args.mode == "biconcat": # text[sep] ws=wt
        tail = " ".join([f"{name} = {word_dict[name]} [SEP]" for name in word_list])
        new_line = f"{line} [SEP] {tail}\n"
    elif args.mode == "concat": # text[sep] wt
        tgt_words = [word_dict[name] for name in word_list]
        tail = " ".join([f"{tgt_word} [SEP]" for tgt_word in tgt_words])
        new_line = f"{line} [SEP] {tail}\n"
    elif args.mode == "replace":  # text ws [sep] ->  text wt [sep]
        for k in word_list:
            new_line = new_line.replace(k, word_dict[k])
        new_line += " [SEP]\n"
    elif args.mode == "insert": # text ws [sep] ->  text ws wt [sep]
        for k in word_list:
            start = new_line.find(k)
            end = start + len(k)
            new_line = new_line[:end] + f" {word_dict[k]} " + new_line[end:]
        new_line += " [SEP]\n"
    elif args.mode == "inserte": # text ws [sep] ->  text ws = wt [sep]
        for k in word_list:
            start = new_line.find(k)
            end = start + len(k)
            new_line = new_line[:end] + f" = {word_dict[k]} " + new_line[end:]
        new_line += " [SEP]\n"
    else:
        modes = "|".join(["replace","insert","inserte","concat","biconcat"])
        raise ValueError(f"mode not in {modes}")
    return new_line

def process_line(line, args=None ,automaton=None,word_dict=None):
    # global total_lines
    # global total_aug_lines
    # total_lines += 1
    # add dict pair
    line = line.strip()
    word_list = search(line, automaton)
    if not args.test:
        word_list = random_sample(args,word_list)
    line = aug_line(line,word_list,word_dict,args)
    return line

def main(args):
    global total_lines
    global total_aug_lines
    word_dict = load_dict(args.dictpath, is_rev=args.is_rev)
    with open(args.infile,"r",encoding="utf-8") as fr,open(args.outfile,"w",encoding="utf-8") as fw:
        pool=Pool(processes=args.workers)
        automaton = make_AC(word_dict.keys())
        process_fn= partial(process_line,args= args, automaton=automaton,word_dict=word_dict)
        sentences = pool.imap(process_fn,fr,chunksize=1000)
        for sent in tqdm(sentences):
            fw.write(sent)
    # print(f"processed [{total_aug_lines}/{total_lines}] lines")

def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-i','--infile',required=True,type=str,default='')
    parser.add_argument('-o','--outfile',required=True,type=str,default='')
    parser.add_argument('-d','--dictpath',required=True,type=str,default="")
    parser.add_argument('-w','--workers',type=int,default=4)
    parser.add_argument('-f','--is-rev', action="store_true", help="reverse build dict")
    parser.add_argument('-m','--mode', type=str,default="insert", choices=["replace","insert","inserte","concat","biconcat"])

    return parser

def rand_parser(parser):
    # 随机性: 每个词以多少几率被保留，每个句子最多不超过几个词，生成几份
    # parser.add_argument('--num-repeat', type=int, default=1) # 再运行一次就行了
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max-pairs', type=int, default=5)
    parser.add_argument('--dict-prob', type=float, default=0.8)
    parser.add_argument('--test', action="store_true")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    parser = rand_parser(parser)
    args = parser.parse_args()
    same_seeds(args.seed)
    main(args)

