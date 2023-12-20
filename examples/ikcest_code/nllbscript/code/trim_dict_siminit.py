'''
ref: https://github.com/facebookresearch/fairseq/issues/2120
改进： 对于微调的词典dict2,比预训练词典dict1多出一些原先没有的中文词，找向量是unk的，此处修改为找若干个近义词的词向量平均

修改： 1.加载相似词典 2.修改mapping，idx全部变成list，然后取到若干个tensor后，取平均
'''
import argparse
import os
from typing import List
import random
import numpy as np
import torch
import torch.nn as nn
import jieba
from fairseq.data import Dictionary

def is_contains_arabic(strs):
    for _char in strs:
        if '\u0600' <= _char <= '\u06ff':
            return True
    return False

def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

def read_file(file):
    with open(file,'r',encoding='utf-8') as f:
        lines=f.readlines()
    return lines


def load_sim_dict(file):
    lines = read_file(file)
    sim_dict = {}
    for line in lines:
        line = line.strip()
        word, sims = line.split("\t")
        sims = sims.strip().split()
        sim_dict[word]  = sims
    return sim_dict

def load_dict(langs: List[str], path: str, is_ptm: bool) -> Dictionary:
    d = Dictionary.load(path)
    if not is_ptm:
        for l in langs:
            d.add_symbol(l)
        d.add_symbol("<mask>")
    return d

def same_seeds(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_avg_weight(tensor,idxs_ls):
    res = []
    for idx in idxs_ls:
        res.append(tensor[idx])
    t = torch.vstack(res)
    return torch.mean(t, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trims pre-trained mBART model for fine-tuning.")
    parser.add_argument("--pre-train-dir", type=str, required=True, help="The pre-trained mBART model directory.(model.pt,dict.txt)")
    parser.add_argument("--ft-dict", type=str, required=True, help="The fine-tuning model dictionary. (dict.txt)")
    parser.add_argument("--langs", type=str, required=True, help="The pre-trained model languages.")
    parser.add_argument("--output", type=str, required=True, help="The trimmed mBART model.")
    parser.add_argument("--simdict", type=str,default="", help="similar words dict for unk")
    parser.add_argument("--topk", type=int,default=3, help="topk similar words for unk")
    args = parser.parse_args()

    num_unk = 0

    langs = args.langs.split(",")
    pre_dict = load_dict(langs,os.path.join(args.pre_train_dir, "dict.txt"),is_ptm=True)
    ft_dict = load_dict(langs,args.ft_dict,is_ptm=False)
    print(len(pre_dict),len(ft_dict))
    # load similar dict for unk
    sim_dict = load_sim_dict(args.simdict)

    data = torch.load(os.path.join(args.pre_train_dir, "model.pt"))
    model = data["model"]
    num_unk = 0
    mapping: List[int] = []
    for i in range(len(ft_dict)):
        word = ft_dict[i]
        idx = pre_dict.index(word)
        #  process unk
        if (i>3) and (idx==3) and (word in sim_dict.keys()):
            num_unk +=1
            sim_words = [word for word in sim_dict[word]]
            sim_words = " ".join(sim_words[:args.topk])
            print(f"raw: {word}, sim: {sim_words}")
            sim_idxs = [pre_dict.index(word) for word in sim_dict[word]]
            idx_ls = [idx for idx in sim_idxs if idx!=3][:args.topk]
            if len(idx_ls)==0:
                idx_ls = [3]
        else:
            idx_ls = [idx]


        mapping.append(idx_ls)
    # for name in ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]:
    for name in ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight","decoder.output_projection.weight"]:
        pre_tensor: torch.Tensor = model[name]
        ft_tensor = torch.zeros(
            [len(ft_dict), pre_tensor.shape[1]], dtype=pre_tensor.dtype, layout=pre_tensor.layout, device=pre_tensor.device,
        )
        for ft_i, pre_idxs in enumerate(mapping):
            # ft_tensor[ft_i] = pre_tensor[pre_i]
            ft_tensor[ft_i] =  load_avg_weight(pre_tensor,pre_idxs)

        model[name] = ft_tensor

    torch.save(data, args.output)
    ft_dict.save(args.ft_dict.replace(".txt",".ft.txt"))
    print(f"Save to {args.output} success, vocab_size: {len(mapping)}, unk: {num_unk}.")

if __name__ == "__main__":
    main()