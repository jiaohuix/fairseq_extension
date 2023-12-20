'''
ref: https://github.com/facebookresearch/fairseq/issues/2120
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

all_words=set()
def main() -> None:
    parser = argparse.ArgumentParser(description="Trims pre-trained mBART model for fine-tuning.")
    parser.add_argument("--pre-train-dir", type=str, required=True, help="The pre-trained mBART model directory.(model.pt,dict.txt)")
    parser.add_argument("--ft-dict", type=str, required=True, help="The fine-tuning model dictionary. (dict.txt)")
    parser.add_argument("--langs", type=str, required=True, help="The pre-trained model languages.")
    parser.add_argument("--output", type=str, required=True, help="The trimmed mBART model.")
    args = parser.parse_args()


    langs = args.langs.split(",")
    pre_dict = load_dict(langs,os.path.join(args.pre_train_dir, "dict.txt"),is_ptm=True)
    ft_dict = load_dict(langs,args.ft_dict,is_ptm=False)
    print(len(pre_dict),len(ft_dict))
    data = torch.load(os.path.join(args.pre_train_dir, "model.pt"))
    model = data["model"]
    num_unk = 0
    mapping: List[int] = []
    num_arb=0
    for i in range(len(ft_dict)):
        word = ft_dict[i]
        idx = pre_dict.index(word)
        if is_contains_arabic(word):
            # print(word,idx)
            num_arb+=1
        if idx==3:
            num_unk+=1
            # print(word,idx,"is contains chinese=",is_contains_chinese(word))
            # print(jieba.lcut(word),idx)
            print(word,idx)
            for w in word:
                if is_contains_chinese(w):
                    all_words.add(w)
        mapping.append(idx)
    print("num arb:",num_arb)
    # for name in ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]:
    for name in ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight","decoder.output_projection.weight"]:
        pre_tensor: torch.Tensor = model[name]
        ft_tensor = torch.zeros(
            [len(ft_dict), pre_tensor.shape[1]], dtype=pre_tensor.dtype, layout=pre_tensor.layout, device=pre_tensor.device,
        )
        for ft_i, pre_i in enumerate(mapping):
            ft_tensor[ft_i] = pre_tensor[pre_i]

        model[name] = ft_tensor

    torch.save(data, args.output)
    ft_dict.save(args.ft_dict.replace(".txt",".ft.txt"))
    print(f"Save to {args.output} success, vocab_size: {len(mapping)}, unk: {num_unk}.")
    print(all_words,f"len unk:{len(all_words)}")

if __name__ == "__main__":
    main()