'''
ref: https://github.com/facebookresearch/fairseq/issues/2120
实验版本：
待修改：
    1.unk替换为随机的某个token，都一样
    2.unk随机替换为若干个token
    3.unk用随机向量初始化（1个固定的）
    4.unk用随机多个向量初始化
    5.用有意义的向量初始化top3平均
    6.有意义的top3和unk平均
'''
import argparse
import os
from typing import List
import random
import numpy as np
import torch
import torch.nn as nn
from fairseq.data import Dictionary
from functools import lru_cache


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

def main() -> None:
    parser = argparse.ArgumentParser(description="Trims pre-trained mBART model for fine-tuning.")
    parser.add_argument("--pre-train-dir", type=str, required=True, help="The pre-trained mBART model directory.(model.pt,dict.txt)")
    parser.add_argument("--ft-dict", type=str, required=True, help="The fine-tuning model dictionary. (dict.txt)")
    parser.add_argument("--langs", type=str, required=True, help="The pre-trained model languages.")
    parser.add_argument("--output", type=str, required=True, help="The trimmed mBART model.")
    parser.add_argument("--mode", type=str,default="unk", choices=["unk","rand_1tok","rand_ntok","rand_1init","rand_ninit"],help="Unk process, choices: unk/rand_1tok/rand_ntok/rand_1init/rand_ninit")
    args = parser.parse_args()
    # mode = "unk"
    # mode = "rand_1tok"
    # mode = "rand_ntok"
    # mode = "rand_1init"
    # mode = "rand_ninit"
    mode = args.mode
    assert mode in ["unk","rand_1tok","rand_ntok","rand_1init","rand_ninit"]

    same_seeds(1)
    cache_size = 1 if mode in ["rand_1tok","rand_1init"] else 0

    # cache 0,则每次返回不同，cache为1则每次相同
    @lru_cache(maxsize=cache_size)
    def gen_rand_vec(embedding_dim):
        vec = torch.randn(embedding_dim)
        nn.init.normal_(vec, mean=0, std=embedding_dim ** -0.5)
        return vec

    @lru_cache(maxsize=cache_size)
    def unk_process(vocab_size):
        # 1. 随机取1个token， 2.随机若干个token  3. 随机初始化tensor
        idx = random.randint(0 + 1000, vocab_size - 1000)
        return idx


    langs = args.langs.split(",")
    pre_dict = load_dict(langs,os.path.join(args.pre_train_dir, "dict.txt"),is_ptm=True)
    ft_dict = load_dict(langs,args.ft_dict,is_ptm=False)
    print(len(pre_dict),len(ft_dict))
    data = torch.load(os.path.join(args.pre_train_dir, "model.pt"))
    model = data["model"]
    num_unk = 0
    mapping: List[int] = []
    for i in range(len(ft_dict)):
        word = ft_dict[i]
        idx = pre_dict.index(word)
        if idx==3:
            num_unk+=1
            # print(f"unk: {word}")
            if (i>3) and (mode in ["rand_1tok","rand_ntok"]): # 对后面的unk从ptm的大词表中随机取嵌入向量（固定1个或若干个）
                idx = unk_process(len(ft_dict))
        mapping.append(idx)

    # for name in ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]:
    for name in ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight","decoder.output_projection.weight"]:
        pre_tensor: torch.Tensor = model[name]
        ft_tensor = torch.zeros(
            [len(ft_dict), pre_tensor.shape[1]], dtype=pre_tensor.dtype, layout=pre_tensor.layout, device=pre_tensor.device,
        )
        for ft_i, pre_i in enumerate(mapping):
            ft_tensor[ft_i] = pre_tensor[pre_i]
            if (ft_i>3) and (pre_i==3) and mode in ["rand_1init","rand_ninit"]:
                vec = gen_rand_vec(embedding_dim=pre_tensor.shape[1])
                ft_tensor[ft_i] = vec


        model[name] = ft_tensor

    torch.save(data, args.output)
    ft_dict.save(args.ft_dict.replace(".txt",".ft.txt"))
    print(f"Save to {args.output} success, vocab_size: {len(mapping)}, unk: {num_unk}.")

if __name__ == "__main__":
    main()