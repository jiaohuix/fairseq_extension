import torch
import torch.nn as nn
from fairseq.data import Dictionary
from functools import lru_cache
import random
import numpy as np

N=0


def same_seeds(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@lru_cache(maxsize=N)
def gen_rand_vec(embedding_dim):
    vec = torch.randn(embedding_dim)
    nn.init.normal_(vec, mean=0, std=embedding_dim**-0.5)
    return vec


same_seeds(1)
for _ in range(10):
    vec = gen_rand_vec(embedding_dim=5)
    print(vec)
