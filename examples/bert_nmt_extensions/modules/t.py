import torch
import torch.nn.functional as F

bsz,bert_len, bert_dim = 4,7,64
n=13
states = torch.randn(bsz,bert_len,n,bert_dim)

nmt_len = 5
nmt_dim = 512
nmt_state = torch.randn(bsz,nmt_len,nmt_dim )


def seq_pooler(x, dest_len):
    x = x.transpose(-2, -1)  # [bsz,seq1_len,dim] -> [bsz,dim, seq1_len]
    x = F.adaptive_max_pool1d(x, output_size=dest_len)  # [bsz,dim, seq1_len] -> [bsz,dim, seq2_len]
    x = x.transpose(-2, -1)  # [bsz,dim, seq2_len] -> [bsz,seq2_len,dim]
    return x


# res = F.adaptive_max_pool1d(states,output_size=nmt_len,)
states = states.permute(0,2,1,3).contiguous().view(bsz*n,bert_len,bert_dim) # [BNCL]

states1  = seq_pooler(states, dest_len=nmt_len)
states2 = F.adaptive_max_pool1d(states1, output_size=nmt_dim)
print(states1.shape)
print(states2.shape)

hidden_states = states2.view(bsz, n, nmt_len, nmt_dim).permute(0, 2, 1, 3)
print(hidden_states.shape) # [bsz,nmt_len,13,nmt_dim]

from numpy.random import uniform

def take_rand_prefix( state):
    # state: [B T C]
    import numpy as np
    bsz, seq_len, dim = state.size()
    rand_len = int(uniform(1, seq_len + 1))
    state_prefix = state[:, :rand_len, :]
    return state_prefix


nmt_state = torch.randn(bsz,nmt_len,nmt_dim )
state_prefix = take_rand_prefix(nmt_state)
print(state_prefix.shape)