import torch
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List
from torch import Tensor
from numpy.random import uniform


def create_select_layer(ls_type, bert_output_layer=-1,is_decoder=False):
    ls_map = {
        # select k
        "selk": {"cls": SelectK, "params": {"k": bert_output_layer}},
        # Stochastic
        "sto1": {"cls":StochasticLS,"params":{"last_k": 1}}, # default
        "sto2": {"cls":StochasticLS,"params":{"last_k": 2}},
        "sto4": {"cls":StochasticLS,"params":{"last_k": 4}},
        "sto6": {"cls":StochasticLS,"params":{"last_k": 8}},
        "sto8": {"cls":StochasticLS,"params":{"last_k": 8}},
        # sparsely gated
        "tok_moe_k3": {"cls": SparselyGatedLS, "params": {"top_k": 3, "token_level": True}},
        "seq_moe_k3": {"cls": SparselyGatedLS, "params": {"top_k": 3, "token_level": False}},
        # gshard
        "tok_moe_k2": {"cls":SparselyGatedLS,"params":{"top_k": 2, "token_level": True}},
        "seq_moe_k2": {"cls":SparselyGatedLS,"params":{"top_k": 2, "token_level": False}},
        # switch gate
        "tok_moe_k1": {"cls": SparselyGatedLS, "params": {"top_k": 1, "token_level": True}},
        "seq_moe_k1": {"cls": SparselyGatedLS, "params": {"top_k": 1, "token_level": False}},

        # sparsely gated dot product # 消融：tok_moe_k2_dot_avg，tok_moe_k2_dot_max
        "tok_moe_k2_dot_avg": {"cls": SparselyGatedDotLS, "params": {"top_k": 2, "token_level": True,"pool_type":"avg","is_decoder":is_decoder}},
        "seq_moe_k2_dot_avg": {"cls": SparselyGatedDotLS, "params": {"top_k": 2, "token_level": False,"pool_type":"avg","is_decoder":is_decoder}},

        "tok_moe_k2_dot_max": {"cls": SparselyGatedDotLS, "params": {"top_k": 2, "token_level": True,"pool_type":"max","is_decoder":is_decoder}},
        "seq_moe_k2_dot_max": {"cls": SparselyGatedDotLS, "params": {"top_k": 2, "token_level": False,"pool_type":"max","is_decoder":is_decoder}},


        # v2: remove decoder nmt state
        "tok_moe_k2_dot_avg_rm": {"cls": SparselyGatedDotLS,
                               "params": {"top_k": 2, "token_level": True, "pool_type": "avg",
                                          "is_decoder": is_decoder,"rm_nmt_state": True}},
        "seq_moe_k2_dot_avg_rm": {"cls": SparselyGatedDotLS,
                               "params": {"top_k": 2, "token_level": False, "pool_type": "avg",
                                          "is_decoder": is_decoder,"rm_nmt_state": True}},

        "tok_moe_k2_dot_max_rm": {"cls": SparselyGatedDotLS,
                               "params": {"top_k": 2, "token_level": True, "pool_type": "max",
                                          "is_decoder": is_decoder,"rm_nmt_state": True}},
        "seq_moe_k2_dot_max_rm": {"cls": SparselyGatedDotLS,
                               "params": {"top_k": 2, "token_level": False, "pool_type": "max",
                                          "is_decoder": is_decoder,"rm_nmt_state": True}},

        # v3: use rand prefix state of decoder
        "tok_moe_k2_dot_avg_rand": {"cls": SparselyGatedDotLS,
                                  "params": {"top_k": 2, "token_level": True, "pool_type": "avg",
                                             "is_decoder": is_decoder,"decoder_rand_prefix":True}},
        "seq_moe_k2_dot_avg_rand": {"cls": SparselyGatedDotLS,
                                  "params": {"top_k": 2, "token_level": False, "pool_type": "avg",
                                             "is_decoder": is_decoder,"decoder_rand_prefix":True}},

        "tok_moe_k2_dot_max_rand": {"cls": SparselyGatedDotLS,
                                  "params": {"top_k": 2, "token_level": True, "pool_type": "max",
                                             "is_decoder": is_decoder,"decoder_rand_prefix":True}},
        "seq_moe_k2_dot_max_rand": {"cls": SparselyGatedDotLS,
                                  "params": {"top_k": 2, "token_level": False, "pool_type": "max",
                                             "is_decoder": is_decoder,"decoder_rand_prefix":True}},
    }
    ls_type = ls_type if ls_type in ls_map.keys() else "sto1"
    ls_cls = ls_map[ls_type]["cls"]
    ls_params = ls_map[ls_type]["params"]
    layer_select = ls_cls(**ls_params)
    return layer_select

class SelectK(nn.Module):
    def __init__(self, k = 1,
                 ):
        super(SelectK,self).__init__()
        self.k = k

    def forward(self, hidden_states: List[Tensor],nmt_state=None, tbc = True):
        return hidden_states[self.k]

class StochasticLS(nn.Module):
    '''Stochastic Layer Selection'''
    def __init__(self, last_k = 1,
                 ):
        super(StochasticLS,self).__init__()
        self.last_k = last_k

    def forward(self,hidden_states: List[Tensor],nmt_state=None, tbc = True):
        if self.last_k == 1:
            return hidden_states[-1]
        if self.training:
            rand_num = torch.rand(1)
            rand_layers = int(rand_num * self.last_k) + 1
            bert_out = hidden_states[-rand_layers]
        else:
            bert_out = torch.mean(torch.stack(hidden_states[-self.last_k:]), dim=0)
        return bert_out


class SparselyGatedLS(nn.Module):
    r"""
    A naive gate implementation that defines the standard behavior of the gate
    which determines which experts the tokens are going to.
    Both the indicies and the score, or confidence, are output to the parent
    module.
    """
    # TODO: FIX HARD CODE FOR BERT_DIM
    def __init__(self, dim=768, top_k=2, token_level=True,
                 ):
        super(SparselyGatedLS,self).__init__()
        self.dim = dim
        self.token_level = token_level
        self.gate = nn.Linear(dim, 1)
        self.top_k = top_k

    def forward(self,hidden_states: List[Tensor],nmt_state=None, tbc = True):
        assert len(hidden_states)>1, "len(hidden_states)>1"
        assert self.top_k >=1 and self.top_k <= len(hidden_states), "topk∈[1,num_layers+1]"
        if tbc:
            hidden_states = [state.permute(1,0,2) for state in hidden_states]
        bsz, seq_len, dim = hidden_states[0].size()
        dest_shape = [bsz,seq_len, self.top_k, dim]
        # # step1. merge input
        hidden_states = torch.stack(hidden_states, dim=-2) # [B L 13 D]

        # step2: forward gate val and idx
        gate_top_k_idx, gate_score = self.forward_gate(
            hidden_states.mean(dim=1) if self.token_level else hidden_states
        ) # token level: [B K], seq level: [B L K]

        if self.token_level:
            gate_top_k_idx = gate_top_k_idx.unsqueeze(1).unsqueeze(-1).expand(dest_shape)  # [B,K]->[B,1,K,1]->[B,L,K,D]
            gate_score = gate_score.unsqueeze(1).unsqueeze(-1)
        else:
            gate_top_k_idx = gate_top_k_idx.unsqueeze(-1).expand(dest_shape)  # [B,L,K]->[B,L,K,1]->[B,L,K,D]
            gate_score = gate_score.unsqueeze(-1)

        # step3: select topk states
        topk_states = hidden_states.gather(dim=-2, index=gate_top_k_idx)
        bert_out = (topk_states * gate_score).mean(-2)

        if tbc:
            bert_out = bert_out.permute(1,0,2)
        return bert_out


    def forward_gate(self, inp, return_all_scores=False):
        r"""
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        gate = self.gate(inp) # [bsz,seq_len,layers,1]
        gate = gate.squeeze(-1)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]

        # B x L  x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate # gate当成原始的logits，gate_score是softmax后的prob
        return gate_top_k_idx, gate_score



class SparselyGatedDotLS(SparselyGatedLS):
    r"""
    A naive gate implementation that defines the standard behavior of the gate
    which determines which experts the tokens are going to.
    Both the indicies and the score, or confidence, are output to the parent
    module.

    改进：之前的SparselyGatedLS借鉴了moe的路由机制，只考虑到bert表征进行层选择，并为nmt模型每层动态分配融合的bert表征。
    缺点：没考虑到每层的nmt表征，因此此类采用点乘将bert和nmt表征融合，为了考虑序列长度问题，简单的使用动态的maxpool。
    """
    # TODO: FIX HARD CODE FOR BERT_DIM
    def __init__(self, dim=512, top_k=2, token_level=True, pool_type="avg",
                 is_decoder = False,
                 decoder_rand_prefix = False,
                 rm_nmt_state = False,
                 ):
        super(SparselyGatedDotLS,self).__init__()
        self.dim = dim
        self.token_level = token_level
        self.gate = nn.Linear(dim, 1)
        self.top_k = top_k

        # 维度pool（默认avg）， 序列pool是max
        self.pool_type=pool_type
        pool_map = {"max":F.adaptive_max_pool1d,"avg":F.adaptive_avg_pool1d}
        assert pool_type in ["max", "avg"]
        self.dim_pooler = pool_map[pool_type]

        # 处理解码器训推长度不一致问题
        self.is_decoder = is_decoder
        self.decoder_rand_prefix = decoder_rand_prefix
        self.rm_nmt_state = rm_nmt_state
        if not is_decoder:
            self.rm_nmt_state = False
        if is_decoder and (rm_nmt_state == decoder_rand_prefix == True):
            self.decoder_rand_prefix=False
            print("rm_nmt_state和decoder_rand_prefix只能有一个为True")

    def seq_pooler(self, x, dest_len):
        x = x.transpose(-2, -1)  # [bsz,seq1_len,dim] -> [bsz,dim, seq1_len]
        x = F.adaptive_max_pool1d(x, output_size=dest_len)  # [bsz,dim, seq1_len] -> [bsz,dim, seq2_len]
        x = x.transpose(-2, -1)  # [bsz,dim, seq2_len] -> [bsz,seq2_len,dim]
        return x

    def take_rand_prefix(self, state):
        # state: [B T C]
        bsz, seq_len, dim = state.size()
        rand_len = int(uniform(1, seq_len + 1))
        state_prefix = state[:,:rand_len,:]
        return state_prefix

    def forward(self,hidden_states: List[Tensor],nmt_state=None, tbc = True):
        #                 bert_encoder_out = self.select_layer(bert_encoder_outs,nmt_state=x)
        assert len(hidden_states)>1, "len(hidden_states)>1"
        assert self.top_k >=1 and self.top_k <= len(hidden_states), "topk∈[1,num_layers+1]"
        if tbc:
            hidden_states = [state.permute(1,0,2) for state in hidden_states]
        n_states = len(hidden_states)
        bsz, bert_len, bert_dim = hidden_states[0].size()
        dest_shape = [bsz,bert_len, self.top_k, bert_dim]
        # # step1. merge input
        hidden_states = torch.stack(hidden_states, dim=-2) # [B L 13 D]

        # dot product on bert and nmt state
        # nmt_state序列长度变到bert
        # bert维度变为nmt
        if (nmt_state is not None) and (not self.rm_nmt_state):
            if tbc: nmt_state = nmt_state.permute(1, 0, 2)

            # take rand prefix of decoder state trick
            if self.is_decoder and self.decoder_rand_prefix:
                nmt_state = self.take_rand_prefix(nmt_state)

            bsz, nmt_len, nmt_dim = nmt_state.size()
            nmt_state = self.seq_pooler(nmt_state, dest_len=bert_len).unsqueeze(-2)  # [bsz,bert_len,1,nmt_dim]
            hidden_states_pool = hidden_states.view(bsz * bert_len, n_states, bert_dim)  # [B*L,13,C]
            hidden_states_pool = self.dim_pooler(hidden_states_pool, output_size=nmt_dim) # dimension pooler
            hidden_states_pool = hidden_states_pool.view(bsz, bert_len, n_states, nmt_dim)
            hidden_states_dot = hidden_states_pool * nmt_state
        else:
            hidden_states_pool = hidden_states.view(bsz * bert_len, n_states, bert_dim)  # [B*L,13,C]
            hidden_states_pool = self.dim_pooler(hidden_states_pool, output_size=self.dim)  # dimension pooler
            hidden_states_pool = hidden_states_pool.view(bsz, bert_len, n_states, self.dim)
            hidden_states_dot = hidden_states_pool
        # step2: forward gate val and idx
        gate_top_k_idx, gate_score = self.forward_gate(
            hidden_states_dot.mean(dim=1) if self.token_level else hidden_states_dot
        ) # token level: [B K], seq level: [B L K]

        if self.token_level:
            gate_top_k_idx = gate_top_k_idx.unsqueeze(1).unsqueeze(-1).expand(dest_shape)  # [B,K]->[B,1,K,1]->[B,L,K,D]
            gate_score = gate_score.unsqueeze(1).unsqueeze(-1)
        else:
            gate_top_k_idx = gate_top_k_idx.unsqueeze(-1).expand(dest_shape)  # [B,L,K]->[B,L,K,1]->[B,L,K,D]
            gate_score = gate_score.unsqueeze(-1)

        # step3: select topk states
        topk_states = hidden_states.gather(dim=-2, index=gate_top_k_idx)
        bert_out = (topk_states * gate_score).mean(-2)

        if tbc:
            bert_out = bert_out.permute(1,0,2)
        return bert_out



if __name__ == '__main__':
    bert_outs = [torch.randn([4, 7, 768]) for _ in range(13)]
    nmt_state  = torch.randn([4, 5, 512])
    # gate = NaiveGate(d_model=768 ,num_expert=13)
    # token_gate = SparselyGatedLS(d_model=768,token_level=True)
    # bert_out = token_gate(bert_outs)
    # print(bert_out.shape)
    #
    # seq_gate = SparselyGatedLS(d_model=768,token_level=False)
    # bert_out = seq_gate(bert_outs)
    # print(bert_out.shape)


    seq_gate = SparselyGatedDotLS(dim=512,token_level=False,pool_type="max")
    bert_out = seq_gate(bert_outs,nmt_state=nmt_state, tbc=False)
    print(bert_out.shape)


    tok_gate = SparselyGatedDotLS(dim=512,token_level=True,pool_type="max")
    bert_out = tok_gate(bert_outs,nmt_state=nmt_state, tbc=False)
    print(bert_out.shape)