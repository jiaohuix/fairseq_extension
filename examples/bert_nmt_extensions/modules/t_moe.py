'''
将moe用于层选择：
输入： bert_hid_states: list[state*13]
计算：根据gate_top_k_idx选择topk的表征，然后与gate_top_k_val进行加权求和
    1. stack，合并13层表征： [bsz,seq_len,layers,dim]
    有点不对劲 bsz,seq_len,  topk原始的moe对每个token取2个gate。
    而我是对sample取topk的融合？还是进行token level的混合?
    2. 计算gate score和idx
    3. 用index_select从13选择topk的states: [bsz,seq,2,dim]
    4. 与gate score相乘，求平均： [bsz,seq,dim]

输出：state=[bsz,seq,dim]
'''
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List
from torch import Tensor
class BaseGate(nn.Module):
    def __init__(self, num_expert):
        super().__init__()
        self.num_expert = num_expert

    def forward(self, x):
        raise NotImplementedError('Base gate cannot be directly used for fwd')

    # def set_loss(self, loss):
        # self.loss = loss
class NaiveGate(nn.Module):
    r"""
    A naive gate implementation that defines the standard behavior of the gate
    which determines which experts the tokens are going to.
    Both the indicies and the score, or confidence, are output to the parent
    module.
    The load-balance strategies are also designed to be implemented within the
    `Gate` module.
    """

    def __init__(self, d_model, top_k=2, token_level=True):
        super(NaiveGate,self).__init__()
        self.token_level = token_level
        self.gate = nn.Linear(d_model, 1)
        self.top_k = top_k

    def forward(self,hidden_states: List[Tensor]):
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
        topk_state = hidden_states.gather(dim=-2, index=gate_top_k_idx)
        topk_state = (topk_state * gate_score).mean(-2)

        return topk_state


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


if __name__ == '__main__':
    # 1.test raw naive gate
    # gate = NaiveGate(d_model=768, num_expert=13)
    # input_ids = torch.randn([4, 7, 768])
    # gate_top_k_idx, gate_score = gate(input_ids)
    # print(gate_top_k_idx.shape, gate_score.shape)
    # # [bsz,seq,topk] [bsz*seq,topk]

    # 2 use naive gate for bert layer select   序列长度保留，其实可以把seq求平均得到768的vec，然后对每层计算[bsz,13,topk]的val和idx
    # token level val: [bsz,2]
    # seq level val: [bsz,seq,2]
    d_model=64
    layers=13
    topk=2
    hidden_states = [torch.randn([4, 7, d_model]) for _ in range(layers)]
    bsz,seq_len,dim = hidden_states[0].size()
    dest_shape = [bsz,seq_len,topk,dim]
    gate = NaiveGate(d_model=d_model,token_level=False)
    ret = gate(hidden_states)
    print(ret.shape)
