'''
dynamic gate: 输入两个相同维度的序列h1和h2，gate=sigmoid(w1h1 + w2h2 + b), h' = gate* h1 + (1-gate) * h2
'''

import collections
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_dynamic_gate(embed_dim, gate_type):
    gate_map = {
                "dynamic": DynamicGate,
                "dynamiclr16": DynamicLR16Gate,
                "dynamiclr32": DynamicLR32Gate,
                "dynamiclr64": DynamicLR64Gate,
                "dynamiclr128": DynamicLR128Gate,
                "glu": DynamicGateGLU,
                "glunorm": DynamicGateGLUNorm,
                "zero": ZeroGate,
                } # 可以把参数添加在这里面，然后不用写那么多乱七八糟类，但是现在很晚了，明天花10分钟优化下
    gate = gate_map.get(gate_type, DynamicGate)(dim=embed_dim)
    return gate

class LowRankMLP(nn.Module):
    def __init__(self, in_dim=512,out_dim=512, k=4, alpha=1,bias1=False,bias2=False):
        super(LowRankMLP, self).__init__()
        self.down_proj = nn.Linear(in_dim,k, bias=bias1)
        self.up_proj =  nn.Linear(k,out_dim, bias=bias2)
        self.k = k
        self.alpha = alpha
        self.scaling = self.alpha / self.k

    def forward(self, x):
        x = self.up_proj(self.down_proj(x)) * self.scaling
        return x

class DynamicGate(nn.Module):
    def __init__(self,dim = 512):
        super(DynamicGate,self).__init__()
        self.w1 = nn.Linear(dim,dim, bias=False) # nmt
        self.w2 = nn.Linear(dim,dim, bias=True) # bert
        # self.reset_parameters()

    def forward(self, nmt_out, bert_out):
        g = F.sigmoid(self.w1(nmt_out) + self.w2(bert_out))
        return g * nmt_out + (1-g) * bert_out

    # def reset_parameters(self):
    #     nn.init.ones_(self.w1.weight)
    #     nn.init.zeros_(self.w2.weight)
    #     nn.init.zeros_(self.w2.bias)


class DynamicLRGate(nn.Module):
    def __init__(self,dim = 512,k = 4): # 这个单独跑吧，或者改成多个类
        super(DynamicLRGate,self).__init__()
        self.w1 = LowRankMLP(dim,dim,k=k ) # nmt
        self.w2 = LowRankMLP(dim,dim,k=k , bias2=True) # bert
        # self.reset_parameters()

    def forward(self, nmt_out, bert_out):
        g = F.sigmoid(self.w1(nmt_out) + self.w2(bert_out))
        return g * nmt_out + (1-g) * bert_out

class DynamicLR16Gate(DynamicLRGate):
    def __init__(self,dim = 512,k = 16):
        super(DynamicLR16Gate,self).__init__(dim=dim,k=k)

class DynamicLR32Gate(DynamicLRGate):
    def __init__(self,dim = 512,k = 32):
        super(DynamicLR32Gate,self).__init__(dim=dim,k=k)

class DynamicLR64Gate(DynamicLRGate):
    def __init__(self,dim = 512,k = 64):
        super(DynamicLR64Gate,self).__init__(dim=dim,k=k)

class DynamicLR128Gate(DynamicLRGate):
    def __init__(self,dim = 512,k = 128):
        super(DynamicLR128Gate,self).__init__(dim=dim,k=k)

class DynamicGateGLU(nn.Module):
    def __init__(self,dim = 512):
        super(DynamicGateGLU,self).__init__()
        self.w1 = nn.Linear(dim,dim, bias=False) # nmt
        self.w2 = nn.Linear(dim,dim, bias=True) # bert
        # self.reset_parameters()

    def forward(self, nmt_out, bert_out):
        g = F.silu(self.w1(nmt_out)) * self.w2(bert_out)
        return g * nmt_out + (1-g) * bert_out

    # def reset_parameters(self):
    #     nn.init.ones_(self.w1.weight)
    #     nn.init.zeros_(self.w2.weight)
    #     nn.init.zeros_(self.w2.bias)

class DynamicGateGLUNorm(nn.Module):
    def __init__(self,dim = 512):
        super(DynamicGateGLUNorm,self).__init__()
        self.w1 = nn.Linear(dim,dim, bias=False) # nmt
        self.w2 = nn.Linear(dim,dim, bias=True) # bert
        # self.reset_parameters()

    def forward(self, nmt_out, bert_out):
        g = F.silu(self.w1(nmt_out)) * self.w2(bert_out)
        g = F.sigmoid(g) # need?
        return g * nmt_out + (1-g) * bert_out

    # def reset_parameters(self):
    #     nn.init.ones_(self.w1.weight)
    #     nn.init.zeros_(self.w2.weight)
    #     nn.init.zeros_(self.w2.bias)



class ZeroGate(nn.Module): # 能不能将这个降维的用在dynamic gate上？
    '''
    low rank gate
    h' = hnmt + transform(hbert)*A*B
    A: Rd*k  B=0: Rk*d
    512*512=26w
    512*4*2=4096
    '''
    # GateCache = collections.namedtuple("Cache", ["A", "B"])
    def __init__(self,dim = 512, k=4, alpha=1):
        super(ZeroGate,self).__init__()
        self.A = nn.Linear(dim,k, bias=False)
        self.B = nn.Linear(k,dim, bias=False)
        self.k = k
        self.alpha = alpha
        self.scaling = self.alpha / self.k
        # self.reset_parameters()

    def forward(self, nmt_out, bert_out):
        nmt_out += self.B(self.A(bert_out)) * self.scaling
        # mean?
        return nmt_out

    def reset_parameters(self):
        nn.init.normal_(self.A.weight)
        nn.init.zeros_(self.B.weight)

    def get_loss(self):
        # 1. l2 norm

        # 2. param update delta mse

        # 3. update loss

        # 4. update cache

        # 5. forward loss in ce criterion
        pass
