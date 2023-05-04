'''
输出随机向量
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class RandPooler(nn.Module):
    def __init__(self,in_features = 768, out_features = 512):
        super(RandPooler,self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, dest_len, pad_mask=None,tbc=True):
        if tbc:
            seq_len,bsz, in_dim = x.size()
        else:
            bsz, seq_len,in_dim = x.size()

        x = torch.randn(bsz, dest_len, self.out_features)
        if tbc:
            x = x.transpose(0,1) # [T,B,C]-> [B,T,C]
        # TODO: DEVICE BUG...
        return x.to(x)

if __name__ == '__main__':
    bsz = 4
    nmt_len, bert_len = 5, 7
    # nmt_len, bert_len = 7, 5
    nmt_dim, bert_dim = 512, 768
    bert_out = torch.randn(bsz,bert_len, bert_dim)
    pooler = RandPooler(in_features=bert_dim, out_features=nmt_dim)
    bert_pool_out = pooler(bert_out, dest_len=nmt_len)
    print(bert_pool_out.shape)
