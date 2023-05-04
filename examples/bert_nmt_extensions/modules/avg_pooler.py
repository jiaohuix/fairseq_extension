'''
pool层将bert_len简单拉到nmt_len,为了方便融合,应该添加模块做维度映射,写成类吧
init:  in_dim, out_dim, 初始权重
forward: [bsz,bert_len,bert_dim]
return: [bsz,nmt_len,nmt_dim]
先合并序列,然后映射维度.
学习下写单侧
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class AveragePooler(nn.Module):
    def __init__(self,in_features = 768, out_features = 512):
        super(AveragePooler,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.proj = nn.Linear(in_features, out_features) # 初始化

    def forward(self, x, dest_len, pad_mask=None,tbc=True):
        if tbc:
            x = x.transpose(0,1) # [T,B,C]-> [B,T,C]
        if pad_mask is not None:
            x[pad_mask] = 0.
        x = self.seq_pooler(x, dest_len) # [bsz,dest_len,in_dim]
        x = self.proj(x) # # [bsz,dest_len,out_dim]

        if tbc:
            x = x.transpose(0,1) # [B,T,C]-> [T,B,C]
        return x

    def seq_pooler(self,x, dest_len):
        x = x.transpose(-2,-1) # [bsz,seq1_len,dim] -> [bsz,dim, seq1_len]
        # x = F.adaptive_max_pool1d(x,output_size=dest_len) # [bsz,dim, seq1_len] -> [bsz,dim, seq2_len]
        x = F.adaptive_avg_pool1d(x,output_size=dest_len) # [bsz,dim, seq1_len] -> [bsz,dim, seq2_len]
        x = x.transpose(-2,-1) # [bsz,dim, seq2_len] -> [bsz,seq2_len,dim]
        return x

class MaxPooler(nn.Module):
    def __init__(self,in_features = 768, out_features = 512):
        super(MaxPooler,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.proj = nn.Linear(in_features, out_features) # 初始化

    def forward(self, x, dest_len, pad_mask=None,tbc=True):
        if tbc:
            x = x.transpose(0,1) # [T,B,C]-> [B,T,C]
        if pad_mask is not None:
            x[pad_mask] = 0.
        x = self.seq_pooler(x, dest_len) # [bsz,dest_len,in_dim]
        x = self.proj(x) # # [bsz,dest_len,out_dim]

        if tbc:
            x = x.transpose(0,1) # [B,T,C]-> [T,B,C]
        return x

    def seq_pooler(self,x, dest_len):
        x = x.transpose(-2,-1) # [bsz,seq1_len,dim] -> [bsz,dim, seq1_len]
        x = F.adaptive_max_pool1d(x,output_size=dest_len) # [bsz,dim, seq1_len] -> [bsz,dim, seq2_len]
        x = x.transpose(-2,-1) # [bsz,dim, seq2_len] -> [bsz,seq2_len,dim]
        return x

    def get_loss(self):
        pass


if __name__ == '__main__':
    bsz = 4
    pad_id = 1
    nmt_len, bert_len = 3, 4
    # nmt_len, bert_len = 7, 5
    nmt_dim, bert_dim = 5, 7
    input_ids = torch.randint(0,5, [bsz,bert_len])
    bert_out = torch.randn(bsz,bert_len, bert_dim)
    pad_mask = input_ids.eq(pad_id)
    print(pad_mask)
    bert_out[pad_mask] = 0
    print(bert_out)

    pooler = AveragePooler(in_features=bert_dim, out_features=nmt_dim)
    bert_pool_out = pooler(bert_out, dest_len=nmt_len)
    print(bert_pool_out.shape)
