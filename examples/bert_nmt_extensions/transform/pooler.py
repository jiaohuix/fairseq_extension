'''
pool层将bert_len简单拉到nmt_len,为了方便融合,应该添加模块做维度映射,写成类吧
init:  in_dim, out_dim, 初始权重
forward: [bsz,bert_len,bert_dim]
return: [bsz,nmt_len,nmt_dim]
疑问: 先合并序列还是先映射维度?
学习下写单侧
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class AveragePooler(nn.Module):
    def __init__(self,in_features = 768, out_features = 512):
        super(AveragePooler,self).__init__()
        self.dim_proj = nn.Linear(in_features, out_features) # 初始化

    def forward(self, input_ids, dest_len):
        x = self.seq_pooler(input_ids, dest_len) # [bsz,dest_len,in_dim]
        x = self.dim_proj(x) # # [bsz,dest_len,out_dim]
        return x

    def seq_pooler(self,x, seq2_len):
        x = x.transpose(-2,-1) # [bsz,seq1_len,dim] -> [bsz,dim, seq1_len]
        x = F.adaptive_avg_pool1d(x,output_size=seq2_len) # [bsz,dim, seq1_len] -> [bsz,dim, seq2_len]
        x = x.transpose(-2,-1) # [bsz,dim, seq2_len] -> [bsz,seq2_len,dim]
        return x

# bsz = 4
# nmt_len, bert_len = 5, 7
# # nmt_len, bert_len = 7, 5
# nmt_dim, bert_dim = 512, 768
# bert_out = torch.randn(bsz,bert_len, bert_dim)
#
# bert_pool_out = seq_pooler(bert_out, dest_len=nmt_len)
# print(bert_pool_out.shape)
