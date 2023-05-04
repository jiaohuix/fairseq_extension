'''
使用一维卷积,先将序列pad到需要的长度,然后卷积到目标长度和通道:
[bsz,bert_len,768] -> [bsz,nmt_len,512]
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvPooler(nn.Module):
    def __init__(self,in_features = 768, out_features = 512, kernel_size=3):
        super(ConvPooler,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.proj = nn.Conv1d(in_channels=in_features, out_channels=out_features,
                              kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x, dest_len, pad_mask=None, tbc=True):
        if tbc:
            x = x.transpose(0,1) # [T,B,C]-> [B,T,C]
        if pad_mask is not None:
            x[pad_mask] = 0.
        x = self.pad_seq(x, dest_len)
        # 在序列维度上进行卷积
        x = self.proj(x) # # [bsz,dest_len,out_dim]
        x = x.transpose(1, 2)  # 将维度转换为[bsz, seq_len2, out_channels]
        # pad
        if dest_len > x.size(1):
            delta = dest_len - x.size(1)
            padding = (
                0, 0,
                0, delta,
                0, 0,
            )
            x = F.pad(x, padding)
        if tbc:
            x = x.transpose(0,1) # [B,T,C] ->  [T,B,C]
        return x

    def pad_seq(self,x, dest_len):
        src_len = x.size(1)
        dilation = 1
        effective_kernel_size = (self.kernel_size - 1) * dilation + 1
        padding = (dest_len - src_len + effective_kernel_size - 1) // 2
        x = x.transpose(1, 2)  # 将维度转换为[bsz, dim, seq]
        x = F.pad(x, (padding, padding), "constant", 0)  # 对序列进行padding
        return x


if __name__ == '__main__':
    bsz = 72
    # nmt_len, bert_len = 5, 7
    nmt_len, bert_len = 51, 56
    nmt_dim, bert_dim = 512, 768
    # bert_out = torch.randn(bsz,bert_len, bert_dim)
    bert_out = torch.randn(bert_len,bsz, bert_dim)
    pooler = ConvPooler(in_features=bert_dim, out_features=nmt_dim)
    bert_pool_out = pooler(bert_out, dest_len=nmt_len)
    print(bert_pool_out.shape)
