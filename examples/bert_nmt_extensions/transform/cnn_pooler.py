import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn

bsz = 4
nmt_len, bert_len = 5, 7
# nmt_len, bert_len = 7, 5
nmt_dim, bert_dim = 512, 768


# 输入数据维度为[bsz,seq,dim]
bert_out = torch.randn(bsz,bert_len, bert_dim)

# 定义卷积层
kernel_size=7
stride=1
conv1d = nn.Conv1d(in_channels=bert_dim, out_channels=nmt_dim, kernel_size=kernel_size, stride=stride, padding=0)

# 计算padding的长度
dilation=1
effective_kernel_size = (kernel_size - 1) * dilation + 1
padding = (nmt_len - bert_len + effective_kernel_size - 1) // 2

# 在序列维度上进行卷积
x = bert_out.transpose(1, 2)  # 将维度转换为[bsz, dim, seq]
x = nn.functional.pad(x, (padding, padding), "constant", 0)  # 对序列进行padding
x = conv1d(x)  # 在序列维度上进行卷积

# 将卷积后的数据维度转换为[bsz,seq_len2,out_channels]
x = x.transpose(1, 2)  # 将维度转换为[bsz, seq_len2, out_channels]
print(x.shape)