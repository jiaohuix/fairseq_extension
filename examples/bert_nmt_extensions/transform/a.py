from torch.autograd import Variable
import torch.nn as nn
import torch
conv1 = nn.Conv1d(in_channels=256, out_channels=100, kernel_size=2)
input = torch.randn(32, 35, 256)
# batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len
input = input.permute(0, 2, 1)
out = conv1(input)
print(out.size())
m = nn.Conv1d(16, 33, 3, stride=2)
input = torch.randn(20, 16, 50)
output = m(input)
print(output.size())
