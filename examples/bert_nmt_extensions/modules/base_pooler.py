import torch
import torch.nn as nn
import torch.nn.functional as F

class BasePooler(nn.Module):
    def __init__(self,in_features = 768, out_features = 512):
        super(BasePooler,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.proj = nn.Linear(in_features, out_features) # 初始化

    def forward(self, x, dest_len, pad_mask=None,tbc=True):
        raise NotImplemented

    def get_loss(self):
        pass

    def update_cache(self):
        pass


