'''
python bert_nmt_extensions/tests/test_avg_pooler.py
'''
import os
import sys
__dir__=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__,"../../")))
import torch
import unittest
from bert_nmt_extensions.modules import AveragePooler

class TestAveragePooler(unittest.TestCase):
    def test_forward(self):
        # 构造输入数据
        bsz = 4
        nmt_len, bert_len = 5, 7
        # # nmt_len, bert_len = 7, 5
        nmt_dim, bert_dim = 512, 768
        x = torch.randn(bsz, bert_len, bert_dim)  # 输入维度为[batch_size, seq_len, input_size]
        pool = AveragePooler(in_features=bert_dim, out_features=nmt_dim)  # 初始化池化层

        # 前向传播
        y = pool(x, dest_len=nmt_len)

        # 验证输出维度是否正确
        self.assertEqual(y.shape, torch.Size([bsz, nmt_len, nmt_dim]))

    # def test_backward(self):
    #     # 构造输入数据
    #     x = torch.randn(2, 5, 10, requires_grad=True)  # 输入维度为[batch_size, seq_len, input_size]
    #     pool = MyPool(3, stride=2)  # 初始化池化层
    #
    #     # 前向传播
    #     y = pool(x)
    #
    #     # 构造输出误差
    #     loss = torch.sum(y)
    #
    #     # 反向传播
    #     loss.backward()
    #
    #     # 验证输入的梯度是否正确
    #     self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))

if __name__ == '__main__':
    unittest.main()
