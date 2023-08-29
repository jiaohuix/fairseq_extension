from .data import LanguageTripleDataset
from .task import BertNMTTask
from .model import *
from .model_qb import *
'''
bert nmt 实验。
model测试直接加attn来转换query tokens
model_qb 是qblock的意思，使用解码器某一层参数初始化。【理想上这个会更好】
'''