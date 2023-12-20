from fairseq.data import Dictionary
import sys

file =sys.argv[1]
dict = Dictionary.load(file)
text = "▁当 您 在 夜 色 中 沉 思 时 ▁, 您 会 感到 自己 正 站在 一 幅 由 星 星 状 的 指 尖 绘 制 并 散 布 出来 的 精 美 画 作 前 。"
num=0

for word in text.split():
    idx = dict.index(word)
    if idx ==3:
        num+=1
print("num unk:",num)