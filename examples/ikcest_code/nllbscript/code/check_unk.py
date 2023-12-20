'''
查看单语含unk数量
'''
from fairseq.data import Dictionary
import sys
import pandas as pd

def read_file(file):
    with open(file,'r',encoding='utf-8') as f:
        lines=f.readlines()
    return lines

def process(lines,dict):
    num_unks=[]
    for line in lines:
        unks=0
        line = line.strip()
        for token in line.split():
            idx = dict.index(token)
            if idx == 3:
                unks+=1
        num_unks.append(unks)
    return num_unks

if __name__ == '__main__':
    print("<infile> <dictfile>")
    infile = sys.argv[1]
    dictfile = sys.argv[2]

    lines = read_file(infile)
    dict = Dictionary.load(dictfile)
    num_unks = process(lines,dict)
    df = pd.DataFrame(data={"unk":num_unks})
    print(df.describe())