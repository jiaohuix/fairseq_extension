'''
删除含unk的双语
'''
from fairseq.data import Dictionary
import sys

def read_file(file):
    with open(file,'r',encoding='utf-8') as f:
        lines=f.readlines()
    return lines

def write_file(res,file):
    with open(file,'w',encoding='utf-8') as f:
        f.writelines(res)
    print(f'write to {file} success, total {len(res)} lines.')

def rm_pairs(src_lines,tgt_lines,dict):
    res_src = []
    res_tgt = []
    for src, tgt in zip(src_lines,tgt_lines):
        src = src.strip()
        tgt = tgt.strip()
        src_ids  = [dict.index(token) for token in src.split()]
        tgt_ids  = [dict.index(token) for token in tgt.split()]
        if (3 not in src_ids) and (3 not in tgt_ids):
            res_src.append(src+"\n")
            res_tgt.append(tgt+"\n")
    return res_src,res_tgt

if __name__ == '__main__':
    print("<inprefix> <src> <tgt> <dictfile>")
    inprefix = sys.argv[1]
    src = sys.argv[2]
    tgt = sys.argv[3]
    dictfile = sys.argv[4]


    src_lines = read_file(f"{inprefix}.{src}")
    tgt_lines = read_file(f"{inprefix}.{tgt}")
    dict = Dictionary.load(dictfile)
    total_lines = len(src_lines)
    res_src, res_tgt = rm_pairs(src_lines,tgt_lines,dict)
    write_file(res_src,f"{inprefix}.nounk.{src}")
    write_file(res_tgt,f"{inprefix}.nounk.{tgt}")

