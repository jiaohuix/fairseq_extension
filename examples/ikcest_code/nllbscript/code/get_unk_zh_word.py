'''
从两个词表（pretrain和finetune），获取finetune中中文unk
'''
import sys


def read_file(file):
    with open(file,'r',encoding='utf-8') as f:
        lines=f.readlines()
    return lines


def write_file(res,file):
    with open(file,'w',encoding='utf-8') as f:
        f.writelines(res)
    print(f'write to {file} success, total {len(res)} lines.')

def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

def get_zh(lines):
    res = []
    for line in lines:
        word,freq = line.strip().split()
        if is_contains_chinese(word):
            res.append(word+"\n")
    return res

if __name__ == '__main__':
    print("<ptm_dict> <fine_dict> <outfile>")
    ptm_dict = sys.argv[1]
    fine_dict = sys.argv[2]
    outfile = sys.argv[3]

    lines_ptm = read_file(ptm_dict)
    lines_fine = read_file(fine_dict)
    res_ptm = get_zh(lines_ptm)
    res_fine = get_zh(lines_fine)

    res = []
    for word in res_fine:
        if word not in res_ptm:
            res.append(word)
    write_file(res,outfile)