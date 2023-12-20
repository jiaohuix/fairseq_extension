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
    print("<infile> <outfile>")
    infile = sys.argv[1]
    outfile = sys.argv[2]
    lines = read_file(infile)
    res = get_zh(lines)
    write_file(res,outfile)