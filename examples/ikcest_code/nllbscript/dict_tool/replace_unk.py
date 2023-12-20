'''
功能：
1.给一个多语言大词表V和一个中文小词表v，V中中文有许多缺失，要从v中取 ; 返回一个新词表V2 （词向量不变）
    1.构建两个有序词典, 其中小词典，可以是ikcest比赛的√
    2.从尾部遍历V，找到两个pos_list（乱码，或韩语的位置）
    3.从头遍历v，找到不存在V中的词unk，从pos_list中取位置，然后替换掉对应的key
    4.返回新的Vocab（随机替换，尝试有意义的替换）

    超过20的unk才能杯替换
    (3000个，首先去掉了好多乱码，然后只选择超过20的)

2.给一个多语言大词表V和一个中阿小词表v，以v为主体，返回对应大词表的索引；
3.根据大词表索引，将词嵌入变小，并替换encoder、decoder的embed和outprojection
'''
import sys
from collections import OrderedDict
def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False


def is_contains_arabic(strs):
    for _char in strs:
        if '\u0600' <= _char <= '\06ff':
            return True
    return False



def is_contains_korean(strs):
    for _char in strs:
        if '\uac00' <= _char <= '\ud7a3':
            return True
    return False

def read_file(file):
    with open(file,'r',encoding='utf-8') as f:
        lines=f.readlines()
    return lines

def write_file(res,file):
    with open(file,'w',encoding='utf-8') as f:
        f.writelines(res)
    print(f'write to {file} success, total {len(res)} lines.')

def build_vocab(lines,check_zh=False):
    dict = OrderedDict()
    for idx,line in enumerate(lines):
        line = line.strip().split()
        if len(line)==2:
            word, freq = line
        else:
            word, freq  = str(idx), 1
            # word, freq  = line[0], 1
        if not check_zh or is_contains_chinese(word):
            dict[word] = freq
    return dict

def get_unk(V1,V2):
    '''find unk index from V1 with V2 '''
    res = []
    for k,v in V2.items():
        if k not in V1.keys():
            res.append(k)
    return res

def get_ko_pos(dict):
    pos_ls, key_ls = [],[]
    for idx,(k,v) in enumerate(dict.items()):
        if is_contains_korean(k):
            pos_ls.append(idx)
            key_ls.append(k)
    return pos_ls[::-1],key_ls[::-1]


def check_is_encode_error(string):
    try:
        string.encode('gbk')
    except UnicodeEncodeError:
        return True
    return False

def is_all_japanese(strs):
    for _char in strs:
        if not '\u0800' <= _char <= '\u4e00':
            return False
    return True

if __name__ == '__main__':
    assert len(sys.argv)==4,f"usage: python {sys.argv[0]} <dict1> <dict2>  <outfile(dict.txt)> "
    file1 = sys.argv[1]
    file2 =  sys.argv[2]
    outfile=  sys.argv[3]

    lines1 = read_file(file1)
    lines2 = read_file(file2)

    dict1 = build_vocab(lines1)
    dict2 = build_vocab(lines2,check_zh=True)
    unk_ls = get_unk(dict1,dict2)
    pos_ls,key_ls = get_ko_pos(dict1)
    # new vocab
    new_vocab = list(dict1.keys())


    # replace unk
    for idx,unk in enumerate(unk_ls):
        rep_tok = key_ls[idx]
        rep_pos = pos_ls[idx]

        new_vocab[rep_pos] = unk

    # new vocab
    # dict3=OrderedDict()
    # for k in new_vocab:dict3[k]=1

    # check unk
    # new_unk = get_unk(dict3,dict2)
    # print(len(new_unk)) # 0
    write_file([tok+" 1\n" for tok in new_vocab],outfile)