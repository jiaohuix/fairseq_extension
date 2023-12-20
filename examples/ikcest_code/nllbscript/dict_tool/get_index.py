'''
功能：给两个vocab，V和v，一大一小，V基本覆盖v； 返回v在V中的索引，若没找到就返回随机索引
'''


import json
import sys
from collections import OrderedDict

def read_file(file):
    with open(file,'r',encoding='utf-8') as f:
        lines=f.readlines()
    return lines

def write_file(res,file):
    with open(file,'w',encoding='utf-8') as f:
        f.writelines(res)
    print(f'write to {file} success, total {len(res)} lines.')

def build_vocab(lines):
    dict = OrderedDict()
    for idx,line in enumerate(lines):
        line = line.strip().split()
        if len(line)==2:
            word, freq = line
        else:
            print("len err...")
            word, freq  = str(idx), 1
            # word, freq  = line[0], 1
        dict[word] = idx

    return dict

def get_unk(V1,V2):
    '''find unk index from V1 with V2 '''
    res = []
    for k,v in V2.items():
        if k not in V1.keys():
            res.append(k)
    return res

def get_indices(dict1,dict2,unk_id=3):
    indices = []
    for key in dict2.keys():
        idx = dict1.get(key,unk_id) # unk
        indices.append(idx)
    return indices

class JsonTool:
    def load(self,filename):
        with open(filename,"r",encoding="utf-8") as f:
            data = json.load(f)
        return data

    def save(self,js_data,filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(js_data,f)
        print(f"save to {filename} success.")


if __name__ == '__main__':
    assert len(sys.argv)==4,f"usage: python {sys.argv[0]} <dict1> <dict2>  <outfile(idx.json)> "
    file1 = sys.argv[1]
    file2 = sys.argv[2] # 小词表
    outfile = sys.argv[3]

    lines1 = read_file(file1)
    lines2 = read_file(file2)

    dict1 = build_vocab(lines1)
    dict2 = build_vocab(lines2)
    # 返回索引
    indices = get_indices(dict1,dict2,unk_id=3)
    # 保存
    jstool = JsonTool()
    jstool.save({"index":indices},filename=outfile)
