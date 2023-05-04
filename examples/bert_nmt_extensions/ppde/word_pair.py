'''
功能：根据词典，从原句中找出词典词，然后拼接到后面
dict={"a":"A"} 词典包含两列，要去重保证一对一，然后有个参数决定前向还是反向映射

'''
#coding:utf-8
import ahocorasick
from functools import partial
from multiprocessing import Pool

def make_AC(word_set):
    automaton  = ahocorasick.Automaton()
    for word in word_set:
        automaton.add_word(word,word)
    automaton.make_automaton()
    return automaton

def search(sent,automaton):
    name_list = set()
    for item in automaton.iter(sent): # item=(pos, word)
        name_list.add(item[1])
    name_list = list(name_list)
    return name_list



def get_ppde(lines, word_dict):
    automaton = make_AC(word_dict.keys())
    res = []
    for line in lines:
        line = line.strip()
        name_list = search(line,automaton)
        if len(name_list)>0:
            tmp = "".join([f"{name}={word_dict[name]}[SEP]" for name in name_list])
            line += "[SEP]"+tmp
        res.append(line+"\n")
    return res

if __name__ == "__main__":
    # test_ahocorasick()
    key_list = ["苹果", "香蕉", "梨", "橙子", "柚子", "火龙果", "柿子", "猕猴挑"]
    test_str_list = ["我最喜欢吃的水果有：苹果、梨和香蕉", "我也喜欢吃香蕉，但是我不喜欢吃梨"]
    key_dict = {
        "苹果":"apple",
        "香蕉":"banana",
        "梨":"pear",
        "橙子": "organge",
        "柚子": "grapefruit",
        "火龙果":"pitaya",
        "柿子": "persimmon",
        "猕猴挑":"macaque pick"
    }
    res = get_ppde(test_str_list,key_dict)
    print(res)

