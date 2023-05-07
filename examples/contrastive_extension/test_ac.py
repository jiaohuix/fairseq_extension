import ahocorasick
from fairseq.data import Dictionary
import os

def build_prefix_starts(input_ids):
    # 用前缀和对前n个索引的长度和，记录第n个位置的索引  n+1
    prefix_starts = [0]
    for idx, ids in enumerate(input_ids):
        prefix_start = prefix_starts[idx]
        cur_start = prefix_start + 1 + len(str(int(ids)))  # 之前长度+空格+当前长度
        prefix_starts.append(cur_start)
    prefix_starts = prefix_starts[:-1]
    return prefix_starts


def build_ac(keys):
    # https://pyahocorasick.readthedocs.io/en/latest/
    automaton = ahocorasick.Automaton()
    for index, word in enumerate(keys):
        automaton.add_word(word, (index, word))
    automaton.make_automaton()
    return automaton

# 只find src的，后面加两边都匹配的
def find_entities_by_ac(input_ids, automaton = None):
    # entity_ids = []
    entity_starts = []
    entity_lens = []
    entity_keys = []
    ids_str = " ".join([str(int(ids)) for ids in input_ids])
    ids_str = f" {ids_str} " # 加空格，防止部分匹配，如string="5883 2" pattern="883 2" -> string=" 5883 2 " pattern=" 883 2 "
    prefix_starts = build_prefix_starts(input_ids)
    for item in automaton.iter_long(ids_str): # 匹配最长的字符串
        end, (ac_idx, key) = item
        start = end - len(key) + 1
        # real_start = start // 2   # woc，如果多位数，那就不是一隔1了
        # real_len = (len(key) + 1) // 2
        real_start = prefix_starts.index(start)
        real_len = len(key.split())
        entity_starts.append(real_start) # 只start
        # entity_ids.extend([real_start+l for l in range(real_len)]) # 所有的实体id
        entity_lens.append(real_len)
        entity_keys.append(key)
    return entity_starts, entity_lens, entity_keys


def find_entity_pairs(src_ids, tgt_ids, entity_dict, ac=None):
    if ac is None:
        ac = build_ac(entity_dict.keys())
    src_entity_ids, src_entity_lens = [],[]
    tgt_entity_ids, tgt_entity_lens = [],[]
    # 使用ac自动机找到原句中包含的实体信息
    src_ent_starts, src_ent_lens, src_ent_keys = find_entities_by_ac(src_ids, ac)
    # 根据entity_key找到tgt中是否匹配，匹配则重新记录
    tgt_ids_str = " ".join([str(int(ids)) for ids in tgt_ids])
    tgt_ids_str = f" {tgt_ids_str} " # 加空格，防止部分匹配，如string="5883 2" pattern="883 2" -> string=" 5883 2 " pattern=" 883 2 "
    print("src_ent_lens", len(src_ent_lens))
    # 记录 字符串位置：索引位置 的映射
    prefix_starts = build_prefix_starts(tgt_ids)
    for src_ent_start, src_ent_len, src_ent_key in zip(src_ent_starts, src_ent_lens, src_ent_keys):
        tgt_ent_key = entity_dict[src_ent_key]
        tgt_ent_start = tgt_ids_str.find(tgt_ent_key)
        # tgt中找到对应的实体词
        if tgt_ent_start != -1:
            tgt_real_start = prefix_starts.index(tgt_ent_start)
            tgt_real_len = len(tgt_ent_key.split())
            # save
            src_entity_ids.extend([src_ent_start + l  for l in range(src_ent_len)]) # 所有的实体token的id
            src_entity_lens.append(src_ent_len)
            tgt_entity_ids.extend([tgt_real_start + l  for l in range(tgt_real_len)])
            tgt_entity_lens.append(tgt_real_len)
    return src_entity_ids, src_entity_lens, tgt_entity_ids, tgt_entity_lens

def load_entity_dict(filename, src_dict=None, tgt_dict=None, topk=-1):
    """Load the entity dictionary from the filename

    Args:
        filename (str): the filename
        topk: the number of words in the dictionary.
    return: tree of ahocorasick
    """
    if topk==-1: topk=1e9
    assert os.path.exists(filename),f"entity dict path {filename} not exists."
    entity_dict = {}
    with open(filename,"r",encoding="utf-8") as f:
        # lines = [line.strip() for line in f.readlines()][:topk] # 截取topk
        lines = [line.strip() for line in f.readlines()] # 截取topk
        for line in lines[::-1]: # 从后往前取，若出现重复的key，取前面频率高的 #TODO: 以后再考虑一词多义
            src_entity, tgt_entity = line.split("\t")
            # 转为id
            src_entity_ids = src_dict.encode_line(src_entity.strip(),append_eos=False).tolist()
            tgt_entity_ids = tgt_dict.encode_line(tgt_entity.strip(),append_eos=False).tolist()
            src_entity_ids_str = " ".join([str(ids) for ids in src_entity_ids])
            tgt_entity_ids_str = " ".join([str(ids) for ids in tgt_entity_ids])
            # 加空格，防止部分匹配，如string="5883 2" pattern="883 2" -> string=" 5883 2 " pattern=" 883 2 "
            src_entity_ids_str = f" {src_entity_ids_str} "
            tgt_entity_ids_str = f" {tgt_entity_ids_str} "
            # forward、backward都要，方便双向训练
            entity_dict[src_entity_ids_str] = tgt_entity_ids_str
            entity_dict[tgt_entity_ids_str] = src_entity_ids_str
    return entity_dict


if __name__ == '__main__':
    a = "entwickeln"
    src = "und diese zwei zusammen zu bringen , erscheint vielleicht wie eine gewal@@ tige aufgabe . aber was ich ihnen zu sagen versuche ist , dass es trotz dieser komplexität einige einfache themen gibt , von denen ich denke , wenn wir diese verstehen , können wir uns wirklich weiter entwickeln ."
    tgt = "and bringing those two together might seem a very da@@ un@@ ting task , but what i &apos;m going to try to say is that even in that complexity , there &apos;s some simple the@@ mes that i think , if we understand , we can really move forward ."
    joint_dict_file = "../data-bin/iwslt14.tokenized.de-en/dict.de.txt"
    entity_dict_file = "../data-bin/iwslt14.tokenized.de-en/dict.de-en.bpe.txt"

    joint_dict = Dictionary.load(joint_dict_file)
    entity_dict = load_entity_dict(entity_dict_file,joint_dict,joint_dict,-1)
    src_ids = joint_dict.encode_line(src,append_eos=False)
    tgt_ids = joint_dict.encode_line(tgt)
    # src_entity_ids, src_entity_lens, tgt_entity_ids, tgt_entity_lens =find_entity_pairs(src_ids, tgt_ids, entity_dict)
    src_entity_ids, src_entity_lens, tgt_entity_ids, tgt_entity_lens =find_entity_pairs(tgt_ids,src_ids, entity_dict)
    print(src_entity_lens)
    print(tgt_entity_lens)
    for src_eids in src_entity_ids:
        idx = src_ids[src_eids]
        print(idx)
        print(joint_dict.string([idx]))