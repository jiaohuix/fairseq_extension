'''
功能：输入两个中文词典，返回fairseq>bert 词典的映射
<s>
</s>
<pad>
<unk>
'''
import sys
import os
import zhconv
import json

def load_vocab(file, remove_bpe = True, add_simple = False):
    vocab = {}
    with open(file,'r',encoding='utf-8') as f:
        for idx,line in enumerate(f.readlines()):
            line = line.strip()
            if remove_bpe:
                line = line.replace("@@","")
            simple_tok = zhconv.convert(line, "zh-cn")
            if add_simple and (simple_tok != line):
                vocab[simple_tok] = idx
            vocab[line] = idx
    return vocab

def write_json(worddict,file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(worddict, f, indent=2, ensure_ascii=False)
    print(f"write to file {file} success.")


def write_file(res,file):
    with open(file,'w',encoding='utf-8') as f:
        f.writelines(res)
    print(f'write to {file} success, total {len(res)} lines.')



def chinese_token_map(token, bert_vocab,special_map):
    # 将中文词对应到多个bert词表的idx
    res_idx = []
    res_tok = []
    # 0. 特例
    if token in special_map.keys():
        map_token = special_map[token]
        return [bert_vocab[map_token]],[map_token]

    for tok in token:
        simple_tok = zhconv.convert(tok, "zh-cn")
        # 1. 直接找
        if bert_vocab.get(tok,None) is not None:
            res_idx.append(bert_vocab[tok])
            res_tok.append(tok)
        # 2. 繁转简
        elif bert_vocab.get(simple_tok,None) is not None:
            res_idx.append(bert_vocab[simple_tok])
            res_tok.append(simple_tok)
        else:
            pass
            # 3. 词向量

        # 4. 随机初始化（unk）
        if not res_idx:
            res_idx.append(bert_vocab["[UNK]"])
            res_tok.append("[UNK]")

    return res_idx,res_tok

def process(bert_vocab, fairseq_vocab, special_map):
    '''返回 vocab_map : {idx:[]}'''
    res_idxs = {}
    res_tokens = {}
    unk_tokens = []
    for token, idx in fairseq_vocab.items():
        res_idx,res_tok = chinese_token_map(token.replace("@@",""), bert_vocab, special_map)
        res_idxs[str(idx)] = [str(idx) for idx in res_idx]
        res_tokens[token] = res_tok
        if res_tok == ["[UNK]"]:
            unk_tokens.append(zhconv.convert(token, "zh-cn")+"\n")
    write_json(res_idxs,"map_idx.json")
    write_json(res_tokens,"map_token.json")
    write_file(unk_tokens, "unk.txt")

if __name__ == '__main__':
    assert len(sys.argv) >= 3, f"usage: python {sys.argv[0]} <bert_dict> <fairseq_dict>"
    special_map = {
        "<s>": "[CLS]",
        "</s>": "[SEP]",
        "<pad>": "[PAD]",
        "<unk>": "[UNK]",
    }
    bert_dict = sys.argv[1]
    fairseq_dict = sys.argv[2]
    bert_vocab = load_vocab(bert_dict,add_simple=True)
    fairseq_vocab = load_vocab(fairseq_dict, remove_bpe = False)
    process(bert_vocab,fairseq_vocab,special_map)
