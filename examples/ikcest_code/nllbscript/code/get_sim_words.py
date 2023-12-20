'''
输入：预训练中文词，微调unk中文词，预训练词向量，unk微调词，topk
输出：unk词的topk近义词（不含unk本身）
'''
from nmt_data_tools.my_tools.annoy_indexer import AnnoyIndexer
import sys
import numpy as np

def read_file(file):
    with open(file,'r',encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
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
    print("<ptm_words> <unk_words> <ptm_npy> <fine_npy> <outfile> <topk>(optional)")
    ptm_words_file = sys.argv[1]
    unk_words_file = sys.argv[2]
    ptm_vec_file = sys.argv[3]
    fine_vec_file = sys.argv[4]
    outfile = sys.argv[5]
    topk=10
    num_trees = 100
    # read file
    ptm_words = read_file(ptm_words_file)
    unk_words = read_file(unk_words_file)

    tokens = ptm_words + unk_words
    ptm_vec = np.load(ptm_vec_file)
    fine_vec = np.load(fine_vec_file)
    vectors = np.vstack([ptm_vec,fine_vec])

    indexer = AnnoyIndexer(tokens,vectors,num_trees=num_trees)
    indexer.build_indexer()
    res = []  # "unk\t sim1 sim2 ... \n"
    for unk in unk_words:
        unk = unk.strip()
        vec = indexer.search_vector(unk)
        ids, distances = indexer.most_similar(vec, topk)
        nearest_tokens = [indexer.idx2token(idx) for idx in ids]
        nearest_tokens = [token for token in nearest_tokens if (token not in unk_words)]
        sim_tokens = " ".join(nearest_tokens)
        if len(nearest_tokens)>0:
            res.append(f"{unk}\t{sim_tokens}\n")

    punc_zh = "，？；！：（）“”‘’《》»«–～‟〞〝【】"
    punc_nllb = ",?;!:()\"\"\'\'<><>-~\"\"\"[]"
    for pz, pn in zip(punc_zh, punc_nllb):
        res.append(f"{pz}\t{pn}\n")
    write_file(res,outfile)