'''
输入一列词，输出一个npy vector
'''
import sys
import fasttext
import numpy as np

def read_file(file):
    with open(file,'r',encoding='utf-8') as f:
        lines=f.readlines()
    return lines


def process(lines,model):
    res = []
    for word in lines:
        word = word.strip()
        vec = model.get_word_vector(word)
        res.append(vec)
    res = np.vstack(res)
    return res


if __name__ == '__main__':
    print("<word_file> <model_path> <outfile(npy)>")
    file = sys.argv[1]
    model_path = sys.argv[2]
    outfile = sys.argv[3]
    model = fasttext.load_model(model_path)
    lines = read_file(file)

    res = process(lines,model)
    np.save(outfile,res)
