import fasttext
import sys

if __name__ == '__main__':
    data = sys.argv[1]
    out = sys.argv[2]

    model = fasttext.train_unsupervised(data, minn=2, maxn=5, dim=300 ,epoch=5)
    model.save_model(out)