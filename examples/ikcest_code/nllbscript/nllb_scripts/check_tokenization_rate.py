'''
tokenization rate, or the average number of tokens per sentence for each language
'''
import sys
import pandas as pd

def read_file(file):
    with open(file,'r',encoding='utf-8') as f:
        lines=[line.strip() for line in f.readlines()]
    return lines


if __name__ == '__main__':
    file = sys.argv[1]
    lines = read_file(file)

    df = pd.DataFrame(data={"text":lines})
    df["tex_len"] = df.text.map(lambda x:len(x.split()))
    print(df.describe())
    tok_rate = df["tex_len"].mean()
    print(f"tokenization rate: {tok_rate}")