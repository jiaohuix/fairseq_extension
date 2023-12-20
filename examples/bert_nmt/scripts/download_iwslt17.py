'''
从iwslt17下载数据，并保存为fairseq需要的格式
@2023/12/1
'''
import os
import argparse

from datasets import load_dataset


def write_lines(res_ls, outfile, sep=None):
    if (sep is not None) and isinstance(sep, (str)):
        res_ls = [res + sep for res in res_ls]
    with open(outfile, "w", encoding="utf-8") as f:
        f.writelines(res_ls)
    print(f'write to {outfile} success, total {len(res_ls)} lines.')


# write
def write_hf2txt(dataset, src="zh", tgt="en", split="train", outdir="out"):
    """
    Write the source and target sentences from the dataset to separate files for a specific split.

    Args:
    dataset: The dataset containing the translation pairs.
    split (str): The split of the dataset to process (e.g., "train", "validation", "test").
    outdir (str): The directory where the output files will be written.

    Returns:
    None
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    src_ls = []
    tgt_ls = []
    dataset_split = dataset[split]

    for data in dataset_split:
        pair = data["translation"]
        src_ls.append(pair[src].strip())
        tgt_ls.append(pair[tgt].strip())

    split = split.replace("validation", "valid")
    outfile_src = os.path.join(outdir, f"{split}.{src}")
    outfile_tgt = os.path.join(outdir, f"{split}.{tgt}")
    write_lines(src_ls, outfile=outfile_src, sep="\n")
    write_lines(tgt_ls, outfile=outfile_tgt, sep="\n")


def get_args():
    parser = argparse.ArgumentParser(description="Download iwslt17 datasets")
    parser.add_argument('-s', '--src-lang', required=True, type=str, default='zh')
    parser.add_argument('-t', '--tgt-lang', required=True, type=str, default='en')
    args = parser.parse_args()
    return args


def process(args):
    data = "iwslt2017"
    src = args.src_lang
    tgt = args.tgt_lang
    lang = f"{src}-{tgt}"
    data_subset = f"{data}-{lang}"
    outdir = os.path.join(data, lang)

    dataset = load_dataset(data, data_subset, cache_dir=data)
    print("Sample data:", dataset["train"][0]["translation"])
    write_hf2txt(dataset, src, tgt, split="train", outdir=outdir)
    write_hf2txt(dataset, src, tgt, split="validation", outdir=outdir)
    write_hf2txt(dataset, src, tgt, split="test", outdir=outdir)


if __name__ == '__main__':
    args = get_args()
    process(args)
