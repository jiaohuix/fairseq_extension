import os
import sys
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from transformers import AutoTokenizer
access_token ="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"


def cut_words(sent,tokenizer):
    sent=sent.strip()
    input_ids = tokenizer(sent)["input_ids"]
    tokens = [tokenizer._convert_id_to_token(int(idx)) for idx in input_ids][:-2] # 去掉eos和langid
    tokenized_sent = " ".join(tokens)
    # tokenized_sent = tokenizer.convert_tokens_to_string(tokens) # 会去掉_
    tokenized_sent+="\n"
    return tokenized_sent

def main(infile,outfile,workers=1,lang="zho_Hans"):
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=access_token,src_lang=lang)
    with open(infile,"r",encoding="utf-8") as fr,open(outfile,"w",encoding="utf-8") as fw:
        pool=Pool(processes=workers)
        cut_words_fn= partial(cut_words,tokenizer=tokenizer)
        sentences = pool.imap(cut_words_fn,fr,chunksize=1000)
        for sent in tqdm(sentences):
            fw.write(sent)

if __name__ == '__main__':
    assert len(sys.argv)==5,f"usage: python {sys.argv[0]} <infile> <outfile>  <workers> <lang>(zh/th) " \
                            f"\n Zh/Thai language multiprocess word cut."
    infile=sys.argv[1]
    outfile=sys.argv[2]
    workers=int(sys.argv[3])
    lang=sys.argv[4]
    assert lang in ["zho_Hans","arb_Arab"]
    main(infile,outfile,workers,lang)