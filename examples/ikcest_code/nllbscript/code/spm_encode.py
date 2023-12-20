#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import argparse
import contextlib
import sys
import sentencepiece as spm
import jieba
from zhconv import convert
from fairseq.data import Dictionary
from multiprocessing import Pool
from snownlp import SnowNLP

def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False


def write_file(res,file):
    with open(file,'w',encoding='utf-8') as f:
        f.writelines(res)
    print(f'write to {file} success, total {len(res)} lines.')


def zh_punc_split(text):
    text = convert(text,"zh-cn")
    words = jieba.lcut(text)
    new_text=""
    last_zh= False
    for word in words:
        if is_contains_chinese(word):
            new_text += word
            last_zh = True
        else:
            if last_zh:
                new_text += f" {word} "
            else:
                new_text += f"{word} "
            last_zh = False

    # new_text = " ".join(new_text.split())
    return new_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="sentencepiece model to use for encoding"
    )
    parser.add_argument(
        "--inputs", nargs="+", default=["-"], help="input files to filter/encode"
    )
    parser.add_argument(
        "--outputs", nargs="+", default=["-"], help="path to save encoded outputs"
    )
    parser.add_argument("--output_format", choices=["piece", "id"], default="piece")
    parser.add_argument(
        "--min-len",
        type=int,
        metavar="N",
        help="filter sentence pairs with fewer than N tokens",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        metavar="N",
        help="filter sentence pairs with more than N tokens",
    )

    parser.add_argument(
        "--use-jieba",
        action="store_true",
        help="whether to split punctuation with jieba." # spm前用jieba，分开标点；spm后对unk用jieba分开
    )

    parser.add_argument(
        "--dict-path",
        default=None,
        type=str,
        help = "Chinese dict, if exists, use jieba to split unk tokens. " # spm 后对unk词，用jieba切分，得到更小粒度的词
    )

    args = parser.parse_args()

    assert len(args.inputs) == len(
        args.outputs
    ), "number of input and output paths should match"

    sp = spm.SentencePieceProcessor()
    sp.Load(args.model)

    assert (args.dict_path is None) or os.path.exists(args.dict_path)

    dict_zh = Dictionary.load(args.dict_path) if args.dict_path is not None else None
    res = []
    if args.output_format == "piece":

        def encode(input):
            if args.use_jieba:
                input = zh_punc_split(input)
            input = sp.EncodeAsPieces(input)
            input = [word for word in input if  word != "▁"]
            # input = [word for (idx,word) in enumerate(input) if  (word != "▁") or (idx>1)]
            if dict_zh is not None:
                words = []
                for word in input:
                    idx = dict_zh.index(word)
                    if idx != 3:
                        words.append(word)
                    else:
                        a=" ".join(jieba.lcut(word))
                        res.append(f"{word}\t{a}\n")
                        # new_words = jieba.lcut(word)
                        new_words = SnowNLP(word).words
                        words.extend(new_words)
                input = words
            return input

    elif args.output_format == "id":

        def encode(input):
            return list(map(str, sp.EncodeAsIds(input)))

    else:
        raise NotImplementedError

    if args.min_len is not None or args.max_len is not None:

        def valid(line):
            return (args.min_len is None or len(line) >= args.min_len) and (
                args.max_len is None or len(line) <= args.max_len
            )

    else:

        def valid(lines):
            return True

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-"
            else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-"
            else sys.stdout
            for output in args.outputs
        ]

        stats = {
            "num_empty": 0,
            "num_filtered": 0,
        }

        def encode_line(line):
            line = line.strip()
            if len(line) > 0:
                line = encode(line)
                if valid(line):
                    return line
                else:
                    stats["num_filtered"] += 1
            else:
                stats["num_empty"] += 1
            return None

        for i, lines in enumerate(zip(*inputs), start=1):
            enc_lines = list(map(encode_line, lines))
            if not any(enc_line is None for enc_line in enc_lines):
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(" ".join(enc_line), file=output_h)
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        print("skipped {} empty lines".format(stats["num_empty"]), file=sys.stderr)
        print("filtered {} lines".format(stats["num_filtered"]), file=sys.stderr)
        if args.use_jieba:
            write_file(res,"out.zh")

if __name__ == "__main__":
    main()
