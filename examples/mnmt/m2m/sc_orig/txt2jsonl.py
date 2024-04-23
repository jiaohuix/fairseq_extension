import json
import argparse


def txt_to_jsonl(infile, outfile, src, tgt):
    with open(infile, 'r', encoding='utf-8') as f_in, \
            open(outfile, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue

            data = {
                src: line[0],
                tgt: line[1]
            }
            json_line = json.dumps(data, ensure_ascii=False)
            f_out.write(json_line + '\n')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert txt translation data to jsonl format')
    parser.add_argument('infile', type=str, help='Input txt file')
    parser.add_argument('outfile', type=str, help='Output jsonl file')
    parser.add_argument('src', type=str, help='Source language code (e.g., zh)')
    parser.add_argument('tgt', type=str, help='Target language code (e.g., th)')
    args = parser.parse_args()

    txt_to_jsonl(args.infile, args.outfile, args.src, args.tgt)
