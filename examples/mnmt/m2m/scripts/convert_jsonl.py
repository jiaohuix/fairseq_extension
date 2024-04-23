import json
import argparse

def convert_jsonl(infile, outfile):
    with open(infile, 'r', encoding='utf-8') as f_in, open(outfile, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            src_lang = list(data.keys())[0]
            tgt_lang = list(data.keys())[1]
            src_text = data[src_lang]
            tgt_text = data[tgt_lang]
            lang =  f"{src_lang}-{tgt_lang}"
            output_data = {
                "translation": {
                    src_lang: src_text,
                    tgt_lang: tgt_text
                },
                "lang": lang
            }
            f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL format from {src: tgt:} to {translations: {src, tgt}}")
    parser.add_argument("infile", type=str, help="Input JSONL file path")
    parser.add_argument("outfile", type=str, help="Output JSONL file path")
    args = parser.parse_args()
    convert_jsonl(args.infile, args.outfile)

