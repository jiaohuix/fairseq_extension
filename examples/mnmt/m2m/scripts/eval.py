'''
todo: 在评估前，先进行分词。比如中文需要jieba、泰语pythainlp
'''
import pandas as pd
import sacrebleu
import argparse
import jieba
from pythainlp import word_tokenize

def tokenizer(text, lang):
    text = text.strip()
    if lang == "zh":
        text = " ".join(jieba.lcut(text))
    if lang == "th":
        toks = word_tokenize(text, keep_whitespace=False)
        text = " ".join(toks)
    return text


def compute_bleu_scores(data):
    bleu_scores = {}
    for model, df_model in data.groupby('model'):
        model_scores = []
        for lang_dir, df_lang in df_model.groupby('lang_dir'):
            tgt_lang = lang_dir.split("-")[1]
            if df_lang.empty:
                bleu_scores[(model, lang_dir)] = '-'
            else:
                references = [[tokenizer(ref, tgt_lang)] for ref in df_lang['ref'].tolist()]
                hypotheses = [tokenizer(pred, tgt_lang) for pred in df_lang['pred'].tolist()]
                bleu = sacrebleu.corpus_bleu(hypotheses, references)
                bleu_scores[(model, lang_dir)] = round(bleu.score,3)
                model_scores.append(bleu.score)
        score = sum(model_scores) / len(model_scores)
        bleu_scores[(model, 'avg')] = round(score, 3)
    return bleu_scores


def main(infile, outfile):
    data = pd.read_json(infile, lines=True)

    # Add lang_dir column
    data['lang_dir'] = data['src_lang'] + '-' + data['tgt_lang']

    # Compute BLEU scores
    bleu_scores = compute_bleu_scores(data)
    # Create DataFrame for output
    models = sorted(data['model'].unique())
    lang_dirs = sorted(data['lang_dir'].unique())
    lang_dirs = data['lang_dir'].unique()
    df_output = pd.DataFrame(index=models, columns=lang_dirs)

    # Fill DataFrame with BLEU scores
    for (model, lang_dir), bleu in bleu_scores.items():
        df_output.at[model, lang_dir] = bleu

    # Write DataFrame to CSV
    df_output.sort_values(by='avg', axis=0, ascending=True, inplace=True)
    df_output.to_csv(outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate machine translation models using sacrebleu.')
    parser.add_argument('infile', type=str, help='Input file in jsonl format')
    parser.add_argument('outfile', type=str, help='Output file in csv format')
    args = parser.parse_args()

    main(args.infile, args.outfile)

