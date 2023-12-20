'''
fairseq 的api进行推理
'''

from fairseq.models.transformer import TransformerModel


ar2zh = TransformerModel.from_pretrained(
    model_name_or_path="F:/IKCEST/nllb_ckpt/",
    checkpoint_file="checkpoint_last.pt",
    data_name_or_path="../data-bin/zhar",
    bpe="sentencepiece",
    bpe_codes="../data-bin/zhar/flores200_sacrebleu_tokenizer_spm.model",
)
print(ar2zh)