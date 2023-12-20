import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def translate(text,tokenizer,model,tgtid):
    inputs = tokenizer(text, return_tensors="pt")
    tokens = [tokenizer._convert_id_to_token(int(idx)) for idx in inputs["input_ids"][0]][:-2] # 去掉eos和langid
    tokenized_sent = " ".join(tokens)
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgtid],max_length=30)
    ret = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print(f"source: {tokenized_sent}, target: {ret}")
    return ret



if __name__ == '__main__':
    assert len(sys.argv)==4,"<src> <tgt> <text>"
    src = sys.argv[1]
    tgt = sys.argv[2]
    text=sys.argv[3]


    access_token = "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"
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=access_token,src_lang=src)
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=access_token)
    ret = translate(text,tokenizer,model,tgt)
    print(ret)