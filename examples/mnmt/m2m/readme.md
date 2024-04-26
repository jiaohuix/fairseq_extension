训练：

run_translation_ikcest.pt

预测：

predict.py

评估：

eval.py

压缩提交文件

submit.py

## 1 train



环境：

```shell
conda create -n nmt python=3.10
conda activate nmt

pip install sacremoses sacrebleu==2.4.2 tensorboardX  apex fastcore omegaconf jieba  sentencepiece pythainlp datasets tokenizers wandb subword-nmt transformers[torch] accelerate protobuf py7zr torch -i https://pypi.tuna.tsinghua.edu.cn/simple

# 源码安装evaluate
git clone https://github.com/huggingface/evaluate.git
cd evaluate && pip install -e . && cd ..
cp -r  evaluate/metrics/sacrebleu .

conda update -n base -c defaults conda
conda install packaging
# https://github.com/NVIDIA/apex/issues/1594
git clone https://github.com/NVIDIA/apex.git
cd apex && pip install -e . && cd ..

```

数据集：

```shell
mkdir datasets && cd datasets
# iwslt17
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/datasets/iwslt2017
wget https://hf-mirror.com/datasets/iwslt2017/resolve/main/data/2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo.zip?download=true -O "iwslt2017/data/2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo.zip"
sed -i '50s#.*#REPO_URL = ""#' iwslt2017/iwslt2017.py


# ikcest22
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/datasets/miugod/ikcest2022
wget https://hf-mirror.com/datasets/miugod/ikcest2022/resolve/main/data/ZhFrRuThArEn.zip?download=true  -O "ZhFrRuThArEn.zip"
cp ZhFrRuThArEn.zip ikcest2022/data/ZhFrRuThArEn.zip

```

模型：

```shell
# m2m 
mkdir models && cd models
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/facebook/m2m100_418M
wget https://hf-mirror.com/facebook/m2m100_418M/resolve/main/pytorch_model.bin?download=true -O "m2m100_418M/pytorch_model.bin"

wget https://hf-mirror.com/facebook/m2m100_418M/resolve/main/sentencepiece.bpe.model?download=true  -O "m2m100_418M/sentencepiece.bpe.model"

# m2m 1.2B
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/facebook/m2m100_1.2B
wget https://hf-mirror.com/facebook/m2m100_1.2B/resolve/main/pytorch_model.bin?download=true -O "m2m100_1.2B/pytorch_model.bin"

wget https://hf-mirror.com/facebook/m2m100_1.2B/resolve/main/sentencepiece.bpe.model?download=true?download=true  -O "m2m100_1.2B/sentencepiece.bpe.model"


# mbart
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/facebook/mbart-large-cc25
wget https://hf-mirror.com/facebook/mbart-large-cc25/resolve/main/pytorch_model.bin?download=true -O "mbart-large-cc25/pytorch_model.bin"

# mt5
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/google/mt5-base
wget https://hf-mirror.com/google/mt5-base/resolve/main/pytorch_model.bin?download=true
 -O "mt5-base/pytorch_model.bin"

# nllb 1.3B
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/facebook/nllb-200-distilled-1.3B
wget https://hf-mirror.com/facebook/nllb-200-distilled-1.3B/resolve/main/pytorch_model.bin?download=true -O "nllb-200-distilled-1.3B/pytorch_model.bin"

wget https://hf-mirror.com/facebook/nllb-200-distilled-1.3B/resolve/main/sentencepiece.bpe.model?download=true -O "sentencepiece.bpe.model"

wget https://hf-mirror.com/facebook/nllb-200-distilled-1.3B/resolve/main/tokenizer.json?download=true -O "nllb-200-distilled-1.3B/tokenizer.json"


# todo: 熟悉hf-mirror.com的下载脚本
```



训练：

```
bash scripts/train.sh
```





推理加速

```
# eval



sed -i '943s#.*#  "num_madeup_words": 0,#' ckpt/ikcest_mft_luchen/checkpoint-20000/tokenizer_config.json
/root/.local/bin/ct2-transformers-converter --model ckpt/ikcest_mft_luchen/checkpoint-20000/  --output_dir ckpt/ct2_m2m --force


time python scripts/predict_ct2.py -d datasets/ikcest2022 -o output/ikcest_m2m_ct2.jsonl -m ckpt/ct2_m2m -n m2m_ct2 -b 64 -l 400 

#16 58s 6.5~6.9g vram  1MIN18S


# INT8 58S 2G-8G 
/root/.local/bin/ct2-transformers-converter --quantization int8 --model ckpt/ikcest_mft_luchen/checkpoint-20000/  --output_dir ckpt/ct2_m2m_q8 --force 
time python scripts/predict_ct2.py -d datasets/ikcest2022 -o output/ikcest_m2m_ct2_q8.jsonl -m  ckpt/ct2_m2m_q8 -n m2m_ct2_q8 -b 64 -l 400 

cat output/ikcest_m2m*.jsonl > ikcest.jsonl
python scripts/eval.py ikcest.jsonl report.csv
```



todo: predict改为ctranslate2	

```shell
pip install ctranslate2 
ct2-transformers-converter --model models/m2m100_418M --output_dir ct_m2m100_418
# https://opennmt.net/CTranslate2/guides/transformers.html#m2m-100
```



```python
import ctranslate2
import transformers

translator = ctranslate2.Translator("ct_m2m100_418")
tokenizer = transformers.AutoTokenizer.from_pretrained("models/m2m100_418M")
tokenizer.src_lang = "en"

source = tokenizer.convert_ids_to_tokens(tokenizer.encode("Hello world!"))
target_prefix = [tokenizer.lang_code_to_token["de"]]
results = translator.translate_batch([source], target_prefix=[target_prefix])
target = results[0].hypotheses[0][1:]

print(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))
```

？int8+vmap

```
--quantization int8
translator = ctranslate2.Translator(model_path, compute_type="int8")
results = translator.translate_batch([tokenize(input)], beam_size=1)
print(detokenize(results[0].hypotheses[0]))
translator = ctranslate2.Translator(model_path, device="cpu", intra_threads=8)
```



badam加速：

```shell
git clone git@github.com:Ledzy/BAdam.git
cd BAdam
pip install -e .
```



## 2 eval

transformers原生fp16推理，保存到ikcest_m2m.jsonl

```shell
mkdir output
time python scripts/predict.py -d datasets/ikcest2022 -o output/ikcest_m2m.jsonl -m ckpt/ikcest_mft_luchen/checkpoint-20000/ -n m2m -b 8 -l 400 
# bsz=8 8g vram
```

ctranslate fp16，保存结果到ikcest_m2m_ct2.jsonl

```shell
# 删除词表中的特殊词num_madeup_words（没有微调的模型不需要调整词表）
sed -i '943s#.*#  "num_madeup_words": 0,#' ckpt/ikcest_mft_luchen/checkpoint-20000/tokenizer_config.json

/root/.local/bin/ct2-transformers-converter --model ckpt/ikcest_mft_luchen/checkpoint-20000/  --output_dir ckpt/ct2_m2m --force
time python scripts/predict_ct2.py -d datasets/ikcest2022 -o output/ikcest_m2m_ct2.jsonl -m ckpt/ct2_m2m -n m2m_ct2 -b 64 -l 400 
```

ctranslate int8，保存结果到ikcest_m2m_ct2_q8.jsonl

```shell
/root/.local/bin/ct2-transformers-converter --quantization int8 --model ckpt/ikcest_mft_luchen/checkpoint-20000/  --output_dir ckpt/ct2_m2m_q8 --force 
time python scripts/predict_ct2.py -d datasets/ikcest2022 -o output/ikcest_m2m_ct2_q8.jsonl -m  ckpt/ct2_m2m_q8 -n m2m_ct2_q8 -b 64 -l 400 

```

合并三个模型的结果，并汇报分数：

```SHELL
cat output/ikcest_m2m*.jsonl > ikcest.jsonl
python scripts/eval.py ikcest.jsonl report.csv
```

结果如下：

| model        | AVG. BLEU | Speed    | VRAM      | Model Size |
| ------------ | --------- | -------- | --------- | ---------- |
| m2m_ct2_fp16 | 28.53     | 58s      | 7G/64bsz  | 1.9G       |
| m2m          | 28.953    | 13min30s | 18G/8bsz  | 1.9G       |
| m2m_ct2_int8 | 29.4      | 78s      | ≈8G/64bsz | 471M       |



## 3 submit

打包预测结果并提交：

```shell
python scripts/submit.py -i output/ikcest_m2m.jsonl -o results
```



## bugs

### 1 ctranslate转换错误

ValueError: Source vocabulary 0 has size 128112 but the model expected a vocabulary of size 128104

因为有madeupword=8，额外添加了8个特殊标记，导致词表与模型嵌入的128104匹配不上。

```
sed -i '943s#.*#  "num_madeup_words": 0,#' ckpt/ikcest_mft_luchen/checkpoint-20000/tokenizer_config.json




root/.local/bin/ct2-transformers-converter --model ckpt/ikcest_mft_luchen/checkpoint-20000/  --output_dir ckpt/ct2_m2m_zhfr --force
```



###  2 badam错误





