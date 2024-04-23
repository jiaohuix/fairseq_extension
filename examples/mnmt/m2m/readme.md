训练：

run_translation_ikcest.pt

预测：

predict.py

评估：

eval.py

压缩提交文件

submit.py



环境：

```shell
conda create -n nmt python=3.10
conda activate nmt

pip install sacremoses sacrebleu tensorboardX  sacrebleu==1.5 apex fastcore omegaconf jieba  sentencepiece pythainlp datasets tokenizers wandb subword-nmt transformers[torch] accelerate protobuf py7zr torch -i https://pypi.tuna.tsinghua.edu.cn/simple

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