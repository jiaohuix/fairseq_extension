iwslt17 ikcest22单向基线

目录：

```shell
scripts/
├── prep_data.py 数据处理
├── bin.sh 二值化
├── train.sh 训练
├── eval.sh 评估
└── pipe.sh 一键数据处理训练评估
```

环境安装：

```shell
# touch==2.0.0
#bash scripts/env.sh
pip install sacremoses tensorboardX  sacrebleu==1.5 apex fairseq==0.12.2 fastcore omegaconf jieba  sentencepiece pythainlp datasets tokenizers wandb subword-nmt -i https://pypi.tuna.tsinghua.edu.cn/simple

```



数据集下载：

```shell
mkdir datasets && cd datasets
# iwslt17
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/datasets/iwslt2017
# 下载DeEnItNlRo-DeEnItNlRo.zip，复制到data/2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/

wget https://hf-mirror.com/datasets/iwslt2017/resolve/main/data/2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo.zip?download=true -O "DeEnItNlRo-DeEnItNlRo.zip"
cp DeEnItNlRo-DeEnItNlRo.zip iwslt2017/data/2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/

# 替换为本地路径
sed -i '50s#.*#REPO_URL = ""#' iwslt2017/iwslt2017.py


# ikcest22
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/datasets/miugod/ikcest2022
wget https://hf-mirror.com/datasets/miugod/ikcest2022/resolve/main/data/ZhFrRuThArEn.zip?download=true  -O "ZhFrRuThArEn.zip"
cp ZhFrRuThArEn.zip ikcest2022/data/ZhFrRuThArEn.zip


```



训练评估：

```shell
bash scripts/train_6layer.sh src tgt data-bin/ikcest2022_multi/ ckpt/ikest2022_multi  ikcest2022
bash scripts/train_6layer.sh src tgt data-bin/iwslt2017_multi/ ckpt/iwslt2017_multi  iwslt2017
```



zhru test缺了一行：

```shell
echo 'Однако они подчеркнули, что это было вполне объяснимое убийство из мести и что они не получили одобрения лидеров движения.' |sacremoses -l ru -j 4 tokenize | subword-nmt apply-bpe -c  train_data/ikcest2022/zh-ru/codes.txt

echo "然而，他们强调说，这属于可以理解的报复性杀人，他们并未得到该运动领导人的批准。" | python -m jieba |sed "s/\///g" | subword-nmt apply-bpe -c  train_data/ikcest2022/zh-ru/codes.txt


sed -i '938i\<to_ru> 然而 ， 他们 强调 说 ， 这 属于 可以 理解 的 报@@ 复@@ 性 杀@@ 人 ， 他们 并未 得到 该 运动 领导人 的 批准 。' train_data/ikcest2022/zh-ru/test.src
sed -i '938i\Однако они подчеркну@@ ли , что это было вполне объя@@ сни@@ мое убий@@ ство из ме@@ сти и чт/ikcest2022/zh-ru/test.tgt

rm -r data-bin/ikcest2022/zh-ru
bash scripts/bin.sh train_data/ikcest2022/zh-ru/ data-bin/ikcest2022/zh-ru src tgt 1 dict/bpe_vocab.txt 

# eval
bash scripts/eval.sh zh ru data-bin/ikcest2022/zh-ru/ ckpt/ikcest2022_multi/checkpoint_best.pt > ckpt/ikcest2022_multi/gen_zh-ru.txt


cat ckpt/ikcest2022_multi/gen_zh-ru.txt | grep -P "^D" | sort -V | cut -f 3- > ckpt/ikcest2022_multi/report/zh_ru.rst
```

