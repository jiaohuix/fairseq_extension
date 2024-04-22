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
bash scripts/prep_dual.sh
bash scripts/pipe.sh
```

