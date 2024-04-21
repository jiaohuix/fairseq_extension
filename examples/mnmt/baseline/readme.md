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
# conda
# https://docs.anaconda.com/free/miniconda/
#https://zhuanlan.zhihu.com/p/459607806
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
export PATH=/root/miniconda3/bin:$PATH
export  PATH=/home/jiahui/anaconda3/bin:$PATH

conda env list
conda create -n nmt python=3.10
conda init 
# 重开终端
conda activate nmt   
pip install --upgrade pip
export PATH="/root/miniconda3/envs/nmt/bin/:$PATH"
```



```shell
# touch==2.0.0
pip install torch==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install sacremoses tensorboardX  sacrebleu==1.5 apex transformers peft fairseq==0.12.2   subword-nmt fastcore omegaconf jieba  sentencepiece pythainlp datasets tokenizers wandb subword-nmt

#bash scripts/env.sh
```



卸载conda：1 删除conda文件 2.修改bashrc，把conda的删掉



数据集下载：

```shell
mkdir datasets && cd datasets
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/datasets/iwslt2017
# 下载DeEnItNlRo-DeEnItNlRo.zip，复制到data/2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/

wget https://hf-mirror.com/datasets/iwslt2017/resolve/main/data/2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo.zip?download=true -O "DeEnItNlRo-DeEnItNlRo.zip"
cp DeEnItNlRo-DeEnItNlRo.zip iwslt2017/data/2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/

# 替换为本地路径
sed -i '50s#.*#REPO_URL = ""#' iwslt2017/iwslt2017.py

# 找common_data目录： find / -type d -name common_data
```



训练评估：

```shell
bash scripts/pipe.sh
```



| model    | it-en | en-it | ro-en | en-ro | nl-en | en-nl | it-ro | ro-it | avg  |
| -------- | :---- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ---- |
| uniTrans | 29.76 | 26.02 | 31.31 | 23.96 | 32.95 | 27.64 | 17.22 | 18.45 |      |
|          |       |       |       |       |       |       |       |       |      |

