# MLM

掩码语言模型(和fairseq关系不大哈哈)

资料：

1. huggingface： https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling



## 1. 安装环境：

```shell
/usr/bin/python3.8 -m pip install --upgrade pip
pip install datasets tokenizers evaluate -i https://pypi.tuna.tsinghua.edu.cn/simple
git clone https://github.com/huggingface/transformers.git
cd transformers/
pip install -e .
cp -r examples/pytorch/language-modeling/ .
```



报错：

`UnencryptedCookieSessionFactoryConfig` error when importing Apex

```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir \
--global-option="--cpp_ext" --global-option="--cuda_ext" ./

# 报错：python pip 安装apex报错：ERROR: Could not build wheels for cryptacular
# 参考：https://blog.csdn.net/qxqxqzzz/article/details/121257126
pip  install -v --no-cache-dir ./
```

源码安装：https://stackoverflow.com/questions/66610378/unencryptedcookiesessionfactoryconfig-error-when-importing-apex



## 2.数据处理：

```shell
cd mlm
bash prepare-data.sh
```



## 3.训练huggingface：

```shell
bash mlm/train_mlm.sh mlm/mlm_dataset/  ckpt_mlm bert-base-multilingual-cased 16

bash mlm/train_mlm.sh mlm/mlm_dataset/  ckpt_mlm ckpt_mlm/bert-base-multilingual-cased/checkpoint-12000/  16
```



colab: https://colab.research.google.com/drive/1p5dzjtEtLFxf0DEO0etcP0DpeNZtxrP3?usp=sharing

参数：https://zhuanlan.zhihu.com/p/363670628

bash train.sh datasets/ ckpt miugod/mbert_trim_ende

```shell
#!/bin/bash
if [ $# -lt 2 ];then
  echo "usage: bash $0 <indir> <outdir> <ptm>(opt) <bsz>(opt)"
  echo "ptm: [xlm-roberta-base,bert-base-multilingual-cased,jhu-clsp/bibert-ende,uklfr/gottbert-base...]"
  exit
fi

indir=$1
outdir=$2
ptm=${3:-"xlm-roberta-base"}
bsz=${4:-"8"}
epochs=1

python run_mlm.py \
    --model_name_or_path $ptm \
    --train_file $indir/train.txt \
    --validation_file $indir/valid.txt \
    --per_device_train_batch_size $bsz \
    --per_device_eval_batch_size $bsz \
    --do_train \
    --do_eval \
    --output_dir $outdir/$ptm --num_train_epochs=$epochs  \
    --fp16 --gradient_accumulation_steps  2  --save_steps 5000 --logging_dir $outdir/visual  --overwrite_output_dir --line_by_line
```

问题：

1 恢复训练

2 保存最佳权重



## 4. 上传hub

```shell
sudo apt-get install git-lfs

#transformers-cli login
huggingface-cli login
transformers-cli repo create your-model-name
git clone https://huggingface.co/username/your-model-name
cd your-model-name
git lfs install
git config --global user.email "email@example.com"
git add .
git commit -m "Initial commit"
git push https://username:password@huggingface.co/username/your-model-name
```

