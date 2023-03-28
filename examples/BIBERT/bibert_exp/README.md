# BIBERT实验

## 1 环境

```shell
/usr/bin/python3.8 -m pip install --upgrade pip
git clone https://github.com/fe1ixxu/BiBERT.git
cd BiBERT
pip install hydra-core==1.0.3 transformers subword-nmt tensorboardX
pip install --editable ./
```

## 2 数据

```shell
cp bibert_exp/*  download_prepare
cd download_prepare
bash download_and_prepare_data.sh
cd ..
```

## 3 训练

```shell
bash bibert_exp/train.sh
```



```
TEXT=./download_prepare/data/
SAVE_DIR=./models/one-way/

tokens=4096
freq=8
fairseq-train ${TEXT}de-en-databin/ --arch transformer_iwslt_de_en --ddp-backend no_c10d --optimizer adam --adam-betas '(0.9, 0.98)' \
--clip-norm 1.0 --lr 0.0004 --lr-scheduler inverse_sqrt --warmup-updates 1000 --warmup-init-lr 1e-07 --dropout 0.3 \
--weight-decay 0.00002 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens $tokens --update-freq $freq \
--attention-dropout 0.1 --activation-dropout 0.1 --max-epoch 75 --save-dir ${SAVE_DIR}  --encoder-embed-dim 768 --decoder-embed-dim 768 \
--no-epoch-checkpoints --save-interval 1 --pretrained_model bert-base-multilingual-cased --use_drop_embedding 8 > | tee -a $SAVE_DIR/train.log

```

不能用fp16:

```
return torch._C._nn.linear(input, weight, bias) RuntimeError: expected scalar type Float but found Half


```

