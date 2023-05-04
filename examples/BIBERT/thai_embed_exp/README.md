# Thai-bert实验

目标语言中文可以用bpe....



## 1 环境

```shell
/usr/bin/python3.8 -m pip install --upgrade pip
git clone https://github.com/fe1ixxu/BiBERT.git
cd BiBERT
pip install hydra-core==1.0.3 transformers subword-nmt tensorboardX tokenizers  jieba
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

char 

```
TEXT=download_prepare/data/th-zh-databin/
SAVE_DIR=ckpt/thzh_bert_char
epoch=50
tokens=4096
freq=8
mkdir -p $SAVE_DIR
ptm=monsoon-nlp/bert-base-thai
fairseq-train ${TEXT} --arch transformer_iwslt_de_en --ddp-backend no_c10d --optimizer adam --adam-betas '(0.9, 0.98)' \
--clip-norm 1.0 --lr 0.0004 --lr-scheduler inverse_sqrt --warmup-updates 1000 --warmup-init-lr 1e-07 --dropout 0.3 \
--weight-decay 0.00002 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens $tokens --update-freq $freq \
--attention-dropout 0.1 --activation-dropout 0.1 --max-epoch $epoch  --save-dir ${SAVE_DIR}  --encoder-embed-dim 768 --decoder-embed-dim 768 \
--no-epoch-checkpoints --save-interval 1 --pretrained_model $ptm --use_drop_embedding 8 >   $SAVE_DIR/train.log

```

bpe:

```
--max-source-positions  500 max_target_positions
```

```
TEXT=download_prepare/data_bpe/th-zh-databin/
SAVE_DIR=ckpt/thzh_bert_bpe
epoch=350
tokens=8192
freq=1
update=60000
mkdir -p $SAVE_DIR

ptm=monsoon-nlp/bert-base-thai
fairseq-train ${TEXT} --arch transformer_iwslt_de_en --max-source-positions  512 --ddp-backend no_c10d --optimizer adam --adam-betas '(0.9, 0.98)' \
--clip-norm 1.0 --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000  --max-update  $update --warmup-init-lr 1e-07 --dropout 0.3 \
--weight-decay 0.00002 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens $tokens --update-freq $freq \
--attention-dropout 0.1 --activation-dropout 0.1 --max-epoch $epoch  --save-dir ${SAVE_DIR}  --encoder-embed-dim 768 --decoder-embed-dim 768 \
--no-epoch-checkpoints --save-interval 1 --pretrained_model $ptm --use_drop_embedding 8 >   $SAVE_DIR/train.log
```

