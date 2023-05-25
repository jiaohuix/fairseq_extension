# span_mask_lm

span corruption 任务

参考：https://colab.research.google.com/drive/1lCkIOjR-uBnIsluzoRLJ22r7DCffwnEF?usp=sharing



## 1 安装环境

```shell
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
pip install --editable ./
git clone https://gitee.com/miugod/nmt_data_tools.git
#git clone https://github.com/jiaohuix/nmt_data_tools.git
pip install sacremoses tensorboardX   sacrebleu==1.5 apex omegaconf jieba  sentencepiece

git clone https://github.com/jiaohuix/fairseq_extension.git
cp -r fairseq_extension/examples/span_mask_lm/  .
```



## 2 数据处理

包含后面三个任务的数据。

```shell
bash span_mask_lm/scripts/prep-data.sh
ls data-bin
#iwslt14.tokenized.de-en(翻译数据)  span_corrup（单语文本段破坏）  trans_span_corrup（翻译文本段破坏）
```



## 3 NMT训练

直接训练翻译iwslt14德英翻译模型，34.6。

```shell
#<data> <save> <ptm>(opt) <src=de>(opt) <tgt=en>(opt)  <tokens=4096>(opt) <updates=60000>(opt)
bash span_mask_lm/scripts/train_nmt.sh  data-bin/iwslt14.tokenized.de-en checkpoints/nmt
# eval
bash bash span_mask_lm/scripts/eval.sh data-bin/iwslt14.tokenized.de-en checkpoints/nmt/checkpoint_best.pt 
```



## 4 span_corrup

4.1 先进行 span corruption的预训练，然后进行nmt微调。

```shell
# <data> <save> <tokens=2048>(opt) <updates=50000>(opt)   <density=0.15>(opt)  <length=3>(opt)
bash span_mask_lm/scripts/pretrain_span.sh data-bin/span_corrup checkpoints/span_lm 4096 50000 0.15 3
```

~~4.2 评估预训练任务~~



4.3 nmt微调

```shell
#<data> <save> <ptm>(opt) <src=de>(opt) <tgt=en>(opt)  <tokens=4096>(opt) <updates=60000>(opt)
bash span_mask_lm/scripts/train_nmt.sh  data-bin/iwslt14.tokenized.de-en checkpoints/span_lm_ft checkpoints/span_lm/checkpoint_best.pt de en 

```

4.4 评估nmt任务

```shell
# eval
bash bash span_mask_lm/scripts/eval.sh data-bin/iwslt14.tokenized.de-en checkpoints/nmt/checkpoint_best.pt 
```



## 5 trans_span_corrup

5.1 先进行 translation span corruption的预训练，然后进行nmt微调。（数据不同）

```shell
# <data> <save> <tokens=2048>(opt) <updates=50000>(opt)  <density=0.15>(opt)  <length=3>(opt)
bash span_mask_lm/scripts/pretrain_span.sh data-bin/trans_span_corrup checkpoints/trans_span_lm 4096 50000 0.5 3

```

~~5.2 评估预训练任务~~



5.3 nmt微调

```shell
#<data> <save> <ptm>(opt) <src=de>(opt) <tgt=en>(opt)  <tokens=4096>(opt) <updates=60000>(opt)
bash span_mask_lm/scripts/train_nmt.sh  data-bin/iwslt14.tokenized.de-en checkpoints/trans_span_lm_ft checkpoints/trans_span_lm/checkpoint_best.pt de en 

```

5.4 评估nmt任务

```shell
# eval
bash bash span_mask_lm/scripts/eval.sh data-bin/iwslt14.tokenized.de-en checkpoints/nmt/checkpoint_best.pt 
```

2023/5/22
