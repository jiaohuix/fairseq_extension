# bart

denoise 任务

参考：https://colab.research.google.com/drive/10RtSW1NAGN4BuF4SbuKnBI1xSmptPenf?usp=sharing



## 1 安装环境

```shell
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
pip install --editable ./
git clone https://gitee.com/miugod/nmt_data_tools.git
#git clone https://github.com/jiaohuix/nmt_data_tools.git
pip install sacremoses tensorboardX   sacrebleu==1.5 apex omegaconf jieba  sentencepiece

git clone https://github.com/jiaohuix/fairseq_extension.git
cp -r fairseq_extension/examples/bart/  .
```



## 2 数据处理

包含后面三个任务的数据。

```shell
bash bart/scripts/prep-data.sh
ls data-bin
#iwslt14.tokenized.de-en(翻译数据)  denoise_data（降噪数据）  
```



## 3 NMT训练

直接训练翻译iwslt14德英翻译模型，34.6。

```shell
#<data> <save> <ptm>(opt) <src=de>(opt) <tgt=en>(opt)  <tokens=4096>(opt) <updates=60000>(opt)
bash bart/scripts/train_nmt.sh  data-bin/iwslt14.tokenized.de-en checkpoints/nmt
# eval
bash bash bart/scripts/eval.sh data-bin/iwslt14.tokenized.de-en checkpoints/nmt/checkpoint_best.pt 
```



## 4 bart

4.1 先进行 span corruption的预训练，然后进行nmt微调。

```shell
# <data> <save> <tokens=2048>(opt) <updates=50000>(opt)
bash bart/scripts/pretrain_bart.sh data-bin/denoise_data checkpoints/bart
```

~~4.2 评估预训练任务~~



4.3 nmt微调

```shell
#<data> <save> <ptm>(opt) <src=de>(opt) <tgt=en>(opt)  <tokens=4096>(opt) <updates=60000>(opt)
bash bart/scripts/train_nmt.sh  data-bin/iwslt14.tokenized.de-en checkpoints/bart_ft checkpoints/bart/checkpoint_best.pt de en 

```

4.4 评估nmt任务

```shell
# eval
bash bart/scripts/eval.sh data-bin/iwslt14.tokenized.de-en checkpoints/bart_ft/checkpoint_best.pt 
```



2023/5/24