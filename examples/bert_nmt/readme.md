# BERT NMT



## 环境安装

```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

git config --global url."https://huggingface.co/".insteadOf "https://hf-mirror.com/"

pip install --editable ./ -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install sacremoses tensorboardX   sacrebleu==1.5 apex transformers peft fairseq==0.12.2    fastcore omegaconf jieba  sentencepiece -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

目录：

```
/bnmt
	data.py # 数据
	task.py # bert_nmt任务
	model_qblock.py # 使用qformer block桥接bert和nmt编码器
preprocess.py # 生成bert_nmt数据脚本
train_bnmt.py # 训练bert_nmt脚本
generate_bnmt.py # 预测脚本（仅使用lora时使用）
```

bert list:
https://huggingface.co/Geotrend/bert-base-15lang-cased
https://huggingface.co/Geotrend/distilbert-base-25lang-cased
dbmdz
git lfs clone https://huggingface.co/dbmdz/bert-base-german-uncased
git lfs clone https://hf-mirror.com/susnato/ernie-m-base_pytorch




## 数据处理

数据处理：

​	首先获取iwslt14数据集并完成分词；其次将分好的文本还原，用于给bert分词；用preprocess.py处理成bin。

```shell
# 1.获取bert权重
# sudo apt-get install git-lfs
mkdir bert && cd bert
git lfs install
git lfs clone https://huggingface.co/dbmdz/bert-base-german-uncased
cd ..

# 2.数据处理 iwslt14.tokenized.de-en
git clone https://github.com/jiaohuix/nmt_data_tools.git
export TOOLS=$PWD/nmt_data_tools/
bash $TOOLS/examples/data_scripts/prepare-iwslt14.sh

# 3.bert数据准备
bash scripts/makedata.sh de iwslt14.tokenized.de-en/
bash scripts/makedata.sh en iwslt14.tokenized.de-en/

# 4.二值化
bash scripts/bin_bert.sh de en iwslt14.tokenized.de-en/ data-bin/iwslt14_dbmdz bert/bert-base-german-uncased/
# bash scripts/bin_bert.sh de en iwslt14.tokenized.de-en/ data-bin/iwslt14_deen_mbert15  bert/bert-base-15lang-cased
```



## 训练

训练：

```shell
# 训练基线（可选）
bash scripts/train_base.sh data-bin/iwslt14_dbmdz/ ckpt/base
# 从头训练一个含bert的模型。
bash scripts/train_bnmt.sh data-bin/iwslt14_dbmdz/ ckpt/dbmdz
```



## 评估

评估：

```shell
# 基线
bash scripts/eval.sh data-bin/iwslt14_dbmdz/ ckpt/base/checkpoint_best.pt
# bert
bash scripts/eval_bnmt.sh de en data-bin/iwslt14_dbmdz/ ckpt/dbmdz/checkpoint_best.pt
```

## 实验1

【整理些实验的脚本。	】

error:

no metric from fairseq
cp fairseq/fairseq/logging/meters.py fairseq/fairseq/


## todo

```shell
1 双向训练
cd iwslt14.tokenized.de-en
cat train.en train.de > train.src
cat train.de train.en > train.tgt
cat valid.en valid.de > valid.src
cat valid.de valid.en > valid.tgt
cat test.en test.de > test.src
cat test.de test.en > test.tgt

注意去掉mose需要de或者en而不是src！
bash jh/makedata.sh de iwslt14.tokenized.de-en/
bash jh/makedata.sh en iwslt14.tokenized.de-en/

cat train.bert.en train.bert.de > train.bert.src
cat valid.bert.en valid.bert.de > valid.bert.src
cat test.bert.en test.bert.de > test.bert.src

2 lora训练的权重平均 [存在问题：ckpt best的 lora部分保存top10，没有更新]
```


## 实验： 更好bert（裁剪并微调的mbert）

data:

```shell
cd bert 
git clone https://huggingface.co/miugod/mbert_trim_ende_mlm
cd ..
# mres + dual (这个效果最好)
mv scripts/dual .
# 随机拼接两句，扩充为2倍
bash dual/multi_res_data.sh de en iwslt14.tokenized.de-en/ iwslt14_mres
# mres 和dual合并，变为4倍
bash dual/dual_data.sh de en iwslt14_mres iwslt14_mres_dual
# 二值化双向数据 todo: 错误顺序参数
bash scripts/bin_bert.sh src tgt iwslt14_mres_dual/ data-bin/iwslt14_mres_dual_mbert bert/mbert_trim_ende_mlm/  

# 单向微调数据de-en
bash scripts/bin_bert_wdict.sh de en iwslt14.tokenized.de-en data-bin-uni/deen  bert/mbert_trim_ende_mlm/  data-bin/iwslt14_mres_dual_mbert/dict.src.txt  
# 单向微调数据en-de
bash scripts/bin_bert_wdict.sh en de  iwslt14.tokenized.de-en data-bin-uni/ende  bert/mbert_trim_ende_mlm/ data-bin/iwslt14_mres_dual_mbert/dict.src.txt  
```

train:

tok=8192


## 实验：lora，从对偶训练好的开始

1. 双向lora微调

2. 单向lora微调


其它：训练bert时候，bert输入cased
git clone https://github.com/jiaohuix/nmt_data_tools.git
export TOOLS=$PWD/nmt_data_tools/
bash $TOOLS/examples/data_scripts/prepare-iwslt14.sh

准备bert
bash scripts/makedata.sh de iwslt14.tokenized.de-en/cased/
bash scripts/makedata.sh en iwslt14.tokenized.de-en/cased/
复制bpe文件
for prefix in train valid test
do
    cp iwslt14.tokenized.de-en/$prefix.de  iwslt14.tokenized.de-en/cased/$prefix.de
    cp iwslt14.tokenized.de-en/$prefix.en  iwslt14.tokenized.de-en/cased/$prefix.en
done
扩充数据

随机拼接两句，扩充为2倍
bash dual/multi_res_data.sh de en iwslt14.tokenized.de-en/cased iwslt14_mres_bcased
mres 和dual合并，变为4倍
bash dual/dual_data.sh de en iwslt14_mres_bcased iwslt14_mres_dual_bcased
二值化
双向
bash scripts/bin_bert.sh src tgt iwslt14_mres_dual_bcased/ data-bin/iwslt14_mres_dual_mbert_bcased bert/bert-base-15lang-cased/ 
单向
bash scripts/bin_bert_wdict.sh de en iwslt14.tokenized.de-en/cased/  data-bin-uni/deen_bcased  bert/bert-base-15lang-cased/ data-bin/iwslt14_mres_dual_mbert_bcased/dict.src.txt
bash scripts/bin_bert_wdict.sh en de iwslt14.tokenized.de-en/cased/  data-bin-uni/ende_bcased  bert/bert-base-15lang-cased/ data-bin/iwslt14_mres_dual_mbert_bcased/dict.src.txt

训练：
bsz=8192,freq=2 epoch=75


## xlm
二值化
双向
config.txt -> config.json
bash scripts/bin_bert.sh src tgt iwslt14_mres_dual_bcased/ data-bin/iwslt14_mres_dual_xlm bert/xlm 
单向
bash scripts/bin_bert_wdict.sh de en iwslt14.tokenized.de-en/cased/  data-bin-uni/deen_xlm  bert/xlm/ data-bin/iwslt14_mres_dual_xlm/dict.src.txt
bash scripts/bin_bert_wdict.sh en de iwslt14.tokenized.de-en/cased/  data-bin-uni/ende_xlm  bert/xlm/ data-bin/iwslt14_mres_dual_xlm/dict.src.txt

## envi

bash scripts/makedata.sh vi iwslt15.tokenized.vi-en 
bash scripts/makedata.sh en iwslt15.tokenized.vi-en 
42是seed
bash dual/multi_res_data.sh vi en iwslt15.tokenized.vi-en  iwslt15_mres 42

mres 和dual合并，变为4倍
bash dual/dual_data.sh vi en iwslt15_mres iwslt15_mres_dual
bin
bash scripts/bin_bert.sh src tgt iwslt15_mres_dual/ data-bin/iwslt15_mres_dual bert/bert-base-15lang-cased/ 
单向
bash scripts/bin_bert_wdict.sh vi en iwslt15.tokenized.vi-en  data-bin-uni/vien  bert/bert-base-15lang-cased/ data-bin/iwslt15_mres_dual/dict.src.txt
bash scripts/bin_bert_wdict.sh en vi iwslt15.tokenized.vi-en/  data-bin-uni/envi  bert/bert-base-15lang-cased/ data-bin/iwslt15_mres_dual/dict.src.txt

微调数据，不同seed
bash dual/multi_res_data.sh vi en iwslt15.tokenized.vi-en  iwslt15_mres_s1 1
bash dual/dual_data.sh vi en iwslt15_mres_s1 iwslt15_mres_dual_s1
bash scripts/bin_bert.sh src tgt iwslt15_mres_dual_s1/ data-bin/iwslt15_mres_dual_s1 bert/bert-base-15lang-cased/ 


# de-en size消融 1w 5w 10w
bash dual/multi_res_data.sh de en iwslt14.tokenized.de-en/cased_1w iwslt14_mres_1w
bash dual/multi_res_data.sh de en iwslt14.tokenized.de-en/cased_5w iwslt14_mres_5w
bash dual/multi_res_data.sh de en iwslt14.tokenized.de-en/cased_10w iwslt14_mres_10w


mres 和dual合并，变为4倍
bash dual/dual_data.sh de en iwslt14_mres_1w iwslt14_mres_dual_1w
bash dual/dual_data.sh de en iwslt14_mres_5w iwslt14_mres_dual_5w
bash dual/dual_data.sh de en iwslt14_mres_10w iwslt14_mres_dual_10w

二值化
双向
bash scripts/bin_bert_wdict.sh src tgt iwslt14_mres_dual_1w  data-bin/iwslt14_mres_dual_1w   bert/bert-base-15lang-cased/ data-bin/iwslt14_mres_dual_mbert_bcased/dict.src.txt
bash scripts/bin_bert_wdict.sh src tgt iwslt14_mres_dual_5w  data-bin/iwslt14_mres_dual_5w   bert/bert-base-15lang-cased/ data-bin/iwslt14_mres_dual_mbert_bcased/dict.src.txt
bash scripts/bin_bert_wdict.sh src tgt iwslt14_mres_dual_10w  data-bin/iwslt14_mres_dual_10w   bert/bert-base-15lang-cased/ data-bin/iwslt14_mres_dual_mbert_bcased/dict.src.txt



200ep 80ep 40ep



## en ar 
pip install datasets -i https://pypi.tuna.tsinghua.edu.cn/simple 
python scripts/down_iwslt17.py -s en -t  ar
bash scripts/bpe.sh data/ar-en data/ar-en-prep ar en 
bash scripts/bin.sh data/ar-en-prep/ data-bin/iwslt17_enar en ar 1 

### bert
bash scripts/makedata.sh ar data/ar-en-prep
bash scripts/makedata.sh en data/ar-en-prep
bash scripts/bin_bert.sh en ar data/ar-en-prep  data-bin/iwslt17_bert15_enar bert/bert-base-15lang-cased
bash scripts/bin_bert.sh ar en data/ar-en-prep  data-bin/iwslt17_bert15_aren bert/bert-base-15lang-cased

### train

bash scripts/train_bnmt_uni.sh  data-bin/iwslt17_bert15_enar/ ckpt_enar/bert15_enar  bert/bert-base-15lang-cased/
bash scripts/train_bnmt_uni.sh  data-bin/iwslt17_bert15_aren/ ckpt_enar/bert15_aren  bert/bert-base-15lang-cased/
# todo : transformer base rather than 37m

bash dual/multi_res_data.sh ar en data/ar-en-prep data/ar-en-mres   42

mres 和dual合并，变为4倍
bash dual/dual_data.sh en ar data/ar-en-mres  data/ar_en_mres_dual
bin
bash scripts/bin_bert.sh src tgt data/ar_en_mres_dual data-bin/iwslt17_enar_dgat bert/bert-base-15lang-cased/ 
单向
bash scripts/bin_bert_wdict.sh ar en data/ar-en-prep  data-bin-uni/aren  bert/bert-base-15lang-cased/ data-bin/iwslt17_enar_dgat/dict.src.txt
bash scripts/bin_bert_wdict.sh en ar data/ar-en-prep  data-bin-uni/enar  bert/bert-base-15lang-cased/ data-bin/iwslt17_enar_dgat/dict.src.txt


## bpedrop
bash nmt_data_tools/my_tools/apply_bpedrop_paral.sh iwslt2017/ar-en-preprocess/train.ar iwslt2017/ar-en-bpedrop/train.ar  iwslt2017/ar-en-preprocess/codes.10000.ar  4 5 0.1
bash nmt_data_tools/my_tools/apply_bpedrop_paral.sh iwslt2017/ar-en-preprocess/train.en iwslt2017/ar-en-bpedrop/train.en  iwslt2017/ar-en-preprocess/codes.10000.en  4 5 0.1

paste iwslt2017/ar-en-bpedrop/train.ar iwslt2017/ar-en-bpedrop/train.en > tmp.txt
grep -v "ErrorTokenize"  tmp.txt >  tmp.ok
cut -f 1  tmp.ok >  iwslt2017/ar-en-bpedrop/train.ar
cut -f 2  tmp.ok >  iwslt2017/ar-en-bpedrop/train.en
rm tmp*

