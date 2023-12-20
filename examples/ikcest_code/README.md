# ikcest复赛

初赛复赛数据都在opus上面找的。 [opus](https://opus.nlpl.eu/)



## 一、初赛

初赛的泰语使用[opsubtitles](https://opus.nlpl.eu/OpenSubtitles.php), 法语和俄语都在[unparallel](https://conferences.unite.un.org/UNCorpus/Home/DownloadOverview)

代码：https://github.com/MiuGod0126/PaddleSeq

## 二、复赛(mRASP)

1. 复赛起初用的paddle的transformer base直接训练五万数据，只有10分。
2. 调整参数后能达到16：① 使用1w6次op的共享bpe ②dropout=0.2 ③ rdrop ④deepnorm 
3. 然后将mrasp2的6层预训练权重转到paddle后，微调分数有18分。
4. 用fairseq直接对6层mrasp2模型微调5w数据，分数直接达到了27分，遂**直接用fairseq了**。
5. 从opus搜集了27m中阿的数据，然后用6层模型单向微调，中阿、阿中各训练1轮（8w多次更新），然后再在比赛的5w数据集上微调，分数达到了 31.545。
6. 用语言模型从27m公开数据中过滤与比赛相近的数据，取10w，与上采样的比赛数据合起来20w微调，到了31.895。
7. 调整数据进步不大，存在瓶颈，于是换了12层的mrasp2预训练模型，分数到了33.69。
8. 尝试使用新的nllb模型，分数一直没上去，见下一节（理论上会更高）。

### 1. 安装环境

cuda版本11。

```shell
# 环境
conda create -n env python=3.7
conda activate env
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# fairseq
git clone https://github.com/pytorch/fairseq 
cd fairseq
pip install --editable ./ -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install sacremoses tensorboardX   sacrebleu==1.5 apex omegaconf  jieba zhconv pythainlp subword-nmt sentencepiece -i https://pypi.tuna.tsinghua.edu.cn/simple/
# mrasp
git clone https://github.com/PANXiao1994/mRASP2.git
cd mRASP2
# data tool
git clone https://github.com/MiuGod0126/nmt_data_tools.git
pip install -r nmt_data_tools/requirements.txt
unzip /zh_ar.zip -d .
bash prepare.sh zh_ar
```

### 2.准备数据

#### 0.目录

将code/mrasp2script 和code/zh_ar.zip放到mRASP2下面。

```shell
echo $PWD
# /hy-tmp/fairseq/mRASP2
# tree:
├── examples
├── mcolt
├── mrasp2script // 数据、训练、评估、生成脚本
│   ├── binarize.sh
│   ├── evaluate.sh
│   ├── extract.sh
│   ├── generate.sh
│   ├── prepare.sh
│   └── train_tiny.sh
│   └── train_huge.sh
├── nmt_data_tools // 数据处理脚本
└── zh_ar.zip // 5w数据
└── zhar_huge.zip // 大量数据


```

#### 1.数据处理

- 在分词之前会从训练集分出1000验证集，用来评估。
- 使用mrasp预训练模型，需要使用他的[bpe code](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/pretrain/dataset/codes.bpe.32000)进行分词，同时训练时要用对应的词表[vocab_bpe](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2021/mrasp2/bpe_vocab) （在prepare.sh会自动下载）
- 对数据要添加语言标识，中文前加 LANG_TOK_ZH, 阿拉伯前加LANG_TOK_AR。

```shell
# 参数： 原始数据目录（含train test.zh_ar.zh  test.ar_zh.zh）
bash mrasp2script/prepare.sh zh_ar
```

#### 2.二值化

```shell
#参数： lang1 lang2 bpe数据  二值化数据
bash mrasp2script/binarize.sh zh ar datasets/bpe/zh_ar/ data-bin/zhar5w
```

### 3.训练模型(5w)

**1.下载权重**

六层 双语数据训练：[6e6d-no-mono](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2021/mrasp2/6e6d_no_mono.pt)

12层含单语数据训练： [12e12d](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2021/mrasp2/12e12d_last.pt) √

本次比赛最终的结果是用12层的mrasp2。

```shell
# 将权重重命名
mkdir -p ckpt/zhar ckpt/arzh
cp 12e12d_last.p ckpt/zhar/checkpoint_last.pt
cp 12e12d_last.p ckpt/arzh/checkpoint_last.pt
```

**3.2 训练**

中->阿拉伯

```shell
#参数： lang1 lang2 二值化数据  权重路径
bash mrasp2script/train_tiny.sh zh ar data-bin/zhar5w/ ckpt/zhar
```

阿拉伯->中

```shell
bash mrasp2script/train_tiny.sh zh ar data-bin/zhar5w/ ckpt/arzh
```

训练参数为：

```shell
echo "bash train_tiny.sh <src> <tgt> <data> <save>"
SRC=$1
TGT=$2
DATA=$3
SAVE=$4

fairseq-train \
    $DATA \
    --user-dir mcolt \
    -s $SRC -t $TGT \
    --dropout 0.2 --weight-decay 0.0001\
    --arch transformer_big_t2t_12e12d \
    --share-all-embeddings --layernorm-embedding  \
    --max-tokens 4096 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --save-dir $SAVE \
    --encoder-learned-pos  --decoder-learned-pos \
    --reset-optimizer --reset-dataloader --fp16 --update-freq 4 --max-epoch 50 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric  --no-epoch-checkpoints

```

### 4.评估模型

中->阿拉伯

```shell
# 参数： lang1 lang2 二值化数据  权重
bash mrasp2script/evaluate.sh zh ar data-bin/zhar5w/ ckpt/arzh/checkpoint_best.pt
```

阿拉伯->中

```shell
bash mrasp2script/evaluate.sh zh ar data-bin/zhar5w/ ckpt/arzh/checkpoint_best.pt
```



### 5.预测结果

中->阿拉伯

```shell
# 参数： lang1 lang2 二值化数据  权重
bash mrasp2script/generate.sh zh ar data-bin/zhar5w/ ckpt/arzh/checkpoint_best.pt> gen.ar
bash mrasp2script/extract.sh gen.ar AR
mv result.txt zh_ar.rst
```

阿拉伯->中

```shell
bash mrasp2script/generate.sh zh ar data-bin/zhar5w/ ckpt/arzh/checkpoint_best.pt> gen.zh
bash mrasp2script/extract.sh gen.zh ZH
mv result.txt ar_zh.rst
# 打包
zip -r trans_result.zip  zh_ar.rst ar_zh.rst
```

### 6.大规模训练

本次比赛从opus找了中阿的数据集，去重后有25m。

####  6.1 数据处理

1. 中文用jieba分词
2. 中文乱码过滤[zh_abnormal_filter](https://github.com/MiuGod0126/nmt_data_tools/blob/main/my_tools/zh_abnormal_filter.py)
3. 去重
4. 长度过滤（2-256，比例1:2.5或 2.5:1）
5. 语言标识过滤 （fasttext过滤非中文和阿拉伯数据）
6. 删除长度超过3的连续英文串
7. 使用mrasp的bpe code对中阿分子词

```shell
# 进入mRASP2/nmt_data_tools
cd mRASP2/nmt_data_tools
bash ../mrasp2script/preprocess_zhar_mrasp.sh ../zhar_huge ../zhar_huge_bpe 
cd ..

# 二值化
cp datasets/bpe/zh_ar/valid.* zhar_huge_bpe
cp datasets/bpe/zh_ar/test.* zhar_huge_bpe
bash mrasp2script/binarize.sh zh ar zhar_huge_bpe  data-bin/zhar25m
```



#### 6.2 训练命令

中->阿拉伯

```shell
#参数： lang1 lang2 二值化数据  权重路径
bash mrasp2script/train_huge.sh zh ar data-bin/zhar25m ckpt/zhar
```

阿拉伯->中

```shell
bash mrasp2script/train_huge.sh zh ar data-bin/zhar25m ckpt/arzh
```

训练1轮就行了

```shell
echo "bash train_huge.sh <src> <tgt> <data> <save>"
SRC=$1
TGT=$2
DATA=$3
SAVE=$4
epoch=1
fairseq-train \
    $DATA \
    --user-dir mcolt \
    -s $SRC -t $TGT \
    --dropout 0.2 --weight-decay 0.0001 \
    --arch transformer_big_t2t_12e12d \
    --share-all-embeddings --layernorm-embedding  \
    --max-tokens 4096 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --save-dir $SAVE \
    --encoder-learned-pos  --decoder-learned-pos \
    --reset-optimizer --reset-dataloader --fp16 --update-freq 4 --max-epoch $epoch \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric  --no-epoch-checkpoints
```

### 6.微调

中->阿拉伯

```shell
#参数： lang1 lang2 二值化数据  权重路径
bash mrasp2script/fine_huge.sh zh ar data-bin/zhar5w/ ckpt/zhar
```

阿拉伯->中

```shell
bash mrasp2script/fine_huge.sh zh ar data-bin/zhar5w/ ckpt/arzh
```

训练参数为（学习率5e-5）：

```shell
echo "bash fine_huge.sh <src> <tgt> <data> <save>"
SRC=$1
TGT=$2
DATA=$3
SAVE=$4
epoch=30
fairseq-train \
    $DATA \
    --user-dir mcolt \
    -s $SRC -t $TGT \
    --dropout 0.2 --weight-decay 0.0001\
    --arch transformer_big_t2t_12e12d \
    --share-all-embeddings --layernorm-embedding  \
    --max-tokens 4096 \
    --lr 5e-5 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --save-dir $SAVE \
    --encoder-learned-pos  --decoder-learned-pos \
    --reset-optimizer --reset-dataloader --fp16 --update-freq 4 --max-epoch $epoch \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric  --no-epoch-checkpoints
```



## 三、复赛（NLLB）

1.训练推理ok，小数据处理 ok，大规模下数据处理， 词表裁剪 数据规模说明。 分数说明

```
pip install snownlp
```

环境同上，在fairseq目录下。

[NLLB](https://github.com/facebookresearch/fairseq/tree/nllb)

```shell
mv nllbscript/* .
cp code/* .
```

### 3.1 数据处理

```shell
unzip zhar5w.zip 
# 使用预训练的spm模型对中阿数据分词
bash nllb/preprocess.sh zh ar nllb_model/flores200_sacrebleu_tokenizer_spm.model nllb_model/dict.nllb.simple.txt zhar5w data-bin/zhar5w_nllb

```

### 3.2 训练(少量)

权重路径：

[NLLB-200-Distilled](https://tinyurl.com/nllb200densedst600mcheckpoint)

```shell
cp nllb200densedst600mcheckpoint ckpt/zhar/checkpoint_last.pt
cp nllb200densedst600mcheckpoint ckpt/arzh/checkpoint_last.pt
```

```shell 
# zhar
bash nllb/train_tiny.sh zh ar data-bin/zhar5w_nllb ckpt/zhar
# arzh
bash nllb/train_tiny.sh ar zh data-bin/zhar5w_nllb ckpt/arzh

```

### 3.3 评估

```shell
# zhar
bash nllb/evaluate.sh zh ar data-bin/zhar5w_nllb/ ckpt/arzh/checkpoint_best.pt
# arzh
bash nllb/evaluate.sh ar zh data-bin/zhar5w_nllb/ ckpt/arzh/checkpoint_best.pt
```

### 3.4 预测

中->阿拉伯

```shell
# 参数： lang1 lang2 二值化数据  权重
bash nllb/generate.sh zh ar data-bin/zhar5w/ ckpt/arzh/checkpoint_best.pt> gen.ar
bash nllb/extract.sh gen.ar ar
mv result.txt zh_ar.rst
```

阿拉伯->中

```shell
bash nllb/generate.sh zh ar data-bin/zhar5w/ ckpt/arzh/checkpoint_best.pt> gen.zh
bash nllb/extract.sh gen.zh zh
mv result.txt ar_zh.rst
# 打包
zip -r trans_result.zip  zh_ar.rst ar_zh.rst
```

### 6.大规模训练

1. 数据处理

将mrasp得到的25m数据去掉bpe，然后重新进行处理

```shell
sed -i "s/@@ //g" zhar_huge_bpe/train.zh
sed -i "s/@@ //g" zhar_huge_bpe/train.ar
sed -i "s/@@ //g" zhar_huge_bpe/valid.zh
sed -i "s/@@ //g" zhar_huge_bpe/valid.ar
sed -i "s/@@ //g" zhar_huge_bpe/test.zh
sed -i "s/@@ //g" zhar_huge_bpe/test.ar
```

```shell
# lang1 lang2 spm_model dict infolder
bash spm_jieba_cut.sh zh ar nllb_model/flores200_sacrebleu_tokenizer_spm.model nllb_model/dict.nllb.simple.txt  zhar_huge_bpe
# 输入 prefix.lang， 输出： prefix.spm.lang
```



2.词表裁剪

```
cp nllb200densedst600mcheckpoint  ckpt/ptm/model.pt
cp nllb_model/dict.nllb.simple.txt ckpt/ptm/dict.txt
```

```shell
python trim_dict_siminit.py --pre-train-dir ckpt/ptm/  --ft-dict zhar25m_clean/dict.zh.txt --langs arb_Arab,zho_Hans --output ckpt/zhar/checkpoint_last.pt --simdict  nllb_model/unk25m_snow_jiebacut.txt --topk 2
```



3.二值化

```
bash nllb/binarize.sh zh ar zhar_huge_bpe data-bin/zhar25m_nllb
```

```
bash nllb/train_huge.sh zh ar data-bin/zhar25m_nllb ckpt/zhar
bash nllb/train_huge.sh ar zh data-bin/zhar25m_nllb ckpt/arzh
```

3.训练

```shell
echo "bash train.sh <src> <tgt> <data> <save>"
SRC=$1
TGT=$2
DATA=$3
SAVE=$4
epoch=1

SRC_Code=zho_Hans
TGT_Code=arb_Arab
if [ "$SRC"x == "ar"x ];then
    tmp=$SRC_Code
    SRC_Code=$TGT_Code
    TGT_Code=$tmp
fi

fairseq-train \
    $DATA \
    --user-dir extension  -s $SRC -t $TGT \
    --task nllb_translation --src-lang-code $SRC_Code --tgt-lang-code  $TGT_Code \
    --dropout 0.1 --weight-decay 0.0001 \
    --arch transformer_12_12 \
    --share-all-embeddings  \
    --max-tokens 4096 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --save-dir $SAVE \
    --reset-optimizer --reset-dataloader --fp16 --update-freq 4 --max-epoch $epoch \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric  --no-epoch-checkpoints

```



