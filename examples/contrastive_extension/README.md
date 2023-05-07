# 各种对比损失



句子对比：

```shell
--user-dir contrastive_extension  --criterion label_smoothed_cross_entropy_with_contrastive --contrastive-lambda 5  --temperature 0.1

```

实体对比：

```shell
--user-dir contrastive_extension  --task entity_ct_translation --criterion label_smoothed_ce_with_entity_contrastive --contrastive-lambda 1.  --temperature 0.1 --use-entity-ct  --entity-dict dict.de-en.bpe.txt --topk -1

```

实体+句子对比：

```shell
 --user-dir contrastive_extension --task entity_ct_translation --criterion label_smoothed_ce_with_multi_granularity_contrastive --contrastive-lambda 1.  --temperature 0.1 --use-entity-ct  --entity-dict dict.de-en.bpe.txt --topk -1

```



```shell
# env
/usr/bin/python3.8 -m pip install --upgrade pip
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./ -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install sacremoses tensorboardX   sacrebleu==1.5 apex  pyahocorasick   fastcore omegaconf jieba  sentencepiece -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# extension
git clone https://github.com/jiaohuix/fairseq_extension.git
mv fairseq_extension/examples/contrastive_extension/ .

# process data
git clone https://gitee.com/miugod/nmt_data_tools.git
bash nmt_data_tools/examples/data_scripts/prepare-iwslt14.sh 
bash contrastive_extension/exps/
bash contrastive_extension/exps/binarize_bi.sh 
cp contrastive_extension/dict.de-en.bpe.txt data-bin/iwslt14.tokenized.bidirection.de-en/

# train
bash contrastive_extension/exps/pipe.sh 
```

