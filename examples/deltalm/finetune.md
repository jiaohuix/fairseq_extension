# deltalm

## 1 安装环境

参考colab： https://colab.research.google.com/drive/1_dHZkA5DjulVRp55AVHjtrTDdeN7z91U?usp=sharing

```
git clone https://github.com/microsoft/unilm.git
cd unilm
git submodule update --init deltalm/fairseq
cd deltalm/
pip install ./fairseq  # 不用-e就不会metric报错
```

源码安装sentencepiece：

```shell
pip install sentencepiece
sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
git clone https://github.com/google/sentencepiece.git 
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v
%cd ../../
```

下载权重：

```shell
# 下载权重
!mkdir ptm
!wget -O ptm/dict.txt https://deltalm.blob.core.windows.net/deltalm/dict.txt
!wget -O ptm/spm.model https://deltalm.blob.core.windows.net/deltalm/spm.model
!wget -O ptm/deltalm-base.pt https://deltalm.blob.core.windows.net/deltalm/deltalm-base.pt

```



## 2 数据处理



```shell
# 预处理
bash examples/prepare_iwslt14.sh /tmp/iwslt14

     /tmp/iwslt14/iwslt14.tokenized.de-en \
# 分词
bash examples/spm_iwslt14.sh \
     /tmp/iwslt14/iwslt14.tokenized.de-en \
     /tmp/iwslt14/iwslt14.spm \
     ptm/spm.model
     
     
# 二值化
bash examples/binary_iwslt14.sh \
     /tmp/iwslt14/iwslt14.spm \
     /tmp/iwslt14/iwslt14.bin \
     ptm/dict.txt
```



## 3 微调

```shell
data_bin=$1
save_dir=$2
PRETRAINED_MODEL=ptm/deltalm-base.pt
lr="5e-4"
batch_size=1024
updates=60000
epochs=50

python train.py $data_bin \
    --save-dir $save_dir \
    --arch deltalm_base \
    --pretrained-deltalm-checkpoint $PRETRAINED_MODEL \
    --share-all-embeddings \
    --max-source-positions 512 --max-target-positions 512 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --lr $lr \
    --warmup-init-lr 1e-07 \
    --stop-min-lr 1e-09 \
    --warmup-updates 4000 \
    --max-update $updates \
    --max-epoch $epochs \
    --max-tokens $batch_size --fp16 \
    --update-freq 1 \
    --seed 1 \
    --log-format simple \
    --skip-invalid-size-inputs-valid-test \
    --tensorboard-logdir $save_dir/vislogs/   >> $save_dir/train.log
```

或：

```shell
bash examples/train_iwslt14.sh \
     /tmp/iwslt14/iwslt14.bin \
     /tmp/iwslt14/checkpoints \
     ptm/deltalm-base.pt
```



## 4 评估

```shell
python generate.py $data_bin \
    --path $save_dir/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe=sentencepiece
#或
bash examples/evaluate_iwslt14.sh \
     /tmp/iwslt14/iwslt14.bin \
     /tmp/iwslt14/checkpoints
```



2023/5/22