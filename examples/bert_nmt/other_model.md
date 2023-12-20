conda install cudatoolkit=10.1 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/linux-64/
conda install cudnn
conda install cudnn=7.6.5 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/
conda list | grep cudatoolkit
git clone https://gitee.com/rrjin/fairseq.git
git clone https://gitee.com/miugod/nmt_data_tools.git


# bert-fused
conda create -n bfuse python=3.6
conda activate bfuse
git clone https://github.com/bert-nmt/bert-nmt
cd bert-nmt
git checkout -f update-20-10
pip install -e . -i  https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==1.5.0 transformers==3.5.0 --ignore-installed certifi -i  https://pypi.tuna.tsinghua.edu.cn/simple
pip install  transformers==3.5.0 --ignore-installed certifi -i  https://pypi.tuna.tsinghua.edu.cn/simple
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch

pip install --force-reinstall torch==1.5.0+cu10.1  -i  https://pypi.tuna.tsinghua.edu.cn/simple


是否可以和上面合并
conda create -n bfuse_base python=3.6
conda activate bfuse_base
git clone https://github.com/pytorch/fairseq
cd fairseq
git checkout a8f28ecb63ee01c33ea9f6986102136743d47ec2
pip install -e . -i  https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==1.5.0 transformers==3.5.0 --ignore-installed certifi -i  https://pypi.tuna.tsinghua.edu.cn/simple


cd ./examples/translation/
bash script/prepare-iwslt14.sh
git clone https://github.com/jiaohuix/nmt_data_tools.git
export TOOLS=$PWD/nmt_data_tools/
bash nmt_data_tools/examples/data_scripts/prepare-iwslt14.sh

cd iwslt14.tokenized.de-en
bash ../makedataforbert.sh de
bash ../makedataforbert.sh en


# bert-jam (跑不通)


conda create -n bjam python=3.7
conda activate bjam
git clone https://github.com/HollowFire/bert-jam
cd bert-jam
pip install -e . -i  https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==1.2 transformers==3.5.0 -i  https://pypi.tuna.tsinghua.edu.cn/simple
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python 


mkdir pretrained && cd pretrained
GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/dbmdz/bert-base-german-uncased
wget https://huggingface.co/dbmdz/bert-base-german-uncased/resolve/main/pytorch_model.bin?download=true -O pytorch_model.bin
mv pytorch_model.bin bert-base-german-uncased/

src=de
tgt=en
TEXT=examples/translation/iwslt14.tokenized.de-en
DATADIR=data-bin
python preprocess.py --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir $DATADIR/iwslt14_de_en/  --joined-dictionary \
  --bert-model-name pretrained/bert-base-german-uncased

export  CUDA_VISIBLE_DEVICES=1
BERT=pretrained/bert-base-german-uncased
src=de
tgt=en
model=bt_glu_joint
ARCH=${model}_iwslt_de_en
DATAPATH=data-bin/iwslt14_${src}_${tgt}
SAVE=save/${model}.iwslt14.$src-$tgt.$BERT.
mkdir -p $SAVE
python train.py $DATAPATH \
-a $ARCH --optimizer adam --lr 0.0005 -s $src -t $tgt --label-smoothing 0.1 \
--dropout 0.3 --max-tokens 4000 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --warmup-updates 4000 --warmup-init-lr '1e-07' --keep-last-epochs 10 \
--adam-betas '(0.9,0.98)' --save-dir $SAVE --share-all-embeddings   \
--encoder-bert-dropout --encoder-bert-dropout-ratio 0.5 \
--bert-model-name pretrained/$BERT \
--user-dir my --no-progress-bar --max-epoch 40 --fp16 \
--ddp-backend=no_c10d \
| tee -a $SAVE/training.log
~                        

# bibert
conda create -n bibert python=3.7
conda activate bibert
git clone https://github.com/fe1ixxu/BiBERT
cd BiBERT
pip install -e . -i  https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers  hydra-core==1.0.3 -i  https://pypi.tuna.tsinghua.edu.cn/simple