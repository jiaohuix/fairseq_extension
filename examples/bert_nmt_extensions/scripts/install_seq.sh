conda create -n seq python=3.6
conda activate seq
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# 或
# conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
git clone https://github.com/pytorch/fairseq
cd fairseq
git checkout a8f28ecb63ee01c33ea9f6986102136743d47ec2
pip install --upgrade pip
# pip install apex # 用不了
## 解决报错：ERROR: Failed building wheel for cffi
### 1.修复坏掉的包
sudo apt install --fix-broken
### 2.安装必要的包
sudo apt-get install build-essential libssl-dev libffi-dev python-dev
# 其他bug：AttributeError: module 'distutils' has no attribute 'version'
pip install setuptools==59.5.0
# 安装fairseq
pip install -e .  -i  https://pypi.tuna.tsinghua.edu.cn/simple

