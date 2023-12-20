conda create -n seq python=3.10
conda activate seq
#pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# 或
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
# conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
git clone https://github.com/pytorch/fairseq
cd fairseq
# 7409af7f
git checkout 7409af7f9a7b6ddac4cbfe7cafccc715b3c1b21e
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

# . Downgrade the protobuf package to 3.20.x or lower.
pip install protobuf==3.20.0 -i   https://pypi.tuna.tsinghua.edu.cn/simple