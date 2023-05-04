/usr/bin/python3.8 -m pip install --upgrade pip
pip install datasets tokenizers evaluate -i https://pypi.tuna.tsinghua.edu.cn/simple
git clone https://github.com/huggingface/transformers.git
cd transformers/
pip install -e .
cp -r examples/pytorch/language-modeling/ .