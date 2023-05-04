/usr/bin/python3.8 -m pip install --upgrade pip
git clone https://github.com/bert-nmt/bert-nmt
cd bert-nmt && git checkout update-20-10	
pip install --editable .  -i  https://pypi.tuna.tsinghua.edu.cn/simple --user
pip install transformers==3.5.0  subword-nmt

