pip config set global.index-url https://mirrors.cernet.edu.cn/pypi/web/simple

cat <<'EOF' > ~/.condarc
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud

EOF


export HF_ENDPOINT=https://hf-mirror.com
pip install sacremoses tensorboardX  sacrebleu==1.5 apex transformers peft fairseq==0.12.2   subword-nmt fastcore omegaconf jieba  sentencepiece pythainlp datasets tokenizers wandb
# pip install sacremoses tensorboardX  sacrebleu==1.5 apex fairseq==0.10.0 numpy==1.23.3  subword-nmt fastcore omegaconf jieba  sentencepiece pythainlp datasets tokenizers wandb subword-nmt -i https://pypi.tuna.tsinghua.edu.cn/simple
