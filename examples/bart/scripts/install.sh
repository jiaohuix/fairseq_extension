git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
pip install --editable ./
git clone https://gitee.com/miugod/nmt_data_tools.git
#git clone https://github.com/jiaohuix/nmt_data_tools.git
pip install sacremoses tensorboardX   sacrebleu==1.5 apex omegaconf jieba  sentencepiece

git clone https://github.com/jiaohuix/fairseq_extension.git
cp -r fairseq_extension/examples/span_mask_lm/  .
