# 注意PRE里面的tgt_vocab变为vocab.txt
cp $PRE/tgt_vocab.txt vocab.txt
STPATH=data-bin/mbert_envi_8000/
MODELPATH=ckpt/mbert_envi_8000/
PRE_SRC=bert/bert-base-15lang-cased/
PRE=./tmp/mbert_envi_8000/
CUDA_VISIBLE_DEVICES=0 fairseq-generate \
${STPATH} --path ${MODELPATH}/checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC} \
--beam 4 --lenpen 1 --remove-bpe --vocab_file=${STPATH}/dict.vi.txt \
--max-len-a 1 --max-len-b 50|tee ${STPATH}/generate.out


STPATH=data-bin/mbert_vien_8000/
MODELPATH=ckpt/mbert_vien_8000/
PRE_SRC=bert/bert-base-15lang-cased/
PRE=./tmp/mbert_vien_8000/
CUDA_VISIBLE_DEVICES=0 fairseq-generate \
${STPATH} --path ${MODELPATH}/checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC} \
--beam 4 --lenpen 1 --remove-bpe --vocab_file=${STPATH}/dict.en.txt \
--max-len-a 1 --max-len-b 50|tee ${STPATH}/generate.out



