ptm=bert-base-multilingual-cased


git clone https://huggingface.co/bert-base-multilingual-cased
mv bert-base-multilingual-cased/ ../models/


textpruner-cli  \
  --pruning_mode vocabulary \
  --configurations ../configurations/vc.json ../configurations/gc.json \
  --model_class BertForMaskedLM \
  --tokenizer_class BertTokenizerFast \
  --model_path ../models/bert-base-multilingual-cased \
  --vocabulary ../datasets/iwslt14/iwsltdeen.txt

# 不用手动下载了，直接用auto类
textpruner-cli  \
  --pruning_mode vocabulary \
  --configurations ../configurations/vc.json ../configurations/gc.json \
  --model_class AutoModel \
  --tokenizer_class AutoTokenizer \
  --model_path bert-base-multilingual-cased \
  --vocabulary ../datasets/iwslt14/iwsltdeen.txt

# 682M ->   392M
# 词表：119547 ->  21443