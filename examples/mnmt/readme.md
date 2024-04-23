# MNMT

数据集：
iwslt17   ikcest22

模型：

- [x] 1 单向transformer： baseline
- [x] 2 双向transformer： dual
- [x] 3 m-transformer： multi
- [ ] 4 m2m(多语言微调+双向微调+单向微调)
  - [ ] 修改训练、预测、评估脚本的数据集加载
  - [ ] 修改双向、单向微调
- [x] 5 mrasp2微调：mrasp2
- [ ] 6 na-nmt
- [ ] 7 lass
- [ ] 8 nllb/mbart（同m2m）
- [ ] 9 多任务训练
- [ ] 10 lora微调
  - [ ] 参考m2m





数据集：

```shell
mkdir datasets && cd datasets
# iwslt17
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/datasets/iwslt2017
# 下载DeEnItNlRo-DeEnItNlRo.zip，复制到data/2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/

wget https://hf-mirror.com/datasets/iwslt2017/resolve/main/data/2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo.zip?download=true -O "DeEnItNlRo-DeEnItNlRo.zip"
cp DeEnItNlRo-DeEnItNlRo.zip iwslt2017/data/2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/

# 替换为本地路径
sed -i '50s#.*#REPO_URL = ""#' iwslt2017/iwslt2017.py


# ikcest22
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/datasets/miugod/ikcest2022
wget https://hf-mirror.com/datasets/miugod/ikcest2022/resolve/main/data/ZhFrRuThArEn.zip?download=true  -O "ZhFrRuThArEn.zip"
cp ZhFrRuThArEn.zip ikcest2022/data/ZhFrRuThArEn.zip


```

