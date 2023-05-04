#  NLLB  MoE

## 说明

fastmoe和NLLB都有实现gshard，fairseq官方的应该会比fastmoe高效些，于是准备以NLLB为codebase作修改。 由于NLLB的训练脚本学习成本实在太大，我把NLLB的MoE部分单独拉了出来，奈何实力有限，目前只能勉强跑通单卡版，欢迎测试。

## 依赖

```shell
pip install pyahocorasick
# tutel
git clone https://github.com/microsoft/tutel --branch main
python -m pip uninstall tutel -y
python ./tutel/setup.py install --user

# 把moe_extension放在fairseq目录下
```

## 训练

在原始的fairseq命令基础上，指明用户目录**moe_extension**，然后添加额外的moe参数，详看：[transformer_config.py](moe_extension/transformer/modules/transformer_config.py)

```shell
python train.py data-bin/iwslt14 --user-dir moe_extension --arch moe_transformer_iwslt_de_en --moe-freq  2 --moe-cmr  --moe-expert-count 8 --fp16 --optimizer adam  --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --eval-bleu  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

# 其他参数
moe_freq
moe_expert_count
moe_local_drop
moe_cmr # 使用cmr
cmr_gate_drop
moe_top1_expert # top1专家
```

## 其他

遇到了两个问题：

1. [transformer_layer.py](moe_extension/transformer/modules/transformer_layer.py)中 get_data_parallel_rank会报错（没有**init_process_group** ），我改成了0，多卡可能会出现错误。

   ```python
   # ddp_rank = dist_utils.get_data_parallel_rank() # 978
   ddp_rank = 0 # 979
   ```

2. [moe_layer.py](moe_extension/transformer/modules/moe/moe_layer.py)中221行对动态组batch的tokens用 all-reduce进行pad，没理解其作用，直接使用max_tokens参数动态组batch会报和1相同的错误： **init_process_group** ，目前有两种解决方法：

   1. 使用batch_size和max_sentences静态组batch
   2. 使用max_tokens动态组batch，把224-260行直接注释掉 √ 【可以运行】



2022/12/13