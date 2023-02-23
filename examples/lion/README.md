# Lion optimizer for fairseq

paper: [《Symbolic Discovery of Optimization Algorithms》](https://arxiv.org/abs/2302.06675)



1 prepare iwslt14 de-en

```shell
bash examples/lion/prepare-iwslt14.sh
```

2 binarize

```shell
bash examples/lion/binarize.sh de en iwslt14.tokenized.de-en data-bin/iwslt14_deen y
```

3 train

-  3.1 adamw：

  - ```shell
    mkdir -p ckpt/adamw
    bash examples/lion/train_base.sh data-bin/iwslt14_deen  ckpt/adamw
    ```

- 3.2 lion:

  - ```shell
    mkdir -p  ckpt/lion
    bash examples/lion/train_base.sh data-bin/iwslt14_deen  ckpt/lion
    
    # --user-dir extension --optimizer lion --lion-betas '(0.95, 0.98)' --lr 5e-5 --weight-decay 0.001 --lr-scheduler cosine
    ```

4 evaluate

```shell
bash examples/lion/evaluate.sh de en data-bin/iwslt14_deen ckpt/adamw/checkpoint_best.pt adamw
bash examples/lion/evaluate.sh de en data-bin/iwslt14_deen ckpt/lion/checkpoint_best.pt lion
```

| Optimizer | BLEU | Memory |
| --------- | ---- | ------ |
| adamw     | 34.6 | 4100M  |
| lion      | 34.7 | 3900M  |

5 conclude

​		Since the paper did not write the corresponding parameters of machine translation, I have not yet achieved better results... For more related information refer to:

- [lion_pytorch](https://github.com/google/automl/blob/master/lion/lion_pytorch.py)
- [Google新搜出的优化器Lion：效率与效果兼得的“训练狮”](https://kexue.fm/archives/9473)
- [lion-pytorch](https://github.com/lucidrains/lion-pytorch)

