# 各种loss



## 1. Rdrop

[R-Drop: Regularized Dropout for Neural Networks](https://arxiv.org/abs/2106.14448)

```shell
--user-dir extension --criterion reg_label_smoothed_cross_entropy --reg-alpha 5
# or fairseq 0.12
--criterion label_smoothed_cross_entropy_with_rdrop --rdrop-alpha 5.

```



## 2.mRASP2

[Contrastive Learning for Many-to-many Multilingual Neural Machine Translation](https://arxiv.org/pdf/2105.09501.pdf)

```shell
--user-dir extension --criterion label_smoothed_cross_entropy_with_contrastive --contrastive-lambda 1.  --temperature 0.1
```

