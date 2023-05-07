# 实体对比损失

句子对比：

```shell
--user-dir contrastive_extension  --criterion label_smoothed_cross_entropy_with_contrastive --contrastive-lambda 5  --temperature 0.1

```

实体对比：

```shell
--user-dir contrastive_extension  --task entity_ct_translation --criterion label_smoothed_ce_with_entity_contrastive --contrastive-lambda 1.  --temperature 0.1 --use-entity-ct  --entity-dict dict.de-en.bpe.txt --topk -1

```

实体+句子对比：

```shell
 --user-dir contrastive_extension --task entity_ct_translation --criterion label_smoothed_ce_with_multi_granularity_contrastive --contrastive-lambda 1.  --temperature 0.1 --use-entity-ct  --entity-dict dict.de-en.bpe.txt --topk -1

```

