加载到weights里

## siku-roberta

```shell
# embed 6
bert.embeddings.position_ids torch.Size([1, 512])
bert.embeddings.word_embeddings.weight torch.Size([29791, 768])
bert.embeddings.position_embeddings.weight torch.Size([512, 768])
bert.embeddings.token_type_embeddings.weight torch.Size([2, 768])
bert.embeddings.LayerNorm.weight torch.Size([768])
bert.embeddings.LayerNorm.bias torch.Size([768])
# layers 16*12=192
bert.encoder.layer.0.attention.self.query.weight torch.Size([768, 768])
bert.encoder.layer.0.attention.self.query.bias torch.Size([768])
bert.encoder.layer.0.attention.self.key.weight torch.Size([768, 768])
bert.encoder.layer.0.attention.self.key.bias torch.Size([768])
bert.encoder.layer.0.attention.self.value.weight torch.Size([768, 768])
bert.encoder.layer.0.attention.self.value.bias torch.Size([768])
bert.encoder.layer.0.attention.output.dense.weight torch.Size([768, 768])
bert.encoder.layer.0.attention.output.dense.bias torch.Size([768])
bert.encoder.layer.0.attention.output.LayerNorm.weight torch.Size([768])
bert.encoder.layer.0.attention.output.LayerNorm.bias torch.Size([768])

bert.encoder.layer.0.intermediate.dense.weight torch.Size([3072, 768])
bert.encoder.layer.0.intermediate.dense.bias torch.Size([3072])
bert.encoder.layer.0.output.dense.weight torch.Size([768, 3072])
bert.encoder.layer.0.output.dense.bias torch.Size([768])
bert.encoder.layer.0.output.LayerNorm.weight torch.Size([768])
bert.encoder.layer.0.output.LayerNorm.bias torch.Size([768])

# head 7
cls.predictions.bias torch.Size([29791])
cls.predictions.transform.dense.weight torch.Size([768, 768])
cls.predictions.transform.dense.bias torch.Size([768])
cls.predictions.transform.LayerNorm.weight torch.Size([768])
cls.predictions.transform.LayerNorm.bias torch.Size([768])
cls.predictions.decoder.weight torch.Size([29791, 768])
cls.predictions.decoder.bias torch.Size([29791])

total params: 205=205
```

```python
# drop cls
drop_keys=["bert.embeddings.position_ids","bert.embeddings.token_type_embeddings.weight","cls"]

# hf-deltalm map
key_map = {
    "bert.":"",
    ##  embed
    "embeddings.word_embeddings": "embed_tokens",
    "embeddings.position_embeddings": "embed_positions",
    "embeddings.LayerNorm": "layernorm_embedding",
    ## attn
    "layer": "layers",
    "attention.self.query": "self_attn.q_proj",
    "attention.self.key": "self_attn.k_proj",
    "attention.self.value": "self_attn.v_proj",
    "attention.output.dense": "self_attn.out_proj",
    "attention.output.LayerNorm": "self_attn_layer_norm",
    ## ffn
    "intermediate.dense": "fc1",
    "output.dense": "fc2",
    "output.LayerNorm": "final_layer_norm",
}

# decoder要交替初始化，两层bert初始化一层nmt
# layer =layer//2
# 偶数为self_attn，fc1-2, 奇数为encoder_attn和fc3-4
{
    "encoder": "decoder",
    "cls.predictions.decoder": "output_projection",
}
# 偶数是self
even_map={
   "final_layer_norm": "ffn_layer_norm", 
}
# 奇数是cross
odd_map = {
    ## attn
    "self_attn": "encoder_attn",
    "attention.output.LayerNorm": "encoder_attn_layer_norm",
    ## ffn
    "fc1": "fc3",
    "fc2": "fc4",
}
```



##  chinese-roberta-wwm-ext

```
# embed 5
bert.embeddings.word_embeddings.weight torch.Size([21128, 768])
bert.embeddings.position_embeddings.weight torch.Size([512, 768])
bert.embeddings.token_type_embeddings.weight torch.Size([2, 768])
bert.embeddings.LayerNorm.weight torch.Size([768])
bert.embeddings.LayerNorm.bias torch.Size([768])
# layers 16*12
bert.encoder.layer.0.attention.self.query.weight torch.Size([768, 768])
bert.encoder.layer.0.attention.self.query.bias torch.Size([768])
bert.encoder.layer.0.attention.self.key.weight torch.Size([768, 768])
bert.encoder.layer.0.attention.self.key.bias torch.Size([768])
bert.encoder.layer.0.attention.self.value.weight torch.Size([768, 768])
bert.encoder.layer.0.attention.self.value.bias torch.Size([768])
bert.encoder.layer.0.attention.output.dense.weight torch.Size([768, 768])
bert.encoder.layer.0.attention.output.dense.bias torch.Size([768])
bert.encoder.layer.0.attention.output.LayerNorm.weight torch.Size([768])
bert.encoder.layer.0.attention.output.LayerNorm.bias torch.Size([768])
bert.encoder.layer.0.intermediate.dense.weight torch.Size([3072, 768])
bert.encoder.layer.0.intermediate.dense.bias torch.Size([3072])
bert.encoder.layer.0.output.dense.weight torch.Size([768, 3072])
bert.encoder.layer.0.output.dense.bias torch.Size([768])
bert.encoder.layer.0.output.LayerNorm.weight torch.Size([768])
bert.encoder.layer.0.output.LayerNorm.bias torch.Size([768])

# head 10
bert.pooler.dense.weight torch.Size([768, 768])
bert.pooler.dense.bias torch.Size([768])
cls.predictions.bias torch.Size([21128])
cls.predictions.transform.dense.weight torch.Size([768, 768])
cls.predictions.transform.dense.bias torch.Size([768])
cls.predictions.transform.LayerNorm.weight torch.Size([768])
cls.predictions.transform.LayerNorm.bias torch.Size([768])
cls.predictions.decoder.weight torch.Size([21128, 768])
cls.seq_relationship.weight torch.Size([2, 768])
cls.seq_relationship.bias torch.Size([2])

total params: 207=207
```



## roberta-large

```
# embed=5
roberta.embeddings.word_embeddings.weight torch.Size([50265, 1024])
roberta.embeddings.position_embeddings.weight torch.Size([514, 1024])
roberta.embeddings.token_type_embeddings.weight torch.Size([1, 1024])
roberta.embeddings.LayerNorm.weight torch.Size([1024])
roberta.embeddings.LayerNorm.bias torch.Size([1024])
# layers 16*24
roberta.encoder.layer.0.attention.self.query.weight torch.Size([1024, 1024])
roberta.encoder.layer.0.attention.self.query.bias torch.Size([1024])
roberta.encoder.layer.0.attention.self.key.weight torch.Size([1024, 1024])
roberta.encoder.layer.0.attention.self.key.bias torch.Size([1024])
roberta.encoder.layer.0.attention.self.value.weight torch.Size([1024, 1024])
roberta.encoder.layer.0.attention.self.value.bias torch.Size([1024])
roberta.encoder.layer.0.attention.output.dense.weight torch.Size([1024, 1024])
roberta.encoder.layer.0.attention.output.dense.bias torch.Size([1024])
roberta.encoder.layer.0.attention.output.LayerNorm.weight torch.Size([1024])
roberta.encoder.layer.0.attention.output.LayerNorm.bias torch.Size([1024])
roberta.encoder.layer.0.intermediate.dense.weight torch.Size([4096, 1024])
roberta.encoder.layer.0.intermediate.dense.bias torch.Size([4096])
roberta.encoder.layer.0.output.dense.weight torch.Size([1024, 4096])
roberta.encoder.layer.0.output.dense.bias torch.Size([1024])
roberta.encoder.layer.0.output.LayerNorm.weight torch.Size([1024])
roberta.encoder.layer.0.output.LayerNorm.bias torch.Size([1024])

# head 8
roberta.pooler.dense.weight torch.Size([1024, 1024])
roberta.pooler.dense.bias torch.Size([1024])
lm_head.bias torch.Size([50265])
lm_head.dense.weight torch.Size([1024, 1024])
lm_head.dense.bias torch.Size([1024])
lm_head.layer_norm.weight torch.Size([1024])
lm_head.layer_norm.bias torch.Size([1024])
lm_head.decoder.weight torch.Size([50265, 1024])
total params: 397=397
```

