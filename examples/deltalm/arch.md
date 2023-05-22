## hf-xlmr

```
git clone https://huggingface.co/xlm-roberta-base
```

```shell
import torch
deltalm_ft="xlm-roberta-base/pytorch_model.bin"
state = torch.load(deltalm_ft)
print(len(state)) # 205
for key,v in state.items():
  print(key, v.shape)
```

```shell
# embed 5
roberta.embeddings.word_embeddings.weight torch.Size([250002, 768])
roberta.embeddings.position_embeddings.weight torch.Size([514, 768])
roberta.embeddings.token_type_embeddings.weight torch.Size([1, 768])
roberta.embeddings.LayerNorm.weight torch.Size([768])
roberta.embeddings.LayerNorm.bias torch.Size([768])

# layers 16 * 12 =192
oberta.encoder.layer.0.attention.self.query.weight torch.Size([768, 768])
roberta.encoder.layer.0.attention.self.query.bias torch.Size([768])
roberta.encoder.layer.0.attention.self.key.weight torch.Size([768, 768])
roberta.encoder.layer.0.attention.self.key.bias torch.Size([768])
roberta.encoder.layer.0.attention.self.value.weight torch.Size([768, 768])
roberta.encoder.layer.0.attention.self.value.bias torch.Size([768])
roberta.encoder.layer.0.attention.output.dense.weight torch.Size([768, 768])
roberta.encoder.layer.0.attention.output.dense.bias torch.Size([768])
roberta.encoder.layer.0.attention.output.LayerNorm.weight torch.Size([768])
roberta.encoder.layer.0.attention.output.LayerNorm.bias torch.Size([768])
roberta.encoder.layer.0.intermediate.dense.weight torch.Size([3072, 768])
roberta.encoder.layer.0.intermediate.dense.bias torch.Size([3072])
roberta.encoder.layer.0.output.dense.weight torch.Size([768, 3072])
roberta.encoder.layer.0.output.dense.bias torch.Size([768])
roberta.encoder.layer.0.output.LayerNorm.weight torch.Size([768])
roberta.encoder.layer.0.output.LayerNorm.bias torch.Size([768])
# other 8(这个要单独加嘛？)
roberta.pooler.dense.weight torch.Size([768, 768])
roberta.pooler.dense.bias torch.Size([768])
lm_head.bias torch.Size([250002])
lm_head.dense.weight torch.Size([768, 768])
lm_head.dense.bias torch.Size([768])
lm_head.layer_norm.weight torch.Size([768])
lm_head.layer_norm.bias torch.Size([768])
lm_head.decoder.weight torch.Size([250002, 768])

```



## fairseq-xlmr

```shell
!wget https://dl.fbaipublicfiles.com/XLM/mlm_en_2048.pth
```

```python
import torch
mlm_en_2048="mlm_en_2048.pth"
state = torch.load(mlm_en_2048)
print(state.keys())
print(len(state["model"])) # 198
for key,v in state["model"].items():
  print(key, v.shape)

```



## deltalm_ptm

```shell
import torch
state = torch.load("ptm/deltalm-base.pt")
print(len(state["weights"])) # 8
for key,v in state["weights"].items():
  print(key, v.shape)
```

```shell
# 8+16*12+32*6=384+8=392
# 8
tgt_embedding.embed_tokens.weight torch.Size([250104, 768])
tgt_embedding.embed_positions.weight torch.Size([514, 768])
tgt_embedding.emb_layer_norm.weight torch.Size([768])
tgt_embedding.emb_layer_norm.bias torch.Size([768])
# 实际dec和enc共享参数
src_embedding.embed_tokens.weight torch.Size([250104, 768])
src_embedding.embed_positions.weight torch.Size([514, 768])
src_embedding.emb_layer_norm.weight torch.Size([768])
src_embedding.emb_layer_norm.bias torch.Size([768])
# 16
encoder.layers.0.self_attn.q_proj.weight torch.Size([768, 768])
encoder.layers.0.self_attn.q_proj.bias torch.Size([768])
encoder.layers.0.self_attn.k_proj.weight torch.Size([768, 768])
encoder.layers.0.self_attn.k_proj.bias torch.Size([768])
encoder.layers.0.self_attn.v_proj.weight torch.Size([768, 768])
encoder.layers.0.self_attn.v_proj.bias torch.Size([768])
encoder.layers.0.self_attn.out_proj.weight torch.Size([768, 768])
encoder.layers.0.self_attn.out_proj.bias torch.Size([768])
encoder.layers.0.ffn.fc1.weight torch.Size([3072, 768])
encoder.layers.0.ffn.fc1.bias torch.Size([3072])
encoder.layers.0.ffn.fc2.weight torch.Size([768, 3072])
encoder.layers.0.ffn.fc2.bias torch.Size([768])
encoder.layers.0.self_attn_layer_norm.weight torch.Size([768])
encoder.layers.0.self_attn_layer_norm.bias torch.Size([768])
encoder.layers.0.final_layer_norm.weight torch.Size([768])
encoder.layers.0.final_layer_norm.bias torch.Size([768])
# 32
decoder.layers.0.self_attn.q_proj.weight torch.Size([768, 768])
decoder.layers.0.self_attn.q_proj.bias torch.Size([768])
decoder.layers.0.self_attn.k_proj.weight torch.Size([768, 768])
decoder.layers.0.self_attn.k_proj.bias torch.Size([768])
decoder.layers.0.self_attn.v_proj.weight torch.Size([768, 768])
decoder.layers.0.self_attn.v_proj.bias torch.Size([768])
decoder.layers.0.self_attn.out_proj.weight torch.Size([768, 768])
decoder.layers.0.self_attn.out_proj.bias torch.Size([768])
decoder.layers.0.encoder_attn.q_proj.weight torch.Size([768, 768])
decoder.layers.0.encoder_attn.q_proj.bias torch.Size([768])
decoder.layers.0.encoder_attn.k_proj.weight torch.Size([768, 768])
decoder.layers.0.encoder_attn.k_proj.bias torch.Size([768])
decoder.layers.0.encoder_attn.v_proj.weight torch.Size([768, 768])
decoder.layers.0.encoder_attn.v_proj.bias torch.Size([768])
decoder.layers.0.encoder_attn.out_proj.weight torch.Size([768, 768])
decoder.layers.0.encoder_attn.out_proj.bias torch.Size([768])

decoder.layers.0.ffn_1.fc1.weight torch.Size([3072, 768])
decoder.layers.0.ffn_1.fc1.bias torch.Size([3072])
decoder.layers.0.ffn_1.fc2.weight torch.Size([768, 3072])
decoder.layers.0.ffn_1.fc2.bias torch.Size([768])

decoder.layers.0.ffn_2.fc1.weight torch.Size([3072, 768])
decoder.layers.0.ffn_2.fc1.bias torch.Size([3072])
decoder.layers.0.ffn_2.fc2.weight torch.Size([768, 3072])
decoder.layers.0.ffn_2.fc2.bias torch.Size([768])

decoder.layers.0.self_attn_layer_norm.weight torch.Size([768])
decoder.layers.0.self_attn_layer_norm.bias torch.Size([768])
decoder.layers.0.encoder_attn_layer_norm.weight torch.Size([768])
decoder.layers.0.encoder_attn_layer_norm.bias torch.Size([768])

decoder.layers.0.final_layer_norm.weight torch.Size([768])
decoder.layers.0.final_layer_norm.bias torch.Size([768])
decoder.layers.0.ffn_layer_norm.weight torch.Size([768])
decoder.layers.0.ffn_layer_norm.bias torch.Size([768])

# 少了out proj
```



## deltalm

```shell
import torch
deltalm_ft="/content/unilm/deltalm/ckpt/deltalm_ft/checkpoint_1_100.pt"
state = torch.load(deltalm_ft)
print(state.keys()) # dict_keys(['args', 'cfg', 'model', 'criterion', 'optimizer_history', 'task_state', 'extra_state', 'last_optimizer_state'])
print(len(state["model"])) # 395
for key,v in state["model"].items():
  print(key, v.shape)

```





```shell
# params: 392+3 (version*2, outproj)
# embed
encoder.version torch.Size([1])
encoder.embed_tokens.weight torch.Size([250001, 768])
encoder.embed_positions.weight torch.Size([514, 768])
encoder.layernorm_embedding.weight torch.Size([768])
encoder.layernorm_embedding.bias torch.Size([768])
# encoder 16*12
encoder.layers.0.self_attn.k_proj.weight torch.Size([768, 768])
encoder.layers.0.self_attn.k_proj.bias torch.Size([768])
encoder.layers.0.self_attn.v_proj.weight torch.Size([768, 768])
encoder.layers.0.self_attn.v_proj.bias torch.Size([768])
encoder.layers.0.self_attn.q_proj.weight torch.Size([768, 768])
encoder.layers.0.self_attn.q_proj.bias torch.Size([768])
encoder.layers.0.self_attn.out_proj.weight torch.Size([768, 768])
encoder.layers.0.self_attn.out_proj.bias torch.Size([768])
encoder.layers.0.self_attn_layer_norm.weight torch.Size([768])
encoder.layers.0.self_attn_layer_norm.bias torch.Size([768])
encoder.layers.0.fc1.weight torch.Size([3072, 768])
encoder.layers.0.fc1.bias torch.Size([3072])
encoder.layers.0.fc2.weight torch.Size([768, 3072])
encoder.layers.0.fc2.bias torch.Size([768])
encoder.layers.0.final_layer_norm.weight torch.Size([768])
encoder.layers.0.final_layer_norm.bias torch.Size([768])

# decoder 32*6 +1
decoder.version torch.Size([1])
decoder.embed_tokens.weight torch.Size([250001, 768])
decoder.embed_positions.weight torch.Size([514, 768])
decoder.layernorm_embedding.weight torch.Size([768])
decoder.layernorm_embedding.bias torch.Size([768])

decoder.layers.0.self_attn.k_proj.weight torch.Size([768, 768])
decoder.layers.0.self_attn.k_proj.bias torch.Size([768])
decoder.layers.0.self_attn.v_proj.weight torch.Size([768, 768])
decoder.layers.0.self_attn.v_proj.bias torch.Size([768])
decoder.layers.0.self_attn.q_proj.weight torch.Size([768, 768])
decoder.layers.0.self_attn.q_proj.bias torch.Size([768])
decoder.layers.0.self_attn.out_proj.weight torch.Size([768, 768])
decoder.layers.0.self_attn.out_proj.bias torch.Size([768])
decoder.layers.0.self_attn_layer_norm.weight torch.Size([768])
decoder.layers.0.self_attn_layer_norm.bias torch.Size([768])
decoder.layers.0.encoder_attn.k_proj.weight torch.Size([768, 768])
decoder.layers.0.encoder_attn.k_proj.bias torch.Size([768])
decoder.layers.0.encoder_attn.v_proj.weight torch.Size([768, 768])
decoder.layers.0.encoder_attn.v_proj.bias torch.Size([768])
decoder.layers.0.encoder_attn.q_proj.weight torch.Size([768, 768])
decoder.layers.0.encoder_attn.q_proj.bias torch.Size([768])
decoder.layers.0.encoder_attn.out_proj.weight torch.Size([768, 768])
decoder.layers.0.encoder_attn.out_proj.bias torch.Size([768])
decoder.layers.0.encoder_attn_layer_norm.weight torch.Size([768])
decoder.layers.0.encoder_attn_layer_norm.bias torch.Size([768])
decoder.layers.0.fc1.weight torch.Size([3072, 768])
decoder.layers.0.fc1.bias torch.Size([3072])
decoder.layers.0.fc2.weight torch.Size([768, 3072])
decoder.layers.0.fc2.bias torch.Size([768])
decoder.layers.0.fc3.weight torch.Size([3072, 768])
decoder.layers.0.fc3.bias torch.Size([3072])
decoder.layers.0.fc4.weight torch.Size([768, 3072])
decoder.layers.0.fc4.bias torch.Size([768])
decoder.layers.0.ffn_layer_norm.weight torch.Size([768])
decoder.layers.0.ffn_layer_norm.bias torch.Size([768])
decoder.layers.0.final_layer_norm.weight torch.Size([768])
decoder.layers.0.final_layer_norm.bias torch.Size([768])
# out 1
decoder.output_projection.weight torch.Size([250001, 768])
```



```shell
# arch:
2023-05-22 08:48:57 | INFO | fairseq_cli.train | DeltaLMModel(
  (encoder): DeltaLMEncoder(
    (dropout_module): FairseqDropout()
    (embed_tokens): Embedding(250001, 768, padding_idx=1)
    (embed_positions): LearnedPositionalEmbedding(514, 768, padding_idx=1)
    (layernorm_embedding): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (layers): ModuleList(
      (0-11): 12 x TransformerEncoderLayerBase(
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (dropout_module): FairseqDropout()
        (activation_dropout_module): FairseqDropout()
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
  (decoder): DeltaLMDecoder(
    (dropout_module): FairseqDropout()
    (embed_tokens): Embedding(250001, 768, padding_idx=1)
    (embed_positions): LearnedPositionalEmbedding(514, 768, padding_idx=1)
    (layernorm_embedding): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (layers): ModuleList(
      (0-5): 6 x DeltaLMDecoderLayer(
        (dropout_module): FairseqDropout()
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (activation_dropout_module): FairseqDropout()
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (encoder_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (encoder_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (fc3): Linear(in_features=768, out_features=3072, bias=True)
        (fc4): Linear(in_features=3072, out_features=768, bias=True)
        (ffn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
    )
    (output_projection): Linear(in_features=768, out_features=250001, bias=False)
  )
)
```

