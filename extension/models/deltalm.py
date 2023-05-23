# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
修改：
5/23
1.加载jsons，reduce:
    embed_tokens[v,dim]
    output_projection[v,dim]
    1.读取json
    2.映射
    3.修改词表 √
2.中文json生成
3.fasttext
4.roberta base德英
'''
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from fairseq import checkpoint_utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerModel,
    TransformerDecoderBase,
    TransformerEncoderBase,
)
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.modules.transformer_layer import (
    TransformerDecoderLayerBase
)
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from fairseq import utils
from fairseq.file_io import PathManager
import logging

logger = logging.getLogger(__name__)


bert_key_map = {
    "bert.":"",
    "roberta.":"",
    ##  embed
    "embeddings.word_embeddings": "embed_tokens",
    "embeddings.position_embeddings": "embed_positions",
    "embeddings.LayerNorm": "layernorm_embedding",
    ## attn
    "layer.": "layers.",
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
lm_head = "lm_head.decoder"
lm_head = "cls.predictions.decoder"

def reduce_func(vecs, reduce="mean"):
  if reduce=="mean":
    return torch.mean(vecs,dim=0)
  elif reduce=="sum":
    return torch.sum(vecs,dim=0)
  return vecs[0]


# 遍历map，获取对应的raw向量
def map_vocab(new: Tensor,
              raw: Tensor,
              vocab_map: Dict[int,List[int]] = None,
              reduce:str ="mean"):
  if vocab_map is None:
    init_len = min(new.size(0),raw.size(0))
    new[:init_len] = raw[:init_len]
    return new
  for new_idx,raw_idxs in vocab_map.items():
    raw_idxs = torch.tensor(raw_idxs)
    raw_vecs = raw.index_select(dim=0,index=raw_idxs)
    reduced_vec = reduce_func(raw_vecs, reduce)
    new[new_idx] = reduced_vec
    return new

def read_vocab_map(file):
    if not os.path.exists(file):
        return None
    with open(file,'r',encoding='utf-8') as f:
        data=json.load(f)
    vocab_map = {} # idx:[idxs...]
    for key, vals in data.items():
        vocab_map[int(key)] = [int(v) for v in vals]
    return vocab_map

def update_key(key,key_map, lm_head="",is_encoder=True):
    # 偶数是self
    # 奇数是cross
    odd_map = {
        ## attn
        "self_attn": "encoder_attn",
        "attention.output.LayerNorm": "encoder_attn_layer_norm",
        ## ffn
        "fc1": "fc3",
        "fc2": "fc4",
    }

    # # skip
    # if is_skip(key, drop_keys):
    #     return key
    # embed
    if "embed" in key:
        prefix = "encoder." if is_encoder else "decoder."
        key = prefix + key
    # map
    for old_key, new_key in key_map.items():
        if old_key in key:
            key = key.replace(old_key, new_key)

    # decoder
    if not is_encoder:
        key = key.replace("encoder", "decoder")
        # key = key.replace("cls.predictions.decoder", "decoder.output_projection")
        key = key.replace(lm_head, "decoder.output_projection")
        # 交替
        if ".layers." in key:
            # 获取layer
            names = key.split(".")
            layer_id = int(names[2])
            names[2] = str(layer_id // 2)
            key = ".".join(names)
            # 交替初始化
            if layer_id % 2 == 0:
                key = key.replace("final_layer_norm", "ffn_layer_norm")
            else:
                for old_key, new_key in odd_map.items():
                    if old_key in key:
                        key = key.replace(old_key, new_key)

    return key

def upgrade_state_dict_for_deltalm(
        state_dict: Dict[str, Any],
        pretrained_checkpoint: str,
        key_map: Dict[str,str],
        vocab_map: Dict[int,List[int]]=None ,
        lm_head: str = "",
        is_encoder=True,
) -> Dict[str, Any]:
    if not os.path.exists(pretrained_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_checkpoint))

    with open(pretrained_checkpoint, "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))
    pretrained_state_dict = state

    new_pretrained_state_dict = {}
    # 修改key
    for key in pretrained_state_dict.keys():
        new_key = update_key(key,key_map=key_map,lm_head=lm_head, is_encoder=is_encoder)
        new_key = new_key.replace('encoder.', '')
        new_key = new_key.replace('decoder.', '')
        new_pretrained_state_dict[new_key] = pretrained_state_dict[key]

    pretrained_state_dict = new_pretrained_state_dict

    # 修改weight
    vocab_keys = ["embed_tokens", "output_projection"]
    for key in state_dict.keys():
        if key in pretrained_state_dict.keys():
            pretrained_weight = pretrained_state_dict[key]
            # 词表映射
            for vocab_key in vocab_keys:
                # 1. 没有映射 3.有映射
                if vocab_key in key:
                    pretrained_weight = map_vocab(state[key], pretrained_weight,vocab_map,reduce)
            state_dict[key] = pretrained_weight
        else:
            print(f"key {key} not initialized.")

    return state_dict


@register_model("deltalm")
class DeltaLMModel(TransformerModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--pretrained-checkpoint",
            type=str,
            metavar="STR",
        )

        parser.add_argument(
            "--reduce",
            type=str,
            default="mean",
            choices = ["mean","sum"],
            help = "reduce vocab method [mean/sum/max]."
        )

        parser.add_argument(
            "--vocab-map",
            type=str,
            default="",
            help = "map new vocab to raw vocab (json file)."
        )

        parser.add_argument(
            "--lm-head",
            type=str,
            default="",
            help = "lm-head."
        )
        

    @classmethod
    def build_encoder(cls, args, tgt_dict, embed_tokens):
        return DeltaLMEncoder(TransformerConfig.from_namespace(args), tgt_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return DeltaLMDecoder(TransformerConfig.from_namespace(args), tgt_dict, embed_tokens)


class DeltaLMEncoder(TransformerEncoderBase):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if getattr(args, "pretrained_checkpoint", "") != "":
            deltalm_loaded_state_dict = upgrade_state_dict_for_deltalm(
                state_dict=self.state_dict(),
                pretrained_checkpoint=args.pretrained_checkpoint,
                key_map=bert_key_map,
                vocab_map = read_vocab_map(args.vocab_map),
                lm_head=args.lm_head,
                is_encoder=True,
            )
            self.load_state_dict(deltalm_loaded_state_dict, strict=True)
            logger.info("Load DeltaLM's encoder from {0}".format(args.pretrained_checkpoint))


class DeltaLMDecoder(TransformerDecoderBase):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if getattr(args, "pretrained_checkpoint", "") != "":
            deltalm_loaded_state_dict = upgrade_state_dict_for_deltalm(
                state_dict=self.state_dict(),
                pretrained_checkpoint=args.pretrained_checkpoint,
                key_map=bert_key_map,
                vocab_map=read_vocab_map(args.vocab_map),
                lm_head=args.lm_head,
                is_encoder=False,
            )
            self.load_state_dict(deltalm_loaded_state_dict, strict=True)
            logger.info("Load DeltaLM's decoder from {0}".format(args.pretrained_checkpoint))

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = DeltaLMDecoderLayer(args, no_encoder_attn)
        if getattr(args, "checkpoint_activations", False):
            layer = checkpoint_wrapper(layer)
        return layer


class DeltaLMDecoderLayer(TransformerDecoderLayerBase):

    def __init__(
            self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super(TransformerDecoderLayerBase, self).__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.fc3 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc4 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.ffn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
            src_lang_id=None,
            tgt_lang_id=None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.ffn_layer_norm(x)

        x = self.activation_fn(self.fc3(x))
        x = self.activation_dropout_module(x)
        x = self.fc4(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.ffn_layer_norm(x)

        ###############################################

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        ###############################################
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


@register_model_architecture(
    "deltalm", "deltalm_base"
)
def base_architecture(args):
    args.encoder_embed_dim = 768
    args.encoder_ffn_embed_dim = 3072
    args.encoder_layers = 12
    args.encoder_attention_heads = 12
    args.encoder_normalize_before = False
    args.encoder_learned_pos = True
    args.decoder_embed_dim = 768
    args.decoder_ffn_embed_dim = 3072
    args.decoder_layers = 6
    args.decoder_attention_heads = 12
    args.decoder_normalize_before = False
    args.decoder_learned_pos = True
    args.activation_fn = "gelu"
    args.no_scale_embedding = True
    args.layernorm_embedding = True
    args.max_positions = 512


@register_model_architecture(
    "deltalm", "deltalm_large"
)
def large_architecture(args):
    base_architecture(args)
    args.encoder_embed_dim = 1024
    args.encoder_ffn_embed_dim = 4096
    args.encoder_layers = 24
    args.encoder_attention_heads = 16
    args.encoder_normalize_before = False
    args.decoder_embed_dim = 1024
    args.decoder_ffn_embed_dim = 4096
    args.decoder_layers = 12
    args.decoder_attention_heads = 16
    args.decoder_normalize_before = False
    args.layernorm_embedding = False
