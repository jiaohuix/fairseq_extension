import torch
from torch import Tensor
from typing import Any, Dict, List, Optional
from fairseq.modules.transformer_layer import TransformerEncoderLayerBase, TransformerDecoderLayerBase,TransformerEncoderLayer,TransformerDecoderLayer
from fairseq.modules import  MultiheadAttention
from numpy.random import  uniform
import torch.nn as nn
import torch.nn.functional as F
from bert_nmt_extensions.modules import AveragePooler,ConvPooler,RandPooler
from bert_nmt_extensions.modules import DynamicGate,DynamicGateGLU,DynamicGateGLUNorm


class BertFusedEncoderLayer(TransformerEncoderLayerBase):
    def __init__(self, cfg,  bert_gate=True):
        super().__init__(cfg)  # 继承过多参数行吗？？
        # bert fuse 参数

        ####################################### BERT-FUSED ##############################################
        self.bert_attn = self.build_bert_attention(self.embed_dim, cfg) if not cfg.use_pooler else None
        self.bert_pooler = self.build_bert_pooler(self.embed_dim, cfg) if  cfg.use_pooler else None
        self.encoder_ratio = cfg.encoder_ratio
        self.bert_ratio = cfg.bert_ratio

        self.encoder_bert_dropout = getattr(cfg, 'encoder_bert_dropout', False)
        self.encoder_bert_dropout_ratio = getattr(cfg, 'encoder_bert_dropout_ratio', 0.25)
        assert self.encoder_bert_dropout_ratio >= 0. and self.encoder_bert_dropout_ratio <= 0.5
        self.encoder_bert_mixup = getattr(cfg, 'encoder_bert_mixup', False)

        if not bert_gate:
            self.bert_ratio = 0.
            self.encoder_bert_dropout = False
            self.encoder_bert_mixup = False
        ####################################### BERT-FUSED ##############################################

        ###  自定义参数
        self.linear_bert_gate_rate = getattr(cfg, 'linear_bert_gate_rate', None)
        if self.linear_bert_gate_rate is not None:
            # self.bert_gate_coef = nn.Parameter(data=torch.zeros([1]))
            self.bert_gate_coef = nn.Parameter(data=torch.tensor([self.linear_bert_gate_rate]))
        else:
            self.bert_gate_coef = None

        # dynamic gate
        self.dynamic_gate = getattr(cfg, 'dynamic_gate', False)
        self.type = getattr(cfg, 'gate_type', "simple")
        self.gate = self.build_dynamic_gate(self.embed_dim,cfg) if self.dynamic_gate else  None

    def build_bert_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            # self_attention=True,
            kdim=cfg.bert_out_dim,
            vdim=cfg.bert_out_dim,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            xformers_att_config=cfg.encoder.xformers_att_config,
        )

    # TODO: Fix hardcoding of bert dimension
    def build_bert_pooler(self, embed_dim,cfg, bert_dim=768):
        pooler_map = {"avg":AveragePooler, "cnn":ConvPooler, "rand":RandPooler}
        pooler = pooler_map.get(cfg.pooler_type,"avg")(in_features = bert_dim, out_features = embed_dim)
        return pooler

    def build_dynamic_gate(self,embed_dim, cfg):
        gate_map = {"simple": DynamicGate, "glu": DynamicGateGLU, "glunorm":DynamicGateGLUNorm}
        gate = gate_map.get(cfg.gate_type, "simple")(dim=embed_dim)
        return gate

    def get_ratio(self):
        if self.encoder_bert_dropout: # 训练时启动dropnet trick
            frand = float(uniform(0, 1))
            if self.encoder_bert_mixup and self.training:
                return [frand, 1 - frand]

            if frand < self.encoder_bert_dropout_ratio and self.training:
                return [1, 0]
            elif frand > 1 - self.encoder_bert_dropout_ratio and self.training:
                return [0, 1]
            else:
                return [0.5, 0.5]
        else: # 简单平均,训练和推理都是1:1,为什么不是0.5??
            return [self.encoder_ratio, self.bert_ratio]

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        bert_encoder_out = None, # <--bert fused
        bert_encoder_padding_mask = None, # <--bert fused
        attn_mask: Optional[Tensor] = None,
    ):

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(
                attn_mask.to(torch.bool), -1e8 if x.dtype == torch.float32 else -1e4
            )

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        ####################################### BERT-FUSED ##############################################
        x1, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )

        if self.bert_attn is not None:
            x2, _ = self.bert_attn(query=x, key=bert_encoder_out, value=bert_encoder_out,
                                   key_padding_mask=bert_encoder_padding_mask)
        else:
            # pooler
            dest_len = x.size(0) # [T,B,C]
            x2 = self.bert_pooler(bert_encoder_out, dest_len = dest_len, pad_mask=bert_encoder_padding_mask)

        x1 = self.dropout_module(x1)
        x2 = self.dropout_module(x2)
        # gate stratygy
        if self.bert_gate_coef is not  None:  # Linear gate
            bert_coef = F.sigmoid(self.bert_gate_coef)
            x =  (1 - bert_coef ) * x1 +  bert_coef * x2
        elif self.gate is not None: # dynamic elem-wise gate
            x = self.gate(x1, x2)
        else: # dropnet trick in raw paper
            ratios = self.get_ratio()
            x =  ratios[0] * x1 + ratios[1] * x2

        x = self.residual_connection(x, residual)
        ####################################### BERT-FUSED ##############################################

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)

        fc_result = x

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.return_fc and not torch.jit.is_scripting():
            return x, fc_result
        return x



class BertFusedDecoderLayer(TransformerDecoderLayerBase):
    def __init__(
        self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False,
        bert_gate = True # <--bert fused
    ):
        super().__init__(
            cfg,
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        # bert fuse 参数
        self.bert_gate = bert_gate

        ####################################### BERT-FUSED ##############################################
        if not no_encoder_attn: # ?
            self.bert_attn = self.build_bert_attention(self.embed_dim, cfg) if not cfg.use_pooler else None
            self.bert_pooler = self.build_bert_pooler(self.embed_dim, cfg) if cfg.use_pooler else None

        self.encoder_ratio = cfg.encoder_ratio
        self.bert_ratio = cfg.bert_ratio

        self.encoder_bert_dropout = getattr(cfg, 'encoder_bert_dropout', False)
        self.encoder_bert_dropout_ratio = getattr(cfg, 'encoder_bert_dropout_ratio', 0.25)
        assert self.encoder_bert_dropout_ratio >= 0. and self.encoder_bert_dropout_ratio <= 0.5
        self.encoder_bert_mixup = getattr(cfg, 'encoder_bert_mixup', False)

        if not bert_gate:
            self.bert_ratio = 0.
            self.encoder_bert_dropout = False
            self.encoder_bert_mixup = False
        ####################################### BERT-FUSED ##############################################

        ###  自定义参数
        # linear gate
        self.linear_bert_gate_rate = getattr(cfg, 'linear_bert_gate_rate', None)
        if self.linear_bert_gate_rate is not None:
            self.bert_gate_coef = nn.Parameter(data=torch.tensor([self.linear_bert_gate_rate]))
        else:
            self.bert_gate_coef = None

        # dynamic gate
        self.dynamic_gate = getattr(cfg, 'dynamic_gate', False)
        self.type = getattr(cfg, 'gate_type', "simple")
        self.gate = self.build_dynamic_gate(self.embed_dim,cfg) if self.dynamic_gate else  None

    def build_bert_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True, #
            # self_attention=True,
            kdim=cfg.bert_out_dim,
            vdim=cfg.bert_out_dim,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            xformers_att_config=cfg.encoder.xformers_att_config,
        )

    # TODO: Fix hardcoding of bert dimension
    def build_bert_pooler(self, embed_dim,cfg, bert_dim=768):
        pooler_map = {"avg":AveragePooler, "cnn":ConvPooler, "rand": RandPooler}
        pooler = pooler_map.get(cfg.pooler_type,"avg")(in_features = bert_dim, out_features = embed_dim)
        return pooler

    def build_dynamic_gate(self,embed_dim, cfg):
        gate_map = {"simple": DynamicGate, "glu": DynamicGateGLU, "glunorm":DynamicGateGLUNorm}
        gate = gate_map.get(cfg.gate_type, "simple")(dim=embed_dim)
        return gate

    # drop-net trick
    def get_ratio(self):
        if self.encoder_bert_dropout:
            frand = float(uniform(0, 1))
            if self.encoder_bert_mixup and self.training:
                return [frand, 1 - frand]
            if frand < self.encoder_bert_dropout_ratio and self.training:
                return [1, 0]
            elif frand > 1 - self.encoder_bert_dropout_ratio and self.training:
                return [0, 1]
            else:
                return [0.5, 0.5]
        else:
            return [self.encoder_ratio, self.bert_ratio]


    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        bert_encoder_out=None,  # <----
        bert_encoder_padding_mask=None,  # <----
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
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
        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.nh, self.head_dim)
            x = torch.einsum("tbhd,h->tbhd", x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

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

            ####################################### BERT-FUSED ##############################################
            x1, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            if self.bert_gate:
                if self.bert_attn is not None:
                    x2, _ = self.bert_attn(
                        query=x,
                        key=bert_encoder_out,
                        value=bert_encoder_out,
                        key_padding_mask=bert_encoder_padding_mask,
                        incremental_state=incremental_state,
                        static_kv=True,
                        need_weights=(not self.training and self.need_attn),
                    )
                else:
                    # pooler
                    dest_len = x.size(0) # [T,B,C]
                    x2 = self.bert_pooler(bert_encoder_out, dest_len = dest_len, pad_mask=bert_encoder_padding_mask)
                x1 = self.dropout_module(x1)
                x2 = self.dropout_module(x2)

                # gate stratygy
                if self.bert_gate_coef is not None:  # Linear gate
                    bert_coef = F.sigmoid(self.bert_gate_coef)
                    x = (1 - bert_coef) * x1 + bert_coef * x2
                elif self.gate is not None:  # dynamic elem-wise gate
                    x = self.gate(x1, x2)
                else:  # dropnet trick in raw paper
                    ratios = self.get_ratio()
                    x = ratios[0] * x1 + ratios[1] * x2

            else: # 替换原来的TransformerStandardDecoderLayer
                x = x1
                x = self.dropout_module(x)

            x = self.residual_connection(x, residual)
            ####################################### BERT-FUSED ##############################################


            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
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
