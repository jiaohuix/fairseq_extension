# 1.导入decoder，2.修改encoder 3.添加model
# TODO: 写个encoder,支持bert_input这个参数(criterion在forward时,取了net_input的参数,以**关键字参数传给encoder  )
# TODO: qblock qformer直接是transformer解码器一层或若干层
import logging
import math
import torch
import torch.nn as nn
from fairseq.modules import PositionalEmbedding
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.transformer_layer import TransformerDecoderLayerBase
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models.transformer import (
    TransformerEncoderBase,  # 需要继承修改
    TransformerDecoderBase,
    TransformerModel,
    TransformerConfig,
    base_architecture,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    transformer_layer,
)
from fairseq.distributed import fsdp_wrap
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional
from fairseq.models.transformer import TransformerConfig
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

logger = logging.getLogger(__name__)


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerEncoderBase":
        return "TransformerEncoder"
    else:
        return module_name


class QBlock(TransformerDecoderLayerBase):
    def __init__(
        self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(cfg, no_encoder_attn, add_bias_kv, add_zero_attn)

    # def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
    #     return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)
    #
    # def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
    #     return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    # def build_self_attention(
    #     self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    # ):
    #     return MultiheadAttention(
    #         embed_dim,
    #         cfg.decoder.attention_heads,
    #         dropout=cfg.attention_dropout,
    #         add_bias_kv=add_bias_kv,
    #         add_zero_attn=add_zero_attn,
    #         self_attention=not cfg.cross_self_attention,
    #         q_noise=self.quant_noise,
    #         qn_block_size=self.quant_noise_block_size,
    #         xformers_att_config=cfg.decoder.xformers_att_config,
    #     )

    def build_encoder_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            # kdim=cfg.encoder.embed_dim,
            # vdim=cfg.encoder.embed_dim,
            kdim=768,
            vdim=768,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            xformers_att_config=cfg.encoder.xformers_att_config,
        )




# 参数写一个类。
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class QFormer(nn.Module):
    def __init__(self, cfg):
        super(QFormer, self).__init__()
        self.cfg = cfg
        self.n_query = cfg.n_query
        self.q_layers = cfg.q_layers

        self.nmt_dim = cfg.encoder_embed_dim
        self.query_embed = PositionalEmbedding(
            self.n_query,
            self.nmt_dim,
            padding_idx=1,
            learned=True,
        )
        # self.qformer = MultiheadAttention(
        #     nmt_dim,
        #     n_heads,
        #     dropout=drop,
        #     self_attention=False,
        #     kdim=bert_dim,
        #     vdim=bert_dim
        # )
        q_layer_ls = [QBlock(cfg=self.cfg) for _ in range(self.q_layers)]
        self.qformer = nn.ModuleList(q_layer_ls)

    def forward(self, bert_out, bert_mask=None): #
        # x: bert输出
        bsz = bert_out.shape[0]
        x = torch.arange(0, self.n_query).unsqueeze(0).expand(bsz, -1)
        x = x.to(device=bert_out.device)
        q = self.query_embed(x)
        # B x T x C -> T x B x C
        q = q.transpose(0, 1)
        bert_out = bert_out.transpose(0, 1)
        # input encoder_out encoder_pad
        q, attn_weights = self.qformer(q, bert_out)
        q = q.transpose(0, 1)

        # if self.ffn is not None:
        #     q = self.ffn(q)
        return q


class BertNMTEncoder(TransformerEncoderBase):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg, dictionary, embed_tokens, return_fc=False):
        super().__init__(cfg, dictionary, embed_tokens, return_fc=return_fc)

        # bert encoder
        self.bert_tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model_name)
        self.bert_encoder = AutoModel.from_pretrained(cfg.bert_model_name)
        self.bert_out_dim = self.bert_encoder.config.hidden_size
        for param in self.bert_encoder.parameters():
            param.requires_grad = False
        print_trainable_parameters(self.bert_encoder)
        # lora...

        # qformer
        self.qformer = QFormer(cfg)

    def forward_bert(self, bert_input):
        if bert_input is None: return bert_input
        # 1. forward bert
        bert_pad_id = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.pad_token)
        bert_cls_id = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.cls_token)
        bert_sep_id = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.sep_token)
        bert_input = bert_input[:, :512]

        bert_encoder_padding_mask = bert_input.eq(bert_pad_id)
        if self.cfg.mask_cls_sep:
            bert_encoder_padding_mask += bert_input.eq(bert_cls_id)
            bert_encoder_padding_mask += bert_input.eq(bert_sep_id)

        bert_out = self.bert_encoder(bert_input, attention_mask=~bert_encoder_padding_mask)[0]  # 直接获取最后一层输出，没返回n层states

        return bert_out

    def fuse_embedding(self, nmt_embed, bert_embed):
        ''' qformer压缩bert输出,然后和nmt输入拼接,拼接右边,因为left-pad source(src tokens和length需要修改)'''
        q_embed = self.qformer(bert_embed)
        fused_embed = torch.cat([nmt_embed, q_embed], dim=1)  # [bsz,nmt_len,dim]->[bsz,nmt_len+n_query, dim]
        return fused_embed

    def forward_embedding(
            self, src_tokens, bert_input=None, token_embedding: Optional[torch.Tensor] = None
    ):
        ''' 需要获取bert输入 '''
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)

        ### BERT EMBED ###
        if bert_input is not None:
            bert_out = self.forward_bert(bert_input)
            x = self.fuse_embedding(nmt_embed=x, bert_embed=bert_out)
            embed = self.fuse_embedding(nmt_embed=embed, bert_embed=bert_out)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)

        # 在这里尽量不修改encoder的定义，而是先获取embed，然后传到encoder里面
        # 注意：如果拿embed和bert的拼接，那么对应的src_tokens需要修改（右边拼接tokens）；src_lengths也要修改

        return x, embed

    def forward(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            bert_input=None,  # <== BERT INPUT TOKENS
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        return self.forward_scriptable(
            src_tokens, src_lengths, bert_input, return_all_hiddens, token_embeddings
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            bert_input=None,  # <== BERT INPUT TOKENS
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        # 1.先计算embed， 2.再拼接src_tok， 3.再获取pad mask
        x, encoder_embedding = self.forward_embedding(src_tokens,bert_input, token_embeddings) # [b, t+n_query, c]

        # 先拼接tokens，再获取mask
        if bert_input is not None:  # 修改tokens,拼接到右侧;修改长度
            bsz, nmt_len = src_tokens.size()
            val = -1
            fake_tokens = torch.full([bsz, self.cfg.n_query], fill_value=val).to(device=src_tokens.device)
            src_tokens = torch.cat([src_tokens, fake_tokens], dim=1)
            src_lengths = src_lengths + self.cfg.n_query

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)  # 这个padding mask应该放到后面啊
        has_pads = (
                torch.tensor(src_tokens.device.type == "xla") or encoder_padding_mask.any()
        )

        # Torchscript doesn't handle bool Tensor correctly, so we need to work around.
        if torch.jit.is_scripting():
            has_pads = torch.tensor(1) if has_pads else torch.tensor(0)

        # account for padding while computing the representation
        x = x * (
                1 - encoder_padding_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x)
        )

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            lr = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # 删除顶层的query tokens
        if self.cfg.del_top_qtok:
            n_query = self.cfg.n_query
            x = x[:- n_query, :, :]  # [T B C]
            src_tokens = src_tokens[:, :- n_query]  # [B T]
            encoder_padding_mask = src_tokens.eq(self.padding_idx)  # 这个padding mask应该放到后面啊
            encoder_embedding = encoder_embedding[:, :-n_query, :]
            encoder_states = [state[:- n_query,:,:] for state in encoder_states if state is not None] # List[T x B x C]
            fc_results = [fc[:- n_query,:,:] for fc in fc_results if fc is not None] # List[T x B x C]

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }


@register_model('bert_nmt_qblock')
class BertNMTModel(TransformerModel):

    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--mask-cls-sep', action='store_true')
        parser.add_argument('--bert-model-name', default="bert-base", type=str)

        ## qformer
        parser.add_argument('--n-query', default=8, type=int)
        parser.add_argument('--q-layers', default=1, type=int)
        parser.add_argument('--q-ffn', action='store_true')
        parser.add_argument('--q-drop', default=0., type=float)
        parser.add_argument('--del-top-qtok', action='store_true')  # 删除顶层n_query个用完的query tokens

        ######## LoRA Config ########
        # parser.add_argument('--use-lora', action='store_true', )
        # # parser.add_argument('--lora-task-type', default="SEQ_CLS",type=str, help="task_type, [SEQ_CLS/TOKEN_CLS/SEQ_CLS]")
        # parser.add_argument('--lora-rank', default=8, type=int, help="lora rank")
        # parser.add_argument('--lora-alpha', default=16, type=int, help="lora alpha")
        # parser.add_argument('--lora-dropout', default=0.1, type=float, help="lora dropout")
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=True, with_prefix=""
        )

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return BertNMTEncoder(cfg, src_dict, embed_tokens)

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            bert_input=None,  # <--bert fuse
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            bert_input=bert_input,
            return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out


@register_model_architecture("bert_nmt_qblock", "bert_nmt_qb")
def bnmt_base_arch(args):
    args.mask_cls_sep = getattr(args, "mask_cls_sep", False)
    args.bert_model_name = getattr(args, "bert_model_name", "bert-base-uncased")
    ## qformer
    args.n_query = getattr(args, "n_query", 8)
    args.q_layers = getattr(args, "q_layers", 1)
    args.q_ffn = getattr(args, "q_ffn", False)
    args.q_drop = getattr(args, "q_drop", 0.)
    args.del_top_qtok = getattr(args, "del_top_qtok", False)

    # lora
    # args.use_lora = getattr(args, "use_lora", False)
    # args.lora_task_type = getattr(args, "lora_task_type", "SEQ_CLS")
    # args.lora_rank = getattr(args, "lora_rank", 18)
    # args.lora_alpha = getattr(args, "lora_alpha", 16)
    # args.lora_dropout = getattr(args, "lora_dropout", 0.1)
    base_architecture(args)


@register_model_architecture('bert_nmt_qblock', 'bnmt_qb_iwslt_de_en')
def bnmt_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    bnmt_base_arch(args)



