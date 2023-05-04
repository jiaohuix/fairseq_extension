'''
@Date:  2023/4/6
@Change:
    1. modify from: https://github.com/bert-nmt/bert-nmt/blob/update-20-10/fairseq/models/transformer.py
?no_encoder_attn?

'''
import logging
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, fields
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models.transformer import (
    TransformerEncoderBase,
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
from fairseq.distributed import fsdp_wrap
from transformers import AutoTokenizer, AutoModel
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from .bert_fused_layer import BertFusedEncoderLayer,BertFusedDecoderLayer
from bert_nmt_extensions.checkpoint_utils import print_trainable_parameters

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PeftConfig,
    PeftModel,
)
logger = logging.getLogger(__name__)
'''
应该这么搞：
1.对照旧版的fairseq和bert-nmt找出修改的地方
2.对照着新版fairseq和bert-nmt,先抄新版，再把修改的地方添加进去
不好。enc-dec可能需要args，cfg是层次结构的
'''

def check_bert_gates(bert_gates):
    if type(bert_gates)==str:
        bert_gates = [int(gate) for gate in bert_gates.split(",")]
    elif type(bert_gates) == list:
        bert_gates = [int(gate) for gate in bert_gates]

    for gate in bert_gates:
        assert gate==1 or gate==0, f"gate=0/1, but gate={gate}"
    bert_gates = [x == 1 for x in bert_gates]
    return bert_gates

# def bool_field(default: bool, metadata):
#     def wrapper(**kwargs):
#         if 'default' not in kwargs:
#             kwargs['default'] = default
#         if 'metadata' not in kwargs:
#             kwargs['metadata'] = metadata
#
#         return field(**kwargs)
#
#     return wrapper
#
#
# def update_bool_fields(instance):
#     for field_name, field_value in instance.__dict__.items():
#         if isinstance(field_value, bool):
#             setattr(instance, field_name, field_value or False)
#

@dataclass
class BertFusedConfig(TransformerConfig):
    # encoder_ratio和bert_ratio是在不用dropnet trick时的简单平均，默认h=nmt*1 + bert*1
    encoder_ratio: float = field(
        default=1, metadata={"help": "nmt ratio in dropnet trick?"}
    ) # ? 和bertdrop啥关系？

    bert_ratio: float = field(
        default=1, metadata={"help": "bert ratio in dropnet trick?"}
    )
    # bert掩码，除了pad，也把cls和sep掩为True，目的是在bert-attn中key_padding_mask=bert_encoder_padding_mask，对这些mask注意力分数变为0
    # mask_cls_sep: bool = field(
    #     default=False, metadata={"help": "mask_cls_sep"}
    # )

    # bert_gates: 是否启动某层的bert-fuse
    # bert_gates: List[int] = field(
    #     default=[1, 1, 1, 1, 1, 1], metadata={"help": "weather to fuse bert states  each layer."}
    # )
    # ValueError: mutable default <class 'list'> for field bert_gates is not allowed: use default_factory
    # bert_gates: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1], metadata={"help": "weather to fuse bert states  each layer."})
    bert_gates: str = field(default="1,1,1,1,1,1", metadata={"help": "weather to fuse bert states  each layer."})

    # ？没作用
    # bert_first: bool = field(
    #     default=False, metadata={"help": "bert_first?"}
    # )
    # 是否启用bert的dropnet
    # encoder_bert_dropout: bool = field(
    #     default=False, metadata={"help": "encoder_bert_dropout?"}
    # )
    # dropnet的ratio，以p/2的概率只用bert，p/2概率只用nmt，其余1-p的概率求平均（训练时）；推理时求平均
    encoder_bert_dropout_ratio: float = field(
        default=0.25, metadata={"help": "encoder_bert_dropout_ratio?"}
    )
    # 使用第几层的bert输出，在FairseqEncoderDecoderModel中 （没用上，在model里直接用的last state）
    bert_output_layer: int = field(
        default=-1, metadata={"help": "bert_output_layer?"}
    )
    # 这参数有点迷
    encoder_bert_mixup: bool = field(
        default=False, metadata={"help": "encoder_bert_mixup?"}
    )
    # encoder_no_bert: bool = field(
    #     default=False, metadata={"help": "encoder_no_bert?"}
    # )
    # # decoder不使用bert，都变成0
    # decoder_no_bert: bool = field(
    #     default=False, metadata={"help": "decoder_no_bert?"}
    # )

    ########## 自定义修改 ############
    # 1. 线性门控，用参数控制; 传给layer，如果存在就创建参数
    # linear_bert_gate_rate: float = field(
    #     default=None, metadata={"help": "linear_bert_gate_rate, (1-p) * nmt + p * bert"}
    # )

    ######## LoRA Config ########
    use_lora: bool = field(
        default=False, metadata={"help": "use_lora"}
    )
    # lora_task_type: str = field( default="SEQ_CLS", metadata={"help": "task_type"})
    lora_rank: int = field(default= 8,metadata={"help": "lora rank"})
    lora_alpha: int = field(default= 16,metadata={"help": "lora alpha"})
    lora_dropout: float = field(default= 0.1,metadata={"help": "lora dropout"})

    # def __post_init__(self):
    #     update_bool_fields(self)



# forward bert
@register_model('bert_fused')
class BertFusedTransformerModel(TransformerModel):
    def __init__(self, args, encoder, decoder,  bertencoder, berttokenizer):
        cfg = TransformerConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args # args和cfg有什么区别?到底用哪个。 答：args扁平的argparse参数namespace，cfg通过convert_namespace_to_omegana,转为嵌套的参数结构，具体的嵌套的范围由Dataclass注册的参数决定，如task，model，等，其他的会归入common。
        self.berttokenizer = berttokenizer
        self.mask_cls_sep = args.mask_cls_sep
        self.bert_output_layer = getattr(args, 'bert_output_layer', -1)
        # paralfuse
        self.parafuse = args.parafuse
        # layer select type
        self.ls_type = args.ls_type
        self.use_lora = args.use_lora

        ######## LoRA Config ########
        if args.use_lora:
            # https://github.com/huggingface/peft/issues/219,  when use lora, remove "task_type"
            # peft_config = LoraConfig(task_type=args.lora_task_type, inference_mode=False, r=args.lora_rank,
            peft_config = LoraConfig(inference_mode=False, r=args.lora_rank,
                                     lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
            bertencoder = get_peft_model(bertencoder, peft_config)
            bertencoder.print_trainable_parameters()
            self.peft_config = peft_config
        else:
            self.peft_config = None

        self.bert_encoder = bertencoder

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        # do not set defaults so that settings defaults from various architectures still works
        parser.add_argument('--mask-cls-sep', action='store_true',)
        parser.add_argument('--encoder-bert-dropout', action='store_true',)
        parser.add_argument('--encoder-no-bert', action='store_true',)
        parser.add_argument('--decoder-no-bert', action='store_true',)

        # pooler experients, in order to replace the bert attention module
        parser.add_argument('--use-pooler', action='store_true',)
        parser.add_argument('--pooler-type', default="avg", help="[avg/max/cnn/rand]") # pool效果不好，不如注意力吧
        # dynamic gate
        parser.add_argument('--dynamic-gate', action='store_true',)
        parser.add_argument('--gate-type', default="dynamic", help="[dynamic/glu/glunorm/zero]")
        # 线性门控，用参数控制; 传给layer，如果存在就创建参数
        parser.add_argument('--linear-bert-gate-rate', default=None,type=float, help="linear_bert_gate_rate, (1-p) * nmt + p * bert")

        # parafuse
        parser.add_argument('--parafuse', action='store_true',)

        # share bert attn
        parser.add_argument('--share-enc-bertattn', action='store_true',)
        parser.add_argument('--share-dec-bertattn', action='store_true',)
        parser.add_argument('--share-all-bertattn', action='store_true',)

        # layer select for bert
        parser.add_argument('--ls-type', default="selk",type=str, help="[selk/sto1/sto2/sto8/tok_moe_k2/seq_moe_k2...]")
        parser.add_argument('--bert-output-layer', default=-1,type=int, help="select bert output layer")

        # lr group, make pre-training and new-added modules have different learning rates
        parser.add_argument('--fuse-lr-multiply', default=1,type=int, help="fuse module")

        ######## LoRA Config ########
        parser.add_argument('--use-lora', action='store_true',)
        # parser.add_argument('--lora-task-type', default="SEQ_CLS",type=str, help="task_type, [SEQ_CLS/TOKEN_CLS/SEQ_CLS]")
        parser.add_argument('--lora-rank', default=8,type=int, help="lora rank")
        parser.add_argument('--lora-alpha', default=16,type=int, help="lora alpha")
        parser.add_argument('--lora-dropout', default=0.1,type=float, help="lora dropout")


        gen_parser_from_dataclass(
            parser, BertFusedConfig(), delete_default=True, with_prefix=""
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        
        ####################################### BERT-FUSED ##############################################
        # tokenizer
        if len(task.datasets) > 0:
            src_berttokenizer = next(iter(task.datasets.values())).berttokenizer
        else:
            src_berttokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)
        # bert_model
        bertencoder = AutoModel.from_pretrained(args.bert_model_name)
        args.bert_out_dim = bertencoder.config.hidden_size
        #args.bert_out_dim = bertencoder.hidden_size
        # make sure all arguments are present in older models
        bert_fuse_base_arch(args)
        cfg = BertFusedConfig.from_namespace(args)

        ####################################### BERT-FUSED ##############################################
        
        
        # COPY FROM TransformerModelBase
        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        elif cfg.merge_src_tgt_embed:
            logger.info(f"source dict size: {len(src_dict)}")
            logger.info(f"target dict size: {len(tgt_dict)}")
            src_dict.update(tgt_dict)
            task.src_dict = src_dict
            task.tgt_dict = src_dict
            logger.info(f"merged dict size: {len(src_dict)}")
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
        # COPY FROM TransformerModelBase



        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        return cls(cfg, encoder, decoder, bertencoder, src_berttokenizer)

    # 没必要写
    # @classmethod
    # def build_embedding(cls, args, dictionary, embed_dim, path=None):
    #     return super().build_embedding(
    #         BertFusedConfig.from_namespace(args), dictionary, embed_dim, path
    #     )

    # 要重写,不是构建base类，而是fuse层
    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return BertFusedEncoder(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return BertFusedDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )


    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        bert_input,   # <--bert fuse
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

        def parafuse_mask(input_ids, sep_id=103):
            # type id
            token_type_ids = (input_ids == sep_id).int()
            token_type_ids = torch.cumsum(token_type_ids, dim=-1)  # ≈token type ids
            # mask for enc-dec
            bert_enc_mask = token_type_ids == 2
            bert_dec_mask = token_type_ids == 1
            return bert_enc_mask, bert_dec_mask

        # 1. forward bert
        bert_pad_id = self.berttokenizer.convert_tokens_to_ids(self.berttokenizer.pad_token)
        bert_cls_id = self.berttokenizer.convert_tokens_to_ids(self.berttokenizer.cls_token)
        bert_sep_id = self.berttokenizer.convert_tokens_to_ids(self.berttokenizer.sep_token)
        bert_input = bert_input[:,:512]

        bert_encoder_padding_mask = bert_input.eq(bert_pad_id)
        # 这0是last hidden state
        # bert_encoder_outs = self.bert_encoder(bert_input, attention_mask= ~bert_encoder_padding_mask)[0] #直接获取最后一层输出，没返回n层states
        last_hidden_state,pooler_output,hidden_states = self.bert_encoder(bert_input, attention_mask= ~bert_encoder_padding_mask,
                                                                          output_hidden_states = True, return_dict = False)
        # bert_encoder_outs = last_hidden_state

        # hidden_states = self.bert_encoder(src_tokens, attention_mask= ~bert_encoder_padding_mask, output_hidden_states=True)[2]
        # bert_encoder_outs = hidden_states[self.bert_output_layer] # default take last hidden state of bert
        if self.mask_cls_sep:
            bert_encoder_padding_mask += bert_input.eq(bert_cls_id)
            bert_encoder_padding_mask += bert_input.eq(bert_sep_id)

        # for parafuse
        bert_decoder_padding_mask = None
        if self.parafuse:
            bert_enc_mask, bert_dec_mask = parafuse_mask(bert_input, sep_id=bert_sep_id) # bert_input
            bert_decoder_padding_mask = bert_encoder_padding_mask + bert_dec_mask # 防止重复加src src2 ,  [src1,tgt2]
            bert_encoder_padding_mask += bert_enc_mask # [src1, src2]

        bert_encoder_out = [state.permute(1,0,2).contiguous()  for state in hidden_states] # [[T B C]...]
        bert_encoder_outs = {
            # 'bert_encoder_outs': bert_encoder_out[-1] if self.ls_type=="sto1" else bert_encoder_out,
            'bert_encoder_outs': bert_encoder_out,
            'bert_encoder_padding_mask': bert_encoder_padding_mask,
            'bert_decoder_padding_mask': bert_decoder_padding_mask,
        }
        # 2. forward encoder
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens,
            bert_encoder_outs=bert_encoder_outs # <--bert fused
        )
        # 3. forward decoder
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            bert_encoder_outs=bert_encoder_outs, # <--bert fused
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out


# enc-dec添加构建layer的参数，以及forward中传bert
class BertFusedEncoder(TransformerEncoderBase):
    def __init__(self, cfg, dictionary, embed_tokens, return_fc=False):
        self.cfg = cfg
        super().__init__(
            cfg,
            dictionary,
            embed_tokens,
            return_fc=return_fc,
        )
        ####################################### BERT-FUSED ##############################################
        bert_gates = getattr(cfg, 'bert_gates', "1,1,1,1,1,1")
        bert_gates = check_bert_gates(bert_gates)
        assert len(bert_gates) == cfg.encoder_layers
        self.layers = nn.ModuleList([])
        encoder_no_bert = getattr(cfg, 'encoder_no_bert', False)
        # 可以在encoder_no_bert时，不forward bert
        if encoder_no_bert:
            bert_gates = [0] * len(bert_gates)

        ###### 共享bertattn参数 ######
        if cfg.share_enc_bertattn or cfg.share_all_bertattn:
            bertattn = BertFusedEncoderLayer.build_bert_attention(cfg.encoder.embed_dim, cfg)
        else:
            bertattn = None
        ###### 共享bertattn参数 ######

        self.layers.extend([
            # BertFusedEncoderLayer(cfg, bert_gate=bert_gates[i])
            self.build_encoder_layer(cfg,bert_gates[i],bertattn)
            for i in range(cfg.encoder.layers)
        ])
        ####################################### BERT-FUSED ##############################################

    def build_encoder_layer(self, cfg, bert_gate=1, bertattn=None):
        layer = BertFusedEncoderLayer(
            cfg, bert_gate=bert_gate,bertattn=bertattn,
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        bert_encoder_outs = None ,  # <--- bert fused
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = (
            torch.tensor(src_tokens.device.type == "xla") or encoder_padding_mask.any()
        )
        # Torchscript doesn't handle bool Tensor correctly, so we need to work around.
        if torch.jit.is_scripting():
            has_pads = torch.tensor(1) if has_pads else torch.tensor(0)

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

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
            ####################################### BERT-FUSED ##############################################
            bert_pad_mask = bert_encoder_outs['bert_encoder_padding_mask']

            lr = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None,
                bert_encoder_outs = bert_encoder_outs['bert_encoder_outs'],
                bert_encoder_padding_mask = bert_pad_mask)
            ####################################### BERT-FUSED ##############################################

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


    # generator用，encoder_out是encoder输出，bert_out是直接forward bert输出
    def reorder_encoder_out(self, encoder_out, bert_out, new_order):
        encoder_out = super().reorder_encoder_out(encoder_out, new_order)
        if bert_out['bert_encoder_outs'] is not None: # [states*13]
            # bert_out['bert_encoder_outs'] =  bert_out['bert_encoder_outs'].index_select(1, new_order)
            bert_out['bert_encoder_outs'] = [state.index_select(1, new_order) for state in bert_out['bert_encoder_outs'] ]
        if bert_out['bert_encoder_padding_mask'] is not None:
            bert_out['bert_encoder_padding_mask'] = \
                bert_out['bert_encoder_padding_mask'].index_select(0, new_order)
        # FOR PARAL FUSE
        if bert_out['bert_decoder_padding_mask'] is not None:
            bert_out['bert_decoder_padding_mask'] = \
                bert_out['bert_decoder_padding_mask'].index_select(0, new_order)
        return encoder_out, bert_out

    def forward_torchscript(self, net_input: Dict[str, Tensor], bert_encoder_outs=None):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """

        if torch.jit.is_scripting():
            return self.forward(
                src_tokens=net_input["src_tokens"],
                src_lengths=net_input["src_lengths"],
                bert_encoder_outs=bert_encoder_outs,
            )
        else:
            return self.forward_non_torchscript(net_input, bert_encoder_outs)

    @torch.jit.unused
    def forward_non_torchscript(self, net_input: Dict[str, Tensor], bert_encoder_outs):
        encoder_input = {
            k: v for k, v in net_input.items()
            if k != 'prev_output_tokens' and k != 'bert_input'
        }
        encoder_input['bert_encoder_outs'] = bert_encoder_outs
        return self.forward(**encoder_input)

class BertFusedDecoder(TransformerDecoderBase):
    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(
            cfg,
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )
        ####################################### BERT-FUSED ##############################################
        # decoder的gate，是否某层用bert-fuse
        bert_gates = getattr(cfg, 'bert_gates', "1,1,1,1,1,1")
        bert_gates = check_bert_gates(bert_gates)
        assert len(bert_gates) == cfg.decoder_layers
        self.layers = nn.ModuleList([])
        # 如果decoder_no_bert，直接用标准的decoderlayer
        decoder_no_bert = getattr(cfg, 'decoder_no_bert', False)
        # 可以在decoder_no_bert时，不forward bert
        if decoder_no_bert:
            bert_gates = [0] * len(bert_gates)


        ###### 共享bertattn参数 ######
        if cfg.share_dec_bertattn or cfg.share_all_bertattn:
            bertattn = BertFusedDecoderLayer.build_bert_attention(cfg.decoder.embed_dim, cfg)
        else:
            bertattn = None
        ###### 共享bertattn参数 ######

        self.layers.extend([
            # TransformerDecoderLayer(cfg, no_encoder_attn, bert_gate=bert_gates[i])
            self.build_decoder_layer(cfg, no_encoder_attn, bert_gate=bert_gates[i],bertattn=bertattn)
            for i in range(cfg.decoder.layers)
        ])


    def build_decoder_layer(self, cfg, no_encoder_attn=False, bert_gate=1,bertattn=None):
        layer =  BertFusedDecoderLayer(cfg, no_encoder_attn, bert_gate = bert_gate,bertattn=bertattn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer


    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        bert_encoder_outs=None, # <--- bert fused
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):


        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            bert_encoder_outs=bert_encoder_outs, # <--- bert fused
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:
            x = self.output_layer(x)
        return x, extra


    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        bert_encoder_outs=None,  # <--- bert fused
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
            bert_pad_mask = bert_encoder_outs['bert_decoder_padding_mask'] \
                            if bert_encoder_outs['bert_decoder_padding_mask'] is not None \
                            else bert_encoder_outs['bert_encoder_padding_mask']
            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                bert_encoder_outs['bert_encoder_outs'],  # <--- bert fused
                bert_pad_mask,  # <--- bert fused
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}




@register_model_architecture("bert_fused", "bert_fused")
def bert_fuse_base_arch(args):
    args.encoder_ratio = getattr(args, "encoder_ratio", 1)
    args.bert_ratio = getattr(args, "bert_ratio", 1)
    args.mask_cls_sep = getattr(args, "mask_cls_sep", False)
    args.bert_gates = getattr(args, "bert_gates", "1,1,1,1,1,1")
    # args.bert_first = getattr(args, "bert_first", False)
    args.encoder_bert_dropout = getattr(args, "encoder_bert_dropout", False)
    args.encoder_bert_dropout_ratio = getattr(args, "encoder_bert_dropout_ratio", 0.25)
    args.bert_output_layer = getattr(args, "bert_output_layer", -1)
    args.encoder_bert_mixup = getattr(args, "encoder_bert_mixup", False)
    args.encoder_no_bert = getattr(args, "encoder_no_bert", False)
    args.decoder_no_bert = getattr(args, "decoder_no_bert", False)

    # pooler experients, in order to replace the bert attention module
    args.use_pooler = getattr(args, "use_pooler", False)
    args.pooler_type = getattr(args, "pooler_type", "avg")
    # gate
    args.dynamic_gate = getattr(args, "dynamic_gate", False)
    args.gate_type = getattr(args, "gate_type", "dynamic")
    args.linear_bert_gate_rate = getattr(args, "linear_bert_gate_rate", None)
    # similar parallel sent
    args.parafuse = getattr(args, "parafuse", False)
    # share bert attn
    args.share_enc_bertattn = getattr(args, "share_enc_bertattn", False)
    args.share_dec_bertattn = getattr(args, "share_dec_bertattn", False)
    args.share_all_bertattn = getattr(args, "share_all_bertattn", False)

    # layer select for bert
    args.ls_type = getattr(args, "ls_type", "sto1")

    # lr group
    args.fuse_lr_multiply = getattr(args, "fuse_lr_multiply", 1)

    # lora
    args.use_lora = getattr(args, "use_lora", False)
    # args.lora_task_type = getattr(args, "lora_task_type", "SEQ_CLS")
    args.lora_rank = getattr(args, "lora_rank", 18)
    args.lora_alpha = getattr(args, "lora_alpha", 16)
    args.lora_dropout = getattr(args, "lora_dropout", 0.1)



    base_architecture(args)

@register_model_architecture('bert_fused', 'bert_fused_iwslt_de_en')
def bert_fused_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    bert_fuse_base_arch(args)

# @register_model_architecture('transformerstack', 'transformerstack_iwslt_de_en')
# def transformerstack_iwslt_de_en(args):
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
#     args.encoder_layers = getattr(args, 'encoder_layers', 6)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
#     args.decoder_layers = getattr(args, 'decoder_layers', 6)
#     base_stack_architecture(args)
#
# @register_model_architecture('transformers2', 'transformer_wmt_en_de')
# def transformer_wmt_en_de(args):
#     base_architecture_s2(args)
#
#
# # parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
# @register_model_architecture('transformer', 'transformer_vaswani_wmt_en_de_big')
# def transformer_vaswani_wmt_en_de_big(args):
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
#     args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
#     args.dropout = getattr(args, 'dropout', 0.3)
#     base_architecture(args)
#
# @register_model_architecture('transformers2', 'transformer_s2_vaswani_wmt_en_de_big')
# def transformer_s2_vaswani_wmt_en_de_big(args):
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
#     args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
#     args.dropout = getattr(args, 'dropout', 0.3)
#     base_architecture_s2(args)
#
# @register_model_architecture('transformer', 'transformer_vaswani_wmt_en_fr_big')
# def transformer_vaswani_wmt_en_fr_big(args):
#     args.dropout = getattr(args, 'dropout', 0.1)
#     transformer_vaswani_wmt_en_de_big(args)
#
#
# @register_model_architecture('transformer', 'transformer_wmt_en_de_big')
# def transformer_wmt_en_de_big(args):
#     args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
#     transformer_vaswani_wmt_en_de_big(args)
#
#
# # default parameters used in tensor2tensor implementation
# @register_model_architecture('transformer', 'transformer_wmt_en_de_big_t2t')
# def transformer_wmt_en_de_big_t2t(args):
#     args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
#     args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
#     args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
#     args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
#     transformer_vaswani_wmt_en_de_big(args)
