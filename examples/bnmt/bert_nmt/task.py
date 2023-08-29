# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
'''
@Date: 2023/4/6
@Change:
    1. copy code from: https://github.com/bert-nmt/bert-nmt/blob/update-20-10/fairseq/tasks/translation.py
    2.  "load_langpair_dataset" 改名为 "load_bert_nmt_dataset"， 将BertTokenizer改为AutoTokenizer
    3. 修改BertNMTTask中add_args的参数为BertNMTConfig，并继承自TranslationConfig
    4. BertNMTTask继承TranslationTask，删除setup_task， max_positions, source_dictionary, target_dictionary
'''
import torch
from fairseq import search, tokenizer, utils
import logging
import itertools
import os
from fairseq import options, utils
from typing import Optional, List
from .data import LanguageTripleDataset
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from fairseq.tasks import  FairseqTask, register_task
from fairseq.tasks.translation import TranslationConfig,TranslationTask
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
logger = logging.getLogger(__name__)

def load_bert_nmt_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,

    bert_model_name = None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    srcbert_datasets = []  # <-----BERT-FUSED
    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
            bertprefix = os.path.join(data_path, '{}.bert.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
            bertprefix = os.path.join(data_path, '{}.bert.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        srcbert_datasets.append( data_utils.load_indexed_dataset(bertprefix + src, dataset_impl )) # <-----BERT-FUSED

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
        srcbert_datasets = srcbert_datasets[0]

    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    berttokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    return LanguageTripleDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        srcbert_datasets, srcbert_datasets.sizes, berttokenizer,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


 # 不对啊，有些不是bert-nmt需要的参数，反而是model的。。。
@dataclass
class BertNMTConfig(TranslationConfig):
    # task参数
    bert_model_name: Optional[str] = field(
        default="bert-base-uncased",
        metadata={"help": "pretrained bert name"},
    )
    # train参数
    finetune_bert: bool = field(
        default=False, metadata={"help": "finetune bert"}
    )
    warmup_from_nmt: bool = field(
        default=False,  metadata= {"help": "warmup_from_nmt"}
    )
    # ckpt utils
    warmup_nmt_file: str = field(
        default="checkpoint_nmt.pt", metadata={"help": "pretrained nmt model to warmup bert-fused model."}
    )
    freeze_nmt: bool = field(
        default=False, metadata={"help": "freeze nmt"}
    )




@register_task('bert_nmt', dataclass = BertNMTConfig)
class BertNMTTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    cfg: BertNMTConfig

    def __init__(self, cfg: BertNMTConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.bert_model_name = cfg.bert_model_name

    # @classmethod
    # def setup_task(cls, cfg: BertNMTConfig, **kwargs):
    #     """Setup the task (e.g., load dictionaries).

        # return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        # paths = self.cfg.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_bert_nmt_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            bert_model_name = self.bert_model_name
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, srcbert, srcbert_sizes, berttokenizer):
        return LanguageTripleDataset(src_tokens, src_lengths, self.source_dictionary, srcbert=srcbert, srcbert_sizes=srcbert_sizes, berttokenizer=berttokenizer)

    # def max_positions(self):
    #     """Return the max sentence length allowed by the task."""
    #     return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    # @property
    # def source_dictionary(self):
    #     """Return the source :class:`~fairseq.data.Dictionary`."""
    #     return self.src_dict

    # @property
    # def target_dictionary(self):
    #     """Return the target :class:`~fairseq.data.Dictionary`."""
    #     return self.tgt_dict

    # def build_generator(
    #     self,
    #     models,
    #     args,
    #     seq_gen_cls=None,
    #     extra_gen_cls_kwargs=None,
    #     prefix_allowed_tokens_fn=None,
    # ):
    #     if getattr(args, "score_reference", False):
    #         from fairseq.sequence_scorer import SequenceScorer

    #         return SequenceScorer(
    #             self.target_dictionary,
    #             compute_alignment=getattr(args, "print_alignment", False),
    #         )


    #     # Choose search strategy. Defaults to Beam Search.
    #     sampling = getattr(args, "sampling", False)
    #     sampling_topk = getattr(args, "sampling_topk", -1)
    #     sampling_topp = getattr(args, "sampling_topp", -1.0)
    #     diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
    #     diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
    #     match_source_len = getattr(args, "match_source_len", False)
    #     diversity_rate = getattr(args, "diversity_rate", -1)
    #     constrained = getattr(args, "constraints", False)
    #     if prefix_allowed_tokens_fn is None:
    #         prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
    #     if (
    #         sum(
    #             int(cond)
    #             for cond in [
    #                 sampling,
    #                 diverse_beam_groups > 0,
    #                 match_source_len,
    #                 diversity_rate > 0,
    #             ]
    #         )
    #         > 1
    #     ):
    #         raise ValueError("Provided Search parameters are mutually exclusive.")
    #     assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
    #     assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

    #     if sampling:
    #         search_strategy = search.Sampling(
    #             self.target_dictionary, sampling_topk, sampling_topp
    #         )
    #     elif diverse_beam_groups > 0:
    #         search_strategy = search.DiverseBeamSearch(
    #             self.target_dictionary, diverse_beam_groups, diverse_beam_strength
    #         )
    #     elif match_source_len:
    #         # this is useful for tagging applications where the output
    #         # length should match the input length, so we hardcode the
    #         # length constraints for simplicity
    #         search_strategy = search.LengthConstrainedBeamSearch(
    #             self.target_dictionary,
    #             min_len_a=1,
    #             min_len_b=0,
    #             max_len_a=1,
    #             max_len_b=0,
    #         )
    #     elif diversity_rate > -1:
    #         search_strategy = search.DiverseSiblingsSearch(
    #             self.target_dictionary, diversity_rate
    #         )
    #     elif constrained:
    #         search_strategy = search.LexicallyConstrainedBeamSearch(
    #             self.target_dictionary, args.constraints
    #         )
    #     elif prefix_allowed_tokens_fn:
    #         search_strategy = search.PrefixConstrainedBeamSearch(
    #             self.target_dictionary, prefix_allowed_tokens_fn
    #         )
    #     else:
    #         search_strategy = search.BeamSearch(self.target_dictionary)

    #     extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}

    #     from .bert_fused_generator import BertFusedeGenerator
    #     bert_output_layer = models[0].args.bert_output_layer
    #     return BertFusedeGenerator(
    #         models,
    #         self.target_dictionary,
    #         beam_size=getattr(args, "beam", 5),
    #         max_len_a=getattr(args, "max_len_a", 0),
    #         max_len_b=getattr(args, "max_len_b", 200),
    #         min_len=getattr(args, "min_len", 1),
    #         normalize_scores=(not getattr(args, "unnormalized", False)),
    #         len_penalty=getattr(args, "lenpen", 1),
    #         unk_penalty=getattr(args, "unkpen", 0),
    #         temperature=getattr(args, "temperature", 1.0),
    #         match_source_len=getattr(args, "match_source_len", False),
    #         no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
    #         search_strategy=search_strategy,
    #         # bert_output_layer = args.bert_output_layer,
    #         # bert_output_layer = -1,
    #         bert_output_layer = bert_output_layer,
    #         **extra_gen_cls_kwargs,
    #     )

