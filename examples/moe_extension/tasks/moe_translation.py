# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
@Time    :   2022/11/29 15:45
@Author  :   Zhu Jiahui
@Contact :   miugod0126@gmail.com
'''
from typing import Optional
import itertools
import json
import logging
import os
from fairseq.tasks.translation import TranslationTask,TranslationConfig
import numpy as np
from ..data import derLangPairDataset
from fairseq import metrics, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    LanguagePairDataset,
    indexed_dataset,
)
from fairseq.data.multi_corpus_dataset import MultiCorpusDataset
from dataclasses import dataclass, field
from fairseq.tasks import FairseqTask, register_task

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)

@dataclass
class MoETranslationConfig(TranslationConfig):
    '''
    entity_dict, topk
    '''
    use_der: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use domain entity routing loss."}
    )
    # 未来支持语言方向，好像不用，直接把双向的拼接在一起
    entity_dict: Optional[str] = field(
        default=None,
        metadata={"help": "Aligned dict path. (src_word\ttgt_word),word has been segmented with bpe "} # 支持含若干词的实体词，如“职业 吸引力”，必须分好bpe
    )

    topk: Optional[int] = field(
        default=-1,
        metadata={"help": "The number of words in the dictionary."}
    )

    der_coef: Optional[float] = field(
        default=0.5,
        metadata={"help": "Coefficient of domain entity routing loss."}
    )


    der_tau: Optional[float] = field(
        default=1.0,
        metadata={"help": "Temperature of domain entity routing loss."}
    )

    der_eps: Optional[float] = field(
        default=0.1,
        metadata={"help": "Label smooth epsilon temperature of domain entity routing loss."}
    )

    # noise_prob: Optional[float] = field(
    #     default=0.1,
    #     metadata={"help": "."}
    # )


    ### moe auxiliary loss ###
    aux_w: float = field(
        default=0.0, # 0.01 for gshard
        metadata={"help": "wight of auxiliary loss"},
    )

def load_langpair_dataset(
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

    ### RAS CONFIG ###
    seed=1,
    use_der=False,
    entity_tree=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )
        from fairseq.data import data_utils
        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source: # False
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
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos: # false
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id: # false
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments: # false
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    # domain entity routing
    dataset  =  derLangPairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        ### DER CONFIG ###
        use_der = use_der,
        entity_tree=entity_tree,
        split=split,
        seed=seed,

        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )
    # from fairseq.data import LanguagePairDataset
    # dataset  =  LanguagePairDataset(
    #     src_dataset,
    #     src_dataset.sizes,
    #     src_dict,
    #     tgt_dataset,
    #     tgt_dataset_sizes,
    #     tgt_dict,
    #     ### DER CONFIG ###
    #
    #     left_pad_source=left_pad_source,
    #     left_pad_target=left_pad_target,
    #     align_dataset=align_dataset,
    #     eos=eos,
    #     num_buckets=num_buckets,
    #     shuffle=shuffle,
    #     pad_to_multiple=pad_to_multiple,
    # )

    return dataset




@register_task("moe_translation", dataclass=MoETranslationConfig)
class MoETranslationTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: MoETranslationConfig

    def __init__(self, cfg: MoETranslationConfig,
                 src_dict, tgt_dict,
                 entity_tree=None,
                 ):
        super().__init__(cfg, src_dict, tgt_dict)
        self.entity_tree = entity_tree

    @classmethod
    def setup_task(cls, cfg: TranslationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        # load entity tree
        # entity_tree, entity_dict = cls.load_entity_tree(
        entity_tree = cls.load_entity_tree(
            os.path.join(paths[0],cfg.entity_dict), topk= cfg.topk
        ) if cfg.use_der else None

        return cls(cfg, src_dict, tgt_dict, entity_tree)


    @classmethod
    def load_entity_tree(cls, filename, topk=1000):
        import ahocorasick
        """Load the entity dictionary from the filename

        Args:
            filename (str): the filename
            topk: the number of words in the dictionary.
        return: tree of ahocorasick
        """
        if topk==-1: topk=1e9
        assert os.path.exists(filename),f"entity dict path {filename} not exists."
        entity_dict = []
        with open(filename,"r",encoding="utf-8") as f:
            for idx, line in enumerate(f.readlines()):
                if idx >= topk: break
                line  = line.strip().lower()
                if len(line.split())<=1: continue # 单个子词作为实体，没必要使用领域路由损失
                entity_dict.append(line)

        entity_tree = ahocorasick.AhoCorasick(*entity_dict)
        return entity_tree

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,

            ### DER CONFIG ###
            seed=epoch,
            use_der=self.cfg.use_der,
            entity_tree=self.entity_tree,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

