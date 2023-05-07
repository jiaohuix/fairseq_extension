# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Optional
from argparse import Namespace
from omegaconf import II

import numpy as np
from fairseq import utils
from fairseq.logging import metrics
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
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.translation import TranslationTask,TranslationConfig
from fairseq.dataclass.utils import gen_parser_from_dataclass
from ..data import LanguagePairDatasetWithEntityCT


EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


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

    ##################### For Entity Contrastive #####################
    seed=1,
    use_entity_ct=False,
    entity_dict=False,
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

    ##################### For Entity Contrastive #####################
    return LanguagePairDatasetWithEntityCT(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        ##################### For Entity Contrastive #####################
        use_entity_ct=use_entity_ct,
        entity_dict=entity_dict,
        split=split,
        seed=seed,
        ##################### For Entity Contrastive #####################
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )



# 该config会将所有参数args中task相关的传过来作为task的cfg
@dataclass
class EntityCTConfig(TranslationConfig):
    '''
    entity_dict, topk
    '''
    use_entity_ct: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use entity contrastive loss."}
    )
    entity_dict: Optional[str] = field(
        default="entity.txt",
        metadata={"help": "Aligned dict path. (src_word\ttgt_word),word has been segmented with bpe "} # 支持含若干词的实体词，如“职业 吸引力”，必须分好bpe
    )

    topk: Optional[int] = field(
        default=-1,
        metadata={"help": "The number of words in the dictionary."}
    )

    # der_coef: Optional[float] = field(
    #     default=0.5,
    #     metadata={"help": "Coefficient of domain entity routing loss."}
    # )
    # tau, loss alpha, use sentence ct



@register_task("entity_ct_translation", dataclass=EntityCTConfig)
class EntityCTTranslationTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: EntityCTConfig

    def __init__(self, cfg: EntityCTConfig,
                 src_dict, tgt_dict,
                 entity_dict=None,
                 ):
        super().__init__(cfg, src_dict, tgt_dict)
        self.entity_dict = entity_dict

    # @classmethod
    # def add_args(cls, parser):
    #     """Add task-specific arguments to the parser."""
    #     ##################### For Entity Contrastive #####################
    #     parser.add_argument('--use-entity-ct', action='store_true',
    #                         help='use entity contrastive loss')
    # #     parser.add_argument('--entity-dict', default="", type=str,
    # #                         help='entity dict file name, in data-bin directory')
    # #     parser.add_argument('--topk', default=-1,type=int,
    # #                         help='topk dict to use')


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

        ##################### For Entity Contrastive #####################
        entity_dict_file = os.path.join(paths[0],cfg.entity_dict)
        entity_dict = None
        if cfg.use_entity_ct:
            assert os.path.isfile(entity_dict_file), f"When use_entity_ct=True, entity_dict_file  {cfg.entity_dict} must exists."
        if os.path.isfile(entity_dict_file):
            entity_dict = cls.load_entity_dict(
                entity_dict_file,
                src_dict,
                tgt_dict,
                topk= cfg.topk
            ) if cfg.use_entity_ct else None

        return cls(cfg, src_dict, tgt_dict, entity_dict)


    @classmethod
    def load_entity_dict(cls, filename, src_dict=None, tgt_dict=None, topk=-1):
        """Load the entity dictionary from the filename

        Args:
            filename (str): the filename
            topk: the number of words in the dictionary.
        return: tree of ahocorasick
        """
        if topk==-1: topk=1e9
        assert os.path.exists(filename),f"entity dict path {filename} not exists."
        entity_dict = {}
        with open(filename,"r",encoding="utf-8") as f:
            # lines = [line.strip() for line in f.readlines()][:topk] # 截取topk
            lines = [line.strip() for line in f.readlines()] # 截取topk
            for line in lines[::-1]: # 从后往前取，若出现重复的key，取前面频率高的 #TODO: 以后再考虑一词多义
                src_entity, tgt_entity = line.split("\t")
                # 转为id
                src_entity_ids = src_dict.encode_line(src_entity.strip(),append_eos=False).tolist() # 不要加eos！
                tgt_entity_ids = tgt_dict.encode_line(tgt_entity.strip(),append_eos=False).tolist()
                src_entity_ids_str = " ".join([str(ids) for ids in src_entity_ids])
                tgt_entity_ids_str = " ".join([str(ids) for ids in tgt_entity_ids])
                # 加空格，防止部分匹配，如string="5883 2" pattern="883 2" -> string=" 5883 2 " pattern=" 883 2 "
                src_entity_ids_str = f" {src_entity_ids_str} "
                tgt_entity_ids_str = f" {tgt_entity_ids_str} "
                # forward、backward都要，方便双向训练
                entity_dict[src_entity_ids_str] = tgt_entity_ids_str
                entity_dict[tgt_entity_ids_str] = src_entity_ids_str
        return entity_dict

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

            ##################### For Entity Contrastive #####################
            seed=epoch,
            use_entity_ct=self.cfg.use_entity_ct,
            entity_dict=self.entity_dict,

        )

