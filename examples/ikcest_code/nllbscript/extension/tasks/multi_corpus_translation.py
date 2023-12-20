# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
eg: zh-en
    folder\
        train.corpus1.zh-en.zh.bin train.corpus1.zh-en.zh.idx   train.corpus2.zh-en.zh.bin train.corpus2.zh-en.zh.idx
config:
    --corpus-names  corpus1,corpus2   --corpus-weights 0.5,0.5 --subsample
    说明：corpus-names只针对训练集，如train.corpus1 ; weights一般是整数，用concat dataset，对数据做1倍或多倍上采样。
    若要均匀混合多个dataset，使用subsample，此时权重w是不同corpus的比例，而非上采样倍数

'''
from typing import Optional
import itertools
import json
import logging
import os
from fairseq.tasks.translation import TranslationTask,TranslationConfig
import numpy as np
from ..data import RASLangPairDataset
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
class MultiCorpusTranslationConfig(TranslationConfig):
    corpus_names: Optional[str] = field(
        default=None,
        metadata={"help": "names of all corpus, separated by commas. (eg: 'corpus1,corpus2')"}
    )

    corpus_weights: Optional[str] = field(
        default=None,
        metadata = {"help": "weights of all corpus, separated by commas. (each weight is int ,like '1,2')"}
    )

    subsample : Optional[bool] = field(
        default=None,
        metadata = {"help": "Whether to subsample corpus, support float 'corpus_weights'. "}
    )


# 将split_k=train变为 train.corpus1



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
    # new config
    corpus_names=None,
    corpus_weights=None,
    subsample = False,
    seed=1,

):

    if (corpus_names is not None) and split=="train":
        assert corpus_names!="","corpus_names !='' "
        combine = True
        corpus_names = [name for name in corpus_names.split(",") if name!=""]
        if corpus_weights is not None:
            corpus_weights = [float(w) for w in corpus_weights.split(",") if w!=""]
            assert len(corpus_weights) == len(corpus_names),"len(corpus_weights) == len(corpus_names)"

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        if (corpus_names is not None) and k>= len(corpus_names): break
        if (split!="train") or (corpus_names is None):
            split_k = split + (str(k) if k > 0 else "")
        else:
            split_k = f"{split}.{corpus_names[k]}"
        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k >= len(corpus_names):
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

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
        sample_ratios = [1] * len(src_datasets) if corpus_weights is None else corpus_weights

        if subsample:
            pair_datasets = []
            for src_dataset, tgt_dataset in zip(src_datasets,tgt_datasets):
                tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
                pair_dataset = LanguagePairDataset(
                                                src_dataset,
                                                src_dataset.sizes,
                                                src_dict,
                                                tgt_dataset,
                                                tgt_dataset_sizes,
                                                tgt_dict,
                                                left_pad_source=left_pad_source,
                                                left_pad_target=left_pad_target,
                                                align_dataset=None,
                                                eos=None,
                                                num_buckets=num_buckets,
                                                shuffle=shuffle,
                                                pad_to_multiple=pad_to_multiple,)
                pair_datasets.append(pair_dataset)

            from collections import OrderedDict
            pair_dataset_dict = OrderedDict()
            for name,pair_dataset in zip(corpus_names,pair_datasets):
                pair_dataset_dict[name] = pair_dataset
            sum_r = sum(sample_ratios)

            distribution = [r/sum_r for r in sample_ratios]
            pair_datasets = MultiCorpusDataset(datasets=pair_dataset_dict, distribution=distribution, seed=seed,
                                                  sort_indices=True, batch_sample=False)
            return pair_datasets

        # sample_ratios[0] = upsample_primary
        sample_ratios = [int(r) for r in sample_ratios]
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
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@register_task("multi_corpus_translation", dataclass=MultiCorpusTranslationConfig)
class MultiCorpusTranslationTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: MultiCorpusTranslationConfig

    def __init__(self, cfg: MultiCorpusTranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

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


        return cls(cfg, src_dict, tgt_dict)

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
            corpus_names = self.cfg.corpus_names,
            corpus_weights = self.cfg.corpus_weights,
            subsample=self.cfg.subsample,
            seed = epoch,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

