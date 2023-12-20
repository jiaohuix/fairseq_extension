# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
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
class RASTranslationConfig(TranslationConfig):
    use_bidirect: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use bidirect training."}
    )

    use_ras: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use random aligned substitution(RAS)."}
    )
    ras_dict: Optional[str] = field(
        default=None,
        metadata={"help": "RAS dict path. (src_word\ttgt_word),word already used bpe "}
    )
    ras_prob: Optional[float] = field(
        default=0.3,
        metadata={"help": "The probability of a sample using RAS."}
    )

    replace_prob: Optional[float] = field(
        default=0.3,
        metadata={"help": "The probability of a word being replaced."}
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
    use_bidirect=False,
    use_ras=False,
    ras_dict=None,
    ras_prob=0.3,
    replace_prob=0.3,
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

    # random aligned substitution
    forward_dataset  =  RASLangPairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        ### RAS CONFIG ###
        use_ras=use_ras,
        ras_dict=ras_dict,
        ras_prob=ras_prob,
        replace_prob=replace_prob,
        seed=seed,

        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )
    if not use_bidirect:
        return forward_dataset

    backward_dataset  =  RASLangPairDataset(
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        src_dataset,
        src_dataset.sizes,
        src_dict,
        ### RAS CONFIG ###
        use_ras=use_ras,
        ras_dict=ras_dict,
        ras_prob=ras_prob,
        replace_prob=replace_prob,
        seed=seed,

        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )
    # multi corpus based bidirect training

    # sort_indices: sort the ordered indices by size
    from collections import OrderedDict
    bidirect_dict = OrderedDict()
    bidirect_dict["forward"] = forward_dataset
    bidirect_dict["backward"] = backward_dataset
    bidirect_dataset = MultiCorpusDataset(datasets=bidirect_dict,distribution=[0.5,0.5],seed=seed,sort_indices=True,batch_sample=False)
    return bidirect_dataset


@register_task("ras_translation", dataclass=RASTranslationConfig)
class RASTranslationTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: RASTranslationConfig

    def __init__(self, cfg: RASTranslationConfig, src_dict, tgt_dict, ras_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.ras_dict = ras_dict

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

        # load ras dictionary # 若使用ras，ras_dict非none且存在
        ras_dict = cls.load_ras_dictionary(
            os.path.join(paths[0],cfg.ras_dict)
        ) if cfg.use_ras else None

        return cls(cfg, src_dict, tgt_dict, ras_dict)


    @classmethod
    def load_ras_dictionary(cls, filename):
        """Load the ras dictionary from the filename

        Args:
            filename (str): the filename
        return: ras_dict ={"key":set(words)}
        """
        def update_dict(ras_dict, src_word, tgt_word):
            if src_word not in ras_dict:
                ras_dict[src_word] = set([tgt_word])
            else:
                cur_set = ras_dict[src_word]
                cur_set.add(tgt_word)
                ras_dict[src_word] = cur_set
            return ras_dict

        assert os.path.exists(filename),f"ras dict path {filename} not exists."
        ras_dict = {}
        with open(filename,"r",encoding="utf-8") as f:
            for line in f.readlines():
                line  = line.strip().split("\t")
                assert len(line)==2,"Each line in ras dict is separated by \t"
                src_word, tgt_word = line[0].strip(),line[1].strip()
                update_dict(ras_dict,src_word,tgt_word)
                update_dict(ras_dict,tgt_word,src_word)

        return ras_dict


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
            ### RAS CONFIG ###
            seed=epoch,
            use_bidirect=self.cfg.use_bidirect,
            use_ras=self.cfg.use_ras,
            ras_dict=self.ras_dict,
            ras_prob=self.cfg.ras_prob,
            replace_prob=self.cfg.replace_prob
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

