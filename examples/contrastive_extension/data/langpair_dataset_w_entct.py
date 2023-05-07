# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import torch
from fairseq.data import LanguagePairDataset
logger = logging.getLogger(__name__)

##################### For Entity Contrastive #####################
import ahocorasick
def build_ac(keys):
    # https://pyahocorasick.readthedocs.io/en/latest/
    automaton = ahocorasick.Automaton()
    for index, word in enumerate(keys):
        automaton.add_word(word, (index, word))
    automaton.make_automaton()
    return automaton

def build_prefix_starts(input_ids):
    # 用前缀和对前n个索引的长度和，记录第n个位置的索引  n+1
    prefix_starts = [0]
    for idx, ids in enumerate(input_ids):
        prefix_start = prefix_starts[idx]
        cur_start = prefix_start + 1 + len(str(int(ids)))  # 之前长度+空格+当前长度
        prefix_starts.append(cur_start)
    prefix_starts = prefix_starts[:-1]
    return prefix_starts


def find_entities_by_ac(input_ids, automaton = None):
    # entity_ids = []
    entity_starts = []
    entity_lens = []
    entity_keys = []
    ids_str = " ".join([str(int(ids)) for ids in input_ids])
    ids_str = f" {ids_str} " # 加空格，防止部分匹配，如string="5883 2" pattern="883 2" -> string=" 5883 2 " pattern=" 883 2 "
    prefix_starts = build_prefix_starts(input_ids)
    for item in automaton.iter_long(ids_str): # 匹配最长的字符串
        end, (ac_idx, key) = item
        start = end - len(key) + 1
        # real_start = start // 2   # woc，如果多位数，那就不是一隔1了
        # real_len = (len(key) + 1) // 2
        real_start = prefix_starts.index(start)
        real_len = len(key.split())
        entity_starts.append(real_start) # 只start
        # entity_ids.extend([real_start+l for l in range(real_len)]) # 所有的实体id
        entity_lens.append(real_len)
        entity_keys.append(key)
    return entity_starts, entity_lens, entity_keys

def find_entity_pairs(src_ids, tgt_ids, entity_dict, ac=None):
    if ac is None:
        ac = build_ac(entity_dict.keys())
    src_entity_ids, src_entity_lens = [],[]
    tgt_entity_ids, tgt_entity_lens = [],[]
    # 使用ac自动机找到原句中包含的实体信息
    src_ent_starts, src_ent_lens, src_ent_keys = find_entities_by_ac(src_ids, ac)
    # 根据entity_key找到tgt中是否匹配，匹配则重新记录
    tgt_ids_str = " ".join([str(int(ids)) for ids in tgt_ids])
    tgt_ids_str = f" {tgt_ids_str} " # 加空格，防止部分匹配，如string="5883 2" pattern="883 2" -> string=" 5883 2 " pattern=" 883 2 "
    # 记录 字符串位置：索引位置 的映射
    prefix_starts = build_prefix_starts(tgt_ids)
    # print("raw source entity num:", len(src_ent_lens))
    for src_ent_start, src_ent_len, src_ent_key in zip(src_ent_starts, src_ent_lens, src_ent_keys):
        tgt_ent_key = entity_dict[src_ent_key]
        tgt_ent_start = tgt_ids_str.find(tgt_ent_key)
        # tgt中找到对应的实体词
        if tgt_ent_start != -1:
            tgt_real_start = prefix_starts.index(tgt_ent_start)
            tgt_real_len = len(tgt_ent_key.split())
            # save
            src_entity_ids.extend([src_ent_start + l  for l in range(src_ent_len)]) # 所有的实体token的id
            src_entity_lens.append(src_ent_len)
            tgt_entity_ids.extend([tgt_real_start + l  for l in range(tgt_real_len)])
            tgt_entity_lens.append(tgt_real_len)

    # print("res entity pair num:", len(src_entity_lens))
    src_entity_ids, src_entity_lens = torch.tensor(src_entity_ids), torch.tensor(src_entity_lens)
    tgt_entity_ids, tgt_entity_lens = torch.tensor(tgt_entity_ids), torch.tensor(tgt_entity_lens)
    return src_entity_ids, src_entity_lens, tgt_entity_ids, tgt_entity_lens
##################### For Entity Contrastive #####################



# 重写后返回偏移信息，左侧加多少？
def collate_tokens(
    values,
    pad_idx,
    eos_idx=None, # 2
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values) # max_len，包含eos
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0: # pad_to_multiple倍数
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx) # [bsz,max_len]个1

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)
    shift_ls = []
    for i, v in enumerate(values):
        pad_len = size - len(v)
        shift_ls.append(pad_len)
        copy_tensor(v, res[i][pad_len :] if left_pad # 填充左边，则把原 tensor放到右边
                                                else res[i][: len(v)])
    # shifts = torch.tensor(shift_ls) if left_pad else torch.full([batch_size],fill_value=1)
    shifts = torch.tensor(shift_ls) if left_pad else torch.full([batch_size],fill_value=0) # fill_value=1是因为之前的prev加了bos，而tgt不需要
    return res,shifts  # <----


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
    ###### For Entity Contrastive ########
    split="train",
    use_entity_ct= False,

):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if (
            alignment[:, 0].max().item() >= src_len - 1
            or alignment[:, 1].max().item() >= tgt_len - 1
        ):
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True
        )
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1.0 / align_weights.float()

    id = torch.LongTensor([s["id"] for s in samples])
    # src_shifts记录源端的偏移量信息
    src_tokens,src_shifts = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    ##################### Source Entity Contrastive #####################
    src_entity_ids, src_entity_lens = None, None
    if split =="train" and use_entity_ct:
        src_shifts = src_shifts.index_select(0,sort_order)
        src_entity_ids, src_entity_lens = [], []
        src_size = src_tokens.shape[1] # 句长
        for new_idx , order in enumerate(sort_order):
            shift = src_shifts[new_idx]
            sample = samples[order]
            entity_ids = sample["src_entity_ids"]
            entity_lens = sample["src_entity_lens"]
            if (entity_ids is None) or (len(entity_ids) == 0): continue
            # 加上pad的偏移量，再加上行的偏移量
            entity_ids = entity_ids + shift + (new_idx * src_size)

            src_entity_ids.extend(entity_ids.numpy().tolist())
            src_entity_lens.extend(entity_lens.numpy().tolist())
        src_entity_ids = torch.tensor(src_entity_ids)
        # src_entity_lens = torch.tensor(src_entity_lens) # src_entity_lens要list

    ##################### Source Entity Contrastive #####################


    prev_output_tokens = None
    target = None
    tgt_entity_ids = None
    tgt_entity_lens = None
    if samples[0].get("target", None) is not None:
        target, tgt_shifts = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        ##################### Target Entity Contrastive #####################
        tgt_entity_ids, tgt_entity_lens = None, None
        if split == "train" and use_entity_ct:
            tgt_shifts = tgt_shifts.index_select(0, sort_order)
            tgt_entity_ids, tgt_entity_lens = [], []
            tgt_size = target.shape[1]  # 句长
            for new_idx, order in enumerate(sort_order):
                shift = tgt_shifts[new_idx]
                sample = samples[order]
                entity_ids = sample["tgt_entity_ids"]
                entity_lens = sample["tgt_entity_lens"]
                if (entity_ids is None) or (len(entity_ids) == 0): continue
                # 加上pad的偏移量，再加上行的偏移量
                entity_ids = entity_ids + shift + (new_idx * tgt_size)
                tgt_entity_ids.extend(entity_ids.numpy().tolist())
                tgt_entity_lens.extend(entity_lens.numpy().tolist())
            tgt_entity_ids = torch.tensor(tgt_entity_ids)
            # tgt_entity_lens = torch.tensor(tgt_entity_lens) # ?

        ##################### Target Entity Contrastive #####################

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens,prev_shifts = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens,prev_shifts = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
    }

    ##################### For Entity Contrastive #####################
    if split == "train" and use_entity_ct:
        # 防止报错：net_output = model(**sample["net_input"])
        batch["entity_info"] = {}
        batch["entity_info"]["src_entity_ids"] = src_entity_ids
        batch["entity_info"]["src_entity_lens"] = src_entity_lens
        batch["entity_info"]["tgt_entity_ids"] = tgt_entity_ids
        batch["entity_info"]["tgt_entity_lens"] = tgt_entity_lens

    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )

    if samples[0].get("alignment", None) is not None:
        bsz, tgt_sz = batch["target"].shape
        src_sz = batch["net_input"]["src_tokens"].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += torch.arange(len(sort_order), dtype=torch.long) * tgt_sz
        if left_pad_source:
            offsets[:, 0] += src_sz - src_lengths
        if left_pad_target:
            offsets[:, 1] += tgt_sz - tgt_lengths

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(
                sort_order, offsets, src_lengths, tgt_lengths
            )
            for alignment in [samples[align_idx]["alignment"].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch["alignments"] = alignments
            batch["align_weights"] = align_weights

    if samples[0].get("constraints", None) is not None:
        # Collate the packed constraints across the samples, padding to
        # the length of the longest sample.
        lens = [sample.get("constraints").size(0) for sample in samples]
        max_len = max(lens)
        constraints = torch.zeros((len(samples), max(lens))).long()
        for i, sample in enumerate(samples):
            constraints[i, 0 : lens[i]] = samples[i].get("constraints")
        batch["constraints"] = constraints.index_select(0, sort_order)

    return batch


class LanguagePairDatasetWithEntityCT(LanguagePairDataset):
    def __init__(self,*args,
                 use_entity_ct=False,
                 entity_dict=None,
                 split = "train",
                 seed=1,
                 **kwargs):
        super(LanguagePairDatasetWithEntityCT, self).__init__(*args, **kwargs)
        self.use_entity_ct = use_entity_ct
        self.entity_dict =  entity_dict
        self.split = split # 如果是train时才需要记录实体信息
        self.seed = seed
        if entity_dict is None: self.use_entity_ct = False
        assert split in ["train","valid","test"]
        # 构造源端的AC自动机
        if self.use_entity_ct:
            self.ac = build_ac(entity_dict.keys())
        else:
            self.ac = None

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        ##################### For Entity Contrastive #####################
        src_entity_ids, src_entity_lens = None, None
        tgt_entity_ids, tgt_entity_lens = None, None
        if self.split == "train" and self.use_entity_ct:
            src_entity_ids, src_entity_lens, tgt_entity_ids, tgt_entity_lens = find_entity_pairs(src_item, tgt_item,
                                                                                             self.entity_dict, self.ac)


        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
            ##################### For Entity Contrastive #####################
            "src_entity_ids": src_entity_ids,
            "src_entity_lens": src_entity_lens,
            "tgt_entity_ids": tgt_entity_ids,
            "tgt_entity_lens": tgt_entity_lens,

        }
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
            split=self.split,
            use_entity_ct = self.use_entity_ct,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        return res



    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

