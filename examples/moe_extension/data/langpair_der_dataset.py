import random
import torch
import numpy as np
import logging
from fairseq.data import data_utils,LanguagePairDataset
import ahocorasick

logger = logging.getLogger(__name__)

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
    shift = torch.tensor(shift_ls) if left_pad else torch.full([batch_size],fill_value=1)
    return res,shift

def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
    split="train",
):
    if len(samples) == 0:
        return {}

    # 组batch，默认eos在句子结尾，move_eos_to_beginning挪到开头
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
    src_tokens,src_shift = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if (pad_to_length is not None) else None,
    )

    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    #### src der position info ####
    src_shift = src_shift.index_select(0,sort_order)
    src_entity_ids, src_entity_len = [], []
    src_size = src_tokens.shape[1]
    for new_idx , order in enumerate(sort_order):
        shift = src_shift[new_idx]
        sample = samples[order]
        entity_ids = sample["src_entity_ids"]
        entity_len = sample["src_entity_len"]
        if (entity_ids is None) or (len(entity_ids) == 0): continue
        # 加上pad的偏移量，再加上行的偏移量
        entity_ids = entity_ids + shift + (new_idx * src_size)

        src_entity_ids.extend(entity_ids.numpy().tolist())
        src_entity_len.extend(entity_len.numpy().tolist())
    src_entity_ids = torch.tensor(src_entity_ids)
    # src_entity_len = torch.tensor(src_entity_len)
    #### src der position info ####

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target,_ = merge(
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

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens, prev_shift  = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding: # <-
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens, prev_shift = merge(
                "target",
                left_pad=left_pad_target, # false
                move_eos_to_beginning=True, #prev的eos挪到开头
                pad_to_length=pad_to_length["target"] # none
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    #### tgt der position info ####
    prev_shift = prev_shift.index_select(0, sort_order)
    prev_entity_ids, prev_entity_len = [], []
    prev_size = prev_output_tokens.shape[1]
    for new_idx , order in enumerate(sort_order):
        shift = prev_shift[new_idx]
        sample = samples[order]
        entity_ids = sample["tgt_entity_ids"]
        entity_len = sample["tgt_entity_len"]
        if (entity_ids is None) or (len(entity_ids) == 0): continue

        # 加上pad的偏移量，再加上行的偏移量
        entity_ids = entity_ids + shift + (new_idx * prev_size)

        prev_entity_ids.extend(entity_ids.numpy().tolist())
        prev_entity_len.extend(entity_len.numpy().tolist())
    prev_entity_ids = torch.tensor(prev_entity_ids)
    # prev_entity_len = torch.tensor(prev_entity_len)
    #### tgt der position info ####

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            # "src_entity_ids": src_entity_ids,
            # "src_entity_len": src_entity_len,
            # "prev_entity_ids": prev_entity_ids,
            # "prev_entity_len": prev_entity_len,
        },
        "target": target,
    }
    if split == "train":
        #### der loss position info ####
        batch["net_input"]["src_entity_ids"] = src_entity_ids
        batch["net_input"]["src_entity_len"] = src_entity_len
        batch["net_input"]["prev_entity_ids"] = prev_entity_ids
        batch["net_input"]["prev_entity_len"] = prev_entity_len


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

class derLangPairDataset(LanguagePairDataset):
    def __init__(self,*args,
                 use_der=False,
                 entity_tree=None,
                 split = "train",
                 seed=1,
                 **kwargs):
        super(derLangPairDataset, self).__init__(*args, **kwargs)
        self.use_der = use_der
        self.entity_tree =  entity_tree
        self.split = split # 如果是train，随机增加约束，如果是valid和test，有约束全部加上
        self.seed = seed
        if entity_tree is None: self.use_der = False
        assert split in ["train","valid","test"]

    def __getitem__(self, index):
        # tgt和src在结尾都会加eos=2，此处不用修改偏移
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        src_entity_ids, src_entity_len = None, None
        tgt_entity_ids, tgt_entity_len = None, None


        # add domain entity routing to both src and tgt (training)
        if self.use_der and (self.split=="train"): # 要加eos！！！  修改！（可以试试句子在有限制词的条件下在依概率决定要不要加cst）
            src_sent = self.src_dict.string(src_item)
            tgt_sent = self.tgt_dict.string(tgt_item) if (tgt_item is not None) else ""
            with data_utils.numpy_seed(self.seed):
                # 输入sentence， 暂时不变，不做修改，暂时只返回实体词的位置信息
                src_entity_ids, src_entity_len = self.find_entities(src_sent)
                tgt_entity_ids, tgt_entity_len = self.find_entities(tgt_sent)


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

        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
            # der
            "src_entity_ids": src_entity_ids,
            "src_entity_len": src_entity_len,
            "tgt_entity_ids": tgt_entity_ids,
            "tgt_entity_len": tgt_entity_len,
        }


        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        return example


    def find_entities(self,sentence):
        res_ls = self.entity_tree.search(sentence.lower(), True)
        entity_ids = []
        entity_len = []
        for entity, (char_start, char_end) in res_ls:
            # TODO: 此处加个概率，限制实体词使用der损失的概率，如<0.3才加入
            length = len(entity.split())
            prefix = sentence[: char_start]
            prefix_subword_num = len(prefix.strip().split())
            start = prefix_subword_num
            end = start + length
            entity_ids.extend([idx for idx in range(start,end)])
            entity_len.append(end - start)

        entity_ids,entity_len  = torch.tensor(entity_ids), torch.tensor(entity_len)
        return entity_ids,entity_len


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
            split= self.split
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