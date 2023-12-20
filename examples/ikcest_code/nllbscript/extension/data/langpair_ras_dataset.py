import random
import torch
import numpy as np
import logging
from fairseq.data import data_utils,LanguagePairDataset
logger = logging.getLogger(__name__)


class RASLangPairDataset(LanguagePairDataset):
    def __init__(self,*args, use_ras=False,ras_dict=None,ras_prob=0.3,replace_prob=0.3,seed=1, **kwargs):
        super(RASLangPairDataset, self).__init__(*args, **kwargs)
        self.use_ras = use_ras
        self.ras_dict = ras_dict
        self.ras_prob = ras_prob
        self.replace_prob = replace_prob
        self.seed = seed
        if ras_dict is None: self.use_ras = False

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # if np.random.randn() >= 0.5:  # random reverse direct
        #     tmp_item = src_item
        #     src_item = tgt_item
        #     tgt_item = tmp_item

        # random aligned substitution(RAS)
        if self.use_ras and (np.random.randn()<=self.ras_prob): # 要加eos！！！
            sent = self.src_dict.string(src_item)
            with data_utils.numpy_seed(self.seed):
                new_sent = self.replace_sent(sent,replace_dic=self.ras_dict,replace_prob=self.replace_prob)
                src_item = torch.tensor([self.src_dict.index(word) for word in new_sent.split()] + [self.eos])
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
        }
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        return example

    def replace_sent(self, sent, replace_dic, replace_prob=0.3):
        new_tokens = []

        words = sent.split()
        slow, fast = 0, 0
        has_next = lambda x: x.endswith("@@")
        while fast < len(words):
            if has_next(words[fast]):  # not whole word
                fast = fast + 1
            else:  # whole word, take words[slow:fast+1]
                cur_word = words[slow:(fast + 1)]
                # token = "".join(cur_word).replace("@@","")
                token = " ".join(cur_word)
                if token in replace_dic and np.random.rand() < replace_prob:
                    new_token = random.choice(list(replace_dic[token]))
                    new_tokens.extend(new_token.split())
                else:
                    new_tokens.extend(cur_word)
                fast = fast + 1
                slow = fast
        new_sent = " ".join(new_tokens)
        return new_sent