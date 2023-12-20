import random
import torch
import numpy as np
import logging
from fairseq.data import LanguagePairDataset
logger = logging.getLogger(__name__)

# src_lang_id, tgt_lang_id
class LangPairDatasetWLID(LanguagePairDataset):
    def __init__(self,*args, src_lang_idx=None,tgt_lang_idx=None, **kwargs):
        super(LangPairDatasetWLID, self).__init__(*args, **kwargs)
        # assert (src_lang_id is not None) and (tgt_lang_id is not None) , "langid should not be None"
        self.src_lang_idx = src_lang_idx
        self.tgt_lang_idx = tgt_lang_idx

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

        ########### FOR NLLB ###########
        # src = 我 吃 苹果   <eos> zho_Hans
        # tgt = i eat apple  <eos>
        # prev = eng_Latn  i eat apple
        src_item = torch.cat([src_item, torch.LongTensor([self.src_lang_idx])])
        prev_item = torch.cat([torch.LongTensor([self.src_lang_idx]),tgt_item[:-1].clone()])
        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
            "prev_output_tokens":prev_item,
        }
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        return example




