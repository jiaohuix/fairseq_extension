'''
ParaFuse:  Fuse parallel sentences into bert for enhancing low-resource NTM
    将 src1 [SEP] src2 [SEP] src3 [SEP]丢入bert，
    取src1 [SEP] src2 [SEP]作为encoder混合信息,
    取 src1 [SEP] src3 [SEP]作为decocder混合信息。

    改进1：对bert做mlm，甚至知识增强
    改进2：nmt模型随机拼接长句训练
    改进3：融入词典信息，做mrasp
    改进4：参照llama，修改act、pos、norm
实际使用： 修改key_pad_mask
'''
import torch
import numpy as np
def same_seeds(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def rand_tokens(seq_len, vocab_size=100):
    return torch.randint(0,vocab_size, [seq_len])

def batch_tokens(bsz=4, seq_len=15):
    inps = []
    for _ in range(bsz):
        # random seq len
        end1 = np.random.randint(1, seq_len//2)
        end2 = end1 + np.random.randint(1, seq_len - end1)
        src1 = rand_tokens(seq_len=end1+1)
        src2 = rand_tokens(seq_len=end2-end1)
        tgt2 = rand_tokens(seq_len=seq_len-end2)
        SEP = torch.tensor([sep_id])
        # concat
        X = torch.cat([src1, SEP, src2, SEP, tgt2, SEP])
        inps.append(X.unsqueeze(0))
    inps = torch.cat(inps, dim=0)
    return inps

def bert_forward(input_tokens , dim=3):
    bsz,seq_len = input_tokens.size()
    bert_out = torch.randn(bsz,seq_len, dim)
    return bert_out

def get_token_type_ids(input_tokens, sep_id=103):
    token_type_ids = (input_tokens == sep_id).int()
    token_type_ids = torch.cumsum(token_type_ids, dim=-1).unsqueeze(-1) # ≈token type ids
    return token_type_ids

def mask_bert_out(bert_out , token_type_ids, pad_val=0.):
    # src mask: src1 + src2
    bert_enc_mask = token_type_ids == 2
    # tgt mask: src1 + tgt2
    bert_dec_mask = token_type_ids == 1
    bert_enc = bert_out.masked_fill(bert_enc_mask, pad_val)
    bert_dec = bert_out.masked_fill(bert_dec_mask, pad_val)
    return bert_enc, bert_dec

def parafuse_mask(input_ids, sep_id=103):
    # type id
    token_type_ids = (input_ids == sep_id).int()
    token_type_ids = torch.cumsum(token_type_ids, dim=-1) # ≈token type ids
    # mask for enc-dec
    bert_enc_mask = token_type_ids == 2
    bert_dec_mask = token_type_ids == 1
    return bert_enc_mask, bert_dec_mask

if __name__ == '__main__':
    # 1.params
    pad_id = 1
    pad_val = 0
    sep_id = 103
    bsz=2
    seq_len=10
    dim = 3
    same_seeds(1)
    # 2.input
    inps = batch_tokens(bsz=bsz, seq_len=seq_len)
    token_type_ids = get_token_type_ids(inps , sep_id=sep_id)
    # 3.forward bert
    bert_out = bert_forward(inps)
    print("inps:\n",inps)
    # 4.mask bert out for encoder and decoder
    bert_enc, bert_dec = mask_bert_out(bert_out,token_type_ids, pad_val=pad_val)
    print("bert_enc:\n",bert_enc)
    print("bert_dec:\n",bert_dec)