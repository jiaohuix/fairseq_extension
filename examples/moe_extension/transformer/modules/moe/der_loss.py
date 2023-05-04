import torch
import torch.nn.functional as F
import numpy as np
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss

# for循环在cuda会报错
def make_mode_label(entity_val: torch.Tensor, entity_idx: torch.Tensor, length: list):
    '''
    inputs:
        entity_val = [0.4,0.5,0.55, 0.6,0.5,0.55]
        entity_idx = [1,2,3, 1,1,2]
        length=[3,3]
    outputs:
        [3,3,3, 1,1,1]
    '''
    # assert entity_val.device == entity_idx.device == torch.device("cpu"), "device error."
    mode_label = []
    max_label = []
    label_mask = [] # true 表示mode

    val_sub_seqs = torch.split(entity_val, length)
    idx_sub_seqs = torch.split(entity_idx, length)
    for val_sub_seq,idx_sub_seq,lens in zip(val_sub_seqs,idx_sub_seqs,length):
        # 可以用torch.where来实现if-else逻辑
        modeVal = torch.mode(idx_sub_seq).values
        maxVal = idx_sub_seq[torch.argmax(val_sub_seq)]
        is_mode = len(torch.unique(idx_sub_seq)) != len(idx_sub_seq) # 有重复，取mode
        mode_label.append(torch.full(size=(lens,), fill_value=modeVal) )
        max_label.append(torch.full(size=(lens,), fill_value= maxVal) )
        label_mask.append(torch.full(size=(lens,), fill_value=is_mode) )

    mode_label = torch.cat(mode_label)
    max_label = torch.cat(max_label)
    label_mask = torch.cat(label_mask)

    expert_label = torch.where(label_mask,mode_label,max_label)
    return expert_label

def make_fake_entity(token_num,entity_num=20, max_span=5 ): # 必须cpu上运行，否则会不同步
    ''' 生成若干随机实体的索引和长度 '''
    fake_starts = np.random.randint(0, token_num, entity_num)
    fake_starts = list(sorted(list(set(fake_starts))))
    fake_starts.append(token_num - 1)

    entity_index = []
    entity_length = []
    for idx in range(len(fake_starts) - 1):
        start, next = fake_starts[idx], fake_starts[idx + 1]
        span_len = np.random.randint(0, max_span + 1)
        end = idx + span_len - 1
        if (end < next) and (span_len > 0) and (start+max_span < token_num):
            entity_length.append(span_len)
            for i in range(span_len):
                entity_index.append(start + i)

    return entity_index, entity_length



def domain_entity_routing_loss(logits, gate_score=None,entity_index: torch.Tensor = None, entity_length= [], tau = 1, epsilon=0.1):
    ''' 额外加些tau啊，labelsmooth的
        遇到的问题： 1.make_mode_label必须在cpu上运行，for循环会导致异步操作失败 2.topk在cpu上不支持半精度
    '''
    raw_device = logits.device
    # logits: [num_tokens, num_experts]
    # tau = 2 # 更大的tau可以更突出最大的概率，tau越小越平滑
    assert len(logits.shape)==2
    # entity_index = torch.tensor(entity_index,device=logits.device)

    if gate_score is None:
        gate_score = F.softmax(logits, dim=-1)

    topk_val, topk_idx = torch.topk(
        gate_score, k=2, dim=-1, largest=True, sorted=False
    ) # RuntimeError: "topk_cpu" not implemented for 'Half'

    # dynamically get expert label for each entity
    top1_val = topk_val[:, 0]
    top1_idx = topk_idx[:, 0]

    # entity_val = torch.index_select(top1_val,dim=0,index=entity_index).to("cpu")
    # entity_idx = torch.index_select(top1_idx,dim=0,index=entity_index).to("cpu")
    # entity_label = make_mode_label(entity_val=entity_val, entity_idx=entity_idx, length=entity_length).to(logits.device)

    # gpu code
    # torch.cuda.synchronize()
    entity_val = torch.index_select(top1_val,dim=0,index=entity_index)
    entity_idx = torch.index_select(top1_idx,dim=0,index=entity_index)
    entity_label = make_mode_label(entity_val=entity_val, entity_idx=entity_idx, length=entity_length).to(logits.device)
    # entity_label = torch.randint(0,logits.shape[1],size=[len(entity_index)],device=logits.device)
    # torch.cuda.synchronize()

    # calculate domain entity routing loss
    entity_logits = torch.index_select(logits,dim=0,index=entity_index)
    entity_lprobs = F.log_softmax(entity_logits/tau,dim=-1)
    loss, nll_loss = label_smoothed_nll_loss(lprobs=entity_lprobs, target=entity_label,
                                             epsilon=epsilon, reduce=False)
    der_loss =  sum(loss)/len(loss)
    return der_loss.to(raw_device)

if __name__ == '__main__':
    torch.manual_seed(2022)
    for idx in range(20):
        # params
        num_experts = 4
        bsz, seq, dim = 100, 13, 256
        num_tokens = bsz * seq

        # make fake input
        entity_index, entity_length = make_fake_entity(token_num= num_tokens,entity_num=20,max_span=5)
        logits  = torch.randn(num_tokens,num_experts)

        # get der loss
        der_loss = domain_entity_routing_loss(logits=logits,
                                              gate_score=None,
                                              entity_index=entity_index,
                                              entity_length=entity_length,
                                              tau=1,
                                              epsilon=0.1)

        print(der_loss)