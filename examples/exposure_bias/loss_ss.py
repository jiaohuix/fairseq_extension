'''
schedule sampling with decoding steps
https://github.com/Adaxry/ss_on_decoding_steps.
--user-dir bert_nmt  --criterion ce_w_ss --ssdecode-lambda 0.5  --sampling-strategy exponential --exp-radix 0.99
调度采样损失：
难点：
1.要获取解码器的embedding
2.生成采样的embedding后需要输入给解码器
3.采样的概率调参

'''
import math

from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq import metrics, utils
import torch
import torch.nn as nn


def sampling_function(
        first_decoder_inputs,  # [sos, input_ids]
        first_decoder_outputs,  # [input_ids, eos]
        max_seq_len,
        tgt_lengths,
        sampling_strategy="exponential",  # 最高
        epsilon=0.2,
        exp_radix=0.99,
        sigmoid_k=20,

):
    '''
    conduct scheduled sampling based on the index of decoded tokens
    param first_decoder_outputs: [batch_size, seq_len, hidden_size], model prediections
    param first_decoder_inputs: [batch_size, seq_len, hidden_size], ground-truth target tokens
    param max_seq_len: scalar, the max lengh of target sequence
    param tgt_lengths: [batch_size], the lenghs of target sequences in a mini-batch
    '''
    batch_size, seq_len, _ = first_decoder_outputs.shape
    # 将outputs往后挪:  [input_ids, <eos>] -> [<sos>, input_ids]
    first_decoder_outputs = torch.cat((first_decoder_inputs[:, :1, :],  # <sos>
                                       first_decoder_outputs[:, :-1, :]),  # input_ids - <eos>
                                      dim=1)

    # indexs of decoding steps
    t = torch.arange(0, max_seq_len)  # [start,end)

    # differenct sampling strategy based on decoding steps
    assert sampling_strategy in ["exponential", "sigmoid", "linear"], "Unknown sampling_strategy %s" % sampling_strategy
    if sampling_strategy == "exponential":
        threshold_table = exp_radix ** t
    elif sampling_strategy == "sigmoid":
        threshold_table = sigmoid_k / (sigmoid_k + torch.exp(t / sigmoid_k))
    elif sampling_strategy == "linear":
        threshold_table = torch.max(torch.tensor(epsilon), 1 - t / max_seq_len)

    # convert threshold_table[max_seq_len,max_seq_len] to thresholds [batch_size, seq_len]
    threshold_table = threshold_table.unsqueeze_(0).repeat(max_seq_len, 1).tril()  # 下三角
    thresholds = threshold_table[tgt_lengths].view(-1, max_seq_len)
    thresholds = thresholds[:, :seq_len]  # [bsz, seq_len]  # 截断到当前batch的最大序列长度，并且pad部分也不算

    # conduct sampling based on the above thresholds
    random_select_seed = torch.rand([batch_size, seq_len, 1])  # [bsz,seq_len,1]
    thresholds = thresholds.view([*thresholds.shape, 1])
    second_decoder_inputs = torch.where(random_select_seed < thresholds,
                                        first_decoder_inputs,
                                        first_decoder_outputs)

    return second_decoder_inputs


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("ce_w_ss")  # cross entropy with schedule sampling
class LabelSmoothedCrossEntropyCriterionWithScheduleSampling(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size=0, report_accuracy=False,
                 ssdecode_lambda=0.5,
                 sampling_strategy="exponential",
                 epsilon=0.2,
                 exp_radix=0.99,
                 sigmoid_k=20
                 ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.ssdecode_lambda = ssdecode_lambda
        self.sampling_strategy = sampling_strategy
        self.epsilon = epsilon
        self.exp_radix = exp_radix
        self.sigmoid_k = sigmoid_k

    @staticmethod
    def add_args(parser):
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument("--ssdecode-lambda", type=float,
                            default=0.5,
                            help="The ssdecode loss weight")
        parser.add_argument("--sampling-strategy", type=str,
                            default="exponential",
                            help="The ssdecode sampling_strategy")
        parser.add_argument("--exp-radix", type=float,
                            default=0.99,
                            help="exponential radix.")
        parser.add_argument("--exp-radix", type=float,
                            default=0.99,
                            help="exponential radix.")
        parser.add_argument("--sigmoid-k", type=float,
                            default=20,
                            help="sigmoid_k.")

        parser.add_argument("--epsilon", type=float,
                            default=0.2,
                            help="linear sampling")

    def swap_sample(self, sample):
        target = sample["target"]
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        src_tokens = torch.cat((prev_output_tokens[:, :1], sample["net_input"]['src_tokens']), dim=-1)
        return {
            "net_input": {
                "src_tokens": target.contiguous(),
                "src_lengths": (target != self.padding_idx).int().sum(dim=1),
                "prev_output_tokens": src_tokens[:, :-1].contiguous()
            },
            'nsentences': sample['nsentences'],
            'ntokens': utils.item((src_tokens[:, 1:] != self.padding_idx).int().sum().data),
            "target": src_tokens[:, 1:].contiguous(),
            "id": sample["id"],
        }

    def forward(self, model, sample, reduce=True):
        '''
        步骤：
            1.前向encoder
            2.前向decoder，并计算损失1
            3.计算调度采样的输入
            4.前向decoder 2次，并计算损失2
            5.合并loss
            6.统计指标
        '''
        net_input = sample["net_input"]
        src_tokens = net_input["src_tokens"]
        src_lengths = net_input["src_lengths"]
        prev_output_tokens = net_input["prev_output_tokens"]
        del net_input["src_tokens"], net_input["src_lengths"], net_input["prev_output_tokens"]

        # net_output = model(**sample["net_input"])
        # loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        # 1 前向并计算损失1
        # 不需要编码器的embedding
        # x, embed = model.encoder.forward_embedding(src_tokens, token_embedding=None, **net_input)
        # token_embeddings = embed * model.encoder.embed_scale

        encoder_out = model.encoder.forward(src_tokens, src_lengths, token_embeddings=token_embeddings, **net_input)[
            "encoder_out"]  # error1: dict no attribute encoder_out
        decoder_out = model.decoder.forward(prev_output_tokens, encoder_out=encoder_out, **net_input)
        loss, nll_loss = self.compute_loss(model, decoder_out, sample, reduce=reduce)

        # 2 调度采样
        sampling_function(first_decoder_inputs=encoder_out, # 解码器的输入embed
                          first_decoder_outputs=decoder_out, # 解码器的输出
                          max_seq_len=model.max_positions())
        # 前向2次

        reverse_sample = self.swap_sample(sample)
        reversed_encoder_out = model.encoder.forward(reverse_sample["net_input"]["src_tokens"],
                                                     reverse_sample["net_input"]["src_lengths"])[
            "encoder_out"]  # error1: dict no attribute encoder_out
        contrastive_loss = self.get_contrastive_loss(
            encoder_out,
            reversed_encoder_out,
            sample,
            reverse_sample,
        )
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        nsentences = sample["target"].size(0)
        ntokens = sample["ntokens"]
        all_loss = loss + contrastive_loss * self.contrastive_lambda * ntokens / nsentences
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        if isinstance(contrastive_loss, int):
            logging_output["contrastive_loss"] = 0
        else:
            logging_output["contrastive_loss"] = utils.item(contrastive_loss.data)

        return all_loss, sample_size, logging_output

    def similarity_function(self, ):
        return nn.CosineSimilarity(dim=-1)

    def get_contrastive_loss(self, encoder_out1, encoder_out2, sample1, sample2):
        def _sentence_embedding(encoder_out, sample):
            encoder_output = encoder_out[0].transpose(0, 1)  # error2: list can't transpose
            src_tokens = sample["net_input"]["src_tokens"]
            mask = (src_tokens != self.padding_idx)
            # float * bool = false部分抹0
            # sum(dim=1)： seq_len求和， [B T C] -> [B C]
            # mask.float().sum(dim=1).unsqueeze(-1)： [B 1] 求序列个数，对非pad的序列求平均
            encoder_embedding = (encoder_output * mask.unsqueeze(-1)).sum(dim=1) / mask.float().sum(dim=1).unsqueeze(
                -1)  # [batch, hidden_size]
            return encoder_embedding

        encoder_embedding1 = _sentence_embedding(encoder_out1, sample1)  # [batch, hidden_size]
        encoder_embedding2 = _sentence_embedding(encoder_out2, sample2)  # [batch, hidden_size]

        batch_size = encoder_embedding2.shape[0]
        feature_dim = encoder_embedding2.shape[1]
        anchor_feature = encoder_embedding1
        contrast_feature = encoder_embedding2

        similarity_function = self.similarity_function()
        anchor_dot_contrast = similarity_function(anchor_feature.expand((batch_size, batch_size, feature_dim)),
                                                  torch.transpose(
                                                      contrast_feature.expand((batch_size, batch_size, feature_dim)), 0,
                                                      1))

        loss = -nn.LogSoftmax(0)(torch.div(anchor_dot_contrast, self.temperature)).diag().sum()

        return loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        contrastive_loss = utils.item(
            sum(log.get("contrastive_loss", 0) for log in logging_outputs)
        )
        metrics.log_scalar(
            "contrastive_loss",
            contrastive_loss / nsentences / math.log(2),
            nsentences,
            round=3,
        )
