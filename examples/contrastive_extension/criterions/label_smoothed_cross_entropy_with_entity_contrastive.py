import math

from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq import metrics, utils
import torch
import torch.nn as nn


@register_criterion("label_smoothed_ce_with_entity_contrastive")
class LabelSmoothedCrossEntropyCriterionWithEntityContrastive(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size=0, report_accuracy=False,
                 contrastive_lambda=0.0,
                 temperature=1.0):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.contrastive_lambda = contrastive_lambda
        self.temperature = temperature

    @staticmethod
    def add_args(parser):
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument("--contrastive-lambda", type=float,
                            default=1.0,
                            help="The contrastive loss weight")
        parser.add_argument("--temperature", type=float,
                            default=0.1,
                            help= "Higher temperature increases the difficulty to distinguish positive sample from negative ones.")



    def get_entity_contrastive_loss(self, encoder_out1, encoder_out2, sample1, sample2):
        def _get_entity_embed(encoder_out, sample, entity_ids, entity_lens):
            encoder_output = encoder_out[0].transpose(0, 1) # [T B C] -> [B T C]

            # make pad zero # 应该不影响，我都取实体了
            src_tokens = sample["net_input"]["src_tokens"]
            mask = (src_tokens != self.padding_idx)
            encoder_output = encoder_output * mask.unsqueeze(-1)

            bsz, seq_len, dim = encoder_output.shape
            entity_embeds = encoder_output.reshape(bsz * seq_len, dim).index_select(0, entity_ids)
            splits = torch.split(entity_embeds, entity_lens)
            means = [split.mean(dim=0) for split in splits]  # <--for循环会不会慢？
            entity_embedding = torch.stack(means)
            return entity_embedding

        entity_info = sample1["entity_info"]
        anchor_feature = _get_entity_embed(encoder_out1, sample1,
                                           entity_info["src_entity_ids"], entity_info["src_entity_lens"])  # [num_entities, hidden_size]
        contrast_feature = _get_entity_embed(encoder_out2, sample2,
                                             entity_info["tgt_entity_ids"], entity_info["tgt_entity_lens"])  # [num_entities, hidden_size]

        entity_num = anchor_feature.shape[0]
        feature_dim = contrast_feature.shape[1]

        similarity_function = nn.CosineSimilarity(dim=-1)
        anchor_dot_contrast = similarity_function(anchor_feature.expand((entity_num, entity_num, feature_dim)),
                                                  torch.transpose(
                                                      contrast_feature.expand((entity_num, entity_num, feature_dim)), 0,
                                                      1))

        loss = -nn.LogSoftmax(0)(torch.div(anchor_dot_contrast, self.temperature)).diag().sum()

        return loss, entity_num


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
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        encoder_out = model.encoder.forward(sample["net_input"]["src_tokens"],
                                            sample["net_input"]["src_lengths"])["encoder_out"] # error1: dict no attribute encoder_out
        reverse_sample = self.swap_sample(sample)
        reversed_encoder_out = model.encoder.forward(reverse_sample["net_input"]["src_tokens"],
                                                     reverse_sample["net_input"]["src_lengths"])["encoder_out"] # error1: dict no attribute encoder_out

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        nsentences = sample["target"].size(0)
        ntokens = sample["ntokens"]
        contrastive_loss = -1
        entity_num = 1
        if sample.get("entity_info", None) is not None:
            contrastive_loss, entity_num = self.get_entity_contrastive_loss(
                encoder_out,
                reversed_encoder_out,
                sample,
                reverse_sample,
            )
            # Since Lctr_entity is calculated on the entity-lecel and Lce is calculated on the token-level,
            # therefore Lctr_entity should be multiplied by the number of entity.
            all_loss= loss + contrastive_loss * self.contrastive_lambda * ntokens / entity_num
        else:
            all_loss = loss
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "nentities": entity_num,
            "sample_size": sample_size,
        }
        if isinstance(contrastive_loss, int):
            logging_output["entity_ct_loss"] = 0
        else:
            logging_output["entity_ct_loss"] = utils.item(contrastive_loss.data)

        return all_loss, sample_size, logging_output

    def similarity_function(self, ):
        return nn.CosineSimilarity(dim=-1)

    def get_contrastive_loss(self, encoder_out1, encoder_out2, sample1, sample2):

        def _sentence_embedding(encoder_out, sample):
            encoder_output = encoder_out[0].transpose(0, 1) # error2: list can't transpose
            src_tokens = sample["net_input"]["src_tokens"]
            mask = (src_tokens != self.padding_idx)
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
        # nsentences = utils.item(
        #     sum(log.get("nsentences", 1) for log in logging_outputs)
        # )
        nentities = utils.item(
            sum(log.get("nentities", 1) for log in logging_outputs)
        )
        entity_ct_loss = utils.item(
            sum(log.get("entity_ct_loss", 0) for log in logging_outputs)
        )
        metrics.log_scalar(
            "entity_ct_loss",
            entity_ct_loss / nentities / math.log(2),
            nentities,
            round=3,
        )