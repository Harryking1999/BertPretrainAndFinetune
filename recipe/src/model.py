from typing import List, Optional, Tuple, Union

import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers  import PreTrainedModel, BertModel, BertPreTrainedModel, XLMRobertaPreTrainedModel, XLMRobertaModel, AutoModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
)

from .loss import MultipleNegativesRankingLoss, kl_div

class BertForRetrieve(PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.new = AutoModel.from_config(config, trust_remote_code=True)

        self.post_init()
    def post_init(self):
        pass

    def forward(
        self,
        input_ids: Optional[th.Tensor] = None,
        attention_mask: Optional[th.Tensor] = None,
        token_type_ids: Optional[th.Tensor] = None,
        position_mask: Optional[th.Tensor] = None,
        head_mask: Optional[th.Tensor] = None,
        inputs_embeds: Optional[th.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.new(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_mask = position_mask,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
        )
        dimension = 768
        embeddings = outputs.last_hidden_state[:, 0][:dimension]
        embeddings = F.normalize(embeddings, p = 2, dim = 1)

        return embeddings
    
class BertForRerankPointwise(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.rdrop = config.rdrop
        self.pointwise_weight = config.pointwise_weight
        self.pointwise_threshold = config.pointwise_threshold
        self.contrast_temperature = config.contrast_temperature

        self.new = AutoModel.from_config(config, trust_remote_code=True)
        classifier_dropoout = (
            config.classifier_dropout
            if config.classifier_dropoout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropoout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std = self.config.initializer_range)
            if module.bias is not None:
                module.weight.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std = self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: Optional[th.Tensor] = None,
        attention_mask: Optional[th.Tensor] = None,
        token_type_ids: Optional[th.Tensor] = None,
        position_mask: Optional[th.Tensor] = None,
        head_mask: Optional[th.Tensor] = None,
        inputs_embeds: Optional[th.Tensor] = None,
        labels: Optional[th.Tensor] = None,
        pointwise_labels: Optional[th.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.new(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_mask = position_mask,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
        )
        dimension = 768
        hidden = outputs.last_hidden_state[:, 0][:dimension]
        pooled_output = self.dropout(hidden)
        orig_logits = self.classifier(pooled_output)
        logits = th.sigmoid(orig_logits.view(-1))
        
        logits_list = [logits]
        outputs_list = [outputs]

        if self.training:
            dimension = 768
            hidden = outputs.last_hidden_state[:, 0][:dimension]
            pooled_output = self.dropout(hidden)
            orig_logits = self.classifier(pooled_output)
            logits = th.sigmoid(orig_logits.view(-1))
            
            logits_list = [logits]
            outputs_list = [outputs]

        loss = None
        if labels is not None:
            logits_list = [logits.view(labels.shape[0], -1) for logits in logits_list]
            kl_loss = kl_div(logits_list[0], logits_list[-1], reverse=True)
            loss = self.rdrop * kl_loss
            loss_fct = CrossEntropyLoss()
            for logits in logits_list:
                logits1 = logits * self.contrast_temperature
                loss += 0.5 * loss_fct(logits1, labels)
        regression_loss = 0
        if pointwise_labels is not None:
            logits_list = [logits.view(-1) for logits in logits_list]
            for logits in logits_list:
                loss_per_position = th.max((logits - (pointwise_labels / 4 + 0.15))**2 - 0.01, th.tensor(0.0))
                regression_loss = loss_per_position.sum()
        loss += self.pointwise_weight * (regression_loss.sum())

        output = (logits,) + outputs[2:]

        return ((loss,) + output) if loss is not None else output