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