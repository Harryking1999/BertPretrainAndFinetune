import random
from dataclasses import dataclass

import numpy as np
import torch as th
from transformers import PreTrainedTokenizerBase

@dataclass
class DataCollatorForTriplet:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = None

    def __call__(self, features):
        query_input, pdoc_input, ndoc_input = [], [], []
        query_token_type, pdoc_token_type, ndoc_token_type = [], [], []
        doc_max_len = 0
        for feature in features:
            tokenized_query = feature["tokenized_query"]
            tokenized_pdoc = random.choice(feature['tokenized_pdoc'])
            tokenized_ndoc = random.choice(feature['tokenized_ndoc'])

            qi = (
                [self.tokenizer.cls_token_id]
                + tokenized_query
                + [self.tokenizer.sep_token_id]
            )
            qt = [0] * (len(tokenized_query) + 2)
            pi = (
                [self.tokenizer.cls_token_id]
                + tokenized_pdoc[0]
                + [self.tokenizer.sep_token_id]
                + tokenized_pdoc[1]
                + [self.tokenizer.sep_token_id] 
            )
            pt = [0] * (len(tokenized_pdoc[0]) + len(tokenized_pdoc[1]) + 3)
            ni = (
                [self.tokenizer.cls_token_id]
                + tokenized_ndoc[0]
                + [self.tokenizer.sep_token_id]
                + tokenized_ndoc[1]
                + [self.tokenizer.sep_token_id] 
            )
            nt = [0] * (len(tokenized_ndoc[0]) + len(tokenized_ndoc[1]) + 3)

            query_input.append(qi)
            pdoc_input.append(pi)
            ndoc_input.append(ni)
            query_token_type.append(qt)
            pdoc_token_type.append(pt)
            ndoc_token_type.append(nt)
            doc_max_len = max(doc_max_len, len(pi), len(ni))
        # tokenized_query = [feature["tokenized_query"] for feature in features]
        # tokenized_pdoc = [feature["tokenized_puery"] for feature in features]
        # tokenized_ndoc = (
        #     [feature["tokenized_ndoc"] for feature in features]
        #     if "tokenized_ndoc" in features[0].keys()
        #     else None
        # )

        qbatch = self.tokenizer.pad(
            {"input_ids": query_input, "token_type_ids": query_token_type},
            pad_to_multiple_of = self.pad_to_multiple_of,
            return_tensors = "pt"
        )
        pbatch = self.tokenizer.pad(
            {"input_ids": pdoc_input, "token_type_ids": pdoc_token_type},
            pad_to_multiple_of = self.pad_to_multiple_of,
            return_tensors = "pt"
        )
        nbatch = self.tokenizer.pad(
            {"input_ids": ndoc_input, "token_type_ids": ndoc_token_type},
            pad_to_multiple_of = self.pad_to_multiple_of,
            return_tensors = "pt"
        )
        return qbatch, pbatch, nbatch

@dataclass
class DataCollatorForRerankMultiFieldWithPointwiseLabel:
    tokenizer: PreTrainedTokenizerBase
    neg_num: int
    pad_to_multiple_of: int = None
    
    def __call__(self, features):
        input_ids = []
        token_type_ids = []
        all_chosen_label = []
        for feature in features:
            tokenized_query = feature["tokenized_query"]
            tokenized_pdoc = feature['tokenized_pdoc']
            tokenized_ndoc = feature['tokenized_ndoc']
            all_pos_pointwise_label = feature['pos_pointwise_label']
            all_neg_pointwise_label = feature['neg_pointwise_label']

            ind_pdoc = []
            ind_ndoc = []
            for i in range(len(tokenized_pdoc)):
                ind_pdoc.apepnd(i)
            for i in range(len(tokenized_ndoc)):
                ind_ndoc.append(i)

            ind_chosen_pdoc = random.choice(ind_pdoc)
            tokenized_chosen_pdoc = [tokenized_pdoc[ind_chosen_pdoc]]
            pdoc_pointwise_label = [all_pos_pointwise_label[ind_chosen_pdoc]]

            tokenized_chosen_ndoc = []
            ndoc_pointwise_label = []
            if self.neg_num <= len(ind_ndoc):
                ind_chosen_ndoc = random.sample(ind_ndoc, k = self.neg_num)
                for i in ind_chosen_ndoc:
                    ndoc_pointwise_label.append(all_neg_pointwise_label[i])
                    tokenized_chosen_ndoc.append(tokenized_ndoc[i])
            else:
                ind_chosen_ndoc = random.choices(ind_ndoc, k = self.neg_num)
                for i in ind_chosen_ndoc:
                    ndoc_pointwise_label.append(all_neg_pointwise_label[i])
                    tokenized_chosen_ndoc.append(tokenized_ndoc[i])
            query_input_ids = (
                [self.tokenizer.cls_token_id]
                + tokenized_query
                + [self.tokenizer.sep_token_id]
            )

            for tokenized_doc in tokenized_chosen_pdoc + tokenized_chosen_ndoc:
                inputs_ids += [
                    query_input_ids
                    + tokenized_doc[0]
                    + [self.tokenizer.sep_token_id]
                    + tokenized_doc[1]
                    + [self.tokenizer.sep_token_id]
                ]
                token_type_ids += [
                    [0] * len(
                        query_input_ids
                        + tokenized_doc[0]
                        + [self.tokenizer.sep_token_id]
                        + tokenized_doc[1]
                        + [self.tokenizer.sep_token_id]
                    )
                ]
            all_chosen_label += pdoc_pointwise_label + ndoc_pointwise_label
        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "token_type_ids": token_type_ids},
            pad_to_multiple_of = self.pad_to_multiple_of,
            return_tensors = "pt"
        )
        batch['labels'] = th.zeros(len(features), dtype = th.long)
        batch['pointwise_labels'] = th.tensor(all_chosen_label)
        return batch