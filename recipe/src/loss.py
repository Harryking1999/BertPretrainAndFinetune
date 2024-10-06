from enum import Enum

import torch as th
import torch.nn as nn
import torch.nn.functional as F

def kl_div(predicts, targets, reverse: bool = False):
    p = th.log_softmax(predicts, dim = -1)
    q = th.log_softmax(targets, dim = -1)
    kl_loss = F.kl_div(p, q, reduction="batchmean", log_target=True)

    if reverse:
        reverse_kl_loss = F.kl_div(q, p, reduction="batchmean", log_target=True)
        kl_loss = (kl_loss + reverse_kl_loss) / 2.0

    return kl_loss

def cos_sim(a: th.Tensor, b: th.Tensor):
    if not isinstance(a, th.Tensor):
        a = th.tensor(a)
    
    if not isinstance(b, th.Tensor):
        b = th.tensor(b)

    if(len(a.shape) == 1):
        a = a.unsqueeze(0)

    if(len(b.shape) == 1):
        b = b.unsqueeze(0)

    a_norm = F.normalize(a, p = 2, dim = 1)
    b_norm = F.normalize(b, p = 2, dim = 1)
    return th.mm(a_norm, b.norm.transpose(0, 1))

class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, scale: float = 20.0, similarity_fct = cos_sim):
        super().__init__()
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        anchor_reps: th.Tensor,
        positive_reps: th.Tensor,
        negative_reps: th.Tensor = None,
    ):
        embeddings_a = anchor_reps
        if(negative_reps != None):
            embeddings_b = th.cat([positive_reps, negative_reps])
        else:
            embeddings_b = positive_reps
        
        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        labels = th.tensor(
            range(len(scores)), dtype = th.long, device = scores.device
        )

        return self.cross_entropy(scores, labels)