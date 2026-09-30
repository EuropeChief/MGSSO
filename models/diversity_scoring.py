import math
from typing import List, Optional

import torch
import torch.nn as nn
from fairseq.token_generation_constraints import (
    ConstraintState,
    OrderedConstraintState,
    UnorderedConstraintState,
)
from torch import Tensor
def ngram_entropy(candidates, n=3):
    all_ngrams = []
    for seq in candidates:
        tokens = seq.split()
        for k in range(1, n+1):
            all_ngrams.extend(zip(*[tokens[i:] for i in range(k)]))
    counter = Counter(all_ngrams)
    total = sum(counter.values())
    entropy = -sum((c/total) * math.log(c/total + 1e-9) for c in counter.values())
    return entropy

def group_score(seq, lm_log_prob, emb, other_group_embs, alpha=0.6, beta=0.3, gamma=0.1):
    rdiv = ngram_entropy([seq])
    psim = max(
        F.cosine_similarity(emb.unsqueeze(0), other_emb.unsqueeze(0)).item()
        for other_emb in other_group_embs
    ) if other_group_embs else 0.0
    score = alpha * lm_log_prob + beta * rdiv - gamma * psim
    return score
