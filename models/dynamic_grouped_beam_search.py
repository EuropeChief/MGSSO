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

class DynamicGroupedBeamSearch:
    def __init__(self, beam_size, base_groups=4, max_entropy=math.log(4), tau=0.5):
        self.beam_size = beam_size
        self.base_groups = base_groups
        self.max_entropy = max_entropy
        self.tau = tau

    def compute_entropy(self, probs):
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1).mean()
        return entropy

    def adjust_groups(self, probs):
        entropy = self.compute_entropy(probs)
        group_count = math.ceil(self.base_groups * entropy / self.max_entropy)
        group_count = max(1, group_count)
        beam_per_group = max(1, self.beam_size // group_count)
        return group_count, beam_per_group

    def cosine_similarity(self, a, b):
        return F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=-1)

    def suppress_redundant_groups(self, group_reps):
        while True:
            n = len(group_reps)
            if n <= 1:
                break
            sim_matrix = self.cosine_similarity(group_reps, group_reps)
            mask = torch.eye(n).bool().to(sim_matrix.device)
            sim_matrix = sim_matrix.masked_fill(mask, -1.0)
            max_sim = torch.max(sim_matrix)
            if max_sim < self.tau:
                break
            i, j = torch.nonzero(sim_matrix == max_sim, as_tuple=True)
            group_reps = torch.cat([group_reps[:i[0]], group_reps[i[0]+1:]], dim=0)
        return group_reps
