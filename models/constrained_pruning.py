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
def has_repeated_ngram(tokens, n=3):
    ngrams = set()
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i:i+n])
        if ng in ngrams:
            return True
        ngrams.add(ng)
    return False

def check_grammar(sequence, parser):
    return parser.is_valid(sequence)  # 假设为预训练依存语法分析器

def check_spatial_logic(sequence, region_refs, regions):
    # 示例判断：左侧描述但坐标不满足左小于右
    for ref in region_refs:
        desc = sequence.lower()
        if "left of" in desc:
            i, j = ref
            if regions[i][1] > regions[j][0]:  # xmax_i > xmin_j
                return False
    return True

def prune_candidates(candidates, parser, regions, n=3, eta=0.5):
    pruned = []
    for cand in candidates:
        tokens = cand["tokens"]
        if has_repeated_ngram(tokens, n):
            continue
        if not check_grammar(" ".join(tokens), parser):
            continue
        if not check_spatial_logic(" ".join(tokens), cand["reg_ref"], regions):
            continue
        pruned.append(cand)
    return pruned
