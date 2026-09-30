import math

from dataclasses import dataclass,field
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from fairseq import utils,metrics # 工具函数和评估指标
from fairseq.criterions import FairseqCriterion, register_criterion #损失函数基类和注册器
from fairseq.dataclass import FairseqDataclass # 配置类基类
from omegaconf import II # 用于处理配置参数的继承和引用

@dataclass
class ImprovedLossConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.1,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "temperature for knowledge distillation"},
    )
    distillation_weight: float = field(
        default=0.5,
        metadata={"help": "weight for distillation loss"},
    )
    distillation_teacher: Optional[str] = field(
        default=None,
        metadata={"help": "teacher model for knowledge distillation"},
    )



