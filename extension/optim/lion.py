# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
name: lion optimizer for fairseq
source code: https://github.com/google/automl/blob/master/lion/lion_pytorch.py
paper: Symbolic Discovery of Optimization Algorithms
       https://arxiv.org/pdf/2302.06675.pdf
date: 2023/2/22
'''

import logging
import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, List

import torch
import torch.distributed as dist
import torch.optim
from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer
from omegaconf import II, OmegaConf


logger = logging.getLogger(__name__)

@dataclass
class FairseqLionConfig(FairseqDataclass):
    lion_betas: Any = field(
        default=(0.9, 0.99), metadata={"help": "betas for Lion optimizer"}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "weight decay"})
    # TODO common vars below in parent
    lr: List[float] = II("optimization.lr")


@register_optimizer("lion", dataclass=FairseqLionConfig)
class FairseqLion(FairseqOptimizer):
    """Lion optimizer for fairseq.
    """

    def __init__(self, cfg: FairseqLionConfig, params):
        super().__init__(cfg)
        self._optimizer = Lion(params, **self.optimizer_config)

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.cfg.lr[0]
            if isinstance(self.cfg.lr, Collection)
            else self.cfg.lr,
            "betas": eval(self.cfg.lion_betas)
            if isinstance(self.cfg.lion_betas, str)
            else OmegaConf.to_container(self.cfg.lion_betas),
            "weight_decay": self.cfg.weight_decay,
        }

    def average_params(self):
        """Reduce Params is only used during BMUF distributed training."""
        state_dict = self.optimizer.state_dict()
        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():
            value["exp_avg"] /= total_gpus
            dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)


class Lion(torch.optim.Optimizer):
    """
    Implements Lion algorithm.
    Initialize the hyperparameters.
    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
      lr (float, optional): learning rate (default: 1e-4)
      betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.99))
      weight_decay (float, optional): weight decay coefficient (default: 0)
    """

    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0,
    ):
        defaults = dict(
            lr=lr, betas=betas, weight_decay=weight_decay
        )
        super(Lion, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data   # gradient
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "Lion does not support sparse gradients, please consider SparseAdam instead"
                    )

                p_data_fp32 = p.data  # paramaters
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state = self.state[p]

                # Perform stepweight decay
                if group["weight_decay"] != 0:
                    p_data_fp32.mul_(1 - group['lr'] * group['weight_decay'])

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].to(p_data_fp32)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p_data_fp32.add_(torch.sign(update), alpha=-group['lr'])
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss
