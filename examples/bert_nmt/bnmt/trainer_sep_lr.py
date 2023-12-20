'''
lora ft时候bert和nmt使用不同的学习率:
# Separate Learning Rates
'''
import os
import sys
import time
import logging
from omegaconf import OmegaConf
from itertools import chain
from collections import OrderedDict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from fairseq.trainer import Trainer
from fairseq.optim import lr_scheduler
from fairseq import models, optim, utils
from fairseq.distributed import utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics
from fairseq.utils import safe_hasattr
from bnmt import checkpoint_utils

logger = logging.getLogger(__name__)


def remove_module_params(model, module_names=[]):
    """
    从模型中删除指定模块的指定参数。

    Args:
        model (nn.Module): 要删除参数的模型。
        module_name (str): 要删除参数的模块名称。
    """

    def do_remove(key):
        for exclude_key in module_names:
            if exclude_key in key:
                return True
        return False

    state_dict = model.state_dict()
    new_state = OrderedDict()
    for key, weight in state_dict.items():
        if not do_remove(key):
            new_state[key] = weight
    return new_state


class BertNMTTrainer(Trainer):
    '''
    bert增强NMT的训练器，加载权重时strict=False，参数加载时不用完全匹配模型结构，防止结构不同加载预先训练的标准transformer时报错。
    注意：后续如果要额外增加lora权重，可以放到extra_state中。
    '''

    def freeze_unfreeze_parameters(self):
        ''' 控制参数的冻结 '''
        freeze_nmt = self.model.cfg.freeze_nmt
        for name, param in self.model.named_parameters():
            # freeze bert
            if "bert" in name:
                param.requires_grad = False
            # freeze nmt
            elif "qformer" not in name:
                param.requires_grad = not freeze_nmt

    def initialize_qformer_by_dec(self, layer=0):
        # print("initialize_qformer_by_dec")
        state_dict = self.model.state_dict()
        # 将qformer参数转换为第n层解码器
        # encoder gformer .qlayers.0.
        for name, param in self.model.named_parameters():
            if "qformer" in name:
                module_name = name.replace("encoder.qformer.q_layers.0.", "")
                dec_name = f"decoder.layers.{layer}." + module_name
                # print(f"dec name: {dec_name}")
                if dec_name in state_dict.keys():
                    dec_param = state_dict[dec_name]
                    if param.shape == dec_param.shape:
                        param.data.copy_(dec_param.data)
                        # print(f"init qformer: {dec_name} ===> {name}")
                    else:
                        # print(f"not init param {name} shape != {dec_name} shape")
                        pass

    def _build_optimizer(self):
        if (
                self.cfg.optimization.debug_param_names
                and self.cfg.common.fp16_no_flatten_grads
        ):
            params = []
            self.param_names = []

            for n, p in chain(
                    self.model.named_parameters(), self.criterion.named_parameters()
            ):
                if p.requires_grad:
                    params.append(p)
                    self.param_names.append(n)
        else:
            # params = list(
            #     filter(
            #         lambda p: p.requires_grad,
            #         chain(self.model.parameters(), self.criterion.parameters()),
            #     )
            # )

            # Separate Learning Rates # torch==1.13.1
            params = []
            lr = self.cfg.optimizer['lr'][0]
            # lr_mult = self.cfg.optimization['lr_mul']
            lr_mult = 0.1
            lr2 = lr * lr_mult
            logger.info(f"Separate Learning Rates, lr1: {lr}, lr2: {lr2}")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # if 'PRETRAINED_MODEL' in name:
                    if 'bert' in name:
                        params.append({'params': [param], 'lr': lr2})
                    else:
                        params.append({'params': [param], 'lr': lr})


        if self.is_fsdp and self.cfg.common.fp16:
            # FullyShardedDataParallel always uses MemoryEfficientFP16 wrapper,
            # mostly for the grad scaling. But if we don't have the
            # --memory-efficient-fp16 flag set, then we're effectively doing
            # regular --fp16 and can allow the use of optimizers that would
            # otherwise be unsupported by MemoryEfficientFP16Optimizer.
            allow_unsupported = not self.cfg.common.memory_efficient_fp16
            self._optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(
                self.cfg, params, allow_unsupported=allow_unsupported
            )
        elif self.cfg.common.fp16 or self.cfg.common.bf16 or self.cfg.common.amp:
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                logger.info(
                    "NOTE: your device does NOT support faster training with --fp16 or --amp, "
                    "please switch to FP32 which is likely to be faster"
                )
            if (
                    self.cfg.common.memory_efficient_fp16
                    or self.cfg.common.memory_efficient_bf16
            ):
                self._optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(
                    self.cfg, params
                )
            elif self.cfg.common.amp:
                self._optimizer = optim.AMPOptimizer.build_optimizer(self.cfg, params)
            else:
                self._optimizer = optim.FP16Optimizer.build_optimizer(self.cfg, params)
        else:
            if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
                logger.info(
                    "NOTE: your device may support faster training with --fp16 or --amp"
                )
            self._optimizer = optim.build_optimizer(self.cfg.optimizer, params)

        if self.is_fsdp:
            assert (
                not self.cfg.optimization.use_bmuf
            ), "--ddp-backend=fully_sharded is not compatible with BMUF"
            assert self._optimizer.supports_flat_params, (
                "--ddp-backend=fully_sharded is only compatible with pointwise "
                "optimizers (e.g., Adam, AdamW, Adadelta, Adamax, SGD, etc.). "
                "However, the sharding will result in slightly different results when "
                "using non-pointwise optimizers (e.g., Adagrad, Adafactor, LAMB)"
            )

        if self.cfg.optimization.use_bmuf:
            self._optimizer = optim.FairseqBMUF(
                self.cfg.bmuf,
                self._optimizer,
            )

        if self.cfg.distributed_training.zero_sharding == "os":
            if (
                    self.cfg.common.fp16
                    and not self.cfg.common.memory_efficient_fp16
                    and not self.cfg.common.memory_efficient_bf16
            ) and not self.cfg.common.fp16_no_flatten_grads:
                raise ValueError(
                    "ZeRO is incomptabile with fp16 and flattened grads. "
                    "Please use --fp16-no-flatten-grads"
                )
            else:
                optim.shard_(self._optimizer, self.data_parallel_process_group)

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(
            self.cfg.lr_scheduler,
            self.optimizer,
        )
        self._lr_scheduler.step_update(0)

    def state_dict(self):
        state_dict = {
            "args": None,  # legacy
            "cfg": (
                OmegaConf.to_container(self.cfg, resolve=True, enum_to_str=True)
                if OmegaConf.is_config(self.cfg)
                else self.cfg
            ),
            # "model": self.model.state_dict(),
            "model": remove_module_params(self.model, module_names=["bert"]),  # 删除bert参数
            "criterion": (
                self.criterion.state_dict()
                if utils.has_parameters(self.criterion)
                else None
            ),
            "optimizer_history": (self._optim_history or [])
                                 + [
                                     {
                                         "criterion_name": self.get_criterion().__class__.__name__,
                                         "optimizer_name": self.optimizer.__class__.__name__,
                                         "lr_scheduler_state": self.lr_scheduler.state_dict(),
                                         "num_updates": self.get_num_updates(),
                                     }
                                 ],
            "task_state": self.task.state_dict() if self.task is not None else {},
            "extra_state": {
                "metrics": metrics.state_dict(),
                "previous_training_time": self.cumulative_training_time(),
            },
        }
        if self.cfg.ema.store_ema:
            # Save EMA model state as extra state
            state_dict["extra_state"]["ema"] = self.ema.get_model().state_dict()
            if self.cfg.ema.ema_fp32:
                # Save EMA params in fp32
                state_dict["extra_state"]["ema_fp32_params"] = self.ema.fp32_params
        if not self.cfg.checkpoint.no_save_optimizer_state:
            if self._gathered_optim_state is not None:
                state_dict["last_optimizer_state"] = self._gathered_optim_state
                self._gathered_optim_state = None
            else:
                state_dict["last_optimizer_state"] = self.optimizer.state_dict()
        if self.is_fsdp:
            # save meta data for recombining checkpoint upon loading
            state_dict["fsdp_metadata"] = self.model.local_metadata_dict()
        return state_dict

    def get_lora_dir(self, filename):
        '''
        filename: ckpt/checkpoint_best.pt
        lora_dir: ckpt/loras/checkpoint_best/
        '''
        # lora 目录
        if not os.path.exists(filename): return None
        save_dir = os.path.dirname(filename)
        ckpt_name = os.path.basename(filename)
        ckpt_name = ckpt_name.replace(".pt", "")
        lora_dir = os.path.join(save_dir, f"loras/{ckpt_name}")
        if not os.path.exists(lora_dir):
            os.makedirs(lora_dir)
        return lora_dir

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file.
            修改：保存lora权重到save_dir/loras
        """

        if self.should_save_checkpoint_on_current_rank:

            logger.info(f"Saving checkpoint to {os.path.abspath(filename)}")
            # call state_dict on all ranks in case it needs internal communication
            state_dict = utils.move_to_cpu(self.state_dict())
            state_dict["extra_state"].update(extra_state)

            checkpoint_utils.torch_persistent_save(
                state_dict,
                filename,
                async_write=self.cfg.checkpoint.write_checkpoints_asynchronously,
            )

            # 如果使用lora
            if self.model.cfg.use_lora:
                # lora 目录
                lora_dir = self.get_lora_dir(filename)
                self.model.encoder.bert_encoder.save_pretrained(lora_dir)
                logger.info(f"Saving BERT LoRA checkpoint to {os.path.abspath(lora_dir)}")

            logger.info(f"Finished saving checkpoint to {os.path.abspath(filename)}")
            return os.path.abspath(filename)
        return None

    def load_checkpoint(
            self,
            filename,
            reset_optimizer=False,
            reset_lr_scheduler=False,
            optimizer_overrides=None,
            reset_meters=False,
    ):
        """
        Load all training state from a checkpoint file.
        rank = 0 will load the checkpoint, and then broadcast it to all
        other ranks.
        """
        strict = False  # <<== 运行部分加载参数

        extra_state, self._optim_history, last_optim_state = None, [], None

        logger.info(f"Preparing to load checkpoint {filename}")
        is_distributed = self.data_parallel_world_size > 1
        bexists = PathManager.isfile(filename)  # 是否存在filename（restore_file）
        if bexists:
            load_on_all_ranks = (
                    self.cfg.checkpoint.load_checkpoint_on_all_dp_ranks
                    # TPUs don't support broadcast yet, so load checkpoints
                    # on every worker for now
                    or self.tpu
                    # FSDP requires loading checkpoint shards on all ranks
                    or (self.is_fsdp and self.cfg.distributed_training.use_sharded_state)
                    or getattr(self.cfg.model, "base_layers", 0) > 0
            )

            if load_on_all_ranks or self.data_parallel_rank == 0:
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    filename, load_on_all_ranks=load_on_all_ranks
                )
                last_optim_state = state.get("last_optimizer_state", None)

                # If doing zero_sharding, do not broadcast global optimizer
                # state. Later we will broadcast sharded states to each rank
                # to avoid memory from exploding.
                if (
                        not load_on_all_ranks
                        and self.cfg.distributed_training.zero_sharding == "os"
                        and "last_optimizer_state" in state
                        and is_distributed
                ):
                    state["last_optimizer_state"] = "SHARDED"
            else:
                last_optim_state = None
                state = None

            if is_distributed and not load_on_all_ranks:
                state = distributed_utils.broadcast_object(
                    state,
                    src_rank=0,
                    group=self.data_parallel_process_group,
                    dist_device=self.device,
                )
                if self.data_parallel_rank > 0:
                    last_optim_state = state.get("last_optimizer_state", None)

            # load model parameters
            try:
                if (
                        "optimizer_history" in state
                        and len(state["optimizer_history"]) > 0
                        and "num_updates" in state["optimizer_history"][-1]
                ):
                    self.model.set_num_updates(
                        state["optimizer_history"][-1]["num_updates"]
                    )

                # this is the code related to AdaPrune
                # In short, it removes redundant heads in multi-head attention module based on heads importance provided
                # For more info, please refer to the paper: https://openreview.net/forum?id=_CMSV7FTzGI
                # The idea of prune in mha can be summarized as
                # Fine tune model (e.g. roberta encoder) on a certain datasets with regularization
                # After the model is trained. User could use get_reserve_head_index and _adaptive_prune_heads functions to get the top X heads with most importance.
                # Then user uses the rank to prune a new roberta encoder and save the pruned ckpt manually.
                # User will fine tune the the new roberta encoder via the ckpt saved above
                # To get rid of registering different pruned version of Roberta, I use the argument --mha-heads-to-keep to prune the Roberta model into a pruned version which matches the pruned ckpt.
                if (
                        safe_hasattr(self.model, "args")
                        and safe_hasattr(self.model.args, "mha_heads_to_keep")
                        and self.model.args.mha_heads_to_keep != -1
                ):
                    logger.info(
                        f"Prune model: keep {self.model.args.mha_heads_to_keep} heads for each multihead attention module"
                    )
                    for layer in self.model.encoder.sentence_encoder.layers:
                        reserve_head_index = layer.self_attn._get_reserve_head_index(
                            num_heads_to_keep=self.model.args.mha_heads_to_keep
                        )
                        layer.self_attn._adaptive_prune_heads(
                            reserve_head_index=reserve_head_index
                        )
                        layer.self_attn._set_skip_embed_dim_check()
                    logger.info(self.model)
                # this is the code related to AdaPrune
                # In short, it removes redundant units in feedforward layer in each transformer layer based on importance
                # For more info, please refer to the paper: https://openreview.net/forum?id=_CMSV7FTzGI
                # The idea of prune in ffn can be summarized as
                # Fine tune model (e.g. roberta encoder) on a certain datasets with regularization
                # After the model is trained. User could use _get_fc_rank and _prune_fc_layer functions to get the top X units with most importance.
                # Then user uses the rank to prune a new roberta encoder and save the pruned ckpt manually.
                # User will fine tune the the new roberta encoder via the ckpt saved above
                # To get rid of registering different pruned version of Roberta, I use the argument --ffn-blocks-to-remove to prune the Roberta model into a pruned version which matches the pruned ckpt.
                if (
                        safe_hasattr(self.model, "args")
                        and safe_hasattr(self.model.args, "ffn_blocks_to_remove")
                        and self.model.args.ffn_blocks_to_remove != -1
                ):
                    logger.info(
                        f"Prune model: remove {self.model.args.ffn_blocks_to_remove} ffn blocks for each transformer layer"
                    )
                    for layer in self.model.encoder.sentence_encoder.layers:
                        remove_index = layer._get_fc_rank(
                            remove_num=self.model.args.ffn_blocks_to_remove
                        )
                        layer._prune_fc_layer(remove_index=remove_index)
                    logger.info(self.model)

                self.model.load_state_dict(
                    state["model"], strict=strict, model_cfg=self.cfg.model
                )
                # save memory for later steps
                del state["model"]
                if utils.has_parameters(self.get_criterion()):
                    self.get_criterion().load_state_dict(
                        state["criterion"], strict=strict
                    )
                    del state["criterion"]

            except Exception:
                raise Exception(
                    "Cannot load model parameters from checkpoint {}; "
                    "please ensure that the architectures match.".format(filename)
                )
            extra_state = state["extra_state"]
            self._optim_history = state["optimizer_history"]

        ####### BERT-NMT #######
        self.freeze_unfreeze_parameters()  # 冻结参数与否
        # 加载lora
        lora_dir = self.get_lora_dir(filename)
        self.model.encoder.load_bert_lora(lora_dir)
        # self.initialize_qformer_by_dec() # 初始化qformer，比如用bert参数

        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert (
                    last_optim["criterion_name"] == self.get_criterion().__class__.__name__
            ), f"Criterion does not match; please reset the optimizer (--reset-optimizer). {last_optim['criterion_name']} vs {self.get_criterion().__class__.__name__}"
            assert (
                    last_optim["optimizer_name"] == self.optimizer.__class__.__name__
            ), f"Optimizer does not match; please reset the optimizer (--reset-optimizer). {last_optim['optimizer_name']} vs {self.optimizer.__class__.__name__}"

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim["lr_scheduler_state"])

            if self.is_fsdp and not self.model.use_sharded_state:
                # if use_sharded_state, the last_optim_state is already sharded, skip this
                last_optim_state = self.model.get_shard_from_optim_state_dict(
                    last_optim_state
                )
            elif not load_on_all_ranks and is_distributed:
                last_optim_state = self.optimizer.broadcast_global_state_dict(
                    last_optim_state
                )

            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self.set_num_updates(last_optim["num_updates"])

        if extra_state is not None:
            itr_state = extra_state["train_iterator"]
            epoch = itr_state["epoch"]

            if "previous_training_time" in extra_state:
                self._previous_training_time = extra_state["previous_training_time"]
                self._start_time = time.time()

            self.lr_step(epoch)

            if (
                    itr_state.get("version", 1) >= 2
                    and itr_state["iterations_in_epoch"] == 0
            ):
                # reset meters at start of epoch
                reset_meters = True

            if "metrics" in extra_state and not reset_meters:
                metrics.load_state_dict(extra_state["metrics"])

                # reset TimeMeters, since their start times don't make sense anymore
                for meter in metrics.get_meters("default"):
                    if isinstance(meter, meters.TimeMeter):
                        meter.reset()

            if self.cfg.ema.store_ema:
                if "ema" not in extra_state:
                    logger.warn(
                        "EMA not found in checkpoint. But store_ema is True. "
                        "EMA is re-initialized from checkpoint."
                    )
                    self.ema.restore(
                        state["model"], build_fp32_params=self.cfg.ema.ema_fp32
                    )
                else:
                    logger.info("Loading EMA from checkpoint")
                    self.ema.restore(extra_state["ema"], build_fp32_params=False)

                    if self.cfg.ema.ema_fp32:
                        if "ema_fp32_params" in extra_state:
                            logger.info("Loading EMA fp32 params from checkpoint")
                            self.ema.build_fp32_params(extra_state["ema_fp32_params"])
                        else:
                            logger.info(
                                "Building EMA fp32 params from EMA model in checkpoint"
                            )
                            self.ema.build_fp32_params()

            logger.info(
                "Loaded checkpoint {} (epoch {} @ {} updates)".format(
                    filename, epoch, self.get_num_updates()
                )
            )

        else:
            logger.info("No existing checkpoint found {}".format(filename))

        return extra_state
