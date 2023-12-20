#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import os
import re
import shutil

import torch

from fairseq.file_io import PathManager


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of checkpoints to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for fpath in inputs:
        with PathManager.open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                ),
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state["model"]

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model"] = averaged_params
    return new_state


def average_loras(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of lora checkpoints to load from.  [ckpt/xxx/loras/check..]

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for fpath in inputs:
        lora_path = fpath + "/adapter_model.bin"
        with PathManager.open(lora_path, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                ),
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        # model_params = state["model"]
        model_params = state

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    # new_state["model"] = averaged_params
    new_state = averaged_params

    return new_state


def last_n_checkpoints(paths, n, update_based, upper_bound=None):
    assert len(paths) == 1
    path = paths[0]
    if update_based:
        pt_regexp = re.compile(r"checkpoint_\d+_(\d+)\.pt")
    else:
        pt_regexp = re.compile(r"checkpoint(\d+)\.pt")
    files = PathManager.ls(path)

    entries = []
    for f in files:
        m = pt_regexp.fullmatch(f)
        if m is not None:
            sort_key = int(m.group(1))
            if upper_bound is None or sort_key <= upper_bound:
                entries.append((sort_key, m.group(0)))
    if len(entries) < n:
        raise Exception(
            "Found {} checkpoint files but need at least {}", len(entries), n
        )
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)[:n]]


def main():
    parser = argparse.ArgumentParser(
        description="Tool to average the params of input checkpoints to "
                    "produce a new checkpoint",
    )
    # fmt: off
    parser.add_argument('--inputs', required=True, nargs='+',
                        help='Input checkpoint file paths.')
    parser.add_argument('--output', required=True, metavar='FILE',
                        help='Write the new checkpoint containing the averaged weights to this path.')
    num_group = parser.add_mutually_exclusive_group()
    num_group.add_argument('--num-epoch-checkpoints', type=int,
                           help='if set, will try to find checkpoints with names checkpoint_xx.pt in the '
                                'path specified by input, and average last this many of them.')
    num_group.add_argument('--num-update-checkpoints', type=int,
                           help='if set, will try to find checkpoints with names checkpoint_ee_xx.pt in the path specified by'
                                ' input, and average last this many of them.')
    num_group.add_argument('--num-best-checkpoints', type=int, default=0,
                           help='if set, will try to find checkpoints with names checkpoint_best_ee_xx.pt in the path specified by'
                                ' input, and average last this many of them.')
    parser.add_argument('--checkpoint-upper-bound', type=int,
                        help='when using --num-epoch-checkpoints, this will set an upper bound on which epoch to use, '
                             'when using --num-update-checkpoints, this will set an upper bound on which update to use'
                             'e.g., with --num-epoch-checkpoints=10 --checkpoint-upper-bound=50, checkpoints 41-50 would be'
                             ' averaged.'
                             'e.g., with --num-update-checkpoints=10 --checkpoint-upper-bound=50000, checkpoints 40500-50000 would'
                             ' be averaged assuming --save-interval-updates 500'
                        )
    parser.add_argument('--use-lora', required=False, action='store_true',
                        help='average lora ckpts')
    # fmt: on
    args = parser.parse_args()
    print(args)

    num = None
    is_update_based = False
    if args.num_update_checkpoints is not None:
        num = args.num_update_checkpoints
        is_update_based = True
    elif args.num_epoch_checkpoints is not None:
        num = args.num_epoch_checkpoints

    assert args.checkpoint_upper_bound is None or (
            args.num_epoch_checkpoints is not None
            or args.num_update_checkpoints is not None
    ), "--checkpoint-upper-bound requires --num-epoch-checkpoints or --num-update-checkpoints"
    assert (
            args.num_epoch_checkpoints is None or args.num_update_checkpoints is None
    ), "Cannot combine --num-epoch-checkpoints and --num-update-checkpoints"

    if num is not None:
        args.inputs = last_n_checkpoints(
            args.inputs,
            num,
            is_update_based,
            upper_bound=args.checkpoint_upper_bound,
        )
        print("averaging checkpoints: ", args.inputs)

    if args.num_best_checkpoints > 0:
        args.inputs = list(
            sorted(
                args.inputs,
                key=lambda x: float(
                    os.path.basename(x).split("_")[-1].replace(".pt", "")
                ),  # 小到大排序
            )
        )
        args.inputs = args.inputs[: args.num_best_checkpoints]  # 取topk
        for path in args.inputs:
            print(os.path.basename(path))  # 打印文件名
    new_state = average_checkpoints(args.inputs)
    with PathManager.open(args.output, "wb") as f:
        torch.save(new_state, f)

    if args.use_lora:
        def pt2lora_dir(pt_path):
            lora_path = os.path.join(os.path.dirname(pt_path), "loras", os.path.basename(pt_path).replace(".pt", "/"))
            return lora_path

        lora_inputs = [pt2lora_dir(path) for path in args.inputs]
        print("lora_inputs", lora_inputs)
        new_lora_state = average_loras(lora_inputs)
        lora_output_dir = pt2lora_dir(args.output)
        if not os.path.exists(lora_output_dir):
            os.makedirs(lora_output_dir)
        lora_output = lora_output_dir + "/adapter_model.bin"
        with PathManager.open(lora_output, "wb") as f:
            torch.save(new_lora_state, f)
        shutil.copy(lora_inputs[-1] + "/adapter_config.json", lora_output_dir + "/adapter_config.json")

    print("Finished writing averaged checkpoint to {}".format(args.output))


if __name__ == "__main__":
    main()
