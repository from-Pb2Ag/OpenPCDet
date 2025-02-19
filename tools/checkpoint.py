#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import os
import tempfile
import traceback
from shutil import copy2, move

import torch

def replace_module_suffix(state_dict, suffix, replace_with=""):
    """
    Replace suffixes in a state_dict
    Needed when loading DDP or classy vision models
    """
    state_dict = {
        (key.replace(suffix, replace_with, 1) if key.startswith(suffix) else key): val
        for (key, val) in state_dict.items()
    }
    return state_dict


def append_module_suffix(state_dict, suffix):
    """
    Append suffixes in a state_dict
    Needed when loading DDP or classy vision models
    """
    state_dict = {f"{suffix}{key}": val for (key, val) in state_dict.items()}
    return state_dict


def init_model_from_weights(
    model,
    state_dict,
    state_dict_key_name="model",
    skip_layers=None,
    print_init_layers=True,
    replace_suffix="module.trunk.",
    freeze_bb=False,
    append_suffix="trunk.base_model.",
):
    """
    Initialize the model from any given params file. This is particularly useful
    during the finetuning process or when we want to evaluate a model on a range
    of tasks.
    skip_layers:     string : layer names with this key are not copied
    replace_suffix: string : remove these suffixes from the layer names
    print_init_layers:   print whether layer was init or ignored
                    indicates whether the layername was copied or not
    """
    # whether it's a model from somewhere else or a model from this codebase
    if state_dict_key_name and len(state_dict_key_name) > 0:
        #state_dict = state_dict["model_state_dict"]
        assert (
            state_dict_key_name in state_dict.keys()
        ), f"Unknown state dict key: {state_dict_key_name}"
        state_dict = state_dict[state_dict_key_name]
    if state_dict_key_name == "classy_state_dict":
        classy_state_dict = state_dict["base_model"]["model"]
        state_dict = {}
        state_dict.update(classy_state_dict["trunk"])
    if replace_suffix:
        state_dict = replace_module_suffix(state_dict, replace_suffix)
    if append_suffix:
        state_dict = append_module_suffix(state_dict, append_suffix)

    all_layers = model.state_dict()
    init_layers = {layername: False for layername in all_layers}

    new_state_dict = {}

    for param_name in init_layers:
        if 'backbone_3d' not in param_name:
            continue
        tempname = param_name[11:]
        if "trunk.base_model.0"+tempname in state_dict:
            new_state_dict[param_name] = state_dict["trunk.base_model.0"+tempname]
        elif "trunk.base_model.2"+tempname in state_dict:
            new_state_dict[param_name] = state_dict["trunk.base_model.2"+tempname]
        else:
            print (param_name)
    state_dict = new_state_dict
            
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    not_found, not_init = [], []
    for layername in all_layers.keys():
        if (
            skip_layers and len(skip_layers) > 0 and layername.find(skip_layers) >= 0
        ) or layername.find("num_batches_tracked") >= 0:
            if print_init_layers and (local_rank == 0):
                not_init.append(layername)
                print(f"Ignored layer:\t{layername}")
            continue
        if layername in state_dict:
            param = state_dict[layername]
            if not isinstance(param, torch.Tensor):
                param = torch.from_numpy(param)
            all_layers[layername].copy_(param)
            init_layers[layername] = True
            if print_init_layers and (local_rank == 0):
                print(f"Init layer:\t{layername}")
        else:
            not_found.append(layername)
            if print_init_layers and (local_rank == 0):
                print(f"Not found:\t{layername}")
    ####################### DEBUG ############################
    # _print_state_dict_shapes(model.state_dict())

    torch.cuda.empty_cache()
    return model
