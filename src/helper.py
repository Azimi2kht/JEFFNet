# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

import src.models.vision_transformer as vit
from src.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

from collections import OrderedDict

import torch


def strip_module_prefix(state_dict, prefix="module."):
    # state_dict can be a dict of tensors (already) or something nested
    new_state = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state[k[len(prefix) :]] = v
        else:
            new_state[k] = v
    return new_state


def load_checkpoint(
    device,
    r_path,
    encoder,
    predictor,
    target_encoder,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device("cpu"))
        epoch = checkpoint["epoch"]

        # -- loading encoder
        pretrained_dict = strip_module_prefix(checkpoint["encoder"])
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

        # -- loading predictor
        pretrained_dict = strip_module_prefix(checkpoint["predictor"])
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

        # -- loading target_encoder
        if target_encoder is not None:
            pretrained_dict = strip_module_prefix(checkpoint["target_encoder"])
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}")

        logger.info(f"read-path: {r_path}")
        del checkpoint

    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")
        epoch = 0

    return encoder, predictor, target_encoder


def init_model(
    device,
    patch_size=14,
    model_name="vit_base",
    crop_size=224,
    pred_depth=12,
    pred_emb_dim=384,
):
    encoder = vit.__dict__[model_name](img_size=[crop_size], patch_size=patch_size)
    predictor = vit.__dict__["vit_predictor"](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
    )

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    
    return encoder, predictor


def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25,
):
    param_groups = [
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": True,
            "weight_decay": 0,
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": True,
            "weight_decay": 0,
        },
    ]

    logger.info("Using AdamW")
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler


def plot_confusion_matrix(
    cm,
    class_names,
    normalize=True,
    title="Confusion Matrix",
    cmap="Blues",
    save_path=None,
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        linewidths=0.5,
    )

    plt.xlabel("Predicted label", fontsize=12)
    plt.ylabel("True label", fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def ensure_path(path: str | Path) -> Path:
    """
    Ensure that all directories for the given path exist.

    - If `path` is a directory, it is created.
    - If `path` is a file path, its parent directories are created.

    Returns the Path object.
    """
    p = Path(path)

    # If path has a suffix, assume it's a file; otherwise a directory
    if p.suffix:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)

    return p


def create_next_run_dir(runs_dir: str, seed: Optional[int] = None) -> str:
    """
    Creates and returns a run directory.

    - If seed is provided: runs/run_seed_<seed>
    - Otherwise:          runs/run_<idx>

    Safe for repeated use.
    """
    os.makedirs(runs_dir, exist_ok=True)

    if seed is not None:
        run_dir = os.path.join(runs_dir, f"run_seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)
        return Path(run_dir)

    pattern = re.compile(r"^run_(\d+)$")
    indices = []

    for name in os.listdir(runs_dir):
        match = pattern.match(name)
        if match:
            indices.append(int(match.group(1)))

    next_idx = max(indices) + 1 if indices else 0
    run_dir = os.path.join(runs_dir, f"run_{next_idx}")
    os.makedirs(run_dir, exist_ok=False)

    return Path(run_dir)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For PyTorch >= 1.8
    torch.use_deterministic_algorithms(True, warn_only=True)
