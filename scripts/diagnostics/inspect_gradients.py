# ==============================================================================
# Copyright (c) 2026.
# All Rights Reserved.
#
# Description:
#     Gradient Similarity Diagnostics for Multi-Task Hydrological Models.
#
# Purpose:
#     Quantify task interaction by measuring cosine similarity between
#     streamflow and evapotranspiration gradients.
#
# Supported Architectures:
#     - Hard-MTL
#     - MMoE
#     - CGC
#
# Output:
#     gradient_similarity.csv
#
# Author:
#     HydroMTL_CGC
# ==============================================================================

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def flatten_gradients(
    gradients: List[torch.Tensor]
) -> torch.Tensor:
    """
    Flatten gradient tensors into a single vector.
    """
    vectors = []

    for grad in gradients:
        if grad is None:
            continue

        vectors.append(
            grad.detach().flatten()
        )

    if len(vectors) == 0:
        return torch.tensor([])

    return torch.cat(vectors)


def cosine_similarity(
    loss_q: torch.Tensor,
    loss_et: torch.Tensor,
    params: List[torch.nn.Parameter],
) -> float:
    """
    Compute cosine similarity between two task gradients.
    """

    grads_q = torch.autograd.grad(
        loss_q,
        params,
        retain_graph=True,
        allow_unused=True,
    )

    grads_et = torch.autograd.grad(
        loss_et,
        params,
        retain_graph=True,
        allow_unused=True,
    )

    vec_q = flatten_gradients(grads_q)
    vec_et = flatten_gradients(grads_et)

    if vec_q.numel() == 0:
        return np.nan

    sim = F.cosine_similarity(
        vec_q,
        vec_et,
        dim=0,
    )

    return float(sim.item())


def collect_parameter_groups(
    model: torch.nn.Module
) -> Dict[str, List[torch.nn.Parameter]]:
    """
    Collect diagnostic parameter groups.
    """

    groups = {}

    groups["Encoder"] = [
        p
        for name, p in model.named_parameters()
        if (
            "encoder" in name.lower()
            or "lstm" in name.lower()
        )
        and p.requires_grad
    ]

    groups["CGCBlock"] = [
        p
        for name, p in model.named_parameters()
        if (
            "cgc_layer" in name.lower()
        )
        and p.requires_grad
    ]

    groups["Gate"] = [
        p
        for name, p in model.named_parameters()
        if (
            "gate" in name.lower()
        )
        and p.requires_grad
    ]

    return groups


def save_results(
    output_path: Path,
    results: Dict[str, float],
) -> None:

    pd.DataFrame(
        [results]
    ).to_csv(
        output_path,
        index=False,
    )


if __name__ == "__main__":

    print(
        "Gradient diagnostics script template."
    )

    print(
        "Load checkpoint, validation dataloader, "
        "and task losses before running."
    )