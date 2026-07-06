# ==============================================================================
# Copyright (c) 2026.
# All Rights Reserved.
#
# Description:
#     Gate Utilization Diagnostics for CGC / MMoE Models.
#
# Purpose:
#     Analyze expert routing behaviour and expert utilization.
#
# Supported Architectures:
#     - MMoE
#     - CGC
#
# Output:
#     gate_utilization.csv
#     expert_utilization.csv
#
# Author:
#     HydroMTL_CGC
# ==============================================================================

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch


def summarize_gate_weights(
    gate_outputs: Dict[str, torch.Tensor]
) -> pd.DataFrame:
    """
    Compute average gate weights.
    """

    rows = []

    for gate_name, weights in gate_outputs.items():

        weights = (
            weights.detach()
            .cpu()
            .numpy()
        )

        mean_weights = weights.mean(
            axis=0
        )

        for idx, value in enumerate(
            mean_weights
        ):

            rows.append(
                {
                    "gate": gate_name,
                    "expert_id": idx,
                    "mean_weight": float(
                        value
                    ),
                }
            )

    return pd.DataFrame(rows)


def compute_expert_utilization(
    gate_outputs: Dict[str, torch.Tensor]
) -> pd.DataFrame:
    """
    Compute expert activation frequency.
    """

    rows = []

    for gate_name, weights in gate_outputs.items():

        weights = (
            weights.detach()
            .cpu()
            .numpy()
        )

        selected = np.argmax(
            weights,
            axis=1,
        )

        total = len(selected)

        for expert_id in range(
            weights.shape[1]
        ):

            frequency = (
                np.sum(
                    selected == expert_id
                )
                / total
            )

            rows.append(
                {
                    "gate": gate_name,
                    "expert_id": expert_id,
                    "frequency": frequency,
                }
            )

    return pd.DataFrame(rows)


def save_gate_tables(
    gate_df: pd.DataFrame,
    expert_df: pd.DataFrame,
    output_dir: Path,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gate_df.to_csv(
        output_dir /
        "gate_utilization.csv",
        index=False,
    )

    expert_df.to_csv(
        output_dir /
        "expert_utilization.csv",
        index=False,
    )


if __name__ == "__main__":

    print(
        "Gate utilization diagnostics template."
    )

    print(
        "Load stored gate outputs from "
        "validation predictions."
    )