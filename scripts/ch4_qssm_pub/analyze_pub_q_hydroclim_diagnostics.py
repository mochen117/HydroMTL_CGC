#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
======================================================================
Chapter 4 PUB Experiment

Hydroclimatic Diagnostics for PUB Streamflow Prediction

Purpose
-------
Diagnose hydroclimatic controls on PUB streamflow prediction
performance.

The analysis investigates whether low prediction skill in specific
hydroclimatic groups is associated with different water-energy
conditions and runoff characteristics.

Target variable
---------------
Streamflow (Q)

Metrics
-------
NSE_Q

Delta NSE_Q:
    CGC-Q - STL-Q

Hydroclimatic attributes
------------------------
aridity
frac_snow

Input
-----
ch4b_pub_effects_with_ch3_metadata.csv


Outputs
-------
diagnostics/

    pub_q_hydroclimate_diagnostics_summary.csv

    fig_nse_q_distribution.png

    fig_delta_nse_aridity.png

======================================================================
"""


from pathlib import Path
import logging
import argparse

import pandas as pd
import matplotlib.pyplot as plt



# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_INPUT = (
    PROJECT_ROOT
    /
    "experiments"
    /
    "ch4_qssm_pub"
    /
    "summary"
    /
    "ch4b_pub_effects_with_ch3_metadata.csv"
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    /
    "experiments"
    /
    "ch4_qssm_pub"
    /
    "hydroclimate_groups"
    /
    "diagnostics"
)



# ============================================================================
# Logger
# ============================================================================

logger = logging.getLogger(__name__)


def setup_logger():

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )



# ============================================================================
# Arguments
# ============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=
        "Diagnose PUB streamflow prediction across hydroclimatic groups."
    )


    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT)
    )


    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT)
    )


    return parser.parse_args()



# ============================================================================
# Load and validate
# ============================================================================

def load_data(path):

    logger.info(
        f"Loading input: {path}"
    )


    df = pd.read_csv(path)


    logger.info(
        f"Loaded basins: {len(df)}"
    )


    return df



def validate_columns(df):

    required = [

        "hydroclimate_group",

        "aridity",

        "frac_snow",

        "STL_Q_streamflow_nse",

        "CGC_streamflow_nse",

        "Delta_NSE_CGC_minus_STLQ"

    ]


    missing = [

        c for c in required
        if c not in df.columns

    ]


    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )



# ============================================================================
# Summary statistics
# ============================================================================

def summarize_groups(df):


    records = []


    for group, gdf in df.groupby(
        "hydroclimate_group"
    ):


        records.append(

            {

                "hydroclimate_group":
                    group,

                "n_basins":
                    len(gdf),

                "median_aridity":
                    gdf["aridity"].median(),

                "median_frac_snow":
                    gdf["frac_snow"].median(),

                "median_STL_NSE_Q":
                    gdf["STL_Q_streamflow_nse"].median(),

                "median_CGC_NSE_Q":
                    gdf["CGC_streamflow_nse"].median(),

                "median_Delta_NSE_Q":
                    gdf["Delta_NSE_CGC_minus_STLQ"].median(),

            }

        )


    return pd.DataFrame(records)



# ============================================================================
# Plotting
# ============================================================================

def plot_nse_distribution(df, output):


    groups = [
        "Wet",
        "Dry",
        "Snow"
    ]


    data = []

    labels = []


    for group in groups:

        values = (

            df.loc[

                df["hydroclimate_group"] == group,

                "STL_Q_streamflow_nse"

            ]

            .dropna()

        )


        if len(values):

            data.append(
                values.values
            )

            labels.append(
                group
            )


    plt.figure(
        figsize=(6,4)
    )


    plt.boxplot(

        data,

        tick_labels=labels,

        showfliers=False

    )


    plt.ylabel(
        "STL NSE_Q"
    )


    plt.xlabel(
        "Hydroclimatic group"
    )


    plt.tight_layout()


    plt.savefig(

        output /
        "fig_nse_q_distribution.png",

        dpi=300

    )


    plt.close()



def plot_delta_aridity(df, output):


    plt.figure(
        figsize=(6,4)
    )


    plt.scatter(

        df["aridity"],

        df["Delta_NSE_CGC_minus_STLQ"],

        s=12

    )


    plt.axhline(
        0,
        linestyle="--"
    )


    plt.xlabel(
        "Aridity index"
    )


    plt.ylabel(
        r"$\Delta NSE_Q$ (CGC-STL)"
    )


    plt.tight_layout()


    plt.savefig(

        output /
        "fig_delta_nse_aridity.png",

        dpi=300

    )


    plt.close()



# ============================================================================
# Main
# ============================================================================

def main():

    setup_logger()

    args = parse_args()


    output = Path(
        args.output
    )

    output.mkdir(
        parents=True,
        exist_ok=True
    )


    df = load_data(
        args.input
    )


    validate_columns(df)


    summary = summarize_groups(df)


    summary.to_csv(

        output /
        "pub_q_hydroclimate_diagnostics_summary.csv",

        index=False

    )


    plot_nse_distribution(
        df,
        output
    )


    plot_delta_aridity(
        df,
        output
    )


    logger.info(
        "PUB Q hydroclimate diagnostics completed."
    )



if __name__ == "__main__":

    main()