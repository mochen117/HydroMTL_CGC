#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
======================================================================
Chapter 4 PUB Experiment

Hydroclimatic Analysis of Streamflow Prediction

Purpose
-------
Analyze whether auxiliary soil moisture information improves
streamflow prediction under PUB conditions.

Target variable
---------------
Streamflow (Q)

Auxiliary variable
------------------
Soil surface moisture (SSM)

Models
------
STL-Q:
    Single-task streamflow prediction

Hard-MTL-Q:
    Hard parameter sharing with auxiliary SSM task

CGC-Q:
    Customized Gate Control multi-task learning model


Evaluation metrics
------------------
Absolute performance:

    NSE_Q

Transfer effect:

    Delta NSE_Q =
        NSE_Q(MTL) - NSE_Q(STL)


Input
-----
experiments/ch4_qssm_pub/summary/

    ch4b_pub_effects_with_ch3_metadata.csv


Required columns
----------------
hydroclimate_group

STL_Q_streamflow_nse

Hard_MTL_streamflow_nse

CGC_streamflow_nse

Delta_NSE_HardMTL_minus_STLQ

Delta_NSE_CGC_minus_STLQ


Outputs
-------
experiments/ch4_qssm_pub/hydroclimate_groups/

    pub_q_hydroclimate_group_summary.csv

    pub_q_absolute_nse_group_summary.csv

    fig_pub_q_delta_nse_boxplot.png

    fig_pub_q_absolute_nse.png

    fig_pub_q_positive_transfer_rate.png


======================================================================
"""


from pathlib import Path
import argparse
import logging

import numpy as np
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
)



# ============================================================================
# Logging
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
        "Analyze PUB streamflow prediction across hydroclimatic groups."
    )


    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to PUB basin-level summary."
    )


    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output directory."
    )


    return parser.parse_args()



# ============================================================================
# Data loading
# ============================================================================

def load_data(path):

    logger.info(
        f"Loading input file: {path}"
    )


    df = pd.read_csv(path)


    logger.info(
        f"Loaded basins: {len(df)}"
    )


    return df



# ============================================================================
# Validation
# ============================================================================

def validate_columns(df):


    required_columns = [

        "hydroclimate_group",

        "STL_Q_streamflow_nse",

        "Hard_MTL_streamflow_nse",

        "CGC_streamflow_nse",

        "Delta_NSE_HardMTL_minus_STLQ",

        "Delta_NSE_CGC_minus_STLQ",

    ]


    missing = [

        col for col in required_columns
        if col not in df.columns

    ]


    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )



# ============================================================================
# Absolute NSE_Q statistics
# ============================================================================

def calculate_absolute_nse_q(df):


    model_columns = {

        "STL-Q":
            "STL_Q_streamflow_nse",

        "Hard-MTL-Q":
            "Hard_MTL_streamflow_nse",

        "CGC-Q":
            "CGC_streamflow_nse",

    }


    records = []


    for group, group_df in df.groupby(
        "hydroclimate_group"
    ):


        for model, column in model_columns.items():

            values = group_df[column].dropna()


            records.append(

                {

                    "hydroclimate_group":
                        group,

                    "model":
                        model,

                    "n_basins":
                        len(values),

                    "median_NSE_Q":
                        values.median(),

                    "mean_NSE_Q":
                        values.mean(),

                    "q25_NSE_Q":
                        values.quantile(0.25),

                    "q75_NSE_Q":
                        values.quantile(0.75)

                }

            )


    return pd.DataFrame(records)



# ============================================================================
# Delta NSE_Q statistics
# ============================================================================

def calculate_delta_nse_q(df):


    comparison_columns = {

        "Hard-MTL-Q minus STL-Q":
            "Delta_NSE_HardMTL_minus_STLQ",

        "CGC-Q minus STL-Q":
            "Delta_NSE_CGC_minus_STLQ",

    }


    records = []


    for group, group_df in df.groupby(
        "hydroclimate_group"
    ):


        for name, column in comparison_columns.items():

            values = group_df[column].dropna()


            records.append(

                {

                    "hydroclimate_group":
                        group,

                    "comparison":
                        name,

                    "n_basins":
                        len(values),

                    "median_Delta_NSE_Q":
                        values.median(),

                    "mean_Delta_NSE_Q":
                        values.mean(),

                    "q25_Delta_NSE_Q":
                        values.quantile(0.25),

                    "q75_Delta_NSE_Q":
                        values.quantile(0.75),

                    "positive_transfer_rate":
                        np.mean(values > 0),

                    "negative_transfer_rate":
                        np.mean(values < 0)

                }

            )


    return pd.DataFrame(records)



# ============================================================================
# Plotting
# ============================================================================

def plot_delta_nse_boxplot(df, output):


    groups = [
        "Wet",
        "Dry",
        "Snow"
    ]


    values = []

    labels = []


    for group in groups:


        data = df.loc[

            df["hydroclimate_group"] == group,

            "Delta_NSE_CGC_minus_STLQ"

        ].dropna()


        if len(data) > 0:

            values.append(
                data.values
            )

            labels.append(
                group
            )


    plt.figure(
        figsize=(6,4)
    )


    plt.boxplot(

        values,

        tick_labels=labels,

        showfliers=False

    )


    plt.axhline(
        0,
        linestyle="--"
    )


    plt.ylabel(
        r"$\Delta NSE_Q$ (CGC-STL)"
    )


    plt.xlabel(
        "Hydroclimatic group"
    )


    plt.tight_layout()


    plt.savefig(

        output /
        "fig_pub_q_delta_nse_boxplot.png",

        dpi=300

    )


    plt.close()



def plot_absolute_nse(summary, output):


    pivot = summary.pivot(

        index="hydroclimate_group",

        columns="model",

        values="median_NSE_Q"

    )


    ax = pivot.plot(

        kind="bar",

        figsize=(6,4)

    )


    ax.set_xlabel(
        "Hydroclimatic group"
    )


    ax.set_ylabel(
        "Median NSE_Q"
    )


    plt.tight_layout()


    plt.savefig(

        output /
        "fig_pub_q_absolute_nse.png",

        dpi=300

    )


    plt.close()



def plot_positive_transfer(summary, output):


    data = summary.loc[

        summary["comparison"]
        ==
        "CGC-Q minus STL-Q"

    ]


    plt.figure(
        figsize=(5,4)
    )


    plt.bar(

        data["hydroclimate_group"],

        data["positive_transfer_rate"]

    )


    plt.ylim(
        0,
        1
    )


    plt.ylabel(
        "Positive transfer rate"
    )


    plt.xlabel(
        "Hydroclimatic group"
    )


    plt.tight_layout()


    plt.savefig(

        output /
        "fig_pub_q_positive_transfer_rate.png",

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



    absolute_summary = (
        calculate_absolute_nse_q(df)
    )


    delta_summary = (
        calculate_delta_nse_q(df)
    )



    absolute_summary.to_csv(

        output /
        "pub_q_absolute_nse_group_summary.csv",

        index=False

    )


    delta_summary.to_csv(

        output /
        "pub_q_hydroclimate_group_summary.csv",

        index=False

    )


    logger.info(
        "Statistics saved."
    )


    plot_delta_nse_boxplot(
        df,
        output
    )


    plot_absolute_nse(
        absolute_summary,
        output
    )


    plot_positive_transfer(
        delta_summary,
        output
    )


    logger.info(
        "PUB Q hydroclimatic analysis completed."
    )



if __name__ == "__main__":

    main()