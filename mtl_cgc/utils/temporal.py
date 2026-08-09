# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Temporal utilities for split validation and N-to-1 sequence
# alignment in HydroMTL.
# ==============================================================================

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import pandas as pd


_MISSING_PERIOD_STRINGS = {"", "none", "null"}


def is_missing_period(period: Any) -> bool:
    """Return whether a temporal period is absent from the configuration."""
    if period is None:
        return True
    if isinstance(period, str):
        return period.strip().lower() in _MISSING_PERIOD_STRINGS
    if isinstance(period, (list, tuple)):
        return len(period) == 0
    return False


def normalize_period(period: Sequence[Any], name: str = "period") -> List[str]:
    """
    Validate and normalize an inclusive two-date period.

    Parameters
    ----------
    period:
        Two-element sequence containing the inclusive start and end dates.
    name:
        Human-readable period name used in validation errors.
    """
    if is_missing_period(period):
        raise ValueError(f"{name} is missing.")
    if len(period) != 2:
        raise ValueError(
            f"{name} must contain exactly two dates [start, end], got: {period}."
        )

    start = pd.to_datetime(period[0])
    end = pd.to_datetime(period[1])
    if start > end:
        raise ValueError(
            f"Invalid {name}: start date {start.date()} is after end date {end.date()}."
        )

    return [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")]


def period_to_list(period: Any) -> Optional[List[str]]:
    """Convert an optional temporal period to a normalized printable list."""
    if is_missing_period(period):
        return None
    return normalize_period(period)


def first_available_period(config_data: Any, *names: str) -> Any:
    """Return the first non-empty period from a mapping or config object."""
    for name in names:
        if isinstance(config_data, dict):
            value = config_data.get(name)
        else:
            value = getattr(config_data, name, None)

        if not is_missing_period(value):
            return value

    return None


def get_target_name(target: Any) -> str:
    """Return a lowercase target name from a mapping, object, or string."""
    if isinstance(target, dict):
        value = target.get("name", target)
    else:
        value = getattr(target, "name", target)
    return str(value).strip().lower()


def expand_period_for_sequence(
    target_period: Sequence[Any],
    sequence_length: int,
) -> List[str]:
    """
    Add the historical context required by an N-to-1 sequence model.

    The configured period denotes target dates. For a sequence length ``rho``,
    the raw input slice begins ``rho - 1`` days before the first target date,
    while the configured target end date remains unchanged.
    """
    normalized = normalize_period(target_period, name="target_period")
    rho = int(sequence_length)
    if rho <= 0:
        raise ValueError(f"sequence_length must be positive, got {rho}.")

    target_start = pd.to_datetime(normalized[0])
    target_end = pd.to_datetime(normalized[1])
    context_start = target_start - pd.Timedelta(days=rho - 1)

    return [
        context_start.strftime("%Y-%m-%d"),
        target_end.strftime("%Y-%m-%d"),
    ]


def count_inclusive_days(period: Sequence[Any]) -> int:
    """Return the number of daily target steps in an inclusive period."""
    start_str, end_str = normalize_period(period)
    start = pd.to_datetime(start_str)
    end = pd.to_datetime(end_str)
    return int((end - start).days) + 1


def build_prediction_dates(
    start_date: Any,
    sequence_length: int,
    num_time_steps: int,
) -> pd.DatetimeIndex:
    """
    Build target dates for context-expanded N-to-1 samples.

    Parameters
    ----------
    start_date:
        First configured target date. This is not the beginning of the
        historical input context.
    sequence_length:
        Number of daily input steps in each N-to-1 sample. The value is
        validated here, while historical context expansion is handled by the
        DataLoader.
    num_time_steps:
        Number of generated target samples.

    Returns
    -------
    pandas.DatetimeIndex
        Consecutive target dates beginning at the configured target-period
        start date.
    """
    rho = int(sequence_length)
    steps = int(num_time_steps)

    if rho <= 0:
        raise ValueError(
            f"sequence_length must be positive, got {rho}."
        )

    if steps <= 0:
        raise ValueError(
            f"num_time_steps must be positive, got {steps}."
        )

    return pd.date_range(
        start=pd.to_datetime(start_date),
        periods=steps,
        freq="D",
    )
