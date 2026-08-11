"""Small tests for the Wet/Dry/Snow metadata convention."""

import pandas as pd


def classify(frame: pd.DataFrame) -> pd.Series:
    group = pd.Series(index=frame.index, dtype="object")
    snow = frame["frac_snow"] > 0.20
    group.loc[snow] = "Snow"
    group.loc[(~snow) & (frame["aridity"] < 1.0)] = "Wet"
    group.loc[(~snow) & (frame["aridity"] >= 1.0)] = "Dry"
    return group


def test_hydroclimate_precedence() -> None:
    frame = pd.DataFrame({"aridity": [0.8, 1.2, 2.0], "frac_snow": [0.1, 0.1, 0.3]})
    assert classify(frame).tolist() == ["Wet", "Dry", "Snow"]
