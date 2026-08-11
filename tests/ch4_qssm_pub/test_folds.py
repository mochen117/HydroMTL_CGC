"""Tests for deterministic HUC2-balanced PUB folds."""

from mtl_cgc.protocols.ch4_qssm_pub.folds import build_balanced_folds, validate_fold_assignment


def test_each_basin_is_target_exactly_once() -> None:
    basins = [f"{i:08d}" for i in range(1, 13)]
    regions = {basin: str((idx % 3) + 1) for idx, basin in enumerate(basins)}
    assignment = build_balanced_folds(basins, 5, 20260701, regions)
    validate_fold_assignment(assignment, basins, 5)
    assert len(assignment) == len(basins)
    assert assignment["basin_id"].nunique() == len(basins)
