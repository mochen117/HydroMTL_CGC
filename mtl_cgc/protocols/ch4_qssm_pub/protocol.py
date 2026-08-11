"""Typed protocol definition and leakage-safety validation for Chapter 4B PUB."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .constants import PUBScenario
from .io_utils import read_basin_ids, resolve_project_path


@dataclass(frozen=True)
class PUBProtocol:
    """Resolved fold-level PUB protocol for one model configuration."""

    protocol_version: str
    fold_id: int
    scenario: PUBScenario
    source_basin_file: Path
    target_basin_file: Path
    scaler_fit_scope: str
    test_basin_scope: str
    target_streamflow_supervision: bool
    target_ssm_supervision: bool
    evaluation_task: str
    same_period_spatial_cv: bool

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        project_root: Path,
    ) -> "PUBProtocol":
        """Build and validate a protocol from a generated YAML configuration."""

        pub = config.get("pub")
        if not isinstance(pub, dict) or not bool(pub.get("enabled", False)):
            raise ValueError("The configuration does not enable the PUB protocol.")

        scenario = PUBScenario(str(pub["scenario"]))
        supervision = pub.get("supervision", {})
        target_supervision = supervision.get("target", {})

        protocol = cls(
            protocol_version=str(pub["protocol_version"]),
            fold_id=int(pub["fold_id"]),
            scenario=scenario,
            source_basin_file=resolve_project_path(
                pub["source_basin_file"], project_root
            ),
            target_basin_file=resolve_project_path(
                pub["target_basin_file"], project_root
            ),
            scaler_fit_scope=str(pub.get("scaler_fit_scope", "")),
            test_basin_scope=str(pub.get("test_basin_scope", "")),
            target_streamflow_supervision=bool(
                target_supervision.get("streamflow", False)
            ),
            target_ssm_supervision=bool(target_supervision.get("ssm", False)),
            evaluation_task=str(pub.get("evaluation_task", "streamflow")).lower(),
            same_period_spatial_cv=bool(pub.get("same_period_spatial_cv", False)),
        )
        protocol.validate(config=config)
        return protocol

    @property
    def source_basins(self) -> list[str]:
        """Return source-basin identifiers."""

        return read_basin_ids(self.source_basin_file)

    @property
    def target_basins(self) -> list[str]:
        """Return target-basin identifiers."""

        return read_basin_ids(self.target_basin_file)

    @property
    def effective_training_basins(self) -> list[str]:
        """Return the basin set that must be loaded by the training DataLoader."""

        if self.scenario.include_target_during_training:
            return sorted(set(self.source_basins) | set(self.target_basins))
        return sorted(self.source_basins)

    @property
    def masked_streamflow_basins(self) -> list[str]:
        """Return basins whose streamflow labels must be fully masked in training."""

        if self.scenario.include_target_during_training:
            return sorted(self.target_basins)
        return []

    def validate(self, config: dict[str, Any] | None = None) -> None:
        """Fail fast on spatial leakage or inconsistent supervision semantics."""

        if self.fold_id < 1:
            raise ValueError(f"fold_id must be positive, got {self.fold_id}")

        source = set(self.source_basins)
        target = set(self.target_basins)
        if not source or not target:
            raise ValueError("PUB source and target basin sets must both be non-empty.")

        overlap = sorted(source & target)
        if overlap:
            raise ValueError(
                "Source and target basin sets overlap: " + ", ".join(overlap[:10])
            )

        if self.target_streamflow_supervision:
            raise ValueError(
                "Target-basin streamflow supervision is forbidden under PUB."
            )

        expected_target_ssm = self.scenario.target_ssm_supervision
        if self.target_ssm_supervision != expected_target_ssm:
            raise ValueError(
                "Target-SSM supervision does not match scenario semantics: "
                f"scenario={self.scenario.value}, configured={self.target_ssm_supervision}, "
                f"expected={expected_target_ssm}."
            )

        if self.scaler_fit_scope != "source_only":
            raise ValueError("PUB requires scaler_fit_scope='source_only'.")

        if self.test_basin_scope != "target_only":
            raise ValueError("PUB requires test_basin_scope='target_only'.")

        if self.evaluation_task != "streamflow":
            raise ValueError("Formal PUB evaluation must use target-basin streamflow.")

        if not self.same_period_spatial_cv:
            raise ValueError(
                "PUB requires same_period_spatial_cv=true."
            )

        if config is not None:
            data_cfg = config.get("data", {})
            train_period = list(data_cfg.get("train_period") or [])
            test_period = list(data_cfg.get("test_period") or [])
            if train_period != test_period:
                raise ValueError(
                    "PUB requires identical train_period and test_period; "
                    f"got train={train_period}, test={test_period}."
                )
            if not bool(data_cfg.get("spatial_split", False)):
                raise ValueError("PUB requires data.spatial_split=true.")
            if data_cfg.get("val_period") not in (None, [], [None, None]):
                raise ValueError("Formal Chapter 4B PUB uses no temporal validation split.")

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable protocol snapshot."""

        payload = asdict(self)
        payload["scenario"] = self.scenario.value
        payload["source_basin_file"] = str(self.source_basin_file)
        payload["target_basin_file"] = str(self.target_basin_file)
        payload["source_basin_count"] = len(self.source_basins)
        payload["target_basin_count"] = len(self.target_basins)
        payload["effective_training_basin_count"] = len(self.effective_training_basins)
        payload["masked_streamflow_basin_count"] = len(self.masked_streamflow_basins)
        return payload
