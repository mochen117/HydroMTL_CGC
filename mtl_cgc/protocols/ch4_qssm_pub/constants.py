"""Constants and enumerations for Chapter 4 Experiment B spatial PUB."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PUBScenario(str, Enum):
    """Supported supervision scenarios for the PUB experiment."""

    STL_Q = "stl_q"
    HPS_TARGET_SSM = "hps_target_ssm"
    CGC_TARGET_SSM = "cgc_target_ssm"
    HPS_SOURCE_ONLY = "hps_source_only"
    CGC_SOURCE_ONLY = "cgc_source_only"

    @property
    def architecture(self) -> str:
        """Return the HydroMTL_CGC model registry key."""

        if self is PUBScenario.STL_Q:
            return "stl"
        if self.value.startswith("hps_"):
            return "hps"
        return "cgc"

    @property
    def active_tasks(self) -> tuple[str, ...]:
        """Return tasks configured in the model output heads."""

        if self is PUBScenario.STL_Q:
            return ("streamflow",)
        return ("streamflow", "ssm")

    @property
    def include_target_during_training(self) -> bool:
        """Whether target basins enter training through SSM supervision."""

        return self in {
            PUBScenario.HPS_TARGET_SSM,
            PUBScenario.CGC_TARGET_SSM,
        }

    @property
    def target_ssm_supervision(self) -> bool:
        """Whether target-basin SSM contributes to the training loss."""

        return self.include_target_during_training

    @property
    def group(self) -> str:
        """Return the runner subset label."""

        if self in {
            PUBScenario.HPS_SOURCE_ONLY,
            PUBScenario.CGC_SOURCE_ONLY,
        }:
            return "ablation"
        return "core"


CORE_SCENARIOS: tuple[PUBScenario, ...] = (
    PUBScenario.STL_Q,
    PUBScenario.HPS_TARGET_SSM,
    PUBScenario.CGC_TARGET_SSM,
)

ABLATION_SCENARIOS: tuple[PUBScenario, ...] = (
    PUBScenario.HPS_SOURCE_ONLY,
    PUBScenario.CGC_SOURCE_ONLY,
)

ALL_SCENARIOS: tuple[PUBScenario, ...] = CORE_SCENARIOS + ABLATION_SCENARIOS


@dataclass(frozen=True)
class ProtocolDefaults:
    """Fixed settings for the formal Chapter 4B spatial PUB experiment.

    The train and evaluation target-date periods are intentionally identical.
    Leakage is controlled spatially: target-basin streamflow is fully withheld
    from training, while target-basin SSM may be used as auxiliary supervision.
    """

    protocol_version: str = "ch4b_pub_v3"
    pub_start: str = "2015-04-01"
    pub_end: str = "2021-09-30"
    sequence_length: int = 365
    batch_size: int = 64
    epochs: int = 100
    streamflow_weight: float = 0.5
    ssm_weight: float = 0.5
    n_folds: int = 5
