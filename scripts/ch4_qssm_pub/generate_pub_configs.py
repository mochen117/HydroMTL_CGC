#!/usr/bin/env python3
"""Generate Chapter 4B spatial PUB configurations.

The generator deliberately starts from the frozen Chapter 4A Q-SSM config so
that architecture, forcing variables, static attributes, scaling, optimizer,
and loss implementation remain consistent across Chapter 4 experiments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mtl_cgc.protocols.ch4_qssm_pub.config_factory import build_pub_config  # noqa: E402
from mtl_cgc.protocols.ch4_qssm_pub.constants import (  # noqa: E402
    ABLATION_SCENARIOS,
    ALL_SCENARIOS,
    CORE_SCENARIOS,
    ProtocolDefaults,
)
from mtl_cgc.protocols.ch4_qssm_pub.io_utils import (  # noqa: E402
    atomic_write_json,
    atomic_write_yaml,
    load_json,
    load_yaml,
    project_relative,
    resolve_project_path,
)
from mtl_cgc.protocols.ch4_qssm_pub.paths import (  # noqa: E402
    CH4A_BASE_CONFIG,
    FOLD_MANIFEST,
    CONFIG_ROOT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=Path,
        default=CH4A_BASE_CONFIG,
        help="Frozen/audited Chapter 4A Q-SSM config used as structural template.",
    )
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=FOLD_MANIFEST,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Config output root. Defaults to mtl_cgc/configs/ch4_qssm_pub/"
            "<profile>."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=["formal", "smoke"],
        default="formal",
        help="Use smoke for isolated one-epoch validation runs.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help=(
            "Fixed initialization seeds. The formal multi-seed workflow may use "
            "multiple pre-declared seeds; numeric values are study-defined unless "
            "explicitly documented by the reference study."
        ),
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--subset",
        choices=["core", "ablation", "all"],
        default="core",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epoch override. Defaults to 1 for smoke and 100 for formal.",
    )
    return parser.parse_args()


def selected_scenarios(subset: str):
    if subset == "core":
        return CORE_SCENARIOS
    if subset == "ablation":
        return ABLATION_SCENARIOS
    return ALL_SCENARIOS


def main() -> None:
    args = parse_args()
    base_config_path = resolve_project_path(args.base_config, PROJECT_ROOT)
    fold_manifest_path = resolve_project_path(args.fold_manifest, PROJECT_ROOT)
    base_config = load_yaml(base_config_path)
    fold_manifest = load_json(fold_manifest_path)
    fold_by_id = {int(item["fold_id"]): item for item in fold_manifest["folds"]}

    epochs = args.epochs if args.epochs is not None else (1 if args.profile == "smoke" else 100)
    defaults = ProtocolDefaults(epochs=epochs)
    scenarios = selected_scenarios(args.subset)
    output_arg = args.output_root or (CONFIG_ROOT / args.profile)
    output_root = resolve_project_path(output_arg, PROJECT_ROOT)

    for seed in args.seeds:
        seed_dir = output_root / f"seed{seed}"
        entries: list[dict[str, object]] = []

        for fold_id in args.folds:
            if fold_id not in fold_by_id:
                raise KeyError(f"Fold {fold_id} is absent from {fold_manifest_path}")

            fold = fold_by_id[fold_id]
            source_file = resolve_project_path(fold["source_basin_file"], PROJECT_ROOT)
            target_file = resolve_project_path(fold["target_basin_file"], PROJECT_ROOT)

            for scenario in scenarios:
                config = build_pub_config(
                    base_config=base_config,
                    scenario=scenario,
                    fold_id=fold_id,
                    seed=seed,
                    source_basin_file=source_file,
                    target_basin_file=target_file,
                    project_root=PROJECT_ROOT,
                    defaults=defaults,
                    run_profile=args.profile,
                )

                config_path = (
                    seed_dir
                    / f"fold{fold_id:02d}"
                    / (
                        f"ch4b_pub_{args.profile}_f{fold_id:02d}_"
                        f"{scenario.value}_seed{seed}.yaml"
                    )
                )
                atomic_write_yaml(config_path, config)

                entries.append(
                    {
                        "config": project_relative(config_path, PROJECT_ROOT),
                        "experiment_name": config["experiment"]["name"],
                        "fold_id": fold_id,
                        "seed": int(seed),
                        "run_profile": args.profile,
                        "scenario": scenario.value,
                        "group": scenario.group,
                        "source_basin_file": project_relative(source_file, PROJECT_ROOT),
                        "target_basin_file": project_relative(target_file, PROJECT_ROOT),
                    }
                )

        manifest = {
            "protocol_version": defaults.protocol_version,
            "seed": int(seed),
            "run_profile": args.profile,
            "epochs": int(defaults.epochs),
            "subset": args.subset,
            "base_config": project_relative(base_config_path, PROJECT_ROOT),
            "fold_manifest": project_relative(fold_manifest_path, PROJECT_ROOT),
            "pub_period": [defaults.pub_start, defaults.pub_end],
            "entries": entries,
            "configs": [entry["config"] for entry in entries],
        }
        manifest_path = seed_dir / (
            f"ch4b_pub_{args.profile}_seed{seed}_manifest.json"
        )
        atomic_write_json(manifest_path, manifest)
        print(f"Generated {len(entries)} configs -> {manifest_path}")


if __name__ == "__main__":
    main()
