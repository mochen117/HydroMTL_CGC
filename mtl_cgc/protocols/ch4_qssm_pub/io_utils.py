"""File-system helpers shared by PUB scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def normalize_basin_id(value: Any) -> str:
    """Normalize a CAMELS basin identifier to an eight-character string."""

    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(8)


def read_basin_ids(path: Path) -> list[str]:
    """Read, normalize, and validate a basin-id text file."""

    if not path.exists():
        raise FileNotFoundError(path)

    basin_ids = [
        normalize_basin_id(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not basin_ids:
        raise ValueError(f"Basin list is empty: {path}")

    duplicates = sorted({item for item in basin_ids if basin_ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"Duplicate basin ids in {path}: {duplicates[:10]}")

    return basin_ids


def write_basin_ids(path: Path, basin_ids: Iterable[str]) -> None:
    """Write basin identifiers atomically, one identifier per line."""

    values = [normalize_basin_id(item) for item in basin_ids]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(values) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_yaml(path: Path, payload: Any) -> None:
    """Write YAML atomically while preserving insertion order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path) -> Any:
    """Load a UTF-8 JSON file."""

    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return payload


def resolve_project_path(value: str | Path, project_root: Path) -> Path:
    """Resolve an absolute or project-relative path."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def project_relative(path: Path, project_root: Path) -> str:
    """Return a portable project-relative path when possible."""

    path = path.resolve()
    try:
        return str(path.relative_to(project_root.resolve()))
    except ValueError:
        return str(path)
