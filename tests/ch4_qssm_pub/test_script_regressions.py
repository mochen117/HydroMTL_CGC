"""Small regressions for executable Chapter 4B orchestration scripts."""

from __future__ import annotations

import json

from scripts.ch4_qssm_pub import build_pub_folds


def test_build_pub_folds_has_json_runtime_dependency() -> None:
    assert build_pub_folds.json is json
    assert build_pub_folds.json.dumps({"ok": True}) == '{"ok": true}'
