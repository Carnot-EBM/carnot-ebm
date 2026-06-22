"""Tests for Exp 4599 TAUTOLOGY null-delta guard.

Spec refs: REQ-CAPSTONE-4599, SCENARIO-CAPSTONE-4599,
SCENARIO-CAPSTONE-4599-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

import scripts.adversarial_verify as av


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"
MODULE_PATH = REPO / "python" / "carnot" / "experiment_4599_tautology_null_delta_guard.py"

_SPEC = importlib.util.spec_from_file_location(
    "experiment_4599_tautology_null_delta_guard", MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)


def _tautology_flags(payload: JsonDict) -> list[av.Flag]:
    flags: list[av.Flag] = []
    av.check_tautology(payload, flags)
    return [flag for flag in flags if flag.kind == "TAUTOLOGY"]


def test_req_capstone_4599_spec_declares_null_delta_guard_contract() -> None:
    """REQ-CAPSTONE-4599: OpenSpec declares the reader guard before code."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4599",
        "SCENARIO-CAPSTONE-4599",
        "SCENARIO-CAPSTONE-4599-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4599_declared_null_delta_downgrades_to_warn() -> None:
    """SCENARIO-CAPSTONE-4599: declared null-delta equality is annotated WARN."""

    flags = _tautology_flags(mod.DECLARED_NULL_DELTA_FIXTURE)

    assert not [flag for flag in flags if flag.severity == "critical"]
    warnings = [flag for flag in flags if flag.severity == "warn"]
    assert warnings
    assert any("declared_null_delta" in flag.detail for flag in warnings)


def test_scenario_capstone_4599_undeclared_tautology_stays_critical() -> None:
    """SCENARIO-CAPSTONE-4599: undeclared distinct-metric bit identity is critical."""

    flags = _tautology_flags(mod.UNDECLARED_TAUTOLOGY_FIXTURE)

    critical = [flag for flag in flags if flag.severity == "critical"]
    assert critical
    assert any("Two distinct metrics" in flag.detail for flag in critical)


def test_scenario_capstone_4599_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4599-FIELD-PRINCIPLES: artifact records the guard proof."""

    artifact = mod.build_artifact(
        root=tmp_path,
        preconditions_checked={
            "ok": True,
            "agents_md_read": True,
            "codex_or_opencode_md_read": True,
            "spec_has_req_4599": True,
            "adversarial_verify_help_exits_0": True,
            "research_conductor_modified": False,
        },
    )

    assert artifact["honest_verdict"] == "shipped: tautology_null_delta_guard_added"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["guard_mechanism"]["annotation"] == "declared_null_delta"
    assert artifact["declared_null_delta_downgraded"]["passed"] is True
    assert artifact["undeclared_tautology_still_critical"]["passed"] is True
    assert artifact["tests_added_pass"]["passed"] is True
    assert mod.validate_artifact(artifact) == []

    path = mod.write_artifact(tmp_path, artifact=artifact)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_scenario_capstone_4599_validation_and_error_branches(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4599: malformed artifacts and partial verdicts fail closed."""

    checks = {
        "ok": True,
        "agents_md_read": True,
        "codex_or_opencode_md_read": True,
        "spec_has_req_4599": True,
        "adversarial_verify_help_exits_0": True,
        "research_conductor_modified": False,
    }
    artifact = mod.build_artifact(tmp_path, preconditions_checked=checks)

    assert mod._honest_verdict({"ok": False}, {"passed": True}, {"passed": True}).startswith(
        "complete: tautology_null_delta_guard_partial_preconditions"
    )
    assert mod._honest_verdict(checks, {"passed": False}, {"passed": True}).endswith(
        "declared_not_downgraded"
    )
    assert mod._honest_verdict(checks, {"passed": True}, {"passed": False}).endswith(
        "undeclared_not_critical"
    )

    bad = dict(artifact)
    bad.update(
        {
            "honest_verdict": "pending",
            "inference_substrate": "wrong",
            "guard_mechanism": "wrong",
            "declared_null_delta_downgraded": "wrong",
            "undeclared_tautology_still_critical": {"passed": False},
            "field_principles": "wrong",
            "reproducibility_checksum": "wrong",
        }
    )
    errors = mod.validate_artifact(bad)
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "inference_substrate mismatch" in errors
    assert "guard_mechanism must be object" in errors
    assert "declared_null_delta_downgraded must be object" in errors
    assert "undeclared_tautology_still_critical must pass" in errors
    assert "field_principles missing" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors

    bad_principles = dict(artifact)
    bad_principles["field_principles"] = {}
    bad_principles["reproducibility_checksum"] = "sha256:bad"
    errors = mod.validate_artifact(bad_principles)
    assert "missing field principle for honest_verdict" in errors
    assert "reproducibility_checksum mismatch" in errors

    with pytest.raises(ValueError):
        mod.write_artifact(tmp_path, artifact=bad)


def test_scenario_capstone_4599_live_boundary_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-CAPSTONE-4599: live helper branches are deterministic and guarded."""

    def raise_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise RuntimeError("git unavailable")

    monkeypatch.setattr(mod.subprocess, "run", raise_run)
    assert mod._git_path_modified(REPO, "scripts/research_conductor.py") is False

    class DirtyResult:
        returncode = 1

    monkeypatch.setattr(mod.subprocess, "run", lambda *_args, **_kwargs: DirtyResult())
    assert mod._git_path_modified(REPO, "scripts/research_conductor.py") is True

    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced error"])
    with pytest.raises(ValueError):
        mod.run(REPO, write=False)


def test_scenario_capstone_4599_run_writes_and_clean_git_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-4599: successful run writes the terminal artifact."""

    class CleanResult:
        returncode = 0

    monkeypatch.setattr(mod.subprocess, "run", lambda *_args, **_kwargs: CleanResult())
    assert mod._git_path_modified(REPO, "scripts/research_conductor.py") is False

    artifact = mod.run(tmp_path, write=True)

    assert artifact["result_path"] == mod.RESULT_RELATIVE_PATH
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
