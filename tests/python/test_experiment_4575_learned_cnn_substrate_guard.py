"""Tests for Exp 4575 learned-CNN substrate guard.

Spec refs: REQ-ARC-FCP-4575, SCENARIO-ARC-FCP-4575.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4575_learned_cnn_substrate_guard as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def test_req_arc_fcp_4575_spec_declares_learned_cnn_guard() -> None:
    """REQ-ARC-FCP-4575: OpenSpec declares the guard and artifact contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4575" in spec
    assert "SCENARIO-ARC-FCP-4575" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_fcp_4575_artifact_records_regression_asserts(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-4575: artifact records clean CNN and flagged fake LLM."""
    artifact = mod.build_artifact(
        REPO,
        tests_added_pass={
            "passed": True,
            "commands": [
                ".venv/bin/pytest tests/python/test_adversarial_verify_guards.py "
                "tests/python/test_experiment_4575_learned_cnn_substrate_guard.py -q --no-cov"
            ],
        },
    )

    assert artifact["honest_verdict"] == "shipped: learned_cnn_substrate_guard_added"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["cnn_artifact_not_flagged"] is True
    assert artifact["fake_llm_still_flagged"] is True
    assert artifact["guard_mechanism"]["cnn_applied_floor_s"] == 1.0
    assert artifact["guard_mechanism"]["fake_llm_applied_floor_s"] == 60.0
    assert artifact["tests_added_pass"]["passed"] is True
    assert artifact["preconditions_checked"]["adversarial_verify_help_exits_0"] is True
    assert mod.validate_artifact(artifact) == []

    output = mod.write_artifact(tmp_path, artifact=artifact)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_req_arc_fcp_4575_validation_fail_closed() -> None:
    """REQ-ARC-FCP-4575: shipped artifact requires both regression assertions."""
    artifact = mod.build_artifact(REPO, tests_added_pass=True)

    broken = dict(artifact)
    broken["fake_llm_still_flagged"] = False
    broken["reproducibility_checksum"] = mod.checksum_from_artifact(broken)

    assert "fake_llm_still_flagged must be true" in mod.validate_artifact(broken)


def test_req_arc_fcp_4575_defensive_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-FCP-4575: validation and wrappers fail closed."""
    assert mod._duration_flag_kinds({"duration_s": 5.0}) == []  # noqa: SLF001
    assert mod._honest_verdict(  # noqa: SLF001
        preconditions_ok=False,
        cnn_artifact_not_flagged=True,
        fake_llm_still_flagged=True,
        tests_passed=True,
    ) == "complete: learned_cnn_substrate_guard_partial_preconditions"
    assert mod._honest_verdict(  # noqa: SLF001
        preconditions_ok=True,
        cnn_artifact_not_flagged=False,
        fake_llm_still_flagged=True,
        tests_passed=True,
    ) == "complete: learned_cnn_substrate_guard_partial_cnn_overflagged"
    assert mod._honest_verdict(  # noqa: SLF001
        preconditions_ok=True,
        cnn_artifact_not_flagged=True,
        fake_llm_still_flagged=False,
        tests_passed=True,
    ) == "complete: learned_cnn_substrate_guard_partial_fake_llm_not_flagged"
    assert mod._honest_verdict(  # noqa: SLF001
        preconditions_ok=True,
        cnn_artifact_not_flagged=True,
        fake_llm_still_flagged=True,
        tests_passed=False,
    ) == "complete: learned_cnn_substrate_guard_partial_tests_pending"

    class FailedHelp:
        returncode = 1

    real_subprocess_run = mod.subprocess.run
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: FailedHelp())
    assert mod._adversarial_verify_help_exits_0(tmp_path) is False  # noqa: SLF001
    monkeypatch.setattr(mod.subprocess, "run", real_subprocess_run)

    artifact = mod.build_artifact(REPO)
    pending = mod.build_artifact(REPO, tests_added_pass=False)
    assert pending["honest_verdict"] == "complete: learned_cnn_substrate_guard_partial_tests_pending"
    assert pending["tests_added_pass"] == {"passed": False, "commands": []}
    assert mod._sha256_prefixed(artifact["reproducibility_checksum"]) is True  # noqa: SLF001
    assert mod._sha256_prefixed("bad") is False  # noqa: SLF001
    assert mod._sha256_prefixed(42) is False  # noqa: SLF001

    invalid_cases: list[tuple[str, Any, str]] = [
        ("__delete__", None, "missing honest_verdict"),
        ("honest_verdict", "not-terminal", "honest_verdict"),
        ("inference_substrate", "aggregation_from_upstream_artifacts", "inference_substrate"),
        ("guard_mechanism", [], "guard_mechanism"),
        ("cnn_artifact_not_flagged", False, "cnn_artifact_not_flagged"),
        ("fake_llm_still_flagged", False, "fake_llm_still_flagged"),
        ("tests_added_pass", {"passed": False}, "tests_added_pass"),
        ("preconditions_checked", [], "preconditions_checked"),
        ("__help_false__", None, "adversarial_verify_help_exits_0"),
        ("field_principles", [], "field_principles"),
        ("__drop_principle__", None, "missing field principle"),
        ("leaderboard_submission", True, "leaderboard_submission"),
        ("reproducibility_checksum", "bad", "reproducibility_checksum"),
        ("__checksum_mismatch__", None, "reproducibility_checksum mismatch"),
    ]
    for field, value, message in invalid_cases:
        changed = json.loads(json.dumps(artifact))
        if field == "__delete__":
            changed.pop("honest_verdict")
        elif field == "__help_false__":
            changed["preconditions_checked"]["adversarial_verify_help_exits_0"] = False
            changed["reproducibility_checksum"] = mod.checksum_from_artifact(changed)
        elif field == "__drop_principle__":
            changed["field_principles"].pop("honest_verdict")
        elif field == "__checksum_mismatch__":
            changed["reproducibility_checksum"] = "sha256:" + "1" * 64
        else:
            changed[field] = value
        assert any(message in error for error in mod.validate_artifact(changed))

    bad = dict(artifact, cnn_artifact_not_flagged=False)
    bad["reproducibility_checksum"] = mod.checksum_from_artifact(bad)
    with pytest.raises(ValueError, match="cnn_artifact_not_flagged"):
        mod.write_artifact(tmp_path, artifact=bad)

    built = {"sentinel": True}
    calls = []
    monkeypatch.setattr(mod, "build_artifact", lambda root: built)
    monkeypatch.setattr(
        mod,
        "write_artifact",
        lambda root, artifact: calls.append((root, artifact)) or (tmp_path / "out.json"),
    )
    assert mod.run(tmp_path, write=True) is built
    assert calls == [(tmp_path, built)]
    calls.clear()
    assert mod.run(tmp_path, write=False) is built
    assert calls == []
