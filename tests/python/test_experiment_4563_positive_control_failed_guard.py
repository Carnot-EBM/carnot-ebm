"""Tests for Exp 4563 positive-control-failed null guard.

Spec refs: REQ-CAPSTONE-4563, SCENARIO-CAPSTONE-4563,
SCENARIO-CAPSTONE-4563-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from carnot import experiment_4563_positive_control_failed_guard as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def test_req_capstone_4563_spec_declares_positive_control_guard() -> None:
    """REQ-CAPSTONE-4563: OpenSpec declares the guard and artifact contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4563" in spec
    assert "SCENARIO-CAPSTONE-4563" in spec
    assert "SCENARIO-CAPSTONE-4563-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4563_artifact_records_regression_asserts(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4563: artifact proves trap fires and clean null does not."""
    artifact = mod.build_artifact(
        REPO,
        tests_added_pass={
            "passed": True,
            "commands": [
                ".venv/bin/pytest tests/python/test_adversarial_verify_guards.py "
                "tests/python/test_experiment_4554_capstone_v420.py "
                "tests/python/test_experiment_4563_positive_control_failed_guard.py -q --no-cov"
            ],
        },
    )

    assert artifact["honest_verdict"] == "shipped: positive_control_failed_guard_added"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["trap_exemplar_flagged"] is True
    assert artifact["clean_null_not_flagged"] is True
    assert artifact["guard_mechanism"]["adversarial_verify_guard"] == (
        "FALSE_NEGATIVE_RISK:false_negative_risk_open"
    )
    assert artifact["guard_mechanism"]["capstone_skip_reason"] == "false_negative_risk_open"
    assert artifact["tests_added_pass"]["passed"] is True
    assert artifact["preconditions_checked"]["adversarial_verify_help_exits_0"] is True
    assert mod.validate_artifact(artifact) == []

    output = mod.write_artifact(tmp_path, artifact=artifact)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_req_capstone_4563_validation_fail_closed() -> None:
    """REQ-CAPSTONE-4563: shipped artifact requires both regression assertions."""
    artifact = mod.build_artifact(
        REPO,
        tests_added_pass={"passed": True, "commands": ["fixture"]},
    )

    broken = dict(artifact)
    broken["clean_null_not_flagged"] = False
    broken["reproducibility_checksum"] = mod.checksum_from_artifact(broken)

    assert "clean_null_not_flagged must be true" in mod.validate_artifact(broken)


def test_req_capstone_4563_defensive_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4563: validation and wrapper branches fail closed."""

    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json_object(list_json)  # noqa: SLF001

    assert mod._default_tests_added_pass()["passed"] is True  # noqa: SLF001
    assert mod._honest_verdict(  # noqa: SLF001
        preconditions_ok=False,
        trap_exemplar_flagged=True,
        clean_null_not_flagged=True,
        tests_passed=True,
    ) == "complete: positive_control_failed_guard_partial_preconditions"
    assert mod._honest_verdict(  # noqa: SLF001
        preconditions_ok=True,
        trap_exemplar_flagged=False,
        clean_null_not_flagged=True,
        tests_passed=True,
    ) == "complete: positive_control_failed_guard_partial_trap_not_flagged"
    assert mod._honest_verdict(  # noqa: SLF001
        preconditions_ok=True,
        trap_exemplar_flagged=True,
        clean_null_not_flagged=False,
        tests_passed=True,
    ) == "complete: positive_control_failed_guard_partial_clean_null_overflagged"
    assert mod._honest_verdict(  # noqa: SLF001
        preconditions_ok=True,
        trap_exemplar_flagged=True,
        clean_null_not_flagged=True,
        tests_passed=False,
    ) == "complete: positive_control_failed_guard_partial_tests_pending"

    class FailedHelp:
        returncode = 1

    real_subprocess_run = mod.subprocess.run
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: FailedHelp())
    assert mod._adversarial_verify_help_exits_0(tmp_path) is False  # noqa: SLF001
    monkeypatch.setattr(mod.subprocess, "run", real_subprocess_run)

    artifact = mod.build_artifact(REPO)
    pending = mod.build_artifact(REPO, tests_added_pass=False)
    assert pending["honest_verdict"] == "complete: positive_control_failed_guard_partial_tests_pending"
    assert pending["tests_added_pass"] == {"passed": False, "commands": []}
    assert mod._sha256_prefixed(artifact["reproducibility_checksum"]) is True  # noqa: SLF001
    assert mod._sha256_prefixed("bad") is False  # noqa: SLF001
    assert mod._sha256_prefixed(42) is False  # noqa: SLF001

    invalid_cases: list[tuple[str, Any, str]] = [
        ("__delete__", None, "missing honest_verdict"),
        ("honest_verdict", "not-terminal", "honest_verdict"),
        ("inference_substrate", "aggregation_from_upstream_artifacts", "inference_substrate"),
        ("guard_mechanism", [], "guard_mechanism"),
        ("trap_exemplar_flagged", False, "trap_exemplar_flagged"),
        ("clean_null_not_flagged", False, "clean_null_not_flagged"),
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

    bad = dict(artifact, clean_null_not_flagged=False)
    bad["reproducibility_checksum"] = mod.checksum_from_artifact(bad)
    with pytest.raises(ValueError, match="clean_null_not_flagged"):
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
