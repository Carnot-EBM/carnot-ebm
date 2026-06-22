"""Tests for Exp 4587 offline ARC methodology guard.

Spec refs: REQ-ARC-WMTE-4587, SCENARIO-ARC-WMTE-4587.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4587_offline_arc_methodology_guard as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def test_req_arc_wmte_4587_spec_declares_offline_arc_guard() -> None:
    """REQ-ARC-WMTE-4587: OpenSpec declares the guard and artifact contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4587" in spec
    assert "SCENARIO-ARC-WMTE-4587" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4587_artifact_records_regression_asserts(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4587: artifact records offline clean and live warned."""
    artifact = mod.build_artifact(
        REPO,
        tests_added_pass={
            "passed": True,
            "commands": [
                ".venv/bin/pytest tests/python/test_adversarial_verify_guards.py "
                "tests/python/test_experiment_4587_offline_arc_methodology_guard.py "
                "-q --no-cov"
            ],
        },
    )

    assert artifact["honest_verdict"] == "shipped: offline_arc_methodology_guard_added"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["offline_arc_artifact_not_warned"] is True
    assert artifact["real_llm_still_warned"] is True
    assert artifact["guard_mechanism"]["recognized_descriptor"]["kind"] == (
        "offline_arc_methodology_descriptor"
    )
    assert "solver_module" in artifact["guard_mechanism"]["recognized_descriptor"][
        "evidence_fields"
    ]
    assert artifact["tests_added_pass"]["passed"] is True
    assert artifact["preconditions_checked"]["adversarial_verify_help_exits_0"] is True
    assert mod.validate_artifact(artifact) == []

    output = mod.write_artifact(tmp_path, artifact=artifact)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_req_arc_wmte_4587_validation_fail_closed() -> None:
    """REQ-ARC-WMTE-4587: shipped artifact requires both regression assertions."""
    artifact = mod.build_artifact(REPO, tests_added_pass=True)

    broken = dict(artifact)
    broken["real_llm_still_warned"] = False
    broken["reproducibility_checksum"] = mod.checksum_from_artifact(broken)

    assert "real_llm_still_warned must be true" in mod.validate_artifact(broken)


def test_req_arc_wmte_4587_defensive_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-4587: validation and wrappers fail closed."""
    assert mod._methodology_flag_kinds({"duration_s": 5.0}) == []  # noqa: SLF001
    assert mod._honest_verdict(  # noqa: SLF001
        preconditions_ok=False,
        offline_arc_artifact_not_warned=True,
        real_llm_still_warned=True,
        tests_passed=True,
    ) == "complete: offline_arc_methodology_guard_partial_preconditions"
    assert mod._honest_verdict(  # noqa: SLF001
        preconditions_ok=True,
        offline_arc_artifact_not_warned=False,
        real_llm_still_warned=True,
        tests_passed=True,
    ) == "complete: offline_arc_methodology_guard_partial_offline_arc_overwarned"
    assert mod._honest_verdict(  # noqa: SLF001
        preconditions_ok=True,
        offline_arc_artifact_not_warned=True,
        real_llm_still_warned=False,
        tests_passed=True,
    ) == "complete: offline_arc_methodology_guard_partial_live_llm_not_warned"
    assert mod._honest_verdict(  # noqa: SLF001
        preconditions_ok=True,
        offline_arc_artifact_not_warned=True,
        real_llm_still_warned=True,
        tests_passed=False,
    ) == "complete: offline_arc_methodology_guard_partial_tests_pending"

    class FailedHelp:
        returncode = 1

    real_subprocess_run = mod.subprocess.run
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: FailedHelp())
    assert mod._adversarial_verify_help_exits_0(tmp_path) is False  # noqa: SLF001
    monkeypatch.setattr(mod.subprocess, "run", real_subprocess_run)

    artifact = mod.build_artifact(REPO)
    pending = mod.build_artifact(REPO, tests_added_pass=False)
    assert pending["honest_verdict"] == (
        "complete: offline_arc_methodology_guard_partial_tests_pending"
    )
    assert pending["tests_added_pass"] == {"passed": False, "commands": []}
    assert mod._sha256_prefixed(artifact["reproducibility_checksum"]) is True  # noqa: SLF001
    assert mod._sha256_prefixed("bad") is False  # noqa: SLF001
    assert mod._sha256_prefixed(42) is False  # noqa: SLF001

    invalid_cases: list[tuple[str, Any, str]] = [
        ("__delete__", None, "missing honest_verdict"),
        ("honest_verdict", "not-terminal", "honest_verdict"),
        ("inference_substrate", "aggregation_from_upstream_artifacts", "inference_substrate"),
        ("guard_mechanism", [], "guard_mechanism"),
        ("offline_arc_artifact_not_warned", False, "offline_arc_artifact_not_warned"),
        ("real_llm_still_warned", False, "real_llm_still_warned"),
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

    bad = dict(artifact, offline_arc_artifact_not_warned=False)
    bad["reproducibility_checksum"] = mod.checksum_from_artifact(bad)
    with pytest.raises(ValueError, match="offline_arc_artifact_not_warned"):
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
