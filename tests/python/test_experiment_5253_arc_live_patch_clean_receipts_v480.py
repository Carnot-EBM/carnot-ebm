"""Tests for Exp 5253 ARC live patch clean receipts.

Spec refs: REQ-REPORT-5253,
SCENARIO-REPORT-5253-CLEAN-NO-BANK-RETIRE,
SCENARIO-REPORT-5253-SOLVE-CREDIT-GATE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5253_arc_live_patch_clean_receipts_v480 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _value(artifact: dict[str, Any], field: str) -> Any:
    return artifact[field]["value"]


def _registry(*, total: int = 69, target_prior: int = 0) -> dict[str, Any]:
    games = {"lp85": 5, "tu93": 5}
    if target_prior:
        games[mod.DEFAULT_TARGET_GAME] = target_prior
    return {
        "present": True,
        "path": mod.REGISTRY_RELATIVE_PATH,
        "reproducible_total_levels": total,
        "games": games,
    }


def _attempt(
    *,
    claimed_level: int = 0,
    reproduced: bool = False,
    registry_validation_passed: bool = False,
    solution_labels: list[str] | None = None,
    solve_provenance: str = mod.SOLVE_PROVENANCE_LIVE,
    forbidden_methods: dict[str, bool] | None = None,
) -> dict[str, Any]:
    return {
        "attempt_id": f"exp5253_{mod.DEFAULT_TARGET_GAME}_seed_{mod.RANDOM_SEED}",
        "target_game": mod.DEFAULT_TARGET_GAME,
        "target_level": 1,
        "prior_reproduced_level": 0,
        "random_seed": mod.RANDOM_SEED,
        "budget": mod.DEFAULT_BUDGET,
        "policy": "arc_competition_agent._recommend_live_approach",
        "solve_provenance": solve_provenance,
        "live_agent_patch_enabled": True,
        "runtime_self_discovery_attempted": True,
        "solution_labels": solution_labels or [],
        "reproduction_gate": {
            "claimed_level": claimed_level,
            "reproduced": reproduced,
            "registry_validation_passed": registry_validation_passed,
            "reached_level": claimed_level if reproduced else 0,
        },
        "forbidden_methods": forbidden_methods
        or {
            "read_hidden_game_source": False,
            "offline_ground_truth_bfs": False,
            "hand_per_game_adapter": False,
            "outer_loop_reverse_engineering": False,
        },
        "provenance_route_receipts": [
            {
                "route": "arc_competition_agent._recommend_live_approach",
                "reached_exp5240_guard": True,
                "guard_enabled": True,
                "failure_mode_targeted": "provenance_routing",
            }
        ],
        "approach_recommendation": {
            "typed_memory_provenance_guard": {
                "enabled": True,
                "failure_mode_targeted": "provenance_routing",
                "blocked_arc_consumer_actions": ["quarantine_gap4_candidate_pool_until_positive_validation"],
            }
        },
    }


def test_req_report_5253_spec_declares_clean_receipt_contract() -> None:
    """REQ-REPORT-5253: OpenSpec anchors the clean receipt and retirement schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5253") : spec.index("### REQ-REPORT-5245")]

    for marker in (
        "REQ-REPORT-5253",
        "SCENARIO-REPORT-5253-CLEAN-NO-BANK-RETIRE",
        "SCENARIO-REPORT-5253-SOLVE-CREDIT-GATE",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "retire_current_provenance_patch",
        "duplicate_solve_claimed.value` SHALL be false",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5253_clean_no_bank_retires_current_patch() -> None:
    """SCENARIO-REPORT-5253-CLEAN-NO-BANK-RETIRE: zero delta has clean receipts."""

    artifact = mod.build_artifact(
        registry_summary=_registry(),
        live_attempt=_attempt(),
        duration_s=0.02,
        attempt_log_path=mod.ATTEMPT_LOG_RELATIVE_PATH,
        input_checksum="sha256:" + "1" * 64,
        output_checksum="sha256:" + "2" * 64,
        tests_run=[],
    )

    mod.validate_artifact(artifact)
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert "level_delta=0" in _value(artifact, "honest_verdict")
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert _value(artifact, "solve_provenance") == mod.SOLVE_PROVENANCE_LIVE
    assert _value(artifact, "registry_precheck")["target_already_reproduced"] is False
    assert _value(artifact, "level_delta") == 0
    assert _value(artifact, "levels_reproduced") == []
    assert _value(artifact, "duplicate_solve_claimed") is False
    assert _value(artifact, "retire_current_provenance_patch") is True
    assert _value(artifact, "attempt_log_path") == str(mod.ATTEMPT_LOG_RELATIVE_PATH)
    assert artifact["forbidden_methods"] == {
        "read_hidden_game_source": False,
        "offline_ground_truth_bfs": False,
        "hand_per_game_adapter": False,
        "outer_loop_reverse_engineering": False,
    }


def test_scenario_report_5253_solve_credit_gate_blocks_duplicates_and_forbidden_methods() -> None:
    """SCENARIO-REPORT-5253-SOLVE-CREDIT-GATE: only clean live solves can add delta."""

    accepted = mod.build_artifact(
        registry_summary=_registry(),
        live_attempt=_attempt(
            claimed_level=1,
            reproduced=True,
            registry_validation_passed=True,
            solution_labels=['{"action": 1, "data": null}'],
        ),
        duration_s=0.02,
        attempt_log_path=mod.ATTEMPT_LOG_RELATIVE_PATH,
        input_checksum="sha256:" + "3" * 64,
        output_checksum="sha256:" + "4" * 64,
        tests_run=[],
    )
    assert _value(accepted, "level_delta") == 1
    assert _value(accepted, "levels_reproduced") == [mod.DEFAULT_TARGET_GAME]
    assert _value(accepted, "retire_current_provenance_patch") is False
    mod.validate_artifact(accepted)

    duplicate = mod.build_artifact(
        registry_summary=_registry(target_prior=1),
        live_attempt=_attempt(
            claimed_level=1,
            reproduced=True,
            registry_validation_passed=True,
            solution_labels=['{"action": 1, "data": null}'],
        ),
        duration_s=0.02,
        attempt_log_path=mod.ATTEMPT_LOG_RELATIVE_PATH,
        input_checksum="sha256:" + "5" * 64,
        output_checksum="sha256:" + "6" * 64,
        tests_run=[],
    )
    assert _value(duplicate, "registry_precheck")["target_already_reproduced"] is True
    assert _value(duplicate, "level_delta") == 0
    assert _value(duplicate, "duplicate_solve_claimed") is False
    assert _value(duplicate, "retire_current_provenance_patch") is True
    mod.validate_artifact(duplicate)

    forbidden = mod.build_artifact(
        registry_summary=_registry(),
        live_attempt=_attempt(
            claimed_level=1,
            reproduced=True,
            registry_validation_passed=True,
            solution_labels=['{"action": 1, "data": null}'],
            forbidden_methods={
                "read_hidden_game_source": False,
                "offline_ground_truth_bfs": True,
                "hand_per_game_adapter": False,
                "outer_loop_reverse_engineering": False,
            },
        ),
        duration_s=0.02,
        attempt_log_path=mod.ATTEMPT_LOG_RELATIVE_PATH,
        input_checksum="sha256:" + "7" * 64,
        output_checksum="sha256:" + "8" * 64,
        tests_run=[],
    )
    assert _value(forbidden, "honest_verdict").startswith("blocked_")
    assert _value(forbidden, "level_delta") == 0
    with pytest.raises(ValueError, match="forbidden_methods"):
        mod.validate_artifact(forbidden)


def test_req_report_5253_writer_records_attempt_log_and_checksums(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-5253: writer emits stable JSON plus attempt-log receipts."""

    monkeypatch.setattr(mod, "load_registry_summary", lambda _root: _registry())
    monkeypatch.setattr(mod, "run_live_agent_patch_attempt", lambda **_kwargs: _attempt())
    output = tmp_path / mod.RESULT_RELATIVE_PATH
    attempt_log = tmp_path / mod.ATTEMPT_LOG_RELATIVE_PATH

    artifact = mod.write_artifact(
        root=tmp_path,
        output_path=output,
        attempt_log_path=attempt_log,
        tests_run=[{"command": "pytest fixture", "outcome": "PASS"}],
    )

    written = json.loads(output.read_text(encoding="utf-8"))
    log_rows = [
        json.loads(line)
        for line in attempt_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert written == artifact
    assert log_rows and log_rows[0]["attempt_id"].startswith("exp5253_")
    assert _value(artifact, "attempt_log_path") == str(attempt_log.relative_to(tmp_path))
    assert _value(artifact, "input_checksum").startswith("sha256:")
    assert _value(artifact, "output_checksum").startswith("sha256:")
    mod.validate_artifact(artifact)


def test_req_report_5253_retirement_manifest_update_is_scoped(tmp_path: Path) -> None:
    """REQ-REPORT-5253: zero-delta retirement touches only the provenance patch scope."""

    manifest = tmp_path / "ops" / "exclusion_manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("retired_extras: []\n", encoding="utf-8")
    artifact = mod.build_artifact(
        registry_summary=_registry(),
        live_attempt=_attempt(),
        duration_s=0.02,
        attempt_log_path=mod.ATTEMPT_LOG_RELATIVE_PATH,
        input_checksum="sha256:" + "9" * 64,
        output_checksum="sha256:" + "a" * 64,
        tests_run=[],
    )

    changed = mod.ensure_retirement_manifest_entry(manifest, artifact)
    unchanged = mod.ensure_retirement_manifest_entry(manifest, artifact)
    text = manifest.read_text(encoding="utf-8")

    assert changed is True
    assert unchanged is False
    assert mod.RETIREMENT_SCOPE_ID in text
    assert "experiment_5240_arc_rubric_to_patch_synthesis_v479" in text


def test_req_report_5253_validation_rejects_schema_breaks() -> None:
    """REQ-REPORT-5253: malformed receipts fail before being trusted."""

    artifact = mod.build_artifact(
        registry_summary=_registry(),
        live_attempt=_attempt(),
        duration_s=0.02,
        attempt_log_path=mod.ATTEMPT_LOG_RELATIVE_PATH,
        input_checksum="sha256:" + "b" * 64,
        output_checksum="sha256:" + "c" * 64,
        tests_run=[],
    )

    missing = dict(artifact)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)
    with pytest.raises(ValueError, match="principle-wrapped"):
        mod.validate_artifact(artifact | {"level_delta": 0})
    with pytest.raises(ValueError, match="level_delta"):
        mod.validate_artifact(
            artifact | {"level_delta": {"value": True, "principle": mod.FIELD_PRINCIPLES["level_delta"]}}
        )
    with pytest.raises(ValueError, match="duplicate_solve_claimed"):
        mod.validate_artifact(
            artifact
            | {
                "duplicate_solve_claimed": {
                    "value": True,
                    "principle": mod.FIELD_PRINCIPLES["duplicate_solve_claimed"],
                }
            }
        )
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(
            artifact
            | {"input_checksum": {"value": "bad", "principle": mod.FIELD_PRINCIPLES["input_checksum"]}}
        )


def test_req_report_5253_repository_artifact_is_valid_when_written() -> None:
    """REQ-REPORT-5253: checked-in artifact remains valid after the live receipt run."""

    if not RESULT_PATH.exists():
        pytest.skip("Exp5253 artifact not written yet")
    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    mod.validate_artifact(artifact)
    assert _value(artifact, "duplicate_solve_claimed") is False
