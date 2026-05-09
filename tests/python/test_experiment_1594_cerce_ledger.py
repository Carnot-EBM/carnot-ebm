"""Tests for Exp 1594 CerCE certificate ledger policy promotion.

Spec: REQ-LEARN-1594, SCENARIO-LEARN-1594, SCENARIO-LEARN-1595.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.pipeline.fr11_event_bus import FR11EventBus, ViolationEvent
from carnot.training import cerce_certificate_ledger as mod


def _event(constraint_type: str = "carry_check") -> ViolationEvent:
    return ViolationEvent(
        query_id="daily_eval:policy-a",
        step_index=2,
        energy_score=0.71,
        probe_confidence=0.93,
        constraint_type=constraint_type,
        question_domain="arithmetic",
        timestamp="2026-05-09T00:00:00Z",
    )


def _promotion_row(
    policy_update_id: str = "daily_eval:policy-a",
    *,
    promoted_false_accept: bool = False,
    false_accept_delta: int = 0,
    soundness_mistakes: int = 0,
) -> dict[str, object]:
    return {
        "row_type": "policy_promotion_evaluation",
        "policy_update_id": policy_update_id,
        "contract_case_id": "grammar_certificate:case-001:schema_only:1",
        "source_family": "grammar_certificate",
        "baseline_false_accept": False,
        "promoted_false_accept": promoted_false_accept,
        "false_accept_delta": false_accept_delta,
        "soundness_mistakes": soundness_mistakes,
        "runtime_contract_validation": {
            "promoted": {
                "expected_label": False,
                "proposed_final_deterministic_accept": promoted_false_accept,
                "false_accept": promoted_false_accept,
            }
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_req_learn_1594_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1594-1/6: bootstrap artifact exposes the ledger contract."""

    output = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(
        output,
        project_root=tmp_path,
        run_date="20260509",
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "in_progress"
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["cerce_ledger_ready"] is False
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    mod.validate_artifact(artifact)


def test_req_learn_1594_fr11_event_bus_hook_records_violation_counts() -> None:
    """REQ-LEARN-1594-3: FR-11 ViolationEvent delivery increments ledger counts."""

    bus = FR11EventBus()
    ledger = mod.CerCECertificateLedger(run_date="20260509")
    mod.attach_fr11_event_bus(bus, ledger, policy_update_id="daily_eval:policy-a")

    bus.publish(_event("carry_check"))
    bus.publish(_event("sign_check"))

    assert bus.events_published == 2
    assert bus.events_acked == 2
    assert ledger.fr11_events_recorded == 2
    assert ledger.violation_counts_by_type() == {"carry_check": 1, "sign_check": 1}
    assert ledger.violation_counts_by_policy() == {"daily_eval:policy-a": 2}


def test_scenario_learn_1594_safe_rows_build_ready_certificate(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1594: safe promotion rows produce non-forgetting certificates."""

    ledger = mod.CerCECertificateLedger(run_date="20260509")
    mod.attach_fr11_event_bus(FR11EventBus(), ledger, policy_update_id="daily_eval:policy-a")
    ledger.on_fr11_violation(_event(), policy_update_id="daily_eval:policy-a")
    assert mod.ingest_promotion_rows(ledger, [_promotion_row()]) == 1

    artifact = mod.build_artifact(ledger, project_root=tmp_path)

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["cerce_ledger_ready"] is True
    assert artifact["policy_certificates_evaluated"] == 1
    assert artifact["constraint_violation_records"] == 1
    assert artifact["fr11_events_recorded"] == 1
    assert artifact["accepted_violation_count"] == 0
    assert artifact["false_accept_delta"] == 0
    assert artifact["nonforgetting_certificate_rate"] == 1.0
    assert artifact["promotion_safe_policy_updates"] == ["daily_eval:policy-a"]
    assert artifact["blocked_policy_updates"] == []
    assert len(artifact["ledger_rows"][0]["certificate_id"]) == 64


def test_scenario_learn_1595_accepted_violation_blocks_promotion(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1595: accepted violations prevent CerCE promotion safety."""

    ledger = mod.CerCECertificateLedger(run_date="20260509")
    mod.ingest_promotion_rows(
        ledger,
        [
            _promotion_row(
                promoted_false_accept=True,
                false_accept_delta=1,
                soundness_mistakes=1,
            )
        ],
    )

    artifact = mod.build_artifact(ledger, project_root=tmp_path)

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["cerce_ledger_ready"] is False
    assert artifact["accepted_violation_count"] == 1
    assert artifact["false_accept_delta"] == 1
    assert artifact["nonforgetting_certificate_rate"] == 0.0
    assert artifact["promotion_safe_policy_updates"] == []
    assert artifact["blocked_policy_updates"] == ["daily_eval:policy-a"]
    assert "accepted_constraint_violation" in artifact["blockers"]


def test_req_learn_1594_run_writes_stable_result_from_manifest(tmp_path: Path) -> None:
    """REQ-LEARN-1594-2/4/5/6: runner writes deterministic terminal artifact."""

    manifest = tmp_path / "fr11_live_policy_promotion_1524.jsonl"
    output = tmp_path / mod.OUTPUT_FILE
    _write_jsonl(manifest, [_promotion_row(), {"row_type": "summary"}])

    first = mod.run_experiment(
        project_root=tmp_path,
        promotion_manifest_path=manifest,
        output_path=output,
        run_date="20260509",
        tests_run=["focused pytest"],
    )
    second = mod.run_experiment(
        project_root=tmp_path,
        promotion_manifest_path=manifest,
        output_path=output,
        run_date="20260509",
        tests_run=["focused pytest"],
    )

    mod.validate_artifact(first)
    assert json.loads(output.read_text(encoding="utf-8")) == second
    assert first["ledger_rows"] == second["ledger_rows"]
    assert first["tests_run"] == ["focused pytest"]
    assert first["honest_verdict"] == "complete: cerce_certificate_ledger_ready"


def test_req_learn_1594_input_normalization_and_blocker_edges(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-LEARN-1594-2/3/6: edge inputs normalize deterministically."""

    ledger = mod.CerCECertificateLedger(run_date="20260509")
    rows = [
        {"row_type": "policy_promotion_evaluation"},
        {
            "row_type": "policy_promotion_evaluation",
            "source_event_id": "policy:missing-runtime-validation",
            "contract_case_id": "temporal:case-002",
        },
        {
            "row_type": "policy_promotion_evaluation",
            "source_event_id": "policy:no-promoted-validation",
            "prompt_or_case_id": "plain-case",
            "runtime_contract_validation": {},
        },
    ]

    assert mod.ingest_promotion_rows(ledger, rows) == 2
    assert [row["constraint_type"] for row in ledger.ledger_rows()] == [
        "temporal",
        "runtime_contract",
    ]

    empty = mod.build_artifact(mod.CerCECertificateLedger(run_date="20260509"))
    assert empty["status"] == "blocked"
    assert "no_policy_certificates" in empty["blockers"]

    missing = mod.run_experiment(
        project_root=tmp_path,
        promotion_manifest_path=Path("missing.jsonl"),
        output_path=Path("missing-output.json"),
        run_date="20260509",
    )
    assert "missing_promotion_manifest" in missing["blockers"]

    blank_manifest = tmp_path / "blank.jsonl"
    blank_manifest.write_text("\n" + json.dumps(_promotion_row()) + "\n", encoding="utf-8")
    cli_output = tmp_path / "cli-output.json"
    assert (
        mod.main(
            [
                "--project-root",
                str(tmp_path),
                "--promotion-manifest",
                str(blank_manifest),
                "--output",
                str(cli_output),
            ]
        )
        == 0
    )
    assert "complete: cerce_certificate_ledger_ready" in capsys.readouterr().out
    assert mod._display_path(tmp_path / "outside.jsonl", project_root=tmp_path / "root") == str(
        tmp_path / "outside.jsonl"
    )


def test_req_learn_1594_validation_rejects_malformed_artifacts(tmp_path: Path) -> None:
    """REQ-LEARN-1594-6: validation stays strict around the ready gate."""

    ledger = mod.CerCECertificateLedger(run_date="20260509")
    mod.ingest_promotion_rows(ledger, [_promotion_row()])
    valid = mod.build_artifact(ledger, project_root=tmp_path)

    missing_status = dict(valid)
    del missing_status["status"]
    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact(missing_status)

    bad_schema = dict(valid, schema="wrong")
    with pytest.raises(AssertionError, match="unsupported schema"):
        mod.validate_artifact(bad_schema)

    bad_status = dict(valid, status="unknown")
    with pytest.raises(AssertionError, match="unsupported status"):
        mod.validate_artifact(bad_status)

    bad_rate = dict(valid, nonforgetting_certificate_rate=1.5)
    with pytest.raises(AssertionError, match="between 0 and 1"):
        mod.validate_artifact(bad_rate)

    bad_count = dict(valid, constraint_violation_records=99)
    with pytest.raises(AssertionError, match="must match ledger_rows"):
        mod.validate_artifact(bad_count)

    ready_bad_status = dict(valid, status="blocked")
    with pytest.raises(AssertionError, match="complete status"):
        mod.validate_artifact(ready_bad_status)

    ready_with_blocker = dict(valid, blockers=["unexpected"])
    with pytest.raises(AssertionError, match="cannot have blockers"):
        mod.validate_artifact(ready_with_blocker)

    ready_with_accepted = dict(valid, accepted_violation_count=1)
    with pytest.raises(AssertionError, match="zero accepted violations"):
        mod.validate_artifact(ready_with_accepted)

    ready_with_false_accept = dict(valid, false_accept_delta=1)
    with pytest.raises(AssertionError, match="cannot increase false accepts"):
        mod.validate_artifact(ready_with_false_accept)

    ready_with_low_rate = dict(valid, nonforgetting_certificate_rate=0.5)
    with pytest.raises(AssertionError, match="rate of 1.0"):
        mod.validate_artifact(ready_with_low_rate)
