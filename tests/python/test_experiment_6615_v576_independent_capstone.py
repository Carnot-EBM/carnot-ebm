"""Focused tests for the V576 independent capstone.

Spec refs: REQ-REPORT-6615, REQ-REPORT-6615-PRECONDITIONS,
REQ-REPORT-6615-ROWS, REQ-REPORT-6615-GATES,
REQ-REPORT-6615-DECODING, REQ-REPORT-6615-LIVE,
REQ-REPORT-6615-SAMPLER, REQ-REPORT-6615-LEARNING,
REQ-REPORT-6615-VERDICTS, REQ-REPORT-6615-BOUNDARIES,
REQ-REPORT-6615-ATTACKS, REQ-REPORT-6615-ATOMIC,
SCENARIO-REPORT-6615-MISSING-AND-BLOCKED,
SCENARIO-REPORT-6615-GATE-SPELLING,
SCENARIO-REPORT-6615-ROW-REPLAY,
SCENARIO-REPORT-6615-CLOSED-VERDICTS,
SCENARIO-REPORT-6615-CLAIM-BOUNDARIES,
SCENARIO-REPORT-6615-DOCUMENT-RECONCILIATION,
SCENARIO-REPORT-6615-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6615_v576_independent_capstone as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


@pytest.fixture(scope="module")
def artifact() -> dict[str, object]:
    """Build the audit from stored rows without invoking external verifiers."""

    return mod.build_artifact(
        repo_root=REPO,
        run_date="20260826",
        adversarial_reports=mod.expected_adversarial_reports_for_stored_sources(),
        tests_run=[
            {
                "command": "focused-exp6615",
                "exit_code": 0,
                "duration_s": 1.0,
                "scope": "focused",
            }
        ],
        duration_s=2.0,
    )


def test_req_report_6615_spec_declares_capstone_contract() -> None:
    """REQ-REPORT-6615: the capability spec owns the required contract."""

    section = SPEC.read_text(encoding="utf-8").split("REQ-REPORT-6615", 1)[1]
    for marker in (
        "SCENARIO-REPORT-6615-MISSING-AND-BLOCKED",
        "SCENARIO-REPORT-6615-GATE-SPELLING",
        "SCENARIO-REPORT-6615-ROW-REPLAY",
        "SCENARIO-REPORT-6615-CLOSED-VERDICTS",
        "SCENARIO-REPORT-6615-CLAIM-BOUNDARIES",
        "SCENARIO-REPORT-6615-DOCUMENT-RECONCILIATION",
        "SCENARIO-REPORT-6615-ATOMIC",
        mod.INFERENCE_SUBSTRATE,
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in mod.REQUIRED_FIELDS:
        assert f"`{field}`" in section or field in section


def test_scenario_report_6615_missing_and_blocked_are_preserved(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6615-MISSING-AND-BLOCKED: all tasks stay visible."""

    task_rows = [row for row in artifact["per_unit_rows"] if row["row_kind"] == "task"]
    assert len(task_rows) == 11
    assert {row["experiment_number"] for row in task_rows} == set(range(6604, 6615))
    by_number = {row["experiment_number"]: row for row in task_rows}
    assert by_number[6610]["source_state"] == "missing"
    assert by_number[6610]["verdict_class"] == "blocked"
    assert by_number[6606]["verdict_class"] == "blocked"
    assert by_number[6606]["gate_check_summary"]
    assert by_number[6614]["verdict_class"] == "disqualified"
    # REQ-CONDUCTOR-VERDICT-3: the finished capstone declares null, not partial.
    assert artifact["verdict_class"] == "null"


def test_scenario_report_6615_gate_contracts_are_exact(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6615-GATE-SPELLING: owner spelling is exact."""

    rows = artifact["roadmap_gate_contract_rows"]
    assert len(rows) == 8
    assert all(row["upstream_exists"] for row in rows)
    assert all(row["owner_declares_identical_field"] for row in rows)
    assert all(row["contract_valid"] for row in rows)
    assert any(not row["observed_gate_passed"] for row in rows)

    tasks = mod.load_v576_tasks(REPO)
    sources = mod.load_source_artifacts(REPO, tasks)
    drifted = deepcopy(tasks)
    owner = next(row for row in drifted if row["id"].startswith("exp6604-"))
    owner["prompt"] = owner["prompt"].replace(
        "headroom_fixture_ready_score:", "headroom_fixture_ready_scores:"
    )
    drift_rows = mod.audit_roadmap_gate_contracts(drifted, sources)
    assert not all(row["contract_valid"] for row in drift_rows)


def test_scenario_report_6615_replays_rows_and_boundaries(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6615-ROW-REPLAY: source rows own all comparisons."""

    decoding = artifact["constrained_decoding_replay"]
    qwen = decoding["direct_arms"]["exp6605-qwen36-direct-headroom"]
    assert qwen["row_count"] == 216
    assert qwen["calibration_exact_success"] == 14
    assert qwen["held_exact_success"] == 28
    assert qwen["held_exact_success_rate"] == pytest.approx(28 / 108)
    assert qwen["charged_failure_count"] == 174
    assert qwen["identity_match"]
    assert not qwen["headroom_eligible"]
    assert (
        decoding["treatment_arms"]["exp6610-constraint-safety-hacking-audit"]["source_state"]
        == "missing"
    )
    assert decoding["verdict_class"] == "blocked"

    live = artifact["live_projection_replay"]
    assert live["row_count"] == 156
    assert live["all_predictions_before_observations"]
    assert live["calibration_held_disjoint"]
    assert live["selected_minus_no_projection_mean_error"] == pytest.approx(0.0)
    assert live["selected_minus_random_mean_error"] == pytest.approx(0.0)
    assert live["verdict_class"] == "null"
    assert not live["arc_solve_claim"]

    sampler = artifact["sampler_replay"]
    assert sampler["row_count"] == 240
    assert sampler["all_references_independent"]
    assert sampler["python_rust_parity_count"] == 60
    assert sampler["all_python_rust_parity"]
    assert sampler["scope"] == "cpu_software_only"
    assert sampler["verdict_class"] == "partial"

    learning = artifact["continuous_learning_replay"]
    assert learning["lifecycle_transition_count"] == 27
    assert learning["all_lifecycle_transitions_passed"]
    assert learning["prospective_row_count"] == 352
    assert learning["all_predictions_before_observations"]
    assert learning["matched_dose"]
    assert learning["held_future_benefit_over_static"] == pytest.approx(0.0)
    assert learning["held_future_benefit_over_shuffled"] == pytest.approx(0.0)
    assert learning["all_restart_equal"]
    assert learning["all_rollback_equal"]
    assert learning["frozen_weights_unchanged"]
    assert learning["scientific_verdict_class"] == "null"
    assert learning["source_artifact_verdict_class"] == "disqualified"


def test_scenario_report_6615_closed_dispositions_and_claims(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6615-CLOSED-VERDICTS keeps claim scope narrow."""

    rows = artifact["task_disposition_rows"]
    assert {row["row_kind"] for row in rows} == {"task", "milestone_question"}
    assert all(row["verdict_class"] in mod.CLOSED_VERDICT_CLASSES for row in rows)
    assert artifact["verdict_class"] in {"null", "partial"}
    assert artifact["verdict_class"] != "positive"
    boundaries = {row["boundary"]: row for row in artifact["claim_boundary_rows"]}
    assert set(boundaries) == {
        "oracle",
        "arc",
        "toy",
        "archive",
        "software",
        "hardware",
        "publication",
    }
    assert not boundaries["arc"]["promotion_allowed"]
    assert not boundaries["hardware"]["promotion_allowed"]
    assert not boundaries["publication"]["promotion_allowed"]
    gaps = {row["gap"]: row for row in artifact["prd_gap_disposition"]}
    assert gaps["FR-11"]["disposition"] == "not_advanced"
    assert gaps["hardware"]["disposition"] == "unchanged"


def test_scenario_report_6615_attacks_fail_closed(
    artifact: dict[str, object],
) -> None:
    """REQ-REPORT-6615-ATTACKS: all twelve mutations are rejected."""

    attacks = artifact["attack_rows"]
    assert {row["attack_id"] for row in attacks} == set(mod.ATTACK_IDS)
    assert all(row["mutation_applied"] for row in attacks)
    assert all(row["fail_closed"] for row in attacks)
    assert all(not row["promotion_allowed"] for row in attacks)


def test_req_report_6615_wrapper_checksum_validation_and_atomic_write(
    artifact: dict[str, object], tmp_path: Path
) -> None:
    """REQ-REPORT-6615-ATOMIC: wrappers, checksum, and replacement are exact."""

    assert mod.unwrap_value({"value": 1, "principle": "owner"}) == 1
    ordinary = {"value": 1, "task": "ordinary mapping"}
    assert mod.unwrap_value(ordinary) is ordinary
    mod.validate_artifact(artifact)

    bad_checksum = deepcopy(artifact)
    bad_checksum["honest_verdict"] = "mutated"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)

    bad_class = deepcopy(artifact)
    bad_class["verdict_class"] = "positive"
    bad_class["reproducibility_checksum"] = mod.reproducibility_checksum(bad_class)
    with pytest.raises(ValueError, match="capstone verdict"):
        mod.validate_artifact(bad_class)

    # REQ-CONDUCTOR-VERDICT-3 / SCENARIO-CONDUCTOR-VERDICT-5: a finished
    # capstone may not declare the may-retry class.
    bad_partial = deepcopy(artifact)
    bad_partial["verdict_class"] = "partial"
    bad_partial["reproducibility_checksum"] = mod.reproducibility_checksum(bad_partial)
    with pytest.raises(ValueError, match="capstone verdict must be null"):
        mod.validate_artifact(bad_partial)

    path = tmp_path / "nested" / "capstone.json"
    mod.write_artifact_atomic(path, artifact)
    assert json.loads(path.read_text(encoding="utf-8")) == artifact
    assert not list(path.parent.glob("*.tmp"))


def test_req_report_6615_provenance_protection_and_reconciliation(
    artifact: dict[str, object],
) -> None:
    """REQ-REPORT-6615-PRECONDITIONS: evidence and documents are explicit."""

    assert set(mod.REQUIRED_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_FIELDS) <= set(artifact["field_provenance"])
    protected = artifact["protected_files_unchanged"]
    assert protected["all_unchanged"]
    assert {row["path"] for row in protected["rows"]} == {
        "research-roadmap.yaml",
        "scripts/research_conductor.py",
    }
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    reconciliation = {row["path"]: row for row in artifact["reconciliation_receipts"]}
    assert reconciliation["_bmad/architecture.md"]["former_evidence_date"] == "2026-07-03"
    assert reconciliation["_bmad/architecture.md"]["new_evidence_date"] == "2026-08-26"
    architecture = (REPO / mod.ARCHITECTURE_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "V576 Independent Capstone Evidence (2026-08-26)" in architecture
    assert "**Former evidence date:** 2026-07-03" in architecture
    for deferred in (
        "_bmad/traceability.md",
        "ops/status.md",
        "ops/changelog.md",
    ):
        assert reconciliation[deferred]["action"] == "deferred_to_conductor"


def test_req_report_6615_input_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-6615-PRECONDITIONS: malformed inputs cannot enter the audit."""

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact root"):
        mod._read_json(invalid_json)

    roadmap = tmp_path / mod.ROADMAP_RELATIVE_PATH
    roadmap.write_text("milestone: wrong\ntasks: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected roadmap milestone"):
        mod.load_v576_tasks(tmp_path)
    roadmap.write_text(f"milestone: '{mod.MILESTONE}'\ntasks: wrong\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tasks must be a list"):
        mod.load_v576_tasks(tmp_path)

    with pytest.raises(ValueError, match="invalid experiment task id"):
        mod._experiment_number("not-an-experiment")
    assert mod._declared_artifact_fields({"prompt": "no field block"}) == set()
    assert mod._compare_gate(2, ">=", 1)
    assert mod._compare_gate(1, "<=", 2)
    assert not mod._compare_gate(None, ">=", 1)
    assert not mod._compare_gate(None, "<=", 1)
    assert not mod._compare_gate(1, "!=", 1)

    hashed_row = {"row_id": "changed", "value": 1, "row_hash": "sha256:bad"}
    assert mod._verify_row_hashes([hashed_row]) == {
        "checked": 1,
        "mismatch_count": 1,
        "mismatches": ["changed"],
    }

    verdict, discrepancies = mod._source_verdict_class(
        {
            "present": True,
            "payload": {
                "status": "complete",
                "honest_verdict": "complete",
                "verdict_class": "outside_enum",
            },
        },
        [],
    )
    assert verdict == "null"
    assert discrepancies == ["source verdict_class is outside closed enum: outside_enum"]

    tasks = mod.load_v576_tasks(REPO)
    sources = mod.load_source_artifacts(REPO, tasks)
    alternate_identity = deepcopy(sources)
    qwen = alternate_identity["exp6605-qwen36-direct-headroom"]["payload"]
    qwen["model_spec_and_identity"] = {
        "repository_id": mod.EXPECTED_MODEL_REGISTRY["exp6605-qwen36-direct-headroom"]
    }
    replay = mod.replay_constrained_decoding(alternate_identity)
    assert replay["direct_arms"]["exp6605-qwen36-direct-headroom"]["identity_match"]


def test_req_report_6615_validator_rejects_each_contract_mutation(
    artifact: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6615-ATOMIC: each schema and durability boundary fails closed."""

    def rejected(mutated: dict[str, object], message: str) -> None:
        mutated["reproducibility_checksum"] = mod.reproducibility_checksum(mutated)
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(mutated)

    missing = deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad = deepcopy(artifact)
    bad["inference_substrate"] = "wrong"
    rejected(bad, "inference substrate")
    bad = deepcopy(artifact)
    bad["verifier_is_oracle"] = False
    rejected(bad, "verifier_is_oracle")
    bad = deepcopy(artifact)
    bad["per_unit_rows"] = [
        row for row in bad["per_unit_rows"] if row.get("experiment_number") != 6604
    ]
    rejected(bad, "task matrix")
    bad = deepcopy(artifact)
    bad["per_unit_rows"] = [row for row in bad["per_unit_rows"] if row["row_kind"] == "task"]
    rejected(bad, "comparative unit")
    bad = deepcopy(artifact)
    bad["roadmap_gate_contract_rows"][0]["contract_valid"] = False
    rejected(bad, "roadmap gate")
    bad = deepcopy(artifact)
    bad["task_disposition_rows"][0]["verdict_class"] = "outside_enum"
    rejected(bad, "task disposition")
    bad = deepcopy(artifact)
    bad["attack_rows"][0]["fail_closed"] = False
    rejected(bad, "attacks did not fail closed")
    bad = deepcopy(artifact)
    bad["attack_rows"].pop()
    rejected(bad, "attack matrix")
    bad = deepcopy(artifact)
    bad["protected_files_unchanged"]["all_unchanged"] = False
    rejected(bad, "protected file")
    bad = deepcopy(artifact)
    bad["field_provenance"].pop("status")
    rejected(bad, "field provenance")
    bad = deepcopy(artifact)
    source_with_hashes = next(
        row for row in bad["source_artifact_receipts"] if row.get("row_hash_receipts")
    )
    next(iter(source_with_hashes["row_hash_receipts"].values()))["mismatch_count"] = 1
    rejected(bad, "source row hash")
    bad = deepcopy(artifact)
    bad["tests_run"] = [{"command": "missing receipt fields"}]
    rejected(bad, "test receipt")

    output = tmp_path / "receipt.json"
    mod.write_artifact_atomic(output, artifact)
    updated = mod.update_test_receipts(
        output,
        [{"command": "focused", "exit_code": 0, "duration_s": 1.0}],
        3.0,
    )
    assert updated["duration_s"] == 3.0
    assert updated["tests_run"][0]["command"] == "focused"

    failed_output = tmp_path / "failed.json"
    monkeypatch.setattr(mod.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("stop")))
    with pytest.raises(OSError, match="stop"):
        mod.write_artifact_atomic(failed_output, artifact)
    assert not list(tmp_path.glob("*.tmp"))
