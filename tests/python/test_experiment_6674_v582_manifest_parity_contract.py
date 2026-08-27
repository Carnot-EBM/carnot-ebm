"""Focused tests for the V582 activation-contract audit.

Spec refs: REQ-REPORT-6674, SCENARIO-REPORT-6674-EXACT-PARITY,
SCENARIO-REPORT-6674-FAIL-CLOSED,
SCENARIO-REPORT-6674-VALIDATOR-MISMATCH, and
SCENARIO-REPORT-6674-ATOMIC-PROVENANCE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_6674_v582_manifest_parity_contract as exp


REPO = Path(__file__).resolve().parents[2]


def _passing_validator_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for definition in exp.VALIDATOR_DEFINITIONS:
        output = "passed"
        exit_code = 0
        if definition[0] == "gate_contract":
            output = json.dumps(
                {
                    "failure_details": [
                        "MODEL_AGENT_COHERENCE exp6675-triggered-tail-scope-receipt: "
                        "agent_type=codex requires model=gpt-5.5, got gpt-5.6-sol"
                    ]
                }
            )
            exit_code = 1
        row, mismatches = exp.classify_validator_result(
            definition[0], definition[1], exit_code, output, "20260827"
        )
        rows.append(row)
        if mismatches:
            mismatch_rows = mismatches
    return rows, mismatch_rows


@pytest.fixture(scope="module")
def source_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Parse the checked-in design and manifest once for focused assertions."""

    design = exp.parse_design_contract((REPO / exp.DESIGN_PATH).read_text(encoding="utf-8"))
    manifest = exp.load_manifest_rows(REPO)
    return design, manifest


@pytest.fixture(scope="module")
def artifact() -> dict[str, object]:
    """Build one deterministic ready artifact without invoking subprocesses."""

    validators, mismatches = _passing_validator_rows()
    before = exp.protected_hashes(REPO)
    return exp.build_artifact(
        REPO,
        run_date="20260827",
        duration_s=1.25,
        validator_rows=validators,
        validator_mismatch_rows=mismatches,
        tests_run=[{"command": "focused-exp6674", "exit": 0, "summary": "passed"}],
        protected_before=before,
    )


def test_req_report_6674_spec_owns_complete_artifact_contract() -> None:
    """REQ-REPORT-6674 names every durable field and audit boundary."""

    section = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8").split("REQ-REPORT-6674", 1)[1]
    assert all(field in section for field in exp.REQUIRED_ARTIFACT_FIELDS)
    for marker in (
        "SCENARIO-REPORT-6674-EXACT-PARITY",
        "SCENARIO-REPORT-6674-FAIL-CLOSED",
        "SCENARIO-REPORT-6674-VALIDATOR-MISMATCH",
        "SCENARIO-REPORT-6674-ATOMIC-PROVENANCE",
        exp.INFERENCE_SUBSTRATE,
        exp.RESULT_PATH.as_posix(),
    ):
        assert marker in section


def test_scenario_report_6674_parses_exact_order_fields_and_commands(
    source_rows: tuple[list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-6674-EXACT-PARITY checks all 14 task contracts."""

    design, manifest = source_rows
    assert [row["task_id"] for row in design] == list(exp.EXPECTED_TASK_IDS)
    assert [row["task_id"] for row in manifest] == list(exp.EXPECTED_TASK_IDS)
    assert [row["order"] for row in design] == list(range(1, 15))
    assert len({row["deliverable"] for row in manifest}) == 14
    assert all(str(row["deliverable"]).startswith("results/") for row in manifest)
    assert all(str(row["deliverable"]).endswith(".json") for row in manifest)
    assert all(all(row["field_checks"].values()) for row in manifest)
    assert manifest[0]["route"] == {"agent_type": "claude", "model": "opus"}
    assert manifest[1]["route"] == {"agent_type": "codex", "model": "gpt-5.6-sol"}
    assert manifest[2]["route"] == {"agent_type": "claude", "model": None}
    assert manifest[2]["requires_gpu"] is True
    assert manifest[2]["run_command"].endswith(
        "carnot.experiment_6676_three_family_triggered_tail_ab --date {date}"
    )


def test_scenario_report_6674_proves_producer_field_spelling(
    source_rows: tuple[list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-6674-EXACT-PARITY binds all ten consumer gates."""

    design, manifest = source_rows
    rows = exp.build_producer_consumer_rows(design, manifest)
    assert len(rows) == 10
    assert all(row["upstream_exists"] for row in rows)
    assert all(row["producer_declares_exact_field"] for row in rows)
    assert all(row["matches_design"] for row in rows)
    assert all(row["upstream_retired"] is False for row in rows)
    numeric = next(row for row in rows if row["artifact_field"] == "eligible_redirect_outcome_rows")
    assert numeric["operator"] == ">="
    assert numeric["value"] == 30
    assert numeric["consumer"] == "exp6682-arc-held-family-supervisor-ab"


def test_scenario_report_6674_prior_lineage_is_complete_and_not_an_upstream(
    source_rows: tuple[list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """REQ-REPORT-6674 preserves retired scope only as prior-failure evidence."""

    _design, manifest = source_rows
    rows = exp.build_prior_failure_rows(REPO, manifest)
    assert len(rows) == 14
    assert all(row["completed_record_found"] for row in rows)
    assert all(row["changed_condition"] for row in rows)
    assert all(row["retirement_signal"] is True for row in rows)
    constraint_ir = next(
        row for row in rows if str(row["prior_experiment_id"]).startswith("exp5923-")
    )
    assert constraint_ir["exclusion_manifest_match"] is True
    assert constraint_ir["reference_role"] == "prior_failure"
    assert constraint_ir["retired_upstream_reference"] is False
    assert all(row["passed"] for row in rows)


def test_scenario_report_6674_classifies_only_stale_codex_rule_as_mismatch() -> None:
    """SCENARIO-REPORT-6674-VALIDATOR-MISMATCH keeps dated drift visible."""

    output = json.dumps(
        {
            "failure_details": [
                "MODEL_AGENT_COHERENCE exp6675-a: agent_type=codex requires "
                "model=gpt-5.5, got gpt-5.6-sol",
                "MODEL_AGENT_COHERENCE exp6678-b: agent_type=codex requires "
                "model=gpt-5.5, got gpt-5.6-sol",
            ]
        }
    )
    row, mismatches = exp.classify_validator_result(
        "gate_contract", "audit gates", 1, output, "20260827"
    )
    assert row["classification"] == "validator_mismatch_nonblocking"
    assert len(mismatches) == 2
    assert all(item["operator_model"] == "gpt-5.6-sol" for item in mismatches)
    assert all(item["validator_model"] == "gpt-5.5" for item in mismatches)

    failed, no_mismatches = exp.classify_validator_result(
        "gate_contract",
        "audit gates",
        1,
        '{"failure_details":["GATE_UPSTREAM_EXISTS bad"]}',
        "20260827",
    )
    assert failed["classification"] == "activation_hard_failure"
    assert no_mismatches == []

    malformed, no_mismatches = exp.classify_validator_result(
        "gate_contract", "audit gates", 1, "not-json", "20260827"
    )
    assert malformed["classification"] == "activation_hard_failure"
    assert no_mismatches == []


def test_scenario_report_6674_ready_artifact_recomputes_every_row(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6674-EXACT-PARITY reduces complete rows to readiness."""

    assert exp.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_ready"
    assert str(artifact["honest_verdict"]).startswith("complete:")
    assert artifact["verdict_class"] == "null"
    assert artifact["gate_check_summary"] == []
    assert artifact["v582_manifest_parity_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    aggregate = artifact["aggregate_row_recomputation"]
    assert aggregate["task_count"] == 14
    assert aggregate["gate_count"] == 10
    assert aggregate["prior_failure_count"] == 14
    assert aggregate["validator_count"] == 5
    assert aggregate["recomputed_ready"] is True
    assert len(artifact["validator_mismatch_rows"]) == 1
    assert set(artifact["field_provenance"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == exp.payload_checksum(artifact)
    assert all(row["before"] == row["after"] for row in artifact["protected_files_unchanged"])
    assert any(
        row["path"].endswith("experiment_6660_v581_evidence_contract.json")
        and row["state"] == "missing"
        for row in artifact["preconditions_checked"]["v581_records"]
    )


def test_scenario_report_6674_fail_closed_reducer_localizes_exact_difference(
    source_rows: tuple[list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-6674-FAIL-CLOSED does not coerce missing evidence."""

    design, manifest = source_rows
    bad_manifest = deepcopy(manifest)
    bad_manifest[0]["field_checks"]["run_command"] = False
    bad_manifest[0]["field_differences"]["run_command"] = {
        "expected": design[0]["run_command"],
        "observed": None,
    }
    validators, _mismatches = _passing_validator_rows()
    aggregate, failures = exp.reduce_readiness(
        design,
        bad_manifest,
        exp.build_producer_consumer_rows(design, bad_manifest),
        exp.build_prior_failure_rows(REPO, bad_manifest),
        validators,
        [{"path": "research-roadmap.yaml", "unchanged": True}],
    )
    assert aggregate["recomputed_ready"] is False
    failure = next(row for row in failures if row["check"] == "task.run_command")
    assert failure["expected_value"] == design[0]["run_command"]
    assert failure["observed_value"] is None

    malformed_design = list(reversed(design))
    malformed_manifest = deepcopy(list(reversed(manifest)))
    malformed_manifest[0]["deliverable"] = malformed_manifest[1]["deliverable"]
    bad_gate = exp.build_producer_consumer_rows(design, manifest)
    bad_gate[0]["matches_design"] = False
    bad_gate[0]["upstream_retired"] = True
    bad_prior = exp.build_prior_failure_rows(REPO, manifest)
    bad_prior[0]["passed"] = False
    hard_validator = deepcopy(validators)
    hard_validator[0]["classification"] = "activation_hard_failure"
    aggregate, failures = exp.reduce_readiness(
        malformed_design,
        malformed_manifest,
        bad_gate,
        bad_prior,
        hard_validator,
        [{"path": "research-roadmap.yaml", "before": "a", "after": "b", "unchanged": False}],
    )
    assert aggregate["recomputed_ready"] is False
    assert {
        "design.task_order",
        "manifest.task_order",
        "manifest.deliverables",
        "gate.matches_design",
        "gate.upstream_not_retired",
        "prior.lineage",
        "validator.activation",
        "protected_file.unchanged",
    } <= {row["check"] for row in failures}


def test_scenario_report_6674_validator_rejects_tampering(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6674-ATOMIC-PROVENANCE validates durable boundaries."""

    def errors_for(**changes: object) -> list[str]:
        changed = deepcopy(artifact)
        changed.update(changes)
        changed["reproducibility_checksum"] = exp.payload_checksum(changed)
        return exp.validate_artifact(changed)

    missing = deepcopy(artifact)
    missing.pop("duration_s")
    missing["reproducibility_checksum"] = exp.payload_checksum(missing)
    assert "required_fields_missing:duration_s" in exp.validate_artifact(missing)
    assert "inference_substrate_mismatch" in errors_for(inference_substrate="wrong")
    assert "verifier_is_oracle_mismatch" in errors_for(verifier_is_oracle=False)
    assert "ready_status_mismatch" in errors_for(status="blocked_wrong")
    assert "ready_verdict_class_mismatch" in errors_for(verdict_class="blocked")
    assert "ready_aggregate_mismatch" in errors_for(
        aggregate_row_recomputation={"recomputed_ready": False}
    )
    assert "ready_honest_verdict_mismatch" in errors_for(honest_verdict="wrong")
    assert "field_provenance_mismatch" in errors_for(field_provenance={})
    assert "protected_file_changed" in errors_for(
        protected_files_unchanged=[
            {"path": "research-roadmap.yaml", "before": "a", "after": "b", "unchanged": False}
        ]
    )

    checksum = deepcopy(artifact)
    checksum["status"] = "changed-without-checksum"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(checksum)

    blocked_shape = deepcopy(artifact)
    blocked_shape["v582_manifest_parity_ready"] = False
    blocked_shape["status"] = "wrong"
    blocked_shape["verdict_class"] = "null"
    blocked_shape["honest_verdict"] = "wrong"
    blocked_shape["reproducibility_checksum"] = exp.payload_checksum(blocked_shape)
    blocked_errors = exp.validate_artifact(blocked_shape)
    assert "blocked_status_mismatch" in blocked_errors
    assert "blocked_verdict_class_mismatch" in blocked_errors
    assert "blocked_honest_verdict_mismatch" in blocked_errors


def test_scenario_report_6674_atomic_write_and_cli_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: dict[str, object]
) -> None:
    """SCENARIO-REPORT-6674-ATOMIC-PROVENANCE covers writer and CLI paths."""

    target = tmp_path / "nested" / "artifact.json"
    exp.write_json_atomic(target, artifact)
    assert json.loads(target.read_text(encoding="utf-8")) == artifact
    assert not list(target.parent.glob("*.tmp"))

    assert exp.main(["--validate", "--output", str(target)]) == 0
    broken = deepcopy(artifact)
    broken["status"] = "tampered"
    target.write_text(json.dumps(broken), encoding="utf-8")
    assert exp.main(["--validate", "--output", str(target)]) == 1

    real_replace = exp.os.replace
    monkeypatch.setattr(
        exp.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("replace failed"))
    )
    with pytest.raises(OSError, match="replace failed"):
        exp.write_json_atomic(tmp_path / "replace-failure.json", artifact)
    assert not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(exp.os, "replace", real_replace)

    validator_rows, mismatches = _passing_validator_rows()
    monkeypatch.setattr(exp, "run_validators", lambda _root, _date: (validator_rows, mismatches))
    generated = tmp_path / "generated.json"
    assert exp.main(["--date", "20260827", "--output", str(generated)]) == 0
    payload = json.loads(generated.read_text(encoding="utf-8"))
    assert payload["v582_manifest_parity_ready"] is True
    assert payload["duration_s"] >= 0.0

    monkeypatch.setattr(exp, "validate_artifact", lambda _payload: ["forced-invalid"])
    assert exp.main(["--date", "20260827", "--output", str(tmp_path / "invalid.json")]) == 1


def test_scenario_report_6674_parse_and_validator_helper_edges(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6674-FAIL-CLOSED rejects malformed source boundaries."""

    with pytest.raises(ValueError, match="exact execution contract"):
        exp.parse_design_contract("# no contract table")
    design_text = (REPO / exp.DESIGN_PATH).read_text(encoding="utf-8")
    one_row = "| 14 | `exp6687-v582-branch-synthesis` | V582 five-branch disposition | `results/experiment_6687_v582_branch_synthesis.json` |\n"
    with pytest.raises(ValueError, match="13 task rows"):
        exp.parse_design_contract(design_text.replace(one_row, ""))
    with pytest.raises(ValueError, match="unsupported design gate condition"):
        exp._parse_gate_condition("contains true")
    with pytest.raises(ValueError, match="invalid task id"):
        exp._experiment_number("bad-task")
    assert exp.extract_required_fields("no artifact contract") == []
    assert exp.extract_run_command("no run command") is None
    with pytest.raises(ValueError, match="roadmap milestone"):
        exp.load_manifest_rows(tmp_path)

    (tmp_path / exp.ROADMAP_PATH).write_text(
        yaml.safe_dump({"milestone": exp.MILESTONE, "tasks": {}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="task list"):
        exp.load_manifest_rows(tmp_path)

    (tmp_path / "ops").mkdir()
    (tmp_path / exp.EXCLUSION_PATH).write_text(
        yaml.safe_dump({"retired": ["not-a-mapping"]}), encoding="utf-8"
    )
    assert exp._retirement_index(tmp_path) == {}

    passed, mismatches = exp.classify_validator_result("yaml_parse", "parse", 0, "ok", "20260827")
    assert passed["classification"] == "passed"
    assert mismatches == []
    failed, mismatches = exp.classify_validator_result("yaml_parse", "parse", 2, "bad", "20260827")
    assert failed["classification"] == "activation_hard_failure"
    assert mismatches == []

    exit_code, output = exp._run_command(tmp_path, "printf 'stdout'; printf 'stderr' >&2; exit 3")
    assert exit_code == 3
    assert output == "stdoutstderr"

    calls: list[str] = []

    def fake_run(_root: Path, command: str) -> tuple[int, str]:
        calls.append(command)
        return 0, "passed"

    original_run = exp._run_command
    exp._run_command = fake_run
    try:
        rows, mismatch_rows = exp.run_validators(tmp_path, "20260827")
    finally:
        exp._run_command = original_run
    assert len(rows) == len(exp.VALIDATOR_DEFINITIONS)
    assert mismatch_rows == []
    assert calls == [definition[1] for definition in exp.VALIDATOR_DEFINITIONS]
