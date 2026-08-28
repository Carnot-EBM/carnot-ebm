"""Focused checks for the V583 execution-contract audit.

Spec refs: REQ-REPORT-6688, SCENARIO-REPORT-6688-EXACT-PARITY,
SCENARIO-REPORT-6688-FAIL-CLOSED, SCENARIO-REPORT-6688-VALIDATOR-MISMATCH,
and SCENARIO-REPORT-6688-ATOMIC-PROVENANCE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_6688_v583_manifest_parity_contract as exp


REPO = Path(__file__).resolve().parents[2]


def _validator_rows() -> list[dict[str, object]]:
    """Build passing validator receipts without starting subprocesses."""

    return [
        exp.classify_validator_result(name, command, 0, "passed", "20260828")[0]
        for name, command in exp.VALIDATOR_DEFINITIONS
    ]


@pytest.fixture(scope="module")
def source_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Parse the immutable design and the selected checked-in manifest once."""

    design = exp.parse_design_contract((REPO / exp.DESIGN_PATH).read_text(encoding="utf-8"))
    manifest = exp.load_manifest_rows(REPO, design)
    return design, manifest


@pytest.fixture(scope="module")
def artifact() -> dict[str, object]:
    """Build the expected blocked artifact from the checked-in ten-task YAML."""

    before = exp.protected_hashes(REPO)
    return exp.build_artifact(
        REPO,
        run_date="20260828",
        duration_s=1.5,
        validator_rows=_validator_rows(),
        tests_run=[{"command": "focused-exp6688", "exit": 0, "summary": "passed"}],
        protected_before=before,
    )


def test_req_report_6688_spec_owns_the_complete_contract() -> None:
    """REQ-REPORT-6688 names each durable field and audit boundary."""

    section = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8").split("REQ-REPORT-6688", 1)[1]
    assert all(field in section for field in exp.REQUIRED_ARTIFACT_FIELDS)
    for marker in (
        "SCENARIO-REPORT-6688-EXACT-PARITY",
        "SCENARIO-REPORT-6688-FAIL-CLOSED",
        "SCENARIO-REPORT-6688-VALIDATOR-MISMATCH",
        "SCENARIO-REPORT-6688-ATOMIC-PROVENANCE",
        exp.INFERENCE_SUBSTRATE,
        exp.RESULT_PATH.as_posix(),
    ):
        assert marker in section


def test_scenario_report_6688_parses_design_and_fails_closed_on_truncated_yaml(
    source_rows: tuple[list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-6688-FAIL-CLOSED retains the four absent tasks."""

    design, manifest = source_rows
    assert [row["task_id"] for row in design] == list(exp.EXPECTED_TASK_IDS)
    assert [row["order"] for row in design] == list(range(1, 15))
    assert len({row["deliverable"] for row in design}) == 14
    assert [row["task_id"] for row in manifest] == list(exp.EXPECTED_TASK_IDS[:10])
    assert manifest[0]["manifest_path"] == "research-roadmap.yaml"
    assert design[0]["route"] == {"agent_type": "claude", "model": "opus"}
    assert design[1]["route"] == {"agent_type": "codex", "model": "gpt-5.6-sol"}
    assert design[3]["requires_gpu"] is True
    assert design[10]["task_category"] == "Torx qualification"
    assert manifest[0]["run_command"].endswith(
        "carnot.experiment_6688_v583_manifest_parity_contract --date {date}"
    )


def test_scenario_report_6688_cross_references_gates_priors_and_routes(
    source_rows: tuple[list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """REQ-REPORT-6688 keeps producer, prior, and route evidence as rows."""

    design, manifest = source_rows
    gates = exp.build_producer_consumer_rows(design, manifest, set())
    assert len(gates) == 9
    assert all(row["upstream_exists"] for row in gates)
    assert all(row["producer_declares_exact_field"] for row in gates)
    assert all(row["upstream_retired"] is False for row in gates)
    assert any(row["matches_design"] is False for row in gates)

    priors = exp.build_prior_failure_rows(REPO, manifest)
    assert {row["prior_experiment_id"] for row in priors} == {
        "exp5747-sota-exact-proposal-utility-panel",
        "exp5163",
        "exp6678-constraint-family-stream",
        "exp6679-prequential-cross-family-csl-ab",
        "exp6680-csl-durability-audit",
    }
    assert all(
        row["completed_record_found"]
        or row["artifact_state"] == "present"
        or row["conductor_state_rows"]
        for row in priors
    )
    assert all(row["retirement_signal"] is True for row in priors)
    assert all(row["retired_upstream_reference"] is False for row in priors)
    assert all(row["passed"] for row in priors)

    routes = exp.build_route_rows(design, manifest)
    assert len(routes) == 14
    assert all(row["validation"] == "passed" for row in routes[:10])
    assert all(row["validation"] == "missing_manifest_task" for row in routes[10:])
    assert routes[0]["max_turns"] == 100
    assert routes[1]["agent_backend"] == "codex"


def test_scenario_report_6688_preserves_the_dated_validator_mismatch() -> None:
    """SCENARIO-REPORT-6688-VALIDATOR-MISMATCH does not weaken the gate audit."""

    output = json.dumps(
        {
            "failure_details": [
                "MODEL_AGENT_COHERENCE exp6689-a: agent_type=codex requires "
                "model=gpt-5.5, got gpt-5.6-sol"
            ]
        }
    )
    row, mismatches = exp.classify_validator_result(
        "gate_contract", "audit gates", 1, output, "20260828"
    )
    assert row["classification"] == "validator_mismatch_nonblocking"
    assert mismatches[0]["conflict_date"] == "2026-08-28"
    assert mismatches[0]["validator_model"] == "gpt-5.5"
    assert mismatches[0]["operator_model"] == "gpt-5.6-sol"

    hard, mismatches = exp.classify_validator_result(
        "gate_contract",
        "audit gates",
        1,
        '{"failure_details":["GATE_UPSTREAM_EXISTS missing"]}',
        "20260828",
    )
    assert hard["classification"] == "activation_hard_failure"
    assert mismatches == []


def test_scenario_report_6688_blocked_artifact_recomputes_exact_failures(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6688-FAIL-CLOSED produces a valid blocked receipt."""

    assert exp.validate_artifact(artifact) == []
    assert artifact["status"] == "blocked_manifest_parity"
    assert str(artifact["honest_verdict"]).startswith("blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["v583_manifest_parity_ready"] is False
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    aggregate = artifact["aggregate_row_recomputation"]
    assert aggregate["design_task_count"] == 14
    assert aggregate["manifest_task_count"] == 10
    assert aggregate["missing_task_ids"] == list(exp.EXPECTED_TASK_IDS[10:])
    assert aggregate["recomputed_ready"] is False
    assert any(row["check"] == "manifest.task_order" for row in artifact["gate_check_summary"])
    assert set(artifact["field_provenance"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == exp.payload_checksum(artifact)
    assert all(row["before"] == row["after"] for row in artifact["protected_files_unchanged"])
    assert any(
        row["path"] == "research-roadmap-next.yaml" and row["state"] == "missing"
        for row in artifact["preconditions_checked"]["inputs"]
    )


def test_scenario_report_6688_validator_rejects_tampering(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6688-ATOMIC-PROVENANCE detects durable mutations."""

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
    assert "blocked_status_mismatch" in errors_for(status="complete_ready")
    assert "blocked_verdict_class_mismatch" in errors_for(verdict_class="null")
    assert "blocked_honest_verdict_mismatch" in errors_for(honest_verdict="complete: wrong")
    assert "field_provenance_mismatch" in errors_for(field_provenance={})
    assert "protected_file_changed" in errors_for(
        protected_files_unchanged=[
            {"path": "research-roadmap.yaml", "before": "a", "after": "b", "unchanged": False}
        ]
    )
    checksum = deepcopy(artifact)
    checksum["status"] = "changed-without-checksum"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(checksum)


def test_scenario_report_6688_source_and_cli_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: dict[str, object]
) -> None:
    """SCENARIO-REPORT-6688-ATOMIC-PROVENANCE covers source and CLI boundaries."""

    with pytest.raises(ValueError, match="fourteen task sections"):
        exp.parse_design_contract("# no V583 task sections")
    assert exp.extract_required_fields("no fields") == []
    assert exp.extract_run_command("no command") is None

    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(ValueError, match="V583 execution manifest"):
        exp.select_manifest_path(missing)
    (missing / "research-roadmap.yaml").write_text(
        yaml.safe_dump({"milestone": "wrong", "tasks": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="V583 execution manifest"):
        exp.select_manifest_path(missing)

    target = tmp_path / "nested" / "artifact.json"
    exp.write_json_atomic(target, artifact)
    assert json.loads(target.read_text(encoding="utf-8")) == artifact
    assert exp.main(["--validate", "--output", str(target)]) == 0
    broken = deepcopy(artifact)
    broken["status"] = "tampered"
    target.write_text(json.dumps(broken), encoding="utf-8")
    assert exp.main(["--validate", "--output", str(target)]) == 1

    validators = _validator_rows()
    monkeypatch.setattr(exp, "run_validators", lambda _root, _date: validators)
    generated = tmp_path / "generated.json"
    assert exp.main(["--date", "20260828", "--output", str(generated)]) == 0
    payload = json.loads(generated.read_text(encoding="utf-8"))
    assert payload["v583_manifest_parity_ready"] is False
    assert payload["duration_s"] >= 0.0

    monkeypatch.setattr(exp, "validate_artifact", lambda _payload: ["forced-invalid"])
    assert exp.main(["--date", "20260828", "--output", str(tmp_path / "bad.json")]) == 1


def test_scenario_report_6688_validator_runner_retains_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6688-VALIDATOR-MISMATCH retains command receipts."""

    exit_code, output = exp._run_command(tmp_path, "printf 'stdout'; printf 'stderr' >&2; exit 3")
    assert exit_code == 3
    assert output == "stdoutstderr"

    calls: list[str] = []

    def fake_run(_root: Path, command: str) -> tuple[int, str]:
        calls.append(command)
        return 0, "passed"

    monkeypatch.setattr(exp, "_run_command", fake_run)
    rows = exp.run_validators(tmp_path, "20260828")
    assert len(rows) == len(exp.VALIDATOR_DEFINITIONS)
    assert all(row["classification"] == "passed" for row in rows)
    assert all(row["operator_schema_mismatch_rows"] == [] for row in rows)
    assert calls == [command for _name, command in exp.VALIDATOR_DEFINITIONS]
