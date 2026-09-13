"""Tests for REQ-REPORT-7259 and SCENARIO-REPORT-7259-*.

The tests keep all generated files in pytest-owned temporary directories. The
checked-in producer artifacts remain read-only evidence.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7259_v638_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def repository_state() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Load the contract and evidence once because source hashing is repeated work."""

    contract = capstone.load_contract(ROOT)
    evidence = capstone.load_repository_payloads(ROOT, contract["tasks"])
    return contract, evidence


@pytest.fixture(scope="module")
def built_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, object], Path, Path]:
    """Build one terminal fixture without changing repository result paths."""

    directory = tmp_path_factory.mktemp("exp7259")
    output = directory / "result.json"
    checkpoint = directory / "checkpoint.json"
    raw_dir = directory / "raw"
    log = directory / "fixture-validation.log"
    log.write_text("fixture command passed\n", encoding="utf-8")
    receipt_path = directory / "validation-receipts.json"
    receipt_path.write_text(
        json.dumps(
            {
                "receipts": [
                    {
                        "name": "fixture_validation",
                        "command": "pytest fixture",
                        "exit_code": 0,
                        "log_path": str(log),
                        "log_sha256": capstone._sha256(log),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    artifact = capstone.build_artifact(
        ROOT,
        capstone.RUN_DATE,
        output,
        checkpoint,
        raw_dir=raw_dir,
        validation_receipt_path=receipt_path,
    )
    return artifact, output, checkpoint


def test_req_report_7259_spec_and_prompt_fields() -> None:
    """REQ-REPORT-7259: the requirement and every prompt field are durable."""

    text = (ROOT / capstone.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-REPORT-7259" in text
    assert "SCENARIO-REPORT-7259-ARTIFACT" in text
    assert set(capstone.FIELD_PRINCIPLES) <= capstone.REQUIRED_ARTIFACT_FIELDS


def test_scenario_report_7259_contract_keeps_fourteen_rows(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-7259-CONTRACT: YAML and design parse independently."""

    contract, _ = repository_state
    assert contract["task_ids"] == list(capstone.EXPECTED_TASK_IDS)
    assert len(contract["contract_rows"]) == 14
    assert contract["yaml_milestone"] == capstone.MILESTONE
    assert contract["markdown_milestone"] == "2026.09.636"
    assert contract["contract_agrees"] is False


def test_scenario_report_7259_missing_and_gate_block_rows_remain_explicit(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-7259-CONTRACT: absent declared files are not dropped."""

    _, evidence = repository_state
    assert evidence["exp7247-duration-class"]["evidence_source"] == "missing"
    assert evidence["exp7249-arc-live"]["evidence_source"] == "conductor_gate_block"
    assert evidence["exp7249-arc-live"]["declared_artifact_present"] is False
    assert evidence["exp7250-mention-canary"]["evidence_source"] == "missing"
    assert evidence["exp7251-mention-heldout"]["evidence_source"] == "missing"


def test_scenario_report_7259_gates_distinguish_value_file_and_success(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-7259-GATES: zero, missing file, and pass differ."""

    contract, evidence = repository_state
    rows = capstone.replay_gates(contract["tasks"], evidence)
    assert len(rows) == 5
    by_consumer = {row["consumer"]: row for row in rows}
    assert by_consumer["exp7249-arc-live"]["outcome"] == "value_mismatch"
    assert by_consumer["exp7250-mention-canary"]["outcome"] == "missing_file"
    assert by_consumer["exp7251-mention-heldout"]["outcome"] == "missing_file"
    assert by_consumer["exp7254-coverage-learning"]["outcome"] == "passed"
    assert by_consumer["exp7257-native-cost"]["outcome"] == "passed"


def test_scenario_report_7259_claims_use_raw_rows(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-7259-CLAIMS: distinct claims reduce from producer rows."""

    _, evidence = repository_state
    rows = capstone.recompute_claims(evidence)
    claims = {row["claim"]: row for row in rows}
    assert claims["usable_source_semantics"]["recomputed_value"] is None
    assert claims["completed_mention_capture"]["recomputed_value"] is None
    assert claims["exact_oracle_conformance"]["recomputed_value"] is True
    assert claims["exact_oracle_conformance"]["claim_class"] == "circular_positive"
    assert claims["causal_learning_observed"]["recomputed_value"] is True
    assert claims["coverage_learning_value"]["recomputed_value"] is False
    assert claims["useful_world_model_prediction"]["recomputed_value"] is None
    assert claims["actual_policy_consumption"]["recomputed_value"] is None
    assert claims["durable_event_cost_captured"]["recomputed_value"] is True
    assert claims["durable_event_cost_value"]["recomputed_value"] is False
    assert claims["official_arc_score"]["recomputed_value"] is None
    assert all(
        {"unit_id", "arm", "seed", "metric", "metric_value", "error", "abstention", "censored"}
        <= row.keys()
        for row in rows
    )


def test_scenario_report_7259_artifact_is_complete_and_blocked(
    built_artifact: tuple[dict[str, object], Path, Path],
) -> None:
    """SCENARIO-REPORT-7259-ARTIFACT: complete scope keeps the science block."""

    artifact, output, checkpoint = built_artifact
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert str(artifact["honest_verdict"]).startswith("blocked_")
    assert artifact["capstone_complete_score"] == 1
    assert len(artifact["evidence_matrix"]) == 14
    assert artifact["evidence_matrix"][-1]["evidence_source"] == "self_synthesis"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["current_invocation_counters"] == {
        "models": 0,
        "loads": 0,
        "generations": 0,
        "calls": 0,
    }
    assert artifact["publication_performed"] is False
    assert output.is_file() and checkpoint.is_file()
    assert capstone.validate_artifact(artifact, root=ROOT) == []


def test_scenario_report_7259_sidecars_and_branches_are_narrow(
    built_artifact: tuple[dict[str, object], Path, Path],
) -> None:
    """REQ-REPORT-7259: sidecars isolate history and decisions stay task-scoped."""

    artifact = built_artifact[0]
    receipts = artifact["historical_and_negative_fixture_sidecars"]
    assert len(receipts) == 2
    assert all(str(row["sha256"]).startswith("sha256:") for row in receipts)
    decisions = artifact["branch_decisions"]
    assert len(decisions) == 14
    assert all(row["broad_family_retirement_invented"] is False for row in decisions)
    assert all(
        row["retire_if_same_verdict_applied"] is False
        or row["exact_same_verdict_recurrence"] is True
        for row in decisions
    )


@pytest.mark.parametrize(
    ("field", "replacement", "expected_error"),
    [
        ("capstone_complete_score", 0, "capstone_complete_score"),
        ("verdict_class", "partial", "verdict_class"),
        ("reproducibility_checksum", "sha256:bad", "reproducibility_checksum"),
    ],
)
def test_scenario_report_7259_validator_rejects_mutations(
    built_artifact: tuple[dict[str, object], Path, Path],
    field: str,
    replacement: object,
    expected_error: str,
) -> None:
    """SCENARIO-REPORT-7259-ARTIFACT: terminal mutations fail closed."""

    changed = deepcopy(built_artifact[0])
    changed[field] = replacement
    assert expected_error in capstone.validate_artifact(changed)


def test_req_report_7259_cli_paths_and_bad_date(
    built_artifact: tuple[dict[str, object], Path, Path], tmp_path: Path
) -> None:
    """REQ-REPORT-7259: build and validation CLI paths return exact status."""

    with pytest.raises(ValueError, match="run date"):
        capstone.build_artifact(
            ROOT,
            "20260912",
            tmp_path / "bad.json",
            tmp_path / "bad-checkpoint.json",
            raw_dir=tmp_path / "bad-raw",
            validation_receipt_path=None,
        )
    output = tmp_path / "artifact.json"
    output.write_text(json.dumps(built_artifact[0]), encoding="utf-8")
    assert capstone.main(["--root", str(ROOT), "--artifact-path", str(output), "--validate"]) == 0
    output.write_text("{}", encoding="utf-8")
    assert capstone.main(["--root", str(ROOT), "--artifact-path", str(output), "--validate"]) == 1


def test_req_report_7259_read_json_rejects_array(tmp_path: Path) -> None:
    """REQ-REPORT-7259: list-root JSON cannot impersonate an artifact."""

    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON root"):
        capstone.read_json(path)


def test_req_report_7259_contract_reader_rejects_wrong_authorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7259: wrong roadmap shape, milestone, design, or order fails closed."""

    with monkeypatch.context() as patch:
        patch.setattr(capstone.yaml, "safe_load", lambda _value: [])
        with pytest.raises(ValueError, match="root"):
            capstone.load_contract(ROOT)
    roadmap = capstone.yaml.safe_load((ROOT / capstone.ROADMAP_PATH).read_bytes())
    with monkeypatch.context() as patch:
        bad = deepcopy(roadmap)
        bad["milestone"] = "2026.09.637"
        patch.setattr(capstone.yaml, "safe_load", lambda _value: bad)
        with pytest.raises(ValueError, match="not V638"):
            capstone.load_contract(ROOT)
    with monkeypatch.context() as patch:
        bad = deepcopy(roadmap)
        bad["milestone_doc"] = "wrong.md"
        patch.setattr(capstone.yaml, "safe_load", lambda _value: bad)
        with pytest.raises(ValueError, match="unexpected design"):
            capstone.load_contract(ROOT)
    with monkeypatch.context() as patch:
        bad = deepcopy(roadmap)
        bad["tasks"] = bad["tasks"][:-1]
        patch.setattr(capstone.yaml, "safe_load", lambda _value: bad)
        patch.setattr(
            capstone.source_map,
            "evaluate_contract",
            lambda _markdown, _yaml: {
                "contract_rows": [],
                "passed": False,
                "yaml_milestone": capstone.MILESTONE,
                "markdown_milestone": "2026.09.636",
                "gate_producer_rows": [],
            },
        )
        with pytest.raises(ValueError, match="task order"):
            capstone.load_contract(ROOT)


def test_req_report_7259_upstream_validator_failure_paths(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7259: producer checks retain checksum and validator failures."""

    _, evidence = repository_state
    source = deepcopy(evidence["exp7246-source-map"]["payload"])
    source["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum" in capstone._validate_payload("exp7246-source-map", source)
    with monkeypatch.context() as patch:
        patch.setattr(capstone.arc_witness, "validate_artifact", lambda _payload: None)
        assert capstone._validate_payload("exp7248-arc-witness", {}) == []


def test_scenario_report_7259_gate_replay_covers_field_and_quarantine() -> None:
    """SCENARIO-REPORT-7259-GATES: field absence and quarantine stay distinct."""

    tasks = [
        {
            "id": "exp1-source",
            "milestone": capstone.MILESTONE,
            "prompt": "- score: principle",
            "gated_on": [],
        },
        {
            "id": "exp2-source",
            "milestone": capstone.MILESTONE,
            "prompt": "- score: principle",
            "gated_on": [],
        },
        {
            "id": "exp3-consumer",
            "milestone": capstone.MILESTONE,
            "prompt": "",
            "gated_on": [
                {"upstream": "exp1-source", "artifact_field": "score", "op": "==", "value": 1},
                {"upstream": "exp2-source", "artifact_field": "score", "op": "==", "value": 1},
            ],
        },
    ]
    base = {
        "selected_evidence_path": "result.json",
        "declared_deliverable_path": "result.json",
        "producer_validation_errors": [],
    }
    evidence = {
        "exp1-source": {
            **base,
            "payload": {},
            "quarantine_state": {"quarantined": False},
        },
        "exp2-source": {
            **base,
            "payload": {"score": 1},
            "quarantine_state": {"quarantined": True},
        },
    }
    outcomes = [row["outcome"] for row in capstone.replay_gates(tasks, evidence)]
    assert outcomes == ["missing_field", "quarantined"]


def test_scenario_report_7259_claim_errors_and_recurrence_branch() -> None:
    """REQ-REPORT-7259: invalid producers and exact recurrence stay explicit."""

    base = {
        "selected_evidence_path": "result.json",
        "payload": {"status": "complete"},
        "producer_validation_errors": ["invalid"],
        "accepted_for_positive_claim": False,
        "embedded_row_count": 1,
        "authenticated": False,
    }
    row = capstone._claim_row("claim", "exp1", True, True, ("rows",), {"exp1": base})
    assert row["error"] == "producer_validation_failed"
    changed = deepcopy(base)
    changed["producer_validation_errors"] = []
    row = capstone._claim_row("claim", "exp1", True, False, ("rows",), {"exp1": changed})
    assert row["error"] == "declared_value_mismatch"
    prior = {"verdict": "same", "retire_if_same_verdict": True}
    decisions = capstone._branch_decisions(
        [{"id": "exp1", "prior_failures": [prior]}],
        [{"honest_verdict": "same", "selected_evidence_path": "result.json"}],
    )
    assert decisions[0]["action"] == "retire"
    assert decisions[0]["retire_if_same_verdict_applied"] is True


def test_req_report_7259_validation_receipt_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-7259: malformed validation receipt containers are rejected."""

    assert capstone._load_validation_receipts(ROOT, None) == []
    path = tmp_path / "receipts.json"
    path.write_text('{"receipts": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a list"):
        capstone._load_validation_receipts(ROOT, path)
    path.write_text('{"receipts": [1]}', encoding="utf-8")
    with pytest.raises(ValueError, match="not an object"):
        capstone._load_validation_receipts(ROOT, path)


def test_req_report_7259_sidecar_ignores_nonmapping_hashes(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]], tmp_path: Path
) -> None:
    """REQ-REPORT-7259: malformed historical hash containers cannot become fixtures."""

    contract, evidence = repository_state
    changed = deepcopy(evidence)
    changed["exp7246-source-map"]["payload"]["source_artifact_hashes"] = []
    receipts = capstone._write_sidecars(ROOT, tmp_path, contract, changed)
    assert len(receipts) == 2


def test_scenario_report_7259_validator_rejects_bad_source_hash_shape(
    built_artifact: tuple[dict[str, object], Path, Path],
) -> None:
    """SCENARIO-REPORT-7259-ARTIFACT: source hash failures cannot pass replay."""

    missing = deepcopy(built_artifact[0])
    missing["source_artifact_hashes"] = {"does-not-exist": "sha256:bad"}
    missing["reproducibility_checksum"] = capstone._artifact_checksum(missing)
    assert "source_artifact_hashes" in capstone.validate_artifact(missing, root=ROOT)
    malformed = deepcopy(built_artifact[0])
    malformed["source_artifact_hashes"] = []
    malformed["reproducibility_checksum"] = capstone._artifact_checksum(malformed)
    assert "source_artifact_hashes" in capstone.validate_artifact(malformed, root=ROOT)


def test_req_report_7259_main_build_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7259: the non-validation CLI delegates to the builder."""

    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        capstone,
        "build_artifact",
        lambda *args, **_kwargs: calls.append(args) or {},
    )
    assert capstone.main(["--root", str(ROOT), "--artifact-path", str(tmp_path / "x")]) == 0
    assert calls and calls[0][1] == capstone.RUN_DATE


def test_scenario_report_7259_terminal_write_guards(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-7259-ARTIFACT: both atomic-write validation guards fail closed."""

    contract, evidence = repository_state
    claims = capstone.recompute_claims(evidence)
    gates = capstone.replay_gates(contract["tasks"], evidence)
    monkeypatch.setattr(capstone, "load_contract", lambda _root: contract)
    monkeypatch.setattr(capstone, "load_repository_payloads", lambda _root, _tasks: evidence)
    monkeypatch.setattr(capstone, "recompute_claims", lambda _evidence: claims)
    monkeypatch.setattr(capstone, "replay_gates", lambda _tasks, _evidence: gates)
    with monkeypatch.context() as patch:
        patch.setattr(capstone, "validate_artifact", lambda _artifact, root=None: ["early"])
        with pytest.raises(RuntimeError, match="capstone validation"):
            capstone.build_artifact(
                ROOT,
                capstone.RUN_DATE,
                tmp_path / "early.json",
                tmp_path / "early-checkpoint.json",
                raw_dir=tmp_path / "early-raw",
                validation_receipt_path=None,
            )
    outcomes = iter(([], ["late"]))
    with monkeypatch.context() as patch:
        patch.setattr(capstone, "validate_artifact", lambda _artifact, root=None: next(outcomes))
        with pytest.raises(RuntimeError, match="terminal artifact"):
            capstone.build_artifact(
                ROOT,
                capstone.RUN_DATE,
                tmp_path / "late.json",
                tmp_path / "late-checkpoint.json",
                raw_dir=tmp_path / "late-raw",
                validation_receipt_path=None,
            )
