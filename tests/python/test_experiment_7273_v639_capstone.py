"""Tests for REQ-REPORT-7273 and SCENARIO-REPORT-7273-*.

The tests use temporary output paths. Producer results stay read-only because
they are historical evidence for this aggregation task.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7273_v639_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def repository_state() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Load producer evidence once because validation repeats its hashes."""

    contract = capstone.load_contract(ROOT)
    evidence = capstone.load_repository_payloads(ROOT, contract["tasks"])
    return contract, evidence


@pytest.fixture(scope="module")
def built_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, object], Path, Path]:
    """Build a measured terminal fixture without changing checked-in results."""

    directory = tmp_path_factory.mktemp("exp7273")
    output = directory / "result.json"
    checkpoint = directory / "checkpoint.json"
    log = directory / "validation.log"
    log.write_text("fixture validation passed\n", encoding="utf-8")
    receipts = directory / "validation-receipts.json"
    receipts.write_text(
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
        raw_dir=directory / "raw",
        validation_receipt_path=receipts,
    )
    return artifact, output, checkpoint


def test_req_report_7273_spec_and_exact_prompt_principles() -> None:
    """REQ-REPORT-7273: the requirement and prompt principles are durable."""

    text = (ROOT / capstone.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-REPORT-7273" in text
    assert "SCENARIO-REPORT-7273-ARTIFACT" in text
    assert set(capstone.FIELD_PRINCIPLES) <= capstone.REQUIRED_ARTIFACT_FIELDS


def test_scenario_report_7273_contract_keeps_stale_design_visible(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-7273-CONTRACT: V636 Markdown is not replaced."""

    contract, _ = repository_state
    assert contract["task_ids"] == list(capstone.EXPECTED_TASK_IDS)
    assert len(contract["contract_rows"]) == 14
    assert contract["yaml_milestone"] == capstone.MILESTONE
    assert contract["markdown_milestone"] == "2026.09.636"
    assert contract["contract_agrees"] is False
    assert all(row["passed"] is False for row in contract["contract_rows"])


def test_scenario_report_7273_intake_preserves_declared_and_actual_paths(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-7273-INTAKE: conductor blocks do not impersonate outputs."""

    _, evidence = repository_state
    semantic = evidence["exp7266-semantic-audit"]
    assert semantic["declared_artifact_present"] is False
    assert semantic["evidence_source"] == "conductor_gate_block"
    assert semantic["selected_evidence_path"] == "results/experiment_7266_semantic_audit.json"
    delta = evidence["exp7271-delta-log"]
    assert delta["declared_artifact_present"] is False
    assert delta["evidence_source"] == "conductor_gate_block"
    assert delta["selected_evidence_path"] == "results/experiment_7271_delta_log.json"
    assert all(row["artifact_sha256"] for row in evidence.values())


def test_scenario_report_7273_gates_keep_failures_distinct(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7273: seven same-milestone gates replay exact fields."""

    contract, evidence = repository_state
    rows = capstone.replay_gates(contract["tasks"], evidence)
    assert len(rows) == 7
    by_consumer = {row["consumer"]: row for row in rows}
    assert by_consumer["exp7266-semantic-audit"]["outcome"] == "value_mismatch"
    assert by_consumer["exp7271-delta-log"]["outcome"] == "value_mismatch"
    assert sum(row["passed"] is True for row in rows) == 5
    assert all(row["producer_declares_field"] is True for row in rows)


def test_scenario_report_7273_claims_separate_science_and_oracle_receipts(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-7273-CLAIMS: raw rows control distinct conclusions."""

    _, evidence = repository_state
    rows = capstone.recompute_claims(evidence)
    claims = {row["claim"]: row for row in rows}
    assert claims["source_capture_accounted"]["recomputed_value"] is True
    assert claims["source_fidelity_exact"]["recomputed_value"] is False
    assert claims["source_semantic_value"]["recomputed_value"] is None
    assert claims["arc_transition_oracle_parity"]["claim_class"] == "circular_positive"
    assert claims["arc_public_game_generalization"]["recomputed_value"] is False
    assert claims["arc_policy_consumption"]["recomputed_value"] is False
    assert claims["official_hidden_score"]["recomputed_value"] is None
    assert claims["recognition_learning_value"]["recomputed_value"] is False
    assert claims["recognition_safety"]["recomputed_value"] is True
    assert claims["durable_cost_limit_value"]["recomputed_value"] is False
    assert claims["durable_log_semantics"]["recomputed_value"] is None
    assert all(
        {"unit_id", "arm", "seed", "metric", "metric_value", "error", "abstention", "censored"}
        <= row.keys()
        for row in rows
    )


def test_scenario_report_7273_complete_matrix_is_terminal_blocked(
    built_artifact: tuple[dict[str, object], Path, Path],
) -> None:
    """SCENARIO-REPORT-7273-BLOCKED: upstream absence never becomes partial."""

    artifact, output, checkpoint = built_artifact
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert str(artifact["honest_verdict"]).startswith("blocked_")
    assert artifact["capstone_complete_score"] == 1
    assert len(artifact["evidence_matrix"]) == 14
    assert artifact["evidence_matrix"][-1]["evidence_source"] == "self_synthesis"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert not any(artifact["invocation_counts"].values())
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["publication_performed"] is False
    assert artifact["gate_check_summary"]["upstream"] == "exp7266-semantic-audit"
    assert output.is_file() and checkpoint.is_file()
    assert capstone.validate_artifact(artifact, root=ROOT) == []


def test_scenario_report_7273_branches_apply_only_exact_recurrence(
    built_artifact: tuple[dict[str, object], Path, Path],
) -> None:
    """SCENARIO-REPORT-7273-DECISIONS: retirements stay task-scoped."""

    decisions = {row["task_id"]: row for row in built_artifact[0]["branch_decisions"]}
    mention = decisions["exp7264-mention-canary"]
    assert mention["exact_same_verdict_recurrence"] is True
    assert mention["action"] == "retire"
    assert mention["retire_if_same_verdict_applied"] is True
    assert decisions["exp7266-semantic-audit"]["action"] == "needs_changed_prerequisite"
    assert decisions["exp7266-semantic-audit"]["retry_current_mechanism"] is False
    assert all(row["broad_family_retirement_invented"] is False for row in decisions.values())


@pytest.mark.parametrize(
    ("field", "replacement", "expected_error"),
    [
        ("capstone_complete_score", 0, "capstone_complete_score"),
        ("verdict_class", "partial", "verdict_class"),
        ("reproducibility_checksum", "sha256:bad", "reproducibility_checksum"),
    ],
)
def test_scenario_report_7273_validator_rejects_mutations(
    built_artifact: tuple[dict[str, object], Path, Path],
    field: str,
    replacement: object,
    expected_error: str,
) -> None:
    """SCENARIO-REPORT-7273-ARTIFACT: terminal mutations fail closed."""

    changed = deepcopy(built_artifact[0])
    changed[field] = replacement
    assert expected_error in capstone.validate_artifact(changed)


def test_req_report_7273_cli_validation_and_bad_date(
    built_artifact: tuple[dict[str, object], Path, Path], tmp_path: Path
) -> None:
    """REQ-REPORT-7273: CLI validation and run-date checks return exact status."""

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


def test_req_report_7273_read_json_and_contract_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-7273: malformed JSON and wrong contract authorities fail."""

    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON root"):
        capstone.read_json(path)
    roadmap = capstone.yaml.safe_load((ROOT / capstone.ROADMAP_PATH).read_bytes())
    with monkeypatch.context() as patch:
        patch.setattr(capstone.yaml, "safe_load", lambda _value: [])
        with pytest.raises(ValueError, match="root"):
            capstone.load_contract(ROOT)
    with monkeypatch.context() as patch:
        bad = deepcopy(roadmap)
        bad["milestone"] = "2026.09.638"
        patch.setattr(capstone.yaml, "safe_load", lambda _value: bad)
        with pytest.raises(ValueError, match="not V639"):
            capstone.load_contract(ROOT)
    with monkeypatch.context() as patch:
        bad = deepcopy(roadmap)
        bad["milestone_doc"] = "wrong.md"
        patch.setattr(capstone.yaml, "safe_load", lambda _value: bad)
        with pytest.raises(ValueError, match="unexpected design"):
            capstone.load_contract(ROOT)
    with monkeypatch.context() as patch:
        bad = deepcopy(roadmap)
        bad["tasks"] = list(bad["tasks"])[:-1]
        patch.setattr(capstone.yaml, "safe_load", lambda _value: bad)
        with pytest.raises(ValueError, match="task order"):
            capstone.load_contract(ROOT)


def test_req_report_7273_negative_intake_and_replay_branches(tmp_path: Path) -> None:
    """REQ-REPORT-7273: malformed blocks and all gate absence classes remain distinct."""

    assert capstone._conductor_block_errors("exp7266-semantic-audit", {}) == [
        "conductor_block_lifecycle",
        "conductor_block_identity",
        "conductor_block_gate",
    ]
    missing = capstone.load_evidence(
        tmp_path,
        {
            "id": "exp7260-source-contract",
            "deliverable": "results/missing.json",
        },
        {},
    )
    assert missing["evidence_source"] == "missing"

    tasks = [
        {
            "id": "exp7260-source-contract",
            "milestone": capstone.MILESTONE,
            "prompt": "- ready:",
        },
        {
            "id": "exp7261-compute-contract",
            "milestone": capstone.MILESTONE,
            "gated_on": [
                {
                    "upstream": "exp7260-source-contract",
                    "artifact_field": "ready",
                    "op": "==",
                    "value": 1,
                }
            ],
        },
    ]
    base = {
        "payload": {},
        "selected_evidence_path": None,
        "declared_deliverable_path": "declared.json",
        "quarantine_state": {"quarantined": False},
    }
    row = capstone.replay_gates(tasks, {"exp7260-source-contract": base})[0]
    assert row["outcome"] == "missing_file"
    base["selected_evidence_path"] = "actual.json"
    row = capstone.replay_gates(tasks, {"exp7260-source-contract": base})[0]
    assert row["outcome"] == "missing_field"
    base["payload"] = {"ready": 1}
    base["quarantine_state"] = {"quarantined": True}
    row = capstone.replay_gates(tasks, {"exp7260-source-contract": base})[0]
    assert row["outcome"] == "quarantined"


def test_req_report_7273_validator_exception_and_claim_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7273: validator and claim failures cannot promote evidence."""

    with monkeypatch.context() as patch:
        patch.setattr(
            capstone.arc_witness,
            "validate_artifact",
            lambda _payload: (_ for _ in ()).throw(ValueError("fixture")),
        )
        errors = capstone._validate_payload("exp7262-arc-witness-receipt", {}, ROOT)
    assert errors == ["validator_exception:ValueError:fixture"]

    evidence = {
        "task": {
            "selected_evidence_path": None,
            "payload": {},
            "producer_validation_errors": [],
            "accepted_for_positive_claim": False,
            "authenticated": False,
            "raw_evidence_counts": {},
        }
    }
    assert capstone._claim_row("missing", "task", 1, 1, (), evidence)["error"] == (
        "missing_producer_artifact"
    )
    evidence["task"].update(
        {
            "selected_evidence_path": "source.json",
            "producer_validation_errors": ["bad"],
        }
    )
    assert capstone._claim_row("invalid", "task", 1, 1, (), evidence)["error"] == (
        "producer_validation_failed"
    )
    evidence["task"].update({"producer_validation_errors": [], "accepted_for_positive_claim": True})
    assert capstone._claim_row("mismatch", "task", 0, 1, (), evidence)["error"] == (
        "declared_value_mismatch"
    )
    assert capstone._claim_row("positive", "task", 1, 1, (), evidence)["claim_class"] == (
        "positive"
    )
    evidence["task"]["accepted_for_positive_claim"] = False
    assert capstone._claim_row("numeric", "task", 2, 2, (), evidence)["claim_class"] == (
        "numeric_check"
    )
    assert capstone._passed_gate(
        {"acceptance_gate_results": [{"criterion": "listed", "passed": True}]},
        "listed",
    )


def test_req_report_7273_receipt_and_publication_failure_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-7273: missing logs and failed G1-G4 subprocesses stay visible."""

    receipt_path = tmp_path / "receipts.json"
    receipt_path.write_text(
        json.dumps(
            {
                "receipts": [
                    {
                        "name": "missing_log",
                        "command": "false",
                        "exit_code": 1,
                        "log_path": "absent.log",
                        "log_sha256": "sha256:missing",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = capstone._load_validation_receipts(tmp_path, receipt_path)
    assert rows[0]["observed_log_sha256"] is None
    assert rows[0]["log_hash_matches"] is False

    completed = capstone.subprocess.CompletedProcess([], 0, stdout="[]", stderr="fixture")
    with monkeypatch.context() as patch:
        patch.setattr(capstone.subprocess, "run", lambda *_args, **_kwargs: completed)
        with pytest.raises(RuntimeError, match="publication gate failed"):
            capstone._publication_gate(ROOT)
    assert capstone._publication_gate_valid({"exit_code": 0, "gates": {}}) is False


def test_scenario_report_7273_source_hash_validation_fails_closed(
    built_artifact: tuple[dict[str, object], Path, Path],
) -> None:
    """SCENARIO-REPORT-7273-ARTIFACT: stale and malformed source maps fail."""

    changed = deepcopy(built_artifact[0])
    changed["source_artifact_hashes"] = {"missing-source": "sha256:bad"}
    changed["reproducibility_checksum"] = capstone._artifact_checksum(changed)
    assert "source_artifact_hashes" in capstone.validate_artifact(changed, root=ROOT)
    changed["source_artifact_hashes"] = []
    changed["reproducibility_checksum"] = capstone._artifact_checksum(changed)
    assert "source_artifact_hashes" in capstone.validate_artifact(changed, root=ROOT)


@pytest.mark.parametrize("failure_call", [1, 2])
def test_scenario_report_7273_build_refuses_invalid_candidate_or_terminal(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_call: int,
) -> None:
    """SCENARIO-REPORT-7273-ARTIFACT: validation blocks both atomic stages."""

    contract, evidence = repository_state
    calls = 0

    def forced_validation(_artifact: object, root: Path | None = None) -> list[str]:
        del root
        nonlocal calls
        calls += 1
        return ["forced"] if calls == failure_call else []

    publication = {
        "paper_ready": True,
        "gates": {name: {"pass": True} for name in ("G1", "G2", "G3", "G4")},
        "unmet_gates": [],
        "command": "fixture",
        "exit_code": 0,
        "elapsed_s": 0.0,
        "stderr": "",
    }
    with monkeypatch.context() as patch:
        patch.setattr(capstone, "load_contract", lambda _root: contract)
        patch.setattr(
            capstone,
            "load_repository_payloads",
            lambda _root, _tasks: evidence,
        )
        patch.setattr(capstone, "_publication_gate", lambda _root: publication)
        patch.setattr(capstone, "validate_artifact", forced_validation)
        with pytest.raises(RuntimeError, match="validation failed"):
            capstone.build_artifact(
                ROOT,
                capstone.RUN_DATE,
                tmp_path / "result.json",
                tmp_path / "checkpoint.json",
                raw_dir=tmp_path / "raw",
                validation_receipt_path=None,
            )


def test_req_report_7273_main_build_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """REQ-REPORT-7273: the non-validation CLI forwards explicit output paths."""

    calls: list[tuple[object, ...]] = []

    def fake_build(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((*args, kwargs))
        return {}

    with monkeypatch.context() as patch:
        patch.setattr(capstone, "build_artifact", fake_build)
        assert (
            capstone.main(
                [
                    "--root",
                    str(ROOT),
                    "--artifact-path",
                    str(tmp_path / "result.json"),
                    "--checkpoint-path",
                    str(tmp_path / "checkpoint.json"),
                    "--raw-dir",
                    str(tmp_path / "raw"),
                ]
            )
            == 0
        )
    assert len(calls) == 1
