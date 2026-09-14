"""Verify REQ-REPORT-7287 and SCENARIO-REPORT-7287-*.

The fixtures use temporary output paths. They keep every producer artifact
read-only because those files are the evidence that this capstone evaluates.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7287_v640_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def repository_state() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Load the fixed roster and producer evidence once for all reductions."""

    contract = capstone.load_contract(ROOT)
    evidence = capstone.load_repository_payloads(ROOT, contract["tasks"])
    return contract, evidence


@pytest.fixture(scope="module")
def built_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, object], Path, Path]:
    """Build a measured terminal artifact without changing checked-in results."""

    directory = tmp_path_factory.mktemp("exp7287")
    log = directory / "fixture.log"
    log.write_text("fixture validation passed\n", encoding="utf-8")
    receipts = directory / "receipts.json"
    receipts.write_text(
        json.dumps(
            {
                "receipts": [
                    {
                        "name": "fixture_validation",
                        "command": "pytest fixture",
                        "exit_code": 0,
                        "duration_s": 0.01,
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
        directory / "result.json",
        directory / "checkpoint.json",
        raw_dir=directory / "raw",
        validation_receipt_path=receipts,
    )
    return artifact, directory / "result.json", directory / "checkpoint.json"


def test_req_report_7287_spec_and_prompt_principles() -> None:
    """REQ-REPORT-7287: the driving requirement covers every prompt field."""

    text = (ROOT / capstone.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-REPORT-7287" in text
    assert "SCENARIO-REPORT-7287-ARTIFACT" in text
    assert set(capstone.FIELD_PRINCIPLES) <= capstone.REQUIRED_ARTIFACT_FIELDS


def test_scenario_report_7287_contract_keeps_stale_design(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-7287-CONTRACT: parsing does not repair stale Markdown."""

    contract, _ = repository_state
    assert contract["task_ids"] == list(capstone.EXPECTED_TASK_IDS)
    assert len(contract["contract_rows"]) == 14
    assert contract["yaml_milestone"] == capstone.MILESTONE
    assert contract["markdown_milestone"] == "2026.09.636"
    assert contract["contract_agrees"] is False


def test_scenario_report_7287_intake_authenticates_exact_paths(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-7287-INTAKE: every producer keeps its declared file."""

    _, evidence = repository_state
    assert len(evidence) == 13
    assert all(row["evidence_source"] == "declared_deliverable" for row in evidence.values())
    assert all(row["declared_artifact_present"] is True for row in evidence.values())
    assert all(row["artifact_sha256"] for row in evidence.values())
    arc = evidence["exp7280-arc-live"]
    assert arc["quarantine_state"]["quarantined"] is True
    assert arc["quarantine_state"]["declared_flags"] == {"flagged_adversarial": True}


def test_scenario_report_7287_replays_eight_same_milestone_gates(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7287: all eight YAML gates retain exact observations."""

    contract, evidence = repository_state
    rows = capstone.replay_gates(contract["tasks"], evidence)
    assert len(rows) == 8
    assert all(row["passed"] is True for row in rows)
    assert all(row["producer_declares_field"] is True for row in rows)


def test_scenario_report_7287_claims_keep_infrastructure_and_value_separate(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-7287-CLAIMS: raw reductions do not promote readiness."""

    _, evidence = repository_state
    rows = capstone.recompute_claims(evidence)
    claims = {row["claim"]: row for row in rows}
    assert claims["source_capture_authenticity"]["recomputed_value"] is True
    assert claims["source_promotion"]["recomputed_value"] is False
    assert claims["arc_identity_ready"]["claim_class"] == "circular_positive"
    assert claims["arc_policy_consumption"]["recomputed_value"] is False
    assert claims["official_hidden_score"]["recomputed_value"] is None
    assert claims["admission_learning_efficacy"]["recomputed_value"] is False
    assert claims["admission_mechanical_safety"]["recomputed_value"] is True
    assert claims["admission_opportunity_loss"]["recomputed_value"] > 0
    assert claims["commit_acknowledgment_complete"]["recomputed_value"] is True
    assert claims["commit_cost_value"]["recomputed_value"] is False
    assert all(
        {
            "unit_id",
            "arm",
            "seed",
            "metric",
            "metric_value",
            "error",
            "abstention",
            "cost",
            "censored",
        }
        <= row.keys()
        for row in rows
    )


def test_scenario_report_7287_complete_matrix_is_terminal_blocked(
    built_artifact: tuple[dict[str, object], Path, Path],
) -> None:
    """SCENARIO-REPORT-7287-BLOCKED: quarantined ARC science blocks terminally."""

    artifact, output, checkpoint = built_artifact
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert str(artifact["honest_verdict"]).startswith("blocked_")
    assert artifact["capstone_complete_score"] == 1
    assert len(artifact["evidence_matrix"]) == 14
    assert artifact["evidence_matrix"][-1]["evidence_source"] == "self_synthesis"
    assert artifact["gate_check_summary"]["upstream"] == "exp7280-arc-live"
    assert artifact["gate_check_summary"]["artifact_field"] == "flagged_adversarial"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert not any(artifact["invocation_counts"].values())
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert output.is_file() and checkpoint.is_file()
    assert capstone.validate_artifact(artifact, root=ROOT) == []


def test_scenario_report_7287_decisions_do_not_retire_scale_canary(
    built_artifact: tuple[dict[str, object], Path, Path],
) -> None:
    """SCENARIO-REPORT-7287-DECISIONS: scale success is not scientific failure."""

    decisions = {row["task_id"]: row for row in built_artifact[0]["branch_decisions"]}
    canary = decisions["exp7277-comparator-canary"]
    assert canary["action"] == "continue_bounded_scale"
    assert canary["exact_same_verdict_recurrence"] is False
    assert canary["broad_family_retirement_invented"] is False
    arc = decisions["exp7280-arc-live"]
    assert arc["action"] == "needs_corrigendum"
    assert arc["retry_current_mechanism"] is False


@pytest.mark.parametrize(
    ("field", "replacement", "expected_error"),
    [
        ("capstone_complete_score", 0, "capstone_complete_score"),
        ("verdict_class", "partial", "verdict_class"),
        ("reproducibility_checksum", "sha256:bad", "reproducibility_checksum"),
    ],
)
def test_scenario_report_7287_validator_rejects_mutation(
    built_artifact: tuple[dict[str, object], Path, Path],
    field: str,
    replacement: object,
    expected_error: str,
) -> None:
    """SCENARIO-REPORT-7287-ARTIFACT: terminal mutations fail closed."""

    changed = deepcopy(built_artifact[0])
    changed[field] = replacement
    assert expected_error in capstone.validate_artifact(changed)


def test_scenario_report_7287_missing_required_science_changes_conclusion(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-7287-ARTIFACT: missing evidence cannot keep old claims."""

    _, evidence = repository_state
    changed = deepcopy(evidence)
    changed["exp7279-source-audit"].update(
        {
            "selected_evidence_path": None,
            "payload": {},
            "authenticated": False,
            "accepted_for_positive_claim": False,
        }
    )
    claims = {row["claim"]: row for row in capstone.recompute_claims(changed)}
    assert claims["source_promotion"]["claim_class"] == "blocked"
    assert claims["source_promotion"]["error"] == "missing_producer_artifact"


def test_req_report_7287_cli_and_input_failures(
    built_artifact: tuple[dict[str, object], Path, Path], tmp_path: Path
) -> None:
    """REQ-REPORT-7287: the CLI and malformed inputs return exact failures."""

    with pytest.raises(ValueError, match="run date"):
        capstone.build_artifact(
            ROOT,
            "20260913",
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


def test_req_report_7287_contract_and_json_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-7287: malformed JSON and wrong contract authorities fail."""

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
        bad["milestone"] = "2026.09.639"
        patch.setattr(capstone.yaml, "safe_load", lambda _value: bad)
        with pytest.raises(ValueError, match="not V640"):
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


def test_req_report_7287_conductor_blocks_and_validator_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7287: canonical blocks and validator failures fail closed."""

    assert capstone._conductor_block_errors("exp7279-source-audit", {}) == [
        "conductor_block_lifecycle",
        "conductor_block_identity",
        "conductor_block_gate",
    ]
    block = {
        "schema": "blocked_gate_check_v1",
        "status": "blocked",
        "experiment": 7279,
        "failed_upstream": "exp7278-source-measurement",
        "failed_field": "source_capture_complete_score",
        "failed_expected": 1,
    }
    assert capstone._validate_payload("exp7279-source-audit", block, ROOT) == []
    assert capstone._payload_identity_matches("exp7279-source-audit", block) is True
    with monkeypatch.context() as patch:
        patch.setattr(
            capstone.source_contract,
            "independent_reduce",
            lambda _payload: (_ for _ in ()).throw(ValueError("fixture")),
        )
        assert capstone._validate_payload("exp7274-source-contract", {}, ROOT) == [
            "validator_exception:ValueError:fixture"
        ]


def test_req_report_7287_exact_evidence_fallback_and_missing(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7287-INTAKE: only the canonical fallback is accepted."""

    task = {
        "id": "exp7279-source-audit",
        "deliverable": "results/declared-missing.json",
    }
    fallback = tmp_path / "results/experiment_7279_source_audit.json"
    fallback.parent.mkdir(parents=True)
    fallback.write_text(
        json.dumps(
            {
                "schema": "blocked_gate_check_v1",
                "status": "blocked",
                "experiment": 7279,
                "failed_upstream": "exp7278-source-measurement",
                "failed_field": "source_capture_complete_score",
                "failed_expected": 1,
            }
        ),
        encoding="utf-8",
    )
    row = capstone.load_evidence(tmp_path, task, {})
    assert row["evidence_source"] == "conductor_gate_block"
    fallback.unlink()
    row = capstone.load_evidence(tmp_path, task, {})
    assert row["evidence_source"] == "missing"


def test_req_report_7287_gate_and_claim_failure_classes() -> None:
    """REQ-REPORT-7287: gate absence and claim failures remain distinct."""

    tasks = [
        {"id": "producer", "milestone": capstone.MILESTONE, "prompt": "- ready:"},
        {
            "id": "consumer",
            "milestone": capstone.MILESTONE,
            "gated_on": [
                {"upstream": "producer", "artifact_field": "ready", "op": "==", "value": 1}
            ],
        },
    ]
    producer = {
        "payload": {},
        "selected_evidence_path": None,
        "declared_deliverable_path": "declared.json",
        "quarantine_state": {"quarantined": False},
    }
    assert capstone.replay_gates(tasks, {"producer": producer})[0]["outcome"] == "missing_file"
    producer["selected_evidence_path"] = "actual.json"
    assert capstone.replay_gates(tasks, {"producer": producer})[0]["outcome"] == "missing_field"
    producer["payload"] = {"ready": 1}
    producer["quarantine_state"] = {"quarantined": True}
    assert capstone.replay_gates(tasks, {"producer": producer})[0]["outcome"] == "quarantined"
    producer["quarantine_state"] = {"quarantined": False}
    producer["payload"] = {"ready": 0}
    assert capstone.replay_gates(tasks, {"producer": producer})[0]["outcome"] == "value_mismatch"

    evidence = {
        "task": {
            "selected_evidence_path": "source.json",
            "payload": {"status": "blocked", "duration_s": 1},
            "producer_validation_errors": [],
            "accepted_for_positive_claim": False,
            "authenticated": True,
            "raw_evidence": {},
            "quarantine_state": {"quarantined": False},
        }
    }
    assert capstone._claim_row("blocked", "task", 1, 1, (), evidence)["error"] == (
        "blocked_producer_artifact"
    )
    evidence["task"]["payload"]["status"] = "complete"
    evidence["task"]["producer_validation_errors"] = ["bad"]
    assert capstone._claim_row("invalid", "task", 1, 1, (), evidence)["error"] == (
        "producer_validation_failed"
    )
    evidence["task"]["producer_validation_errors"] = []
    assert (
        capstone._claim_row(
            "unavailable", "task", None, None, (), evidence, unavailable_error="absent"
        )["error"]
        == "absent"
    )
    numeric = capstone._claim_row("numeric", "task", 2, 2, (), evidence)
    assert numeric["claim_class"] == "numeric_check"
    assert capstone._passed_gate(
        {"acceptance_gate_results": [{"criterion": "listed", "passed": True}]}, "listed"
    )


def test_scenario_report_7287_terminal_variants_and_matrix_dispositions(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-7287-TERMINAL: null, positive, and blocked inputs differ."""

    contract, evidence = repository_state
    available = deepcopy(evidence)
    for task_id in capstone.REQUIRED_SCIENCE_TASKS:
        available[task_id]["quarantine_state"] = {
            "declared_flags": {},
            "exclusion_manifest_match": False,
            "quarantined": False,
        }
        available[task_id]["producer_validation_errors"] = []
        available[task_id]["payload"]["status"] = "complete"
    names = (
        "source_promotion",
        "arc_method_value",
        "admission_promotion",
        "commit_cost_value",
    )
    null_claims = [{"claim": name, "recomputed_value": False} for name in names]
    assert capstone._terminal_state(available, null_claims)["verdict_class"] == "null"
    positive_claims = [{"claim": name, "recomputed_value": True} for name in names]
    assert capstone._terminal_state(available, positive_claims)["verdict_class"] == "positive"

    missing = deepcopy(available)
    missing["exp7279-source-audit"]["selected_evidence_path"] = None
    assert capstone._required_science_failure(missing)["failed_check"] == (
        "required_science_evidence_available"
    )
    blocked = deepcopy(available)
    blocked["exp7279-source-audit"]["payload"]["status"] = "blocked"
    assert capstone._required_science_failure(blocked)["failed_check"] == (
        "required_science_terminal_complete"
    )
    invalid = deepcopy(available)
    invalid["exp7279-source-audit"]["producer_validation_errors"] = ["bad"]
    assert capstone._required_science_failure(invalid)["failed_check"] == (
        "required_science_authentic"
    )

    base = deepcopy(evidence["exp7279-source-audit"])
    task = contract["tasks"][5]
    base["selected_evidence_path"] = None
    assert capstone._matrix_row(6, task, base, [], [])["final_disposition"].startswith(
        "blocked_missing"
    )
    base["selected_evidence_path"] = "actual.json"
    base["payload"]["status"] = "blocked"
    base["quarantine_state"] = {"quarantined": False}
    assert capstone._matrix_row(6, task, base, [], [])["final_disposition"].startswith("complete_")
    base["payload"]["status"] = "complete"
    base["producer_validation_errors"] = ["bad"]
    assert capstone._matrix_row(6, task, base, [], [])["final_disposition"].startswith(
        "disqualified_"
    )


def test_req_report_7287_receipts_publication_and_hash_failures(
    built_artifact: tuple[dict[str, object], Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-REPORT-7287: missing logs, bad G1-G4, and stale hashes fail closed."""

    receipt_path = tmp_path / "receipts.json"
    receipt_path.write_text(
        json.dumps(
            {
                "receipts": [
                    {
                        "name": "missing",
                        "command": "false",
                        "exit_code": 1,
                        "duration_s": 0.1,
                        "log_path": "absent.log",
                        "log_sha256": "sha256:missing",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert (
        capstone._load_validation_receipts(tmp_path, receipt_path)[0]["observed_log_sha256"] is None
    )
    completed = capstone.subprocess.CompletedProcess([], 1, stdout="{}", stderr="fixture")
    with monkeypatch.context() as patch:
        patch.setattr(capstone.subprocess, "run", lambda *_args, **_kwargs: completed)
        with pytest.raises(RuntimeError, match="publication gate failed"):
            capstone._publication_gate(ROOT)
    assert capstone._publication_gate_valid({"exit_code": 0, "gates": {}}) is False

    changed = deepcopy(built_artifact[0])
    changed["source_artifact_hashes"] = {"missing-source": "sha256:bad"}
    changed["reproducibility_checksum"] = capstone._artifact_checksum(changed)
    assert "source_artifact_hashes" in capstone.validate_artifact(changed, root=ROOT)
    changed["source_artifact_hashes"] = []
    changed["reproducibility_checksum"] = capstone._artifact_checksum(changed)
    assert "source_artifact_hashes" in capstone.validate_artifact(changed, root=ROOT)


@pytest.mark.parametrize("failure_call", [1, 2])
def test_scenario_report_7287_build_refuses_invalid_atomic_stage(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_call: int,
) -> None:
    """SCENARIO-REPORT-7287-ARTIFACT: both atomic validation stages fail closed."""

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
        patch.setattr(capstone, "load_repository_payloads", lambda _root, _tasks: evidence)
        patch.setattr(capstone, "_publication_gate", lambda _root: publication)
        patch.setattr(capstone, "validate_artifact", forced_validation)
        with pytest.raises(RuntimeError, match="validation failed"):
            capstone.build_artifact(
                ROOT,
                capstone.RUN_DATE,
                tmp_path / f"result-{failure_call}.json",
                tmp_path / f"checkpoint-{failure_call}.json",
                raw_dir=tmp_path / f"raw-{failure_call}",
                validation_receipt_path=None,
            )


def test_req_report_7287_main_build_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """REQ-REPORT-7287: the CLI forwards explicit build paths."""

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
