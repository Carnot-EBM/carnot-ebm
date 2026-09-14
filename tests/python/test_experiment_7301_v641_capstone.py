"""Verify REQ-REPORT-7301 and SCENARIO-REPORT-7301-*.

The tests read the thirteen immutable V641 producer artifacts but write only to
temporary directories. They independently check the reductions that control
the capstone verdict instead of accepting producer headline prose.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7301_v641_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def repository_state() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Load the fixed V641 roster and all thirteen evidence slots once."""

    contract = capstone.load_contract(ROOT)
    evidence = capstone.load_repository_payloads(ROOT, contract["tasks"])
    return contract, evidence


@pytest.fixture(scope="module")
def built_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, object], Path, Path]:
    """Build one validated terminal fixture without touching checked-in results."""

    directory = tmp_path_factory.mktemp("exp7301")
    log = directory / "fixture.log"
    log.write_text("fixture validation passed\n", encoding="utf-8")
    receipt_path = directory / "receipts.json"
    receipt_path.write_text(
        json.dumps(
            {
                "receipts": [
                    {
                        "name": "fixture_validation",
                        "command": "pytest fixture",
                        "exit_code": 0,
                        "duration_s": 0.01,
                        "log_path": str(log),
                        "log_sha256": capstone.sha256(log),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = directory / "result.json"
    checkpoint = directory / "checkpoint.json"
    artifact = capstone.build_artifact(
        ROOT,
        capstone.RUN_DATE,
        output,
        checkpoint,
        raw_dir=directory / "raw",
        validation_receipt_path=receipt_path,
    )
    return artifact, output, checkpoint


def test_req_report_7301_spec_and_prompt_principles() -> None:
    """REQ-REPORT-7301: the spec and every required prompt field are present."""

    text = (ROOT / capstone.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-REPORT-7301" in text
    assert "SCENARIO-REPORT-7301-ARTIFACT" in text
    assert set(capstone.FIELD_PRINCIPLES) == set(capstone.REQUIRED_ARTIFACT_FIELDS)


def test_scenario_report_7301_contract_has_exact_literal_roster(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-7301-CONTRACT: YAML and Markdown retain fourteen rows."""

    contract, _ = repository_state
    assert contract["task_ids"] == list(capstone.EXPECTED_TASK_IDS)
    assert len(contract["contract_rows"]) == 14
    assert contract["contract_agrees"] is True
    assert all(row["checks"]["phase"] is True for row in contract["contract_rows"])


def test_scenario_report_7301_intake_preserves_real_dispositions(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-7301-INTAKE: outcome classes are not collapsed."""

    contract, evidence = repository_state
    assert len(evidence) == 13
    assert sum(row["evidence_source"] == "declared_deliverable" for row in evidence.values()) == 12
    assert evidence["exp7290-arc-selfparse"]["evidence_source"] == "conductor_gate_block"
    assert evidence["exp7290-arc-selfparse"]["declared_artifact_present"] is False
    assert evidence["exp7289-arc-boundary"]["quarantine_state"]["quarantined"] is True
    assert evidence["exp7290-arc-selfparse"]["disposition_class"] == "blocked"
    dispositions = capstone.task_dispositions(contract["tasks"], evidence, {"status": "blocked"})
    assert len(dispositions) == 14
    assert dispositions[-1]["task_id"] == "exp7301-capstone"
    assert dispositions[-1]["artifact_sha256"] is None
    assert {row["disposition_class"] for row in dispositions} >= {
        "blocked",
        "disqualified",
        "null",
        "circular_positive",
        "positive",
    }


def test_scenario_report_7301_replays_selected_yaml_gates(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7301: all eight gates retain exact observed values."""

    contract, evidence = repository_state
    rows = capstone.replay_gates(contract["tasks"], evidence)
    assert len(rows) == 8
    failed = [row for row in rows if row["passed"] is False]
    assert [(row["upstream"], row["artifact_field"]) for row in failed] == [
        ("exp7289-arc-boundary", "arc_boundary_ready_score")
    ]
    assert failed[0]["observed_value"] == 0
    assert failed[0]["expected_value"] == 1


def test_scenario_report_7301_audit_checks_include_every_required_score(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-7301-CLAIMS: required scores keep expected and observed."""

    _, evidence = repository_state
    rows = capstone.audit_score_rows(evidence)
    assert [(row["task_id"], row["artifact_field"]) for row in rows] == list(
        capstone.REQUIRED_SCORE_FIELDS
    )
    by_field = {(row["task_id"], row["artifact_field"]): row for row in rows}
    assert (
        by_field[("exp7290-arc-selfparse", "arc_capture_complete_score")]["observed_value"] is None
    )
    assert by_field[("exp7294-reuse-audit", "reuse_audit_complete_score")]["passed"] is True
    assert by_field[("exp7297-mixture-audit", "mixture_promotion_score")]["observed_value"] == 0
    assert by_field[("exp7299-snapshot-cost", "snapshot_value_score")]["passed"] is False
    assert (
        by_field[("exp7300-board-continuity", "board_continuity_complete_score")]["passed"] is True
    )


def test_scenario_report_7301_claims_recompute_real_denominators(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-7301-CLAIMS: costs, labels, and memory come from rows."""

    _, evidence = repository_state
    rows = capstone.recompute_branch_rows(evidence)
    by_branch = {row["branch"]: row for row in rows}
    reuse = by_branch["source_materialization"]
    assert reuse["denominators"]["comparative_rows"] == 384
    assert reuse["denominators"]["unique_claims"] == 128
    assert reuse["denominators"]["unique_model_calls"] == 672
    assert reuse["cost"]["cold_total_s"] == pytest.approx(789.7677396875806)
    mixture = by_branch["online_hypothesis_retention"]
    assert mixture["denominators"]["prediction_rows"] == 150_528
    assert mixture["denominators"]["shared_label_arrivals"] == 6_144
    assert mixture["denominators"]["maximum_bounded_memory_bytes"] == 13_831
    assert mixture["verdict_class"] == "null"
    storage = by_branch["host_durability"]
    assert storage["denominators"]["event_rows"] == 36_864
    assert storage["denominators"]["trial_units"] == 144
    assert storage["value_score"] == 0
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


def test_scenario_report_7301_terminal_block_has_all_diagnostics(
    built_artifact: tuple[dict[str, object], Path, Path],
) -> None:
    """SCENARIO-REPORT-7301-BLOCKED: unavailable ARC science blocks once."""

    artifact, output, checkpoint = built_artifact
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert str(artifact["honest_verdict"]).startswith("blocked_")
    assert artifact["capstone_complete_score"] == 1
    assert len(artifact["task_dispositions"]) == 14
    assert len(artifact["contract_rows"]) == 14
    failures = artifact["gate_check_summary"]["failures"]
    assert any(
        row["upstream"] == "exp7290-arc-selfparse"
        and row["artifact_field"] == "arc_capture_complete_score"
        and row["observed_value"] is None
        and row["expected_value"] == 1
        for row in failures
    )
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert not any(artifact["invocation_counts"].values())
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["prd_gap_assessment"]["FR11"]["closed"] is False
    assert artifact["prd_gap_assessment"]["FR12"]["closed"] is False
    assert artifact["prd_gap_assessment"]["NFR-01"]["target_speedup"] == 10.0
    assert output.is_file() and checkpoint.is_file()
    assert capstone.validate_artifact(artifact, root=ROOT) == []


def test_scenario_report_7301_retirement_stays_at_exact_scope(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-7301-RETIREMENT: immutable failed mechanisms stay closed."""

    contract, evidence = repository_state
    rows = capstone.retirement_decisions(contract["tasks"], evidence)
    guards = {row["scope"]: row for row in rows if row["decision"] == "remain_retired"}
    assert set(capstone.PROTECTED_RETIRED_SCOPES) <= set(guards)
    assert all(row["retry_current_mechanism"] is False for row in rows)
    mixture = next(row for row in rows if row["task_id"] == "exp7297-mixture-audit")
    assert mixture["producer_retirement_signal"] is True
    assert "bounded fixed-share" in mixture["scope"]
    assert mixture["lawful_next_condition"]


@pytest.mark.parametrize(
    ("field", "replacement", "expected_error"),
    [
        ("capstone_complete_score", 0, "capstone_complete_score"),
        ("verdict_class", "partial", "terminal_state"),
        ("reproducibility_checksum", "sha256:bad", "reproducibility_checksum"),
    ],
)
def test_scenario_report_7301_validator_rejects_mutations(
    built_artifact: tuple[dict[str, object], Path, Path],
    field: str,
    replacement: object,
    expected_error: str,
) -> None:
    """SCENARIO-REPORT-7301-ARTIFACT: terminal mutations fail closed."""

    changed = deepcopy(built_artifact[0])
    changed[field] = replacement
    assert expected_error in capstone.validate_artifact(changed)


def test_req_report_7301_input_and_cli_failures(
    built_artifact: tuple[dict[str, object], Path, Path], tmp_path: Path
) -> None:
    """REQ-REPORT-7301: malformed inputs and dates cannot publish."""

    with pytest.raises(ValueError, match="run date"):
        capstone.build_artifact(
            ROOT,
            "20260913",
            tmp_path / "bad.json",
            tmp_path / "bad-checkpoint.json",
            raw_dir=tmp_path / "bad-raw",
        )
    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON root"):
        capstone.read_json(path)
    output = tmp_path / "artifact.json"
    output.write_text(json.dumps(built_artifact[0]), encoding="utf-8")
    assert capstone.main(["--root", str(ROOT), "--artifact-path", str(output), "--validate"]) == 0
    output.write_text("{}", encoding="utf-8")
    assert capstone.main(["--root", str(ROOT), "--artifact-path", str(output), "--validate"]) == 1


def test_scenario_report_7301_missing_evidence_changes_terminal_state(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-7301-ARTIFACT: missing required science stays blocked."""

    _, evidence = repository_state
    changed = deepcopy(evidence)
    changed["exp7297-mixture-audit"].update(
        {"selected_evidence_path": None, "payload": {}, "authenticated": False}
    )
    terminal = capstone.terminal_state(changed, capstone.recompute_branch_rows(changed))
    assert terminal["verdict_class"] == "blocked"
    assert any(
        row["upstream"] == "exp7297-mixture-audit"
        and row["failed_check"] == "required_science_evidence_available"
        for row in terminal["gate_check_summary"]["failures"]
    )


def test_req_report_7301_contract_and_block_inputs_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7301: malformed authorities and conductor blocks are rejected."""

    roadmap = capstone.yaml.safe_load((ROOT / capstone.ROADMAP_PATH).read_bytes())
    with monkeypatch.context() as patch:
        patch.setattr(capstone.yaml, "safe_load", lambda _value: [])
        with pytest.raises(ValueError, match="root"):
            capstone.load_contract(ROOT)
    with monkeypatch.context() as patch:
        bad = deepcopy(roadmap)
        bad["milestone"] = "2026.09.640"
        patch.setattr(capstone.yaml, "safe_load", lambda _value: bad)
        with pytest.raises(ValueError, match="not V641"):
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
    assert capstone._conductor_block_errors("exp7290-arc-selfparse", {}) == [
        "conductor_block_lifecycle",
        "conductor_block_identity",
        "conductor_block_gate",
    ]
    with monkeypatch.context() as patch:
        patch.setattr(
            capstone.source_contract,
            "validate_artifact",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("fixture")),
        )
        assert capstone._validate_payload("exp7288-source-contract", {}, ROOT) == [
            "validator_exception:ValueError:fixture"
        ]


def test_req_report_7301_missing_path_and_gate_outcomes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7301-INTAKE: missing files and gate values remain distinct."""

    task = {
        "id": "exp7294-reuse-audit",
        "deliverable": "results/declared-missing.json",
    }
    row = capstone.load_evidence(tmp_path, task, {})
    assert row["evidence_source"] == "missing"
    assert row["disposition_class"] == "absent"

    tasks = [
        {"id": "producer", "milestone": capstone.MILESTONE},
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
    producer["payload"] = {"ready": 0}
    assert capstone.replay_gates(tasks, {"producer": producer})[0]["outcome"] == "value_mismatch"


def test_req_report_7301_branch_promotion_classes() -> None:
    """SCENARIO-REPORT-7301-CLAIMS: oracle and admissibility control promotion."""

    source = {
        "task_id": "fixture",
        "selected_evidence_path": "fixture.json",
        "payload": {"status": "complete", "verifier_is_oracle": True},
        "quarantine_state": {"quarantined": False},
        "accepted_for_positive_claim": True,
        "disposition_class": "circular_positive",
    }
    circular = capstone._branch_row("fixture", source, 1, 1, {}, None)
    assert circular["verdict_class"] == "circular_positive"
    source["payload"]["verifier_is_oracle"] = False
    assert capstone._branch_row("fixture", source, 1, 1, {}, None)["verdict_class"] == "positive"
    source["accepted_for_positive_claim"] = False
    assert capstone._branch_row("fixture", source, 1, 1, {}, None)["verdict_class"] == (
        "disqualified"
    )


def test_scenario_report_7301_terminal_null_positive_and_authentication(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """SCENARIO-REPORT-7301-BLOCKED: complete null, positive, and blocks differ."""

    _, original = repository_state
    evidence = deepcopy(original)
    evidence["exp7289-arc-boundary"]["quarantine_state"] = {"quarantined": False}
    field_by_task = dict(capstone.REQUIRED_SCORE_FIELDS)
    for task_id in capstone.REQUIRED_SCIENCE_TASKS:
        source = evidence[task_id]
        source["selected_evidence_path"] = "fixture.json"
        source["payload"]["status"] = "complete"
        source["quarantine_state"] = {"quarantined": False}
        source["producer_validation_errors"] = []
        source["disposition_class"] = "null"
        source["accepted_for_positive_claim"] = True
    for task_id, field in capstone.REQUIRED_SCORE_FIELDS:
        evidence[task_id]["payload"][field] = 1 if field in capstone.COMPLETENESS_FIELDS else 0
    branches = capstone.recompute_branch_rows(evidence)
    assert capstone.terminal_state(evidence, branches)["verdict_class"] == "null"

    evidence["exp7294-reuse-audit"]["payload"][field_by_task["exp7294-reuse-audit"]] = 1
    evidence["exp7294-reuse-audit"]["payload"]["verifier_is_oracle"] = False
    branches = capstone.recompute_branch_rows(evidence)
    assert capstone.terminal_state(evidence, branches)["verdict_class"] == "positive"

    evidence["exp7297-mixture-audit"]["quarantine_state"] = {"quarantined": True}
    blocked = capstone.terminal_state(evidence, branches)
    assert any(
        row["failed_check"] == "required_science_not_quarantined"
        for row in blocked["gate_check_summary"]["failures"]
    )
    evidence["exp7297-mixture-audit"]["quarantine_state"] = {"quarantined": False}
    evidence["exp7297-mixture-audit"]["producer_validation_errors"] = ["fixture"]
    evidence["exp7297-mixture-audit"]["disposition_class"] = "null"
    blocked = capstone.terminal_state(evidence, branches)
    assert any(
        row["failed_check"] == "required_science_authentic"
        for row in blocked["gate_check_summary"]["failures"]
    )


def test_req_report_7301_missing_receipts_and_stale_hash(
    built_artifact: tuple[dict[str, object], Path, Path], tmp_path: Path
) -> None:
    """REQ-REPORT-7301: absent receipts and changed source bytes are visible."""

    assert capstone._load_validation_receipts(ROOT, None) == []
    changed = deepcopy(built_artifact[0])
    path = tmp_path / "changed-source.txt"
    path.write_text("current\n", encoding="utf-8")
    changed["source_artifact_hashes"] = {
        str(path): {
            "sha256": "sha256:bad",
            "authority": "fixture",
            "terminal_class": None,
            "quarantined": False,
            "retired": False,
        }
    }
    changed["reproducibility_checksum"] = capstone.artifact_checksum(changed)
    assert "source_artifact_hashes" in capstone.validate_artifact(changed, root=ROOT)


@pytest.mark.parametrize("failure_call", [1, 2])
def test_scenario_report_7301_build_refuses_invalid_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_call: int
) -> None:
    """SCENARIO-REPORT-7301-ARTIFACT: both atomic validation stages fail closed."""

    calls = 0

    def forced_validation(_artifact: object, root: Path | None = None) -> list[str]:
        del root
        nonlocal calls
        calls += 1
        return ["forced"] if calls == failure_call else []

    with monkeypatch.context() as patch:
        patch.setattr(capstone, "validate_artifact", forced_validation)
        with pytest.raises(RuntimeError, match="validation failed"):
            capstone.build_artifact(
                ROOT,
                capstone.RUN_DATE,
                tmp_path / f"result-{failure_call}.json",
                tmp_path / f"checkpoint-{failure_call}.json",
                raw_dir=tmp_path / f"raw-{failure_call}",
            )


def test_req_report_7301_main_build_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """REQ-REPORT-7301: the thin CLI forwards explicit build paths."""

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
