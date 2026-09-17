"""Behavior tests for the V646 capstone.

Spec refs: REQ-REPORT-7368 and SCENARIO-REPORT-7368-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import pytest
import yaml

from carnot import experiment_7368_v646_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def repository_state() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Load the fixed contract and evidence once for the repository checks."""

    contract = capstone.load_contract(ROOT)
    evidence = capstone.collect_evidence(ROOT, contract["tasks"])
    return contract, evidence


def _passing_validation() -> dict[str, object]:
    receipts = [
        {
            "name": name,
            "command": f"command for {name}",
            "command_argv": ["python", name],
            "scope": "test",
            "exit_code": 0,
            "duration_s": 0.01,
            "log_path": f"/tmp/{name}.log",
            "log_sha256": "sha256:" + "a" * 64,
            "passed": True,
            "timed_out": False,
        }
        for name in capstone.REQUIRED_VALIDATION_NAMES
    ]
    return {
        "validation_receipts": receipts,
        "affected_validation_passed": True,
        "full_suite_passed": True,
        "terminal_validation_passed": True,
        "flagged_adversarial": False,
        "repository_health": {
            "status": "healthy",
            "as_of": "2026-09-17",
            "affects_required_checks": False,
            "historical_failures": [],
        },
    }


# REQ-REPORT-7368 / SCENARIO-REPORT-7368-DISPOSITIONS
def test_contract_has_exact_twelve_ordered_tasks(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    contract, _ = repository_state
    assert contract["milestone"] == capstone.MILESTONE
    assert [row["id"] for row in contract["tasks"]] == list(capstone.EXPECTED_TASK_IDS)
    assert contract["tasks"][-1]["deliverable"] == capstone.RESULT_PATH.as_posix()
    assert contract["contract_sha256"].startswith("sha256:")


# REQ-REPORT-7368 / SCENARIO-REPORT-7368-DISPOSITIONS
def test_evidence_uses_only_declared_artifacts_or_canonical_gate_records(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    _, evidence = repository_state
    assert set(evidence) == set(capstone.EXPECTED_TASK_IDS[:-1])
    canonical = {
        task_id for task_id, row in evidence.items() if row["source_kind"] == "conductor_pre_gate"
    }
    assert canonical == {"exp7363-learning-audit", "exp7366-supervisor-live"}
    assert evidence["exp7363-learning-audit"]["actual_path"] == (
        "results/experiment_7363_learning_audit.json"
    )
    assert evidence["exp7366-supervisor-live"]["actual_path"] == (
        "results/experiment_7366_supervisor_live.json"
    )
    assert evidence["exp7362-prospective-learning"]["verdict_class"] == "disqualified"
    assert evidence["exp7365-supervisor-support"]["verdict_class"] == "null"
    assert evidence["exp7367-board-disposition"]["verdict_class"] == "blocked"
    assert all(row["sha256"].startswith("sha256:") for row in evidence.values())


# REQ-REPORT-7368 / SCENARIO-REPORT-7368-DISPOSITIONS
def test_ineligible_or_missing_producers_fail_closed(tmp_path: Path) -> None:
    task = {
        "id": "exp7363-learning-audit",
        "deliverable": "results/experiment_7363_v646_learning_audit.json",
    }
    missing = capstone.load_evidence_slot(tmp_path, task, {})
    assert missing["source_kind"] == "missing"
    assert missing["authenticated"] is False
    assert missing["accepted_for_science"] is False

    payload = {
        "experiment_id": "exp7363-learning-audit",
        "milestone": capstone.MILESTONE,
        "status": "complete_partial",
        "verdict_class": "partial",
        "flagged_adversarial": False,
    }
    selected = tmp_path / str(task["deliverable"])
    selected.parent.mkdir(parents=True)
    selected.write_text(json.dumps(payload), encoding="utf-8")
    partial = capstone.load_evidence_slot(tmp_path, task, {})
    assert partial["verdict_class"] == "partial"
    assert partial["accepted_for_science"] is False

    payload["status"] = "complete"
    payload["verdict_class"] = "null"
    payload["flagged_adversarial"] = True
    selected.write_text(json.dumps(payload), encoding="utf-8")
    flagged = capstone.load_evidence_slot(tmp_path, task, {})
    assert flagged["flagged_adversarial"] is True
    assert flagged["accepted_for_science"] is False


# REQ-REPORT-7368 / SCENARIO-REPORT-7368-CLAIMS
def test_claim_reduction_separates_current_historical_live_and_synthetic(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    _, evidence = repository_state
    rows = capstone.reduce_claim_rows(ROOT, evidence)
    by_name = {row["claim"]: row for row in rows}
    assert list(by_name) == list(capstone.CLAIM_NAMES)

    source = by_name["fresh_source_fidelity"]
    assert source["evidence_period"] == "current"
    assert source["truth_authority"] == "oracle_defined"
    assert source["metrics"] == {
        "row_count": 128,
        "parser_valid_count": 128,
        "public_semantic_correct_count": 128,
        "executor_valid_count": 6,
        "all_fidelity_dimension_count": 128,
    }

    learning = by_name["structural_learning"]
    assert learning["metrics"]["raw_row_count"] == 1792
    assert learning["metrics"]["synthetic_stream_count"] == 128
    assert learning["metrics"]["live_stream_count"] == 32
    assert learning["metrics"]["structural_witness_count"] == 0
    assert learning["completion_score"] == 0
    assert learning["value_score"] == 0

    assert by_name["learning_audit"]["verdict_class"] == "blocked"
    assert by_name["acquisition_adjudication"]["metrics"]["paired_context_count"] == 42
    assert by_name["acquisition_adjudication"]["metrics"]["complete_cost_upper_95"] == (
        pytest.approx(0.9652615826317683)
    )
    supervisor = by_name["supervisor_support_and_live"]
    assert supervisor["completion_score"] == 1
    assert supervisor["value_score"] == 0
    assert supervisor["metrics"]["supported_decision_count"] == 0
    assert supervisor["metrics"]["live_trial_executed"] is False
    assert supervisor["metrics"]["live_skip_accounted"] is True
    boards = by_name["board_dispositions"]
    assert boards["metrics"]["board_row_count"] == 3
    assert boards["metrics"]["native_tenfold_speed_gate"] is None


# REQ-REPORT-7368 / SCENARIO-REPORT-7368-CLAIMS
def test_historical_live_arc_pairs_recompute_from_raw_episode_rows() -> None:
    rows = capstone.reduce_historical_live_arc_pairs(ROOT)
    assert [row["game"] for row in rows] == ["r11l", "re86"]
    assert all(row["evidence_period"] == "historical" for row in rows)
    assert all(row["cohort"] == "live" for row in rows)
    assert all(row["metrics"]["level_delta_resume_minus_withheld"] == 0 for row in rows)
    assert all(row["metrics"]["action_delta_resume_minus_withheld"] == 0 for row in rows)
    assert all(row["metrics"]["usable_answer_delta_resume_minus_withheld"] == 0 for row in rows)
    assert all(row["verdict_class"] == "null" for row in rows)


# REQ-REPORT-7368 / SCENARIO-REPORT-7368-NULLS
def test_required_failure_disqualifies_but_optional_blocks_do_not(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    _, evidence = repository_state
    terminal = capstone.terminal_state(evidence, required_validation_passed=True)
    assert terminal["verdict_class"] == "disqualified"
    assert terminal["required_science_complete_score"] == 0
    failures = terminal["gate_check_summary"]["failures"]
    assert failures[0]["upstream"] == "exp7362-prospective-learning"
    assert failures[0]["artifact_field"] == "acceptance_gate_results.affected_validation"
    assert failures[0]["expected"][-1] == "verdict_row_consistency_strict"
    assert "full_python_suite" not in failures[0]["observed"]
    assert all(row["upstream"] != "exp7366-supervisor-live" for row in failures)
    assert all(row["upstream"] != "exp7367-board-disposition" for row in failures)

    missing = deepcopy(evidence)
    missing["exp7361-fresh-plan-capture"] = {
        **missing["exp7361-fresh-plan-capture"],
        "source_kind": "missing",
        "authenticated": False,
        "actual_path": None,
    }
    blocked = capstone.terminal_state(missing, required_validation_passed=True)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["first_failure"]["upstream"] == (
        "exp7361-fresh-plan-capture"
    )


# REQ-REPORT-7368 / SCENARIO-REPORT-7368-RETIREMENTS
def test_exact_repeat_retires_and_manifest_records_it(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    contract, evidence = repository_state
    terminal = capstone.terminal_state(evidence, required_validation_passed=True)
    decisions = capstone.retirement_decisions(ROOT, contract["tasks"], evidence, terminal)
    exact = [row for row in decisions if row.get("decision") == "retire_exact_repeat"]
    assert [(row["task_id"], row["prior_experiment_id"]) for row in exact] == [
        ("exp7363-learning-audit", "exp7349-prospective-learning")
    ]
    assert exact[0]["recorded_predecessor_honest_verdict"] == "blocked_gate_check_failed"
    branches = {row["branch"]: row for row in decisions if row["kind"] == "branch"}
    assert branches["native_tenfold_null"]["decision"] == "preserve_retirement"
    assert branches["external_text_scoring"]["decision"] == "preserve_retirement"
    assert branches["unchanged_acquisition"]["decision"] == "preserve_retirement"
    assert branches["result_resume"]["decision"] == "preserve_retirement"

    manifest = yaml.safe_load((ROOT / capstone.EXCLUSION_PATH).read_text(encoding="utf-8"))
    receipts = {row["id"]: row for row in manifest["retired_extras"] if "id" in row}
    assert receipts[capstone.EXCLUSION_RECEIPT_ID]["retired_by_artifact"] == (
        "results/experiment_7363_learning_audit.json"
    )
    assert receipts[capstone.EXCLUSION_RECEIPT_ID]["recorded_by_artifact"] == (
        capstone.RESULT_PATH.as_posix()
    )


# REQ-REPORT-7368 / SCENARIO-REPORT-7368-PUBLICATION
def test_publication_gates_are_independent_and_fail_closed(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    _, evidence = repository_state
    gates = capstone.publication_gate_results(
        evidence,
        capstone.reduce_claim_rows(ROOT, evidence),
        current_validation_passed=True,
    )
    assert list(gates) == ["G1", "G2", "G3", "G4"]
    assert all(set(row) >= {"expected", "observed", "passed"} for row in gates.values())
    assert gates["G1"]["passed"] is False
    assert gates["G2"]["passed"] is False
    assert gates["G3"]["passed"] is True
    assert gates["G4"]["passed"] is True


# REQ-REPORT-7368 / SCENARIO-REPORT-7368-ARTIFACT
def test_terminal_artifact_has_all_required_fields_and_replays(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
    tmp_path: Path,
) -> None:
    contract, evidence = repository_state
    sidecar = tmp_path / "historical-model-receipts.json"
    capstone.write_historical_inference_sidecar(sidecar, evidence, ROOT)
    artifact = capstone.build_artifact(
        root=ROOT,
        contract=contract,
        evidence=evidence,
        validation=_passing_validation(),
        historical_sidecar=sidecar,
        started_at_utc="2026-09-17T15:00:00+00:00",
        completed_at_utc="2026-09-17T15:00:01+00:00",
        duration_s=1.0,
        phase_spans=capstone.zero_test_phase_spans(),
    )
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"]["current"] == capstone.ZERO_INVOCATION_COUNTS
    assert artifact["milestone_disposition_complete_score"] == 1
    assert artifact["required_science_complete_score"] == 0
    assert artifact["publication_ready_score"] == 0
    assert (
        artifact["readiness_score"] == artifact["value_score"] == artifact["promotion_score"] == 0
    )
    assert len(artifact["disposition_rows"]) == 12
    assert artifact["disposition_rows"][-1]["task_id"] == "exp7368-capstone"
    assert artifact["disposition_rows"][-1]["artifact_sha256"] is None
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["flagged_adversarial"] is False
    assert artifact["estimated_operational_savings"] == 0
    assert artifact["publication_performed"] is False
    assert capstone.validate_artifact(artifact, root=ROOT, replay=True) == []

    mutated = deepcopy(artifact)
    mutated["rows"][0]["metrics"]["row_count"] = 127
    assert "rows" in capstone.validate_artifact(mutated, root=ROOT, replay=True)
    mutated = deepcopy(artifact)
    mutated["MODEL_SPECS"] = [{"hf_id": "unsloth/Qwen3.8-27B-GGUF"}]
    assert "model_boundary" in capstone.validate_artifact(mutated, root=ROOT)


# REQ-REPORT-7368 / SCENARIO-REPORT-7368-ARTIFACT
def test_exp7358_plan_is_explicit_and_entrypoint_is_thin(tmp_path: Path) -> None:
    commands = capstone.build_validation_plan(ROOT, tmp_path)
    assert capstone.validate_validation_plan(ROOT, commands) == []
    assert [row.name for row in commands] == list(capstone.AFFECTED_VALIDATION_NAMES)
    assert all(
        "tests/python/test_experiment_7368_v646_capstone.py" in " ".join(row.argv)
        or row.name in {"worktree_imports", "changed_module_coverage_report", "changed_module_mypy"}
        for row in commands
    )

    full = capstone.build_full_suite_command(ROOT, tmp_path)
    assert full.name == "full_python_suite"
    assert full.argv[1:6] == ("tests/python", "-q", "-n", "0", "-o")
    assert "--no-cov" in full.argv
    assert any(arg.startswith("--basetemp=") for arg in full.argv)

    wrapper = (ROOT / capstone.WRAPPER_PATH).read_text(encoding="utf-8")
    assert "experiment_7368_v646_capstone import main" in wrapper
    assert "def " not in wrapper
    assert "subprocess" not in wrapper


# REQ-REPORT-7368 / SCENARIO-REPORT-7368-ARTIFACT
def test_utility_failures_and_disposition_mutations_are_detected(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
    tmp_path: Path,
) -> None:
    contract, evidence = repository_state
    terminal = capstone.terminal_state(evidence, required_validation_passed=True)
    rows = capstone.build_disposition_rows(contract["tasks"], evidence, terminal)
    assert [row["task_id"] for row in rows] == list(capstone.EXPECTED_TASK_IDS)
    assert rows[-1]["evidence_source"] == "capstone_self"

    bad = deepcopy(evidence)
    bad["exp7365-supervisor-support"]["payload"] = {}
    with pytest.raises(ValueError, match="support_rows"):
        capstone.reduce_claim_rows(ROOT, bad)

    with pytest.raises(ValueError, match="four rows"):
        capstone.reduce_historical_live_arc_pairs(tmp_path)

    assert capstone.validate_artifact([]) == ["artifact_not_mapping"]


# REQ-REPORT-7368 / SCENARIO-REPORT-7368-DISPOSITIONS
def test_contract_and_canonical_gate_reject_identity_drift(tmp_path: Path) -> None:
    source = yaml.safe_load((ROOT / capstone.ROADMAP_PATH).read_text(encoding="utf-8"))
    target = tmp_path / capstone.ROADMAP_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    def rejected(document: dict[str, object], message: str) -> None:
        target.write_text(yaml.safe_dump(document), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            capstone.load_contract(tmp_path)

    wrong = deepcopy(source)
    wrong["milestone"] = "wrong"
    rejected(wrong, "must name milestone")

    wrong = deepcopy(source)
    wrong["tasks"] = wrong["tasks"][:-1]
    rejected(wrong, "exact twelve")

    wrong = deepcopy(source)
    wrong["tasks"][0]["milestone"] = "wrong"
    rejected(wrong, "task milestone mismatch")

    wrong = deepcopy(source)
    wrong["tasks"][0]["deliverable"] = "results/wrong.json"
    rejected(wrong, "task deliverable mismatch")

    upstream = capstone.EXPECTED_EVIDENCE_STATES["exp7362-prospective-learning"]["path"]
    upstream_path = tmp_path / upstream
    upstream_path.parent.mkdir(parents=True, exist_ok=True)
    upstream_path.write_text("{}\n", encoding="utf-8")
    canonical = tmp_path / "canonical.json"
    canonical.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical conductor pre-gate record mismatch"):
        capstone._canonical_gate_payload(
            tmp_path,
            "exp7363-learning-audit",
            canonical,
            capstone.CANONICAL_GATE_RECORDS["exp7363-learning-audit"],
        )


# REQ-REPORT-7368 / SCENARIO-REPORT-7368-CLAIMS
def test_reducers_reject_each_malformed_required_row_family(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
    tmp_path: Path,
) -> None:
    _, evidence = repository_state
    cases = (
        ("exp7361-fresh-plan-capture", "source_fidelity_rows", "source_fidelity_rows"),
        ("exp7362-prospective-learning", "rows", "learning rows"),
        ("exp7364-acquisition-adjudication", "paired_cost_rows", "paired_cost_rows"),
        ("exp7367-board-disposition", "board_rows", "board_rows"),
    )
    for task_id, field, message in cases:
        malformed = deepcopy(evidence)
        malformed[task_id]["payload"][field] = None
        with pytest.raises(ValueError, match=message):
            capstone.reduce_claim_rows(ROOT, malformed)

    raw_path = tmp_path / capstone.HISTORICAL_ARC_ROWS
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(json.dumps({"rows": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="four rows"):
        capstone.reduce_historical_live_arc_pairs(tmp_path)
    raw_path.write_text(
        json.dumps(
            {
                "rows": [
                    {"game": "r11l", "arm": "wrong"},
                    {"game": "r11l", "arm": "also_wrong"},
                    {"game": "re86", "arm": "wrong"},
                    {"game": "re86", "arm": "also_wrong"},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="four rows"):
        capstone.reduce_historical_live_arc_pairs(tmp_path)


# REQ-REPORT-7368 / SCENARIO-REPORT-7368-NULLS
def test_terminal_reduction_covers_validation_block_and_completed_null(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    _, evidence = repository_state
    invalid = capstone.terminal_state(evidence, required_validation_passed=False)
    assert invalid["gate_check_summary"]["first_failure"]["failed_check"] == (
        "current_required_validation"
    )

    blocked = deepcopy(evidence)
    blocked["exp7362-prospective-learning"]["verdict_class"] = "null"
    blocked["exp7362-prospective-learning"]["accepted_for_science"] = True
    blocked["exp7363-learning-audit"]["accepted_for_science"] = True
    assert capstone.terminal_state(blocked, required_validation_passed=True)["verdict_class"] == (
        "blocked"
    )

    completed = deepcopy(blocked)
    completed["exp7362-prospective-learning"]["payload"]["learning_capture_complete_score"] = 1
    assert (
        capstone.terminal_state(completed, required_validation_passed=True)["verdict_class"]
        == "null"
    )


# REQ-REPORT-7368 / SCENARIO-REPORT-7368-RETIREMENTS
def test_retirement_helpers_fail_closed_and_preserve_declaration_mismatch(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
    tmp_path: Path,
) -> None:
    contract, evidence = repository_state
    with pytest.raises(ValueError, match="one predecessor artifact"):
        capstone._predecessor(tmp_path, "not-an-experiment")

    manifest = tmp_path / capstone.EXCLUSION_PATH
    manifest.parent.mkdir(parents=True)
    manifest.write_text("retired_extras: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one V646 learning-audit retirement"):
        capstone._retirement_manifest_receipt(tmp_path)

    tasks = deepcopy(contract["tasks"])
    task_with_prior = next(task for task in tasks if task.get("prior_failures"))
    task_with_prior["prior_failures"][0]["verdict"] = "deliberate_mismatch"
    terminal = capstone.terminal_state(evidence, required_validation_passed=True)
    decisions = capstone.retirement_decisions(ROOT, tasks, evidence, terminal)
    assert any(row.get("decision") == "preserve_declaration_mismatch" for row in decisions)


# REQ-REPORT-7368 / SCENARIO-REPORT-7368-ARTIFACT
def test_sidecars_receipt_shape_and_replay_missing_source_fail_closed(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
    tmp_path: Path,
) -> None:
    contract, evidence = repository_state
    historical = tmp_path / "historical.json"
    capstone.write_historical_inference_sidecar(historical, evidence, ROOT)
    independent = tmp_path / "independent.json"
    sidecar = capstone.write_independent_reduction_sidecar(independent, ROOT, evidence)
    assert sidecar["schema"].endswith("independent_raw_reduction.v1")
    hashes = capstone._source_hashes(ROOT, contract, evidence, historical, independent)
    assert str(independent.resolve()) in hashes

    assert capstone._receipt_names_complete(None) is False
    assert capstone._receipt_names_complete([]) is False
    malformed = [{"name": name} for name in capstone.REQUIRED_VALIDATION_NAMES]
    assert capstone._receipt_names_complete(malformed) is False

    artifact = capstone.build_artifact(
        root=ROOT,
        contract=contract,
        evidence=evidence,
        validation=_passing_validation(),
        historical_sidecar=historical,
        independent_sidecar=independent,
        started_at_utc="2026-09-17T15:00:00+00:00",
        completed_at_utc="2026-09-17T15:00:01+00:00",
        duration_s=1.0,
        phase_spans=capstone.zero_test_phase_spans(),
    )
    missing = deepcopy(artifact)
    missing["historical_inference_sidecars"][0]["path"] = str(tmp_path / "absent.json")
    assert "source_artifact_hashes" in capstone.validate_artifact(missing, root=ROOT, replay=True)


# REQ-REPORT-7368 / SCENARIO-REPORT-7368-ARTIFACT
def test_execution_helpers_delegate_exact_bounded_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capstone.progress(time.monotonic(), "test", "start")
    assert "phase=test event=start" in capsys.readouterr().out

    command = capstone.scoped.CommandSpec("focused_pytest", ("pytest",), "one-test")
    captured: dict[str, object] = {}

    def fake_run(root: Path, planned: object, *, log_dir: Path) -> list[dict[str, object]]:
        captured["planned"] = planned
        return [{"name": "focused_pytest"}]

    def fake_reduce(root: Path, manifest: object, receipts: object) -> dict[str, object]:
        captured["receipts"] = receipts
        return {"required_checks_passed": True}

    monkeypatch.setattr(capstone.validation_boundary, "run_categorized_commands", fake_run)
    monkeypatch.setattr(capstone.validation_boundary, "reduce_affected_receipts", fake_reduce)
    receipts, reduced = capstone.run_affected_validation(ROOT, [command], tmp_path)
    assert receipts == [{"name": "focused_pytest"}]
    assert reduced["required_checks_passed"] is True
    assert len(captured["planned"]) == 1

    def fake_commands(root: Path, commands: object, *, log_dir: Path) -> list[dict[str, object]]:
        captured["terminal_names"] = [row.name for row in commands]
        return []

    monkeypatch.setattr(capstone.scoped, "run_commands", fake_commands)
    assert capstone.run_terminal_validation(ROOT, tmp_path / "candidate.json", tmp_path) == []
    assert captured["terminal_names"] == [
        "independent_raw_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]

    assert capstone._task_number("not-an-experiment") is None
    assert capstone._path_label(tmp_path, ROOT) == str(tmp_path.resolve())
    assert capstone._resolve_label(ROOT, str(tmp_path)) == tmp_path
    assert capstone._span("test", 2.0, 3.0, 1.0, 4) == {
        "phase": "test",
        "start_s": 1.0,
        "end_s": 2.0,
        "duration_s": 1.0,
        "units": 4,
    }
