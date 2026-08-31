"""Tests for the V595 terminal branch disposition.

Spec refs: REQ-REPORT-6823 and SCENARIO-REPORT-6823-*.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6823_v595_branch_disposition as exp


REPO = Path(__file__).resolve().parents[2]


def _record(
    verdict: str = "positive",
    rows: list[dict] | None = None,
    *,
    state: str = "present",
    flagged: bool = False,
    payload: dict | None = None,
) -> dict:
    body = {"verdict_class": verdict, "honest_verdict": f"complete: {verdict}"}
    body["rows"] = rows or []
    if payload:
        body.update(payload)
    return {
        "artifact_state": state,
        "path": "results/source.json",
        "sha256": "sha256:" + "a" * 64 if state == "present" else None,
        "payload": body if state == "present" else None,
        "source_verdict_class": verdict if state == "present" else state,
        "terminal_class": "flagged" if flagged else verdict if state == "present" else state,
        "eligible": state == "present" and verdict in {"positive", "null"} and not flagged,
        "adversarial_report": {"flag_count": 1 if flagged else 0, "flags": []},
        "row_lint": {"status": "ok", "findings": []},
        "error": None,
    }


def _records() -> dict[str, dict]:
    return {task_id: _record(state="missing") for task_id in exp.SOURCE_TASK_IDS}


def test_spec_anchor_design_and_manifest_have_fourteen_exact_tasks() -> None:
    """SCENARIO-REPORT-6823-IDENTITY: design, spec, and roadmap retain all IDs."""

    spec = (REPO / exp.REPORT_SPEC_PATH).read_text(encoding="utf-8")
    section = spec.split("REQ-REPORT-6823", 1)[1]
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(set(section.split("`")))
    for name in (
        "SCENARIO-REPORT-6823-IDENTITY",
        "SCENARIO-REPORT-6823-ELIGIBILITY",
        "SCENARIO-REPORT-6823-MANIFEST",
        "SCENARIO-REPORT-6823-DISPOSITION",
    ):
        assert name in section

    design = exp.parse_design((REPO / exp.DESIGN_PATH).read_text(encoding="utf-8"))
    manifest = exp.load_manifest(REPO)
    assert design["phase_count"] == 4
    assert [row["task_id"] for row in design["tasks"]] == list(exp.TASK_IDS)
    assert [row["task_id"] for row in manifest] == list(exp.TASK_IDS)
    assert exp.compare_manifest(design, manifest, exp.CONTRACT_OWNER_MAP)["differences"] == []


def test_source_loader_preserves_terminal_missing_malformed_and_current(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6823-IDENTITY: each exact path keeps its source state."""

    (tmp_path / "results").mkdir()
    good = tmp_path / exp.TASK_PATHS["exp6810"]
    good.write_text(
        json.dumps({"verdict_class": "null", "honest_verdict": "complete: fixture", "rows": []}),
        encoding="utf-8",
    )
    (tmp_path / exp.TASK_PATHS["exp6811"]).write_text("[]", encoding="utf-8")
    blocked = tmp_path / exp.TASK_PATHS["exp6813"]
    blocked.write_text(
        json.dumps(
            {
                "verdict_class": "blocked",
                "honest_verdict": "complete_blocked: fixture",
                "gate_check_summary": [
                    {"check": "resource", "expected": "ready", "observed": "missing"}
                ],
                "rows": [],
            }
        ),
        encoding="utf-8",
    )
    loaded = exp.load_source_artifacts(tmp_path)
    assert loaded["exp6810"]["terminal_class"] == "null"
    assert loaded["exp6811"]["terminal_class"] == "malformed"
    assert loaded["exp6812"]["terminal_class"] == "missing"
    assert loaded["exp6813"]["terminal_class"] == "blocked"
    assert loaded["exp6823"]["terminal_class"] == "current_synthesis"
    assert loaded["exp6813"]["gate_check_summary"][0]["observed"] == "missing"


def test_source_hashes_cover_design_roadmap_specs_validators_and_artifacts(tmp_path: Path) -> None:
    """REQ-REPORT-6823: every owned input path has a hash or explicit absence."""

    for relative in (*exp.OWNED_INPUT_PATHS, exp.TASK_PATHS["exp6810"]):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative.as_posix(), encoding="utf-8")
    hashes = exp.collect_source_hashes(tmp_path)
    expected = {path.as_posix() for path in exp.OWNED_INPUT_PATHS}
    expected.update(exp.TASK_PATHS[task].as_posix() for task in exp.SOURCE_TASK_IDS)
    assert set(hashes) == expected
    assert hashes[exp.TASK_PATHS["exp6810"].as_posix()].startswith("sha256:")
    assert hashes[exp.TASK_PATHS["exp6812"].as_posix()] is None


def test_validators_flag_and_exclude_evidence_without_erasing_it(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6823-ELIGIBILITY: flagged evidence stays visible and excluded."""

    records = _records()
    source = tmp_path / "results/source.json"
    source.parent.mkdir()
    source.write_text("{}", encoding="utf-8")
    records["exp6813"] = _record(rows=[{"arm": "selective_priority"}])
    records["exp6813"]["path"] = "results/source.json"

    def adversarial(_path: Path) -> dict:
        return {"flag_count": 1, "flags": [{"kind": "TEST_FLAG", "severity": "warn"}]}

    def row_lint(_path: Path) -> tuple[str, list[str]]:
        return "findings", ["TEST_ROW_FINDING"]

    findings = exp.apply_validators(tmp_path, records, adversarial, row_lint)
    assert records["exp6813"]["terminal_class"] == "flagged"
    assert records["exp6813"]["eligible"] is False
    assert {row["validator"] for row in findings} == {"adversarial", "row_verdict"}
    assert records["exp6813"]["payload"]["rows"] == [{"arm": "selective_priority"}]


def test_selective_reducer_uses_held_rows_and_keeps_safety_separate() -> None:
    """SCENARIO-REPORT-6823-ELIGIBILITY: hard safety and utility reduce separately."""

    records = _records()
    rows = [
        {
            "split": "held",
            "pair_id": "a",
            "arm": "flat_reject_retry",
            "accepted_hard_violation": False,
            "accepted_progress": 0,
            "retry_count": 1,
            "false_intervention": True,
            "base_already_valid": True,
            "harmful_selection": False,
        },
        {
            "split": "held",
            "pair_id": "a",
            "arm": "selective_priority",
            "accepted_hard_violation": False,
            "accepted_progress": 1,
            "retry_count": 0,
            "false_intervention": False,
            "base_already_valid": True,
            "harmful_selection": False,
        },
        {
            "split": "development",
            "pair_id": "ignored",
            "arm": "selective_priority",
            "accepted_hard_violation": True,
            "accepted_progress": 99,
            "retry_count": 99,
        },
    ]
    records["exp6813"] = _record(rows=rows)
    claim = exp.recompute_selective(records)
    assert claim["eligible_row_count"] == 2
    assert claim["hard_safety"]["selective_priority"]["rate"] == 0.0
    assert claim["utility"]["paired_progress_delta"]["mean"] == 1.0
    assert claim["utility"]["flat_minus_selective_retry_delta"]["mean"] == 1.0
    assert claim["false_intervention"]["selective_priority"]["rate"] == 0.0


def test_memory_and_live_reducers_keep_causality_portability_transport_and_progress_separate() -> (
    None
):
    """REQ-REPORT-6823: memory and live evidence classes do not pool."""

    records = _records()
    records["exp6816"] = _record(
        rows=[
            {
                "arm": "frozen_memory",
                "exact_utility": 0.2,
                "credited_write": False,
                "support_harm": False,
                "retention_harm": False,
                "hard_case_harm": False,
            },
            {
                "arm": "residual_pressure",
                "exact_utility": 0.6,
                "credited_write": True,
                "support_harm": False,
                "retention_harm": False,
                "hard_case_harm": False,
            },
        ]
    )
    records["exp6817"] = _record(
        rows=[
            {"arm": "frozen_memory", "exact_utility": 0.1, "rotation": "qwen"},
            {"arm": "residual_pressure", "exact_utility": 0.3, "rotation": "qwen"},
        ]
    )
    records["exp6820"] = _record(
        rows=[{"typed_canary_passed": True}],
        payload={"tool_gap_obligation_transport_ready": True},
    )
    records["exp6821"] = _record(
        rows=[
            {
                "pair_id": "g",
                "arm": "control_unset",
                "actions_to_progress": 10,
                "hard_violation": False,
                "false_intervention": False,
            },
            {
                "pair_id": "g",
                "arm": "obligation_routed",
                "actions_to_progress": 7,
                "hard_violation": False,
                "false_intervention": False,
            },
        ]
    )
    memory = exp.recompute_memory(records)
    live = exp.recompute_live_arc(records)
    assert memory["causality"]["credited_writes"] == 1
    assert memory["utility"]["residual_minus_frozen"] == pytest.approx(0.4)
    assert memory["portability"]["residual_minus_frozen"] == pytest.approx(0.2)
    assert memory["safety"]["harm_count"] == 0
    assert live["transport"]["ready"] is True
    assert live["progress"]["control_minus_treatment_actions"] == 3.0
    assert live["hard_safety"]["hard_violation_count"] == 0


def test_missing_or_flagged_inputs_make_only_the_affected_branch_blocked() -> None:
    """SCENARIO-REPORT-6823-DISPOSITION: missing audits fail closed by branch."""

    records = _records()
    records["exp6813"] = _record()
    records["exp6814"] = _record(flagged=True)
    decisions = exp.build_dispositions(records, exp.recompute_claims(records))
    assert decisions["selective_arbiter"]["disposition"] == "blocked"
    assert decisions["verified_route_memory"]["disposition"] == "blocked"
    assert decisions["live_arc"]["disposition"] == "blocked"
    assert decisions["selective_arbiter"]["failed_checks"][0]["observed"] == "flagged"


@pytest.mark.parametrize(
    ("producer", "audit", "positive_gate", "expected"),
    [
        ("positive", "positive", True, "adopt"),
        ("positive", "null", True, "narrow"),
        ("null", "null", False, "retire"),
        ("blocked", "positive", True, "blocked"),
    ],
)
def test_closed_decision_enum(
    producer: str, audit: str, positive_gate: bool, expected: str
) -> None:
    """SCENARIO-REPORT-6823-DISPOSITION: all evidence patterns close."""

    records = {"producer": _record(producer), "audit": _record(audit)}
    decision = exp.decide_disposition(
        "branch", records, ("producer", "audit"), positive_gate=positive_gate
    )
    assert decision["disposition"] == expected
    assert decision["disposition"] in exp.DISPOSITION_ENUM


def test_artifact_has_task_and_branch_rows_counts_principles_and_stable_checksum() -> None:
    """REQ-REPORT-6823: the complete synthesis schema validates and replays."""

    records = _records()
    records["exp6810"] = _record("null")
    records["exp6811"] = _record("null")
    records["exp6812"] = _record("positive")
    records["exp6813"] = _record("positive")
    artifact = exp.assemble_artifact(
        run_date="20260831",
        duration_s=0.125,
        manifest=exp.load_manifest(REPO),
        design=exp.parse_design((REPO / exp.DESIGN_PATH).read_text(encoding="utf-8")),
        records=records,
        source_hashes=exp.collect_source_hashes(REPO),
        adversarial_findings=[],
    )
    assert artifact["task_count"] == 14
    assert len([row for row in artifact["rows"] if row["row_type"] == "task"]) == 14
    assert len([row for row in artifact["rows"] if row["row_type"] == "branch"]) == 3
    assert artifact["terminal_class_counts"] == {
        "current_synthesis": 1,
        "missing": 9,
        "null": 2,
        "positive": 2,
    }
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(artifact)
    assert exp.validate_artifact(artifact) == []
    assert exp.reproducibility_checksum(artifact) == artifact["reproducibility_checksum"]

    broken = copy.deepcopy(artifact)
    broken["task_count"] = 13
    broken["rows"][0]["eligible_for_positive_claim"] = True
    broken["rows"][0]["terminal_class"] = "blocked"
    broken["selective_arbiter_disposition"]["disposition"] = "invented"
    assert set(exp.validate_artifact(broken)) == {
        "task_count must equal fourteen",
        "terminal class counts do not replay",
        "excluded task row is positive-eligible",
        "selective_arbiter_disposition has an invalid decision",
        "reproducibility checksum mismatch",
    }


def test_fail_closed_parsers_and_manifest_difference_receipts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6823-MANIFEST: malformed design and roadmap inputs stay errors."""

    with pytest.raises(ValueError, match="invalid task id"):
        exp._short_task_id("not-a-task")
    with pytest.raises(ValueError, match="deliverable missing"):
        exp.parse_design("### Exp6810: missing deliverable\n### Exp6811: next")
    no_milestone = exp.parse_design("")
    assert no_milestone == {"milestone": None, "phase_count": 0, "tasks": []}

    (tmp_path / exp.ROADMAP_PATH).write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="task list"):
        exp.load_manifest(tmp_path)
    (tmp_path / exp.ROADMAP_PATH).write_text(
        "tasks:\n  - ignored\n  - id: exp6810-test\n    prompt: ''\n", encoding="utf-8"
    )
    assert exp.load_manifest(tmp_path)[0]["task_id"] == "exp6810"

    design = exp.parse_design((REPO / exp.DESIGN_PATH).read_text(encoding="utf-8"))
    manifest = exp.load_manifest(REPO)
    changed = copy.deepcopy(manifest)
    changed[0]["deliverable"] = "results/wrong.json"
    comparison = exp.compare_manifest(design, changed, {})
    assert comparison["matches"] is False
    assert {row["field"] for row in comparison["differences"]} >= {
        "exp6810.deliverable",
        "contract_owner_map",
    }


def test_malformed_json_no_split_rows_and_gate_receipts_are_covered(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6823-ELIGIBILITY: alternate source shapes still fail closed."""

    (tmp_path / "results").mkdir()
    (tmp_path / exp.TASK_PATHS["exp6810"]).write_text("{", encoding="utf-8")
    loaded = exp.load_source_artifacts(tmp_path)
    assert loaded["exp6810"]["terminal_class"] == "malformed"
    assert "JSONDecodeError" in loaded["exp6810"]["error"]

    records = _records()
    records["exp6813"] = _record(
        rows=[
            {
                "pair_id": "p",
                "arm": "flat_reject_retry",
                "accepted_progress": 0,
                "retry_count": 1,
                "accepted_hard_violation": False,
                "harmful_selection": False,
            },
            {
                "pair_id": "p",
                "arm": "selective_priority",
                "accepted_progress": 1,
                "retry_count": 0,
                "accepted_hard_violation": False,
                "harmful_selection": False,
            },
        ]
    )
    assert exp.recompute_selective(records)["eligible_row_count"] == 2

    records["exp6814"]["gate_check_summary"] = [
        {"check": "audit", "expected": True, "observed": None}
    ]
    task_rows = exp._task_rows(exp.load_manifest(REPO), records)
    gates = exp._gate_summary(task_rows, {"differences": []})
    assert any(row.get("check") == "audit" for row in gates)
    records.pop("exp6810")
    with pytest.raises(ValueError, match="source record missing"):
        exp._task_rows(exp.load_manifest(REPO), records)


def test_atomic_write_and_schema_error(tmp_path: Path) -> None:
    """REQ-REPORT-6823: publication is atomic and schema drift is rejected."""

    target = tmp_path / "nested/result.json"
    exp.atomic_write_json(target, {"value": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}
    assert not target.with_suffix(".json.tmp").exists()
    assert "artifact fields do not match the required schema" in exp.validate_artifact({})
    assert callable(exp._load_validator(REPO, exp.ROW_LINT_PATH, "check_artifact"))


def test_run_success_and_validation_failure_use_the_same_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6823: the command validates before it writes."""

    design_path = tmp_path / exp.DESIGN_PATH
    design_path.parent.mkdir(parents=True)
    design_path.write_text("design", encoding="utf-8")
    calls: list[object] = []
    monkeypatch.setattr(exp, "parse_design", lambda _text: {"tasks": []})
    monkeypatch.setattr(exp, "load_manifest", lambda _root: [])
    monkeypatch.setattr(exp, "load_source_artifacts", lambda _root: {})
    monkeypatch.setattr(exp, "_load_validator", lambda *_args: lambda _path: {})
    monkeypatch.setattr(exp, "apply_validators", lambda *_args: [])
    monkeypatch.setattr(exp, "collect_source_hashes", lambda _root: {})
    monkeypatch.setattr(exp, "assemble_artifact", lambda **_kwargs: {"ok": True})
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: [])
    monkeypatch.setattr(
        exp, "atomic_write_json", lambda path, artifact: calls.append((path, artifact))
    )
    assert exp.run(tmp_path, "20260831") == {"ok": True}
    assert calls == [(tmp_path / exp.RESULT_PATH, {"ok": True})]

    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["bad"])
    with pytest.raises(ValueError, match="bad"):
        exp.run(tmp_path, "20260831")


def test_main_prints_closed_branch_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-6823-DISPOSITION: the wrapper reports all three decisions."""

    artifact = {
        "verdict_class": "partial",
        "selective_arbiter_disposition": {"disposition": "blocked"},
        "verified_route_memory_disposition": {"disposition": "retire"},
        "live_arc_disposition": {"disposition": "narrow"},
    }
    monkeypatch.setattr(exp, "run", lambda root, date: artifact)
    assert exp.main(["--date", "20260831"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["dispositions"] == {
        "selective_arbiter": "blocked",
        "verified_route_memory": "retire",
        "live_arc": "narrow",
    }
