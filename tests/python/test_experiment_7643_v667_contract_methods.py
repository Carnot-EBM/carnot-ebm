"""Focused V667 contract checks. Spec: REQ-REPORT-7643."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_7643_v667_contract_methods as subject


ROOT = Path(__file__).resolve().parents[2]
DESIGN = (ROOT / subject.DESIGN_PATH).read_text(encoding="utf-8")
ROADMAP = yaml.safe_load((ROOT / "research-roadmap.yaml").read_text(encoding="utf-8"))


def test_authority_and_private_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7643-AUTHORITY rejects corrupt copies and stale staging."""

    assert subject.compare_authorities(DESIGN, ROADMAP)["passed"]
    for name in (
        "missing_task",
        "reordered_task",
        "changed_producer",
        "self_input",
        "changed_gate",
        "changed_model",
    ):
        changed = subject.mutate_authority(ROADMAP, name)
        assert not subject.compare_authorities(DESIGN, changed)["passed"], name
    active = tmp_path / "research-roadmap.yaml"
    active.write_text(yaml.safe_dump(ROADMAP), encoding="utf-8")
    assert subject.resolve_authority(tmp_path)[0] == active
    staged = tmp_path / "research-roadmap-next.yaml"
    staged.write_text(yaml.safe_dump(ROADMAP), encoding="utf-8")
    assert subject.resolve_authority(tmp_path)[0] == staged
    stale = deepcopy(ROADMAP)
    stale["milestone"] = "2026.09.666"
    staged.write_text(yaml.safe_dump(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        subject.resolve_authority(tmp_path)


def test_prior_custody_preserves_distinct_failures() -> None:
    """SCENARIO-REPORT-7643-CUSTODY keeps pre-gates apart from absent producers."""

    rows = subject.collect_prior_dispositions(ROOT)
    assert len(rows) == 14
    assert sum(row["custody_kind"] == "terminal_producer" for row in rows) == 6
    assert sum(row["custody_kind"] == "current_self" for row in rows) == 1
    assert all(row["authenticated"] for row in rows)
    by_id = {row["task_id"]: row for row in rows}
    assert by_id["exp7631-schema-pilot"]["verdict_class"] == "blocked"
    assert by_id["exp7639-arc-goal-dedup"]["verdict_class"] == "disqualified"
    assert by_id["exp7632-fit-evidence"]["custody_kind"] == "conductor_pre_gate"
    assert by_id["exp7635-evidence-energy"]["custody_kind"] == "missing_work"


def test_reduction_and_block_operands() -> None:
    """SCENARIO-REPORT-7643-TERMINAL separates readiness and exact input blocks."""

    contract = subject.compare_authorities(DESIGN, ROADMAP)
    assert subject.independent_reduce(contract["rows"]) == {"matched": 14, "total": 14}
    damaged = deepcopy(contract["rows"])
    damaged[0]["matched"] = False
    assert subject.independent_reduce(damaged) == {"matched": 13, "total": 14}
    check = subject.input_check("prior", "missing.json", "bytes", "exists", True, False)
    assert check == {
        "check": "prior",
        "upstream": "missing.json",
        "path": "missing.json",
        "field": "bytes",
        "operator": "exists",
        "expected": True,
        "observed": False,
        "passed": False,
    }
    assert subject.classify(False, True) == (
        "complete_disqualified_v667_validation",
        "disqualified",
    )
    assert subject.classify(True, False) == ("complete_blocked_v667_external_evidence", "blocked")
    assert subject.classify(True, True) == ("complete_null_v667_contract_methods", "null")


def test_artifact_fields_and_cold_reader(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7643-TERMINAL recomputes rows and binds input bytes."""

    value = subject.build_artifact_for_test(ROOT)
    assert value["MODEL_SPECS"] == []
    assert value["inference_substrate_class"] == "aggregation"
    assert value["model_invoked"] is False
    assert value["contract_ready_score"] == 1
    assert len(value["prior_dispositions"]) == 14
    assert value["acceptance_gate_results"]["probability_benefit"]["passed"] is False
    assert subject.validate_artifact(value, ROOT)
    altered = deepcopy(value)
    altered["rows"][0]["matched"] = False
    assert not subject.validate_artifact(altered, ROOT)
    output = tmp_path / "candidate.json"
    output.write_text(json.dumps(value), encoding="utf-8")
    assert subject.cold_read(output, ROOT)


def test_bad_authorities_and_prior_orders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7643-AUTHORITY rejects malformed private authorities."""

    assert not subject.compare_authorities(DESIGN, {**ROADMAP, "milestone": "old"})["passed"]
    with pytest.raises(ValueError, match="unknown"):
        subject.mutate_authority(ROADMAP, "unknown")
    with pytest.raises(ValueError, match="unavailable"):
        subject.resolve_authority(tmp_path)
    with pytest.raises(ValueError, match="missing"):
        subject._machine_contract("no machine block")
    with pytest.raises(ValueError, match="list"):
        subject._machine_contract("<!-- V667-TASK-CONTRACT-BEGIN -->\n```json\n{}\n```")
    capstone = json.loads((ROOT / subject.PRIOR_PATH).read_text())
    old = subject.json.loads
    capstone["milestone_dispositions"].pop()
    monkeypatch.setattr(subject.json, "loads", lambda value: capstone)
    with pytest.raises(ValueError, match="count"):
        subject.collect_prior_dispositions(ROOT)
    capstone["milestone_dispositions"].append(deepcopy(capstone["milestone_dispositions"][-1]))
    capstone["milestone_dispositions"][0]["order"] = 2
    with pytest.raises(ValueError, match="order"):
        subject.collect_prior_dispositions(ROOT)
    monkeypatch.setattr(subject.json, "loads", old)


def test_cold_reader_rejects_tampering() -> None:
    """SCENARIO-REPORT-7643-TERMINAL rejects changed source or reduction operands."""

    value = subject.build_artifact_for_test(ROOT)
    for mutate in (
        lambda row: row.update(selected_roadmap_path="absent.yaml"),
        lambda row: row["rows"].pop(),
        lambda row: row["sample_size_budget"].update(eligible=0),
        lambda row: row["source_artifact_hashes"][-1].update(sha256="bad"),
        lambda row: row["source_artifact_hashes"][0].update(exists=False),
        lambda row: row["source_artifact_hashes"][0].update(sha256="bad"),
    ):
        changed = deepcopy(value)
        mutate(changed)
        changed["reproducibility_checksum"] = subject._checksum(changed)
        assert not subject.validate_artifact(changed, ROOT)
    assert subject.input_check("x", "y", "z", "!=", 1, 2)["passed"]
    assert subject.input_check("x", "y", "z", "!=", 1, 1)["passed"] is False


def test_plans_and_mocked_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7643-TERMINAL covers orchestration without touching tracked output."""

    authority = ROOT / "research-roadmap.yaml"
    assert subject.prompt_path_findings(ROOT, authority) == []
    plan = subject._validation_plan(ROOT, authority, tmp_path)
    assert any(row.name == "prompt_path" for row in plan)
    terminal_plan = subject._terminal_plan(ROOT, tmp_path / "candidate.json")
    assert len(terminal_plan) == 4
    with pytest.raises(ValueError, match="date"):
        subject.run_experiment(ROOT, "wrong", subject.RESULT_PATH)
    original_atomic = subject.atomic_json
    published = []

    def private_atomic(path: Path, value: dict) -> None:
        if path == ROOT / subject.RESULT_PATH:
            published.append(value)
        else:
            original_atomic(path, value)

    calls = 0

    def fake_commands(root: Path, commands: list, *, log_dir: Path) -> list[dict]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [{"name": "scoped", "passed": True, "exit_code": 0, "log_sha256": "sha256:ok"}]
        return [
            {"name": item.name, "passed": True, "exit_code": 0, "log_sha256": "sha256:ok"}
            for item in commands
        ]

    monkeypatch.setattr(subject, "atomic_json", private_atomic)
    monkeypatch.setattr(subject, "run_commands", fake_commands)
    value = subject.run_experiment(ROOT, subject.RUN_DATE, subject.RESULT_PATH)
    assert calls == 3
    assert published[0]["reproducibility_checksum"] == value["reproducibility_checksum"]
    assert len(value["terminal_reader_outcomes"]) == 4
    original_cold = subject.cold_read
    monkeypatch.setattr(subject, "cold_read", lambda *args: False)
    with pytest.raises(RuntimeError, match="cold validation"):
        subject.run_experiment(ROOT, subject.RUN_DATE, subject.RESULT_PATH)
    monkeypatch.setattr(subject, "cold_read", original_cold)
    replay_calls = 0

    def changed_replay(root: Path, commands: list, *, log_dir: Path) -> list[dict]:
        nonlocal replay_calls
        replay_calls += 1
        rows = [
            {"name": item.name, "passed": True, "exit_code": 0, "log_sha256": "sha256:ok"}
            for item in commands
        ]
        if replay_calls == 3:
            rows[0]["passed"] = False
        return rows

    monkeypatch.setattr(subject, "run_commands", changed_replay)
    with pytest.raises(RuntimeError, match="outcomes changed"):
        subject.run_experiment(ROOT, subject.RUN_DATE, subject.RESULT_PATH)


def test_main_reader_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7643-TERMINAL exposes both independent reader modes."""

    value = subject.build_artifact_for_test(ROOT)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(value), encoding="utf-8")
    assert subject.main(["--cold-validate", str(candidate)]) == 0
    assert subject.main(["--independent-reduce", str(candidate)]) == 0
    value["rows"][0]["matched"] = False
    candidate.write_text(json.dumps(value), encoding="utf-8")
    assert subject.main(["--cold-validate", str(candidate)]) == 1
    assert subject.main(["--independent-reduce", str(candidate)]) == 1
    monkeypatch.setattr(subject, "run_experiment", lambda *args: {})
    assert subject.main(["--date", subject.RUN_DATE]) == 0
