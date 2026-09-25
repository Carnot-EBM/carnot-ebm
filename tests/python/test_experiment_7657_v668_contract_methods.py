"""V668 contract checks. Spec: REQ-REPORT-7657."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_7657_v668_contract_methods as subject
from carnot.reporting import v668_contract as contract


ROOT = Path(__file__).resolve().parents[2]
DESIGN = (ROOT / contract.DESIGN_PATH).read_text(encoding="utf-8")
ROADMAP = yaml.safe_load((ROOT / "research-roadmap.yaml").read_text(encoding="utf-8"))


def test_authority_selection_and_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7657-AUTHORITY checks staged and consumed cases."""

    active = tmp_path / "research-roadmap.yaml"
    active.write_text(yaml.safe_dump(ROADMAP), encoding="utf-8")
    assert contract.resolve_authority(tmp_path)[0] == active
    staged = tmp_path / "research-roadmap-next.yaml"
    stale = deepcopy(ROADMAP)
    stale["milestone"] = "2026.09.667"
    staged.write_text(yaml.safe_dump(stale), encoding="utf-8")
    assert contract.resolve_authority(tmp_path)[0] == active
    staged.write_text(yaml.safe_dump(ROADMAP), encoding="utf-8")
    assert contract.resolve_authority(tmp_path)[0] == staged
    assert contract.compare_authorities(DESIGN, ROADMAP)["passed"]
    for mutation in ("delete", "reorder", "gate", "self_input", "model", "stale"):
        changed = contract.mutate_authority(ROADMAP, mutation)
        assert not contract.compare_authorities(DESIGN, changed)["passed"], mutation
    active.unlink()
    staged.unlink()
    with pytest.raises(ValueError, match="unavailable"):
        contract.resolve_authority(tmp_path)


def test_prior_custody_and_failure_boundaries() -> None:
    """SCENARIO-REPORT-7657-CUSTODY keeps absences and distinct failures."""

    rows = contract.collect_prior_dispositions(ROOT)
    assert len(rows) == 14
    assert [row["order"] for row in rows] == list(range(1, 15))
    assert all(row["authenticated"] for row in rows)
    assert {row["custody_kind"] for row in rows} == {
        "terminal_producer",
        "conductor_pre_gate",
        "missing_work",
        "current_self",
    }
    assert rows[4]["actual_path"] == "results/experiment_7647_witness_energy.json"
    assert rows[5]["honest_verdict"] is None
    assert rows[6]["verdict_class"] is None
    assert rows[-1]["authentication_path"] == contract.PRIOR_PATH.as_posix()
    assert contract.classify(False, True)[1] == "disqualified"
    assert contract.classify(True, False)[1] == "blocked"
    assert contract.classify(True, True)[1] == "null"


def test_artifact_cold_reduction_and_gate_split(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7657-TERMINAL recomputes rows and keeps science closed."""

    value = subject.build_artifact_for_test(ROOT)
    assert len(value["rows"]) == len(value["prior_dispositions"]) == 14
    assert value["MODEL_SPECS"] == []
    assert value["model_specs"] == [{"no_model": True}]
    assert value["contract_ready_score"] == 0  # no validation receipts in fixture
    assert value["acceptance_gate_results"]["probability_benefit"]["passed"] is False
    assert value["verdict_class"] == "disqualified"
    assert subject.validate_artifact(value, ROOT)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(value), encoding="utf-8")
    assert subject.cold_read(candidate, ROOT)
    changed = deepcopy(value)
    changed["rows"][0]["matched"] = False
    changed["reproducibility_checksum"] = subject.checksum(changed)
    assert not subject.validate_artifact(changed, ROOT)


def test_validation_plan_and_block_operands(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7657-TERMINAL freezes exact scope and exact blocks."""

    plan = subject.validation_plan(ROOT, ROOT / "research-roadmap.yaml", tmp_path)
    assert {"focused_pytest", "changed_module_coverage_report", "roadmap_schema"} <= {
        item.name for item in plan
    }
    assert not any("tests/python -q" in " ".join(item.argv) for item in plan)
    terminal = subject.terminal_plan(ROOT, tmp_path / "candidate.json")
    assert {item.name for item in terminal} == {
        "cold_replay",
        "independent_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
    check = contract.input_check("source", "missing.json", "bytes", "==", True, False)
    assert check == {
        "check": "source",
        "upstream": "missing.json",
        "path": "missing.json",
        "field": "bytes",
        "operator": "==",
        "expected": True,
        "observed": False,
        "passed": False,
    }
    with pytest.raises(ValueError, match="date"):
        subject.run_experiment(ROOT, "wrong", subject.RESULT_PATH)


def test_reducer_edge_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7657-AUTHORITY and CUSTODY reject corrupt sources."""

    with pytest.raises(ValueError, match="machine contract missing"):
        contract.machine_contract("no contract")
    with pytest.raises(ValueError, match="must be a list"):
        contract.machine_contract("<!-- V668-TASK-CONTRACT-BEGIN -->\n```json\n{}\n```")
    with pytest.raises(ValueError, match="unknown mutation"):
        contract.mutate_authority(ROADMAP, "unknown")
    with pytest.raises(ValueError, match="equality"):
        contract.input_check("x", "y", "z", "!=", 1, 2)
    capstone = json.loads((ROOT / contract.PRIOR_PATH).read_text(encoding="utf-8"))
    original_loads = contract.json.loads
    capstone["milestone_dispositions"].pop()
    monkeypatch.setattr(contract.json, "loads", lambda _: capstone)
    with pytest.raises(ValueError, match="count"):
        contract.collect_prior_dispositions(ROOT)
    capstone["milestone_dispositions"].append(deepcopy(capstone["milestone_dispositions"][-1]))
    capstone["milestone_dispositions"][0]["order"] = 3
    with pytest.raises(ValueError, match="order"):
        contract.collect_prior_dispositions(ROOT)
    monkeypatch.setattr(contract.json, "loads", original_loads)
    assert contract.classify(True, True) == ("complete_null_v668_contract_methods", "null")
    empty = tmp_path / "research-roadmap-next.yaml"
    empty.write_text("milestone: old\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unavailable"):
        contract.resolve_authority(tmp_path)


def test_cold_reader_rejects_modified_bytes_and_rows() -> None:
    """SCENARIO-REPORT-7657-TERMINAL refuses stale inputs and row counts."""

    value = subject.build_artifact_for_test(ROOT)
    for modify in (
        lambda item: item.update(selected_roadmap_path="missing.yaml"),
        lambda item: item["sample_size_budget"].update(eligible=0),
        lambda item: item["prior_dispositions"].pop(),
        lambda item: item["source_artifact_hashes"][0].update(sha256="bad"),
        lambda item: item["source_artifact_hashes"][0].update(exists=False),
        lambda item: item["source_artifact_hashes"][-1].update(sha256="bad"),
    ):
        altered = deepcopy(value)
        modify(altered)
        altered["reproducibility_checksum"] = subject.checksum(altered)
        assert not subject.validate_artifact(altered, ROOT)
    corrupt = deepcopy(value)
    corrupt["reproducibility_checksum"] = "bad"
    assert not subject.validate_artifact(corrupt, ROOT)


def test_mocked_run_and_main_readers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7657-TERMINAL publishes only a checked candidate."""

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
        return [
            {
                "name": spec.name,
                "passed": True,
                "exit_code": 0,
                "log_sha256": "sha256:ok",
                "command": "mock",
            }
            for spec in commands
        ]

    monkeypatch.setattr(subject, "atomic_json", private_atomic)
    monkeypatch.setattr(subject, "run_commands", fake_commands)
    value = subject.run_experiment(ROOT, subject.RUN_DATE, subject.RESULT_PATH)
    assert calls == 3
    assert value["contract_ready_score"] == 1
    assert value["verdict_class"] == "null"
    assert published[0]["reproducibility_checksum"] == value["reproducibility_checksum"]
    candidate = tmp_path / "candidate.json"
    original_atomic(candidate, value)
    assert subject.main(["--cold-validate", str(candidate)]) == 0
    assert subject.main(["--independent-reduce", str(candidate)]) == 0
    bad = deepcopy(value)
    bad["rows"][0]["matched"] = False
    original_atomic(candidate, bad)
    assert subject.main(["--cold-validate", str(candidate)]) == 1
    assert subject.main(["--independent-reduce", str(candidate)]) == 1
    monkeypatch.setattr(subject, "run_experiment", lambda *args: {})
    assert subject.main(["--date", subject.RUN_DATE]) == 0


def test_failed_terminal_reader_zeroes_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7657-TERMINAL never credits a failed required reader."""

    original_atomic = subject.atomic_json
    original_run = subject.run_commands
    calls = 0

    def private_atomic(path: Path, value: dict) -> None:
        if path != ROOT / subject.RESULT_PATH:
            original_atomic(path, value)

    def fake_commands(root: Path, commands: list, *, log_dir: Path) -> list[dict]:
        nonlocal calls
        calls += 1
        return [
            {
                "name": spec.name,
                "passed": calls == 1 or spec.name != "adversarial_verify",
                "exit_code": 0 if calls == 1 or spec.name != "adversarial_verify" else 1,
                "log_sha256": "sha256:ok",
            }
            for spec in commands
        ]

    monkeypatch.setattr(subject, "atomic_json", private_atomic)
    monkeypatch.setattr(subject, "run_commands", fake_commands)
    value = subject.run_experiment(ROOT, subject.RUN_DATE, subject.RESULT_PATH)
    assert value["contract_ready_score"] == 0
    assert value["flagged_adversarial"] is True
    assert value["verdict_class"] == "disqualified"
    monkeypatch.setattr(subject, "run_commands", original_run)


def test_terminal_replay_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7657-TERMINAL rejects cold and replay disagreement."""

    original_atomic = subject.atomic_json
    original_cold = subject.cold_read
    calls = 0

    def private_atomic(path: Path, value: dict) -> None:
        if path != ROOT / subject.RESULT_PATH:
            original_atomic(path, value)

    def fake_commands(root: Path, commands: list, *, log_dir: Path) -> list[dict]:
        nonlocal calls
        calls += 1
        return [
            {
                "name": spec.name,
                "passed": calls != 3 or spec.name != "cold_replay",
                "exit_code": 0,
                "log_sha256": "sha256:ok",
            }
            for spec in commands
        ]

    monkeypatch.setattr(subject, "atomic_json", private_atomic)
    monkeypatch.setattr(subject, "run_commands", fake_commands)
    monkeypatch.setattr(subject, "cold_read", lambda *args: False)
    with pytest.raises(RuntimeError, match="cold validation"):
        subject.run_experiment(ROOT, subject.RUN_DATE, subject.RESULT_PATH)
    monkeypatch.setattr(subject, "cold_read", original_cold)
    calls = 0
    with pytest.raises(RuntimeError, match="outcomes changed"):
        subject.run_experiment(ROOT, subject.RUN_DATE, subject.RESULT_PATH)
