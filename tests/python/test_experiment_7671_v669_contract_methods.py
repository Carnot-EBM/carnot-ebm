"""V669 contract accounting tests. REQ-REPORT-7671 and REQ-HARNESS-7671."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_7671_v669_contract_methods as subject
from carnot.reporting import v669_contract as contract

ROOT = Path(__file__).resolve().parents[2]
DESIGN = (ROOT / contract.DESIGN_PATH).read_text(encoding="utf-8")
ROADMAP = yaml.safe_load((ROOT / "research-roadmap.yaml").read_text(encoding="utf-8"))


def test_authority_and_six_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7671-AUTHORITY checks both lifecycle copies and all attacks."""

    active = tmp_path / "research-roadmap.yaml"
    staged = tmp_path / "research-roadmap-next.yaml"
    active.write_text(yaml.safe_dump(ROADMAP), encoding="utf-8")
    assert contract.resolve_authority(tmp_path)[0] == active
    stale = deepcopy(ROADMAP)
    stale["milestone"] = "2026.09.668"
    staged.write_text(yaml.safe_dump(stale), encoding="utf-8")
    assert contract.resolve_authority(tmp_path)[0] == active
    staged.write_text(yaml.safe_dump(ROADMAP), encoding="utf-8")
    assert contract.resolve_authority(tmp_path)[0] == staged
    assert contract.compare_authorities(DESIGN, ROADMAP)["passed"]
    for attack in ("delete", "reorder", "gate", "self_input", "model", "stale"):
        changed = contract.mutate_authority(ROADMAP, attack)
        assert not contract.compare_authorities(DESIGN, changed)["passed"], attack
    active.unlink()
    staged.unlink()
    with pytest.raises(ValueError, match="unavailable"):
        contract.resolve_authority(tmp_path)


def test_prior_custody_is_not_inferred_science() -> None:
    """SCENARIO-REPORT-7671-CUSTODY keeps missing producers distinct from nulls."""

    rows = contract.collect_prior_dispositions(ROOT)
    assert len(rows) == 14
    assert [row["task_id"].split("-")[0] for row in rows] == [f"exp{n}" for n in range(7657, 7671)]
    assert sum(row["custody_kind"] == "producer_file" for row in rows) == 10
    assert sum(row["custody_kind"] == "missing_producer_log" for row in rows) == 4
    assert all(row["authenticated"] for row in rows)
    assert all(row["honest_verdict"] is None for row in rows[10:])
    assert rows[8]["exact_supported_proposals"] == 0
    source = json.loads((ROOT / rows[8]["planned_path"]).read_text(encoding="utf-8"))
    assert (
        rows[8]["exact_supported_proposals"]
        == source["acceptance_gate_results"]["coverage"]["measured_operands"]["exact_supported"]
    )
    assert contract.classify(False, True)[1] == "disqualified"
    assert contract.classify(True, False)[1] == "blocked"
    assert contract.classify(True, True)[1] == "null"


def test_artifact_cold_reduction_and_gate_split(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7671-TERMINAL recomputes independent task rows."""

    value = subject.build_artifact_for_test(ROOT)
    assert len(value["rows"]) == len(value["prior_dispositions"]) == 14
    assert value["MODEL_SPECS"] == []
    assert value["model_invoked"] is False
    assert value["contract_ready_score"] == 0
    assert value["verdict_class"] == "disqualified"
    assert value["acceptance_gate_results"]["probability"]["passed"] is False
    assert value["v668_findings"]["covered_groups"] == 66
    assert value["v668_findings"]["prior_exposed_groups"] == 248
    assert "No V669 fresh relation" in value["publication_gates"]["missing_evidence"]
    assert subject.validate_artifact(value, ROOT)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(value), encoding="utf-8")
    assert subject.cold_read(candidate, ROOT)
    changed = deepcopy(value)
    changed["rows"][0]["matched"] = False
    changed["reproducibility_checksum"] = subject.checksum(changed)
    assert not subject.validate_artifact(changed, ROOT)


def test_frozen_validation_and_exact_blocks(tmp_path: Path) -> None:
    """SCENARIO-HARNESS-7671-ACCOUNTING pins focused checks and block operands."""

    plan = subject.validation_plan(ROOT, ROOT / "research-roadmap.yaml", tmp_path)
    assert {"focused_pytest", "changed_module_coverage_report", "roadmap_schema"} <= {
        row.name for row in plan
    }
    assert not any("tests/python -q" in " ".join(row.argv) for row in plan)
    assert {row.name for row in subject.terminal_plan(ROOT, tmp_path / "candidate.json")} == {
        "cold_replay",
        "independent_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
    check = contract.input_check("source", "missing.json", "bytes", "==", True, False)
    assert {
        key: check[key]
        for key in ("check", "upstream", "path", "field", "operator", "expected", "observed")
    } == {
        "check": "source",
        "upstream": "missing.json",
        "path": "missing.json",
        "field": "bytes",
        "operator": "==",
        "expected": True,
        "observed": False,
    }
    with pytest.raises(ValueError, match="date"):
        subject.run_experiment(ROOT, "wrong", subject.RESULT_PATH)


@pytest.mark.parametrize("reader_passed", [True, False])
def test_terminal_lifecycle_retains_real_check_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reader_passed: bool,
) -> None:
    """SCENARIO-REPORT-7671-TERMINAL applies required reader exits to readiness."""

    original_atomic = subject.atomic_json
    published = tmp_path / "published.json"

    def private_write(path: Path, value: dict) -> None:
        original_atomic(published if path == ROOT / subject.RESULT_PATH else path, value)

    def receipts(_root: Path, plan: list, **_kwargs: object) -> list[dict]:
        return [
            {
                "name": spec.name,
                "passed": reader_passed if spec.name == "adversarial_verify" else True,
                "exit_code": 0 if reader_passed or spec.name != "adversarial_verify" else 1,
                "log_sha256": "sha256:test",
                "command": "fixture",
            }
            for spec in plan
        ]

    monkeypatch.setattr(subject, "atomic_json", private_write)
    monkeypatch.setattr(subject, "run_commands", receipts)
    value = subject.run_experiment(ROOT, subject.RUN_DATE, subject.RESULT_PATH)
    assert published.is_file()
    assert subject.validate_artifact(value, ROOT)
    assert value["contract_ready_score"] == int(reader_passed)
    assert value["verdict_class"] == ("null" if reader_passed else "disqualified")
    assert value["flagged_adversarial"] is not reader_passed
    assert len(value["terminal_reader_outcomes"]) == 4


def test_cold_rejection_and_cli_modes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7671-TERMINAL rejects changed bytes and runs both readers."""

    value = subject.build_artifact_for_test(ROOT)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert subject.main(["--cold-validate", str(path)]) == 0
    assert subject.main(["--independent-reduce", str(path)]) == 0
    value["source_artifact_hashes"][0]["sha256"] = "sha256:wrong"
    value["reproducibility_checksum"] = subject.checksum(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    assert subject.main(["--cold-validate", str(path)]) == 1
    assert subject.main(["--independent-reduce", str(path)]) == 1
    called = []
    monkeypatch.setattr(subject, "run_experiment", lambda *args: called.append(args))
    assert subject.main(["--date", subject.RUN_DATE, "--output", str(subject.RESULT_PATH)]) == 0
    assert called[0][1:] == (subject.RUN_DATE, subject.RESULT_PATH)


def test_malformed_design_and_input_checks() -> None:
    """SCENARIO-REPORT-7671-AUTHORITY rejects missing machine contracts."""

    with pytest.raises(ValueError, match="machine contract missing"):
        contract.machine_contract("missing")
    with pytest.raises(ValueError, match="must be a list"):
        contract.machine_contract("<!-- V669-TASK-CONTRACT-BEGIN -->\n```json\n{}\n```")
    with pytest.raises(ValueError, match="unknown mutation"):
        contract.mutate_authority(ROADMAP, "unsupported")
    with pytest.raises(ValueError, match="only equality"):
        contract.input_check("x", "y", "z", "in", True, False)


def test_cold_reader_rejects_each_custody_tamper() -> None:
    """SCENARIO-REPORT-7671-TERMINAL checks more than its own checksum."""

    original = subject.build_artifact_for_test(ROOT)

    def rejected(change: str, edit: object) -> None:
        value = deepcopy(original)
        edit(value)
        if change != "checksum":
            value["reproducibility_checksum"] = subject.checksum(value)
        assert not subject.validate_artifact(value, ROOT), change

    rejected("checksum", lambda value: value.update(honest_verdict="wrong"))
    rejected("authority", lambda value: value.update(selected_roadmap_path="absent.yaml"))
    rejected("eligible", lambda value: value["sample_size_budget"].update(eligible=0))
    rejected("prior", lambda value: value["prior_dispositions"].pop())
    rejected("self_input", lambda value: value["source_artifact_hashes"][-1].update(sha256="x"))
    rejected("existence", lambda value: value["source_artifact_hashes"][0].update(exists=False))


def test_preserved_contract_malformed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7671-CUSTODY rejects changed prior-design structure."""

    path = tmp_path / "prior.md"
    monkeypatch.setattr(contract, "PRIOR_DESIGN_PATH", Path("prior.md"))
    path.write_text("missing", encoding="utf-8")
    with pytest.raises(ValueError, match="preserved contract missing"):
        contract.prior_machine_contract(tmp_path)
    path.write_text("<!-- V668-TASK-CONTRACT-BEGIN -->\n```json\n[]\n```", encoding="utf-8")
    with pytest.raises(ValueError, match="task count"):
        contract.prior_machine_contract(tmp_path)


@pytest.mark.parametrize("failure", ["cold", "changed_reader"])
def test_terminal_fail_closed_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """SCENARIO-REPORT-7671-TERMINAL never publishes an unstable candidate."""

    calls = 0
    original_cold = subject.cold_read

    def fake_commands(_root: Path, plan: list, **_kwargs: object) -> list[dict]:
        nonlocal calls
        calls += 1
        return [
            {
                "name": spec.name,
                "passed": not (
                    failure == "changed_reader" and calls == 3 and spec.name == "adversarial_verify"
                ),
                "exit_code": 0,
                "log_sha256": "sha256:test",
                "command": "fixture",
            }
            for spec in plan
        ]

    monkeypatch.setattr(subject, "run_commands", fake_commands)
    if failure == "cold":
        monkeypatch.setattr(subject, "cold_read", lambda *_args: False)
    with pytest.raises(RuntimeError, match="cold validation|reader outcomes"):
        subject.run_experiment(ROOT, subject.RUN_DATE, subject.RESULT_PATH)
    monkeypatch.setattr(subject, "cold_read", original_cold)
