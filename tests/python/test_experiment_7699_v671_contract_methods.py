"""V671 accounting checks. REQ-REPORT-7699; REQ-HARNESS-7699."""

from copy import deepcopy
import json
from pathlib import Path
import time

import pytest
import yaml

from carnot import experiment_7699_v671_contract_methods as subject
from carnot.reporting import v671_contract as contract

ROOT = Path(__file__).resolve().parents[2]
DESIGN = (ROOT / contract.DESIGN_PATH).read_text(encoding="utf-8")
ROADMAP = yaml.safe_load((ROOT / "research-roadmap.yaml").read_text(encoding="utf-8"))


def test_authority_parity_and_private_mutations() -> None:
    """SCENARIO-REPORT-7699-AUTHORITY: all fields and gates bind in order."""

    comparison = contract.compare_authorities(DESIGN, ROADMAP)
    assert comparison["passed"]
    assert len(comparison["rows"]) == 14
    assert all(row["matched"] for row in comparison["rows"])
    for mutation in ("delete", "reorder", "producer_field", "model", "stale"):
        assert not contract.compare_authorities(
            DESIGN, contract.mutate_authority(ROADMAP, mutation)
        )["passed"], mutation


def test_staged_selection_and_active_fallback(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7699-AUTHORITY: stale staged bytes do not win."""

    (tmp_path / "research-roadmap.yaml").write_text(yaml.safe_dump(ROADMAP))
    staged = deepcopy(ROADMAP)
    staged["milestone"] = "2026.09.670"
    (tmp_path / "research-roadmap-next.yaml").write_text(yaml.safe_dump(staged))
    path, _, candidates = contract.resolve_authority(tmp_path)
    assert path.name == "research-roadmap.yaml"
    assert candidates[0]["observed"] == "2026.09.670"
    staged["milestone"] = contract.MILESTONE
    (tmp_path / "research-roadmap-next.yaml").write_text(yaml.safe_dump(staged))
    path, _, _ = contract.resolve_authority(tmp_path)
    assert path.name == "research-roadmap-next.yaml"


def test_v670_dispositions_are_log_custody() -> None:
    """SCENARIO-REPORT-7699-CUSTODY: absent producers get no verdict."""

    rows = contract.collect_prior_dispositions(ROOT)
    assert len(rows) == 14
    assert sum(row["disposition"] == "usage_limit_three_attempts" for row in rows) == 6
    assert sum(row["disposition"] == "gate_skipped" for row in rows) == 8
    assert all(row["custody_kind"] == "not_emitted" for row in rows)
    assert all(row["honest_verdict"] is None for row in rows)
    assert all(row["authenticated"] and row["log_lines"] for row in rows)


def test_artifact_is_administrative_and_cold_reducible() -> None:
    """SCENARIO-REPORT-7699-TERMINAL: fixture cannot open science gates."""

    value = subject.build_artifact_for_test(ROOT)
    assert value["verdict_class"] == "disqualified"  # no fabricated receipts
    assert value["MODEL_SPECS"] == [] and not value["model_invoked"]
    assert value["contract_ready_score"] == 0
    assert len(value["rows"]) == 14
    assert subject.validate_artifact(value, ROOT)
    changed = deepcopy(value)
    changed["rows"][0]["matched"] = False
    assert not subject.validate_artifact(changed, ROOT)
    changed = deepcopy(value)
    changed["reproducibility_checksum"] = "bad"
    assert not subject.validate_artifact(changed, ROOT)


def test_missing_external_input_is_blocked() -> None:
    """REQ-REPORT-7699: external absence is a terminal blocked state."""

    assert contract.classify(True, False) == ("complete_blocked_v671_external_evidence", "blocked")
    check = contract.input_check("required_input", "upstream", "field", "==", 1, None)
    assert not check["passed"]
    assert all(
        key in check
        for key in ("check", "upstream", "path", "field", "operator", "expected", "observed")
    )


def test_exact_candidate_reader(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7699-TERMINAL: disk bytes govern cold validation."""

    path = tmp_path / "candidate.json"
    value = subject.build_artifact_for_test(ROOT)
    path.write_text(json.dumps(value))
    assert subject.cold_read(path, ROOT)
    path.write_text("{}")
    assert not subject.cold_read(path, ROOT)


@pytest.mark.parametrize("mode", ["success", "terminal_fail", "cold_fail", "drift"])
def test_orchestration_freezes_and_publishes_private_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """SCENARIO-REPORT-7699-TERMINAL: owned phases retain exact reader exits."""

    (tmp_path / contract.DESIGN_PATH).parent.mkdir(parents=True)
    (tmp_path / contract.DESIGN_PATH).write_text(DESIGN)
    (tmp_path / "research-roadmap.yaml").write_text(yaml.safe_dump(ROADMAP))
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    monkeypatch.setattr(
        subject,
        "preconditions",
        lambda *_: [contract.input_check("fixture", "local", "exists", "==", True, True)],
    )
    monkeypatch.setattr(subject, "source_hashes", lambda *_: [])
    monkeypatch.setattr(subject.contract, "collect_prior_dispositions", lambda *_: [])
    monkeypatch.setattr(
        subject,
        "validation_plan",
        lambda *_: [subject.CommandSpec("fixture_check", ("true",), "fixture")],
    )
    monkeypatch.setattr(
        subject, "reduce_required_checks", lambda *_: {"required_checks_passed": True}
    )
    monkeypatch.setattr(subject, "cold_read", lambda *_: mode != "cold_fail")
    monkeypatch.setattr(subject, "sha256_file", lambda *_: "sha256:fixture")
    calls = []

    def fake_commands(_root: Path, commands: list, **_kwargs: object) -> list[dict]:
        calls.append([command.name for command in commands])
        return [
            {
                "name": command.name,
                "passed": not (
                    (mode == "terminal_fail" and command.name == "adversarial_verify")
                    or (mode == "drift" and len(calls) == 3 and command.name == "cold_replay")
                ),
                "exit_code": 0,
                "log_sha256": "sha256:fixture",
            }
            for command in commands
        ]

    monkeypatch.setattr(subject, "run_commands", fake_commands)
    if mode in {"cold_fail", "drift"}:
        with pytest.raises(RuntimeError, match="cold validation|outcomes changed"):
            subject.run_experiment(tmp_path, subject.RUN_DATE, subject.RESULT_PATH)
        return
    value = subject.run_experiment(tmp_path, subject.RUN_DATE, subject.RESULT_PATH)
    if mode == "success":
        assert value["honest_verdict"] == "complete_null_v671_contract_methods"
        assert value["contract_ready_score"] == 1
    else:
        assert value["verdict_class"] == "disqualified"
        assert value["flagged_adversarial"]
        assert value["contract_ready_score"] == 0
    assert [span["phase"] for span in value["phase_spans"]] == [
        "preconditions",
        "model_load",
        "generation",
        "benchmark",
        "validation",
        "terminal_readers",
    ]
    assert calls[-1] == calls[-2]
    assert (tmp_path / subject.RESULT_PATH).is_file()


def test_plans_and_guard_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-HARNESS-7699: frozen commands and bad input guards are explicit."""

    plan = subject.validation_plan(ROOT, ROOT / "research-roadmap.yaml", tmp_path)
    assert {"focused_pytest", "changed_module_coverage_report", "prompt_path"}.issubset(
        {row.name for row in plan}
    )
    terminal = subject.terminal_plan(ROOT, tmp_path / "candidate.json")
    assert [row.name for row in terminal] == [
        "cold_replay",
        "independent_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    assert subject.span("unit", time.monotonic() - 1, time.monotonic(), 2)["completed_units"] == 2
    with pytest.raises(ValueError, match="run date"):
        subject.run_experiment(ROOT, "20260925", subject.RESULT_PATH)
    monkeypatch.setattr(subject, "cold_read", lambda *_: True)
    assert subject.main(["--cold-validate", "unused"]) == 0
    monkeypatch.setattr(subject, "cold_read", lambda *_: False)
    assert subject.main(["--cold-validate", "unused"]) == 1


def test_cold_reader_rejects_changed_custody_and_inputs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7699-TERMINAL: every hashed input binds the artifact."""

    base = subject.build_artifact_for_test(ROOT)
    for change in ("authority", "prior", "eligible", "output_hash", "missing_path", "input_hash"):
        value = deepcopy(base)
        if change == "authority":
            value["selected_roadmap_path"] = "missing.yaml"
        elif change == "prior":
            value["prior_dispositions"] = []
        elif change == "eligible":
            value["sample_size_budget"]["eligible"] = 0
        elif change == "output_hash":
            next(
                row
                for row in value["source_artifact_hashes"]
                if row["role"] == "planned_output_not_input"
            )["sha256"] = "bad"
        elif change == "missing_path":
            next(
                row for row in value["source_artifact_hashes"] if row["role"] == "missing_producer"
            )["exists"] = True
        else:
            next(row for row in value["source_artifact_hashes"] if row["role"] == "current_input")[
                "sha256"
            ] = "bad"
        value["reproducibility_checksum"] = subject.checksum(value)
        assert not subject.validate_artifact(value, ROOT), change
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(base))
    assert subject.main(["--independent-reduce", str(path)]) == 0
    path.write_text("{}")
    assert subject.main(["--independent-reduce", str(path)]) == 1


def test_main_dispatches_declared_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-HARNESS-7699: the thin entrypoint uses the declared output."""

    calls = []
    monkeypatch.setattr(subject, "run_experiment", lambda *args: calls.append(args))
    assert subject.main([]) == 0
    assert calls == [(subject.ROOT, subject.RUN_DATE, subject.RESULT_PATH)]


def test_contract_rejects_unavailable_inputs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7699-AUTHORITY: absent authority and malformed design fail."""

    with pytest.raises(ValueError, match="matching roadmap"):
        contract.resolve_authority(tmp_path)
    with pytest.raises(ValueError, match="machine contract missing"):
        contract.design_contract("## Exact Task Contract\nno machine block")
    with pytest.raises(ValueError, match="unknown mutation"):
        contract.mutate_authority(ROADMAP, "unsupported")
    with pytest.raises(ValueError, match="only equality"):
        contract.input_check("bad", "x", "f", "in", 1, 1)
    (tmp_path / contract.PRIOR_DESIGN_PATH).parent.mkdir(parents=True)
    wrong = DESIGN.replace(contract.MILESTONE, "2026.09.999")
    (tmp_path / contract.PRIOR_DESIGN_PATH).write_text(wrong)
    with pytest.raises(ValueError, match="preserved contract changed"):
        contract.collect_prior_dispositions(tmp_path)
