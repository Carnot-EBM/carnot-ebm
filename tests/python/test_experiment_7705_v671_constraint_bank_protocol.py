"""REQ-REPORT-7705 and REQ-CL-7705-BOUNDED-BANK regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting.constraint_bank_protocol import Bank, advisory_status, grammar
from carnot.experiment_7705_v671_constraint_bank_protocol import (
    cold_reduce,
    fixture_cases,
    preflight,
    run_experiment,
)


def _bank(tmp_path: Path, scheduler: str = "priority", period: int = 2) -> Bank:
    return Bank(tmp_path / "state.json", grammar(), scheduler, 0.1, period)


def _feature(i: int) -> dict[str, float]:
    return {name: float((i + j) % 3 > 0) for j, name in enumerate(grammar()["primitives"])}


def test_7705_preflight_exact_missing_gate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7705-CUSTODY: missing inputs have exact operands."""
    checks, hashes = preflight(tmp_path)
    assert any(
        not c["passed"]
        and c["upstream_id"] == "exp7703-typed-decision-energy"
        and c["field"] == "exists"
        and c["observed"] is False
        for c in checks
    )
    assert hashes["producers"] == {}
    assert hashes["missing_inputs"]


def test_7705_grammar_and_authority(tmp_path: Path) -> None:
    """REQ-CL-7705-BOUNDED-BANK: candidates never become verifier axioms."""
    frozen = grammar()
    assert len(frozen["primitives"]) == 8
    assert len(frozen["pairs"]) == 28
    assert advisory_status("contradicted", True) == "contradicted"
    assert advisory_status("unknown", True) == "unknown"
    assert advisory_status("supported", True) == "supported"
    with pytest.raises(ValueError, match="grammar_invalid"):
        Bank(tmp_path / "bad.json", {"primitives": ["x"], "pairs": []}, "priority", 0.1, 2)


def test_7705_feedback_order_and_missing(tmp_path: Path) -> None:
    """SCENARIO-CL-7705-DURABLE: only the oldest due feedback applies."""
    bank = _bank(tmp_path)
    bank.predict("a", 0, _feature(0), 0.2, "unknown", "update", "源/α")
    bank.predict("b", 1, _feature(1), 0.3, "unknown", "update", "源/β")
    with pytest.raises(ValueError, match="future_feedback"):
        bank.release("a", 1, 7)
    with pytest.raises(ValueError, match="feedback_order"):
        bank.release("b", 1, 9)
    bank.mark_missing("a", 8)
    with pytest.raises(ValueError, match="duplicate_feedback"):
        bank.release("a", 1, 10)
    bank.release("b", 1, 9)
    with pytest.raises(ValueError, match="duplicate_feedback"):
        bank.release("b", 1, 10)
    with pytest.raises(ValueError, match="missing_feedback"):
        bank.propose()
    assert (
        Bank(tmp_path / "state.json", grammar(), "priority", 0.1, 2).state_hash == bank.state_hash
    )


def test_7705_overflow_and_rejected_credit(tmp_path: Path) -> None:
    """REQ-CL-7705-BOUNDED-BANK: overflow and rejection spend bounded credits."""
    bank = _bank(tmp_path)
    for i in range(16):
        bank.predict(str(i), i, _feature(i), 0.1, "unknown", "update", f"src/{i}")
    with pytest.raises(ValueError, match="pending_overflow"):
        bank.predict("overflow", 16, _feature(16), 0.1, "unknown", "update", "s")
    bank.release("0", 1, 16)
    proposal = bank.propose()
    assert proposal is not None
    assert proposal["gradient_steps"] <= 50
    assert bank.budget["proposal_credits_spent"] == 1
    bank.reject("heldout_worse")
    assert bank.budget["admission_credits_spent"] == 1
    assert bank.state["templates"] == []
    assert bank.budget["proposal_credits_spent"] == 1


def test_7705_restart_commit_once_and_cold_ledger(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7705-REPLAY: pending labels and future forecasts survive restart."""
    path = tmp_path / "bank.json"
    bank = Bank(path, grammar(), "priority", 0.1, 2)
    first = bank.predict("a", 0, _feature(0), 0.1, "unknown", "update", "源/α")
    bank = Bank(path, grammar(), "priority", 0.1, 2)
    assert bank.state["predictions"]["a"]["probability"] == first["probability"]
    bank.release("a", 1, 8)
    proposal = bank.propose()
    assert proposal is not None
    bank.simulate_crash_before_commit()
    bank = Bank(path, grammar(), "priority", 0.1, 2)
    assert not bank.state["templates"]
    bank.admit(True)
    bank = Bank(path, grammar(), "priority", 0.1, 2)
    assert len(bank.state["templates"]) == 1
    with pytest.raises(ValueError, match="no_proposal"):
        bank.admit(True)
    forecast = bank.predict("b", 9, _feature(1), 0.1, "unknown", "update", "源/β")
    reloaded = Bank(path, grammar(), "priority", 0.1, 2)
    assert reloaded.state["predictions"]["b"]["probability"] == forecast["probability"]
    assert reloaded.replay_ledger()["state_hash"] == reloaded.state_hash
    altered = json.loads(path.read_text())
    altered["ledger"].pop(0)
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="ledger_mismatch"):
        Bank(path, grammar(), "priority", 0.1, 2)


def test_7705_fixture_lifecycle_and_untouched_holdouts(tmp_path: Path) -> None:
    """SCENARIO-CL-7705-ACQUISITION: 96 lifecycle units and 32 untouched holdouts."""
    cases = fixture_cases()
    assert len(cases["update"]) == 64
    assert len(cases["admission"]) == 32
    assert len(cases["retention"]) == 32
    assert len({row["unit_id"] for group in cases.values() for row in group}) == 128
    assert all(
        row["verifier_status"] in {"supported", "contradicted", "unknown"}
        for row in cases["update"]
    )
    bank = _bank(tmp_path)
    for i, case in enumerate(cases["update"]):
        bank.predict(
            case["unit_id"],
            i,
            case["features"],
            case["base_probability"],
            case["verifier_status"],
            "update",
            case["source_id"],
        )
        if i >= 8:
            old = cases["update"][i - 8]
            bank.release(old["unit_id"], old["label"], i)
            candidate = bank.propose()
            if candidate:
                bank.reject("fixture_admission_not_yet_released")
    for i in range(56, 64):
        old = cases["update"][i]
        bank.release(old["unit_id"], old["label"], i + 8)
    assert len(bank.state["feedback"]) == 64
    assert bank.budget["proposal_credits_spent"] <= 6
    assert bank.replay_ledger()["state_hash"] == bank.state_hash
    assert not any(r["unit_id"] in bank.state["predictions"] for r in cases["retention"])


def test_7705_full_protocol_artifact_and_independent_reduction(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7705-TERMINAL: exact rows reduce from a fresh reader."""
    from carnot import experiment_7705_v671_constraint_bank_protocol as exp

    result = exp.measure_fixture(tmp_path, grammar(), 0.1)
    assert len({r["unit_id"] for r in result["rows"]}) == 128
    assert len(result["rows"]) >= 128 * 10
    assert result["budget_accounting"]["priority"]["proposal_credits_spent"] <= 6
    assert result["budget_accounting"]["priority"]["gradient_steps_spent"] <= 300
    assert result["restart_exact_parity"] is True
    assert result["ledger_reduction_valid"] is True
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "rows": result["rows"],
                "lifecycle_rows": result["lifecycle_rows"],
                "bank_state_paths": result["bank_state_paths"],
            }
        )
    )
    reduced = cold_reduce(candidate)
    assert reduced["independent_units"] == 128
    assert reduced["lifecycle_valid"] is True


def test_7705_terminal_blocked_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7705-CUSTODY: absent producers close without partial verdict."""
    output = tmp_path / "blocked.json"
    artifact = run_experiment(tmp_path, "20260926", output, validate=False)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["constraint_bank_ready_score"] == 0
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["gate_check_summary"]
    assert output.is_file()


def test_7705_current_preflight_authenticates_frozen_bytes() -> None:
    """SCENARIO-REPORT-7705-CUSTODY: current producers match frozen hashes."""
    from carnot.experiment_7705_v671_constraint_bank_protocol import ROOT

    checks, hashes = preflight(ROOT)
    assert checks and all(check["passed"] for check in checks)
    assert hashes["producers"] and hashes["pre_gate_receipts"]
    assert hashes["missing_inputs"] == []


def test_7705_rejects_bad_bank_inputs_and_mutated_ledger(tmp_path: Path) -> None:
    """SCENARIO-CL-7705-DURABLE: malformed state, sources and events fail closed."""
    from carnot.reporting.constraint_bank_protocol import replay_ledger

    for scheduler, threshold, period, error in (
        ("wrong", 0.1, 2, "scheduler_invalid"),
        ("fixed", 0.1, 3, "period_invalid"),
        ("priority", -1, 2, "threshold_invalid"),
    ):
        with pytest.raises(ValueError, match=error):
            Bank(tmp_path / f"{error}.json", grammar(), scheduler, threshold, period)
    bank = _bank(tmp_path)
    for args, error in (
        (("", 0, _feature(0), 0.2, "unknown", "update", "s"), "duplicate_prediction"),
        (("x", 0, {"x": 1.0}, 0.2, "unknown", "update", "s"), "features_invalid"),
        (("x", 0, _feature(0), 1.1, "unknown", "update", "s"), "base_probability_invalid"),
        (("x", 0, _feature(0), 0.2, "bad", "update", "s"), "exact_status_invalid"),
        (("x", 0, _feature(0), 0.2, "unknown", "bad", "s"), "partition_invalid"),
        (("x", 0, _feature(0), 0.2, "unknown", "update", ""), "source_identity_missing"),
    ):
        with pytest.raises(ValueError, match=error):
            bank.predict(*args)
    bank.predict("x", 0, _feature(0), 0.2, "unknown", "update", "源")
    with pytest.raises(ValueError, match="duplicate_prediction"):
        bank.predict("x", 1, _feature(0), 0.2, "unknown", "update", "源")
    with pytest.raises(ValueError, match="prediction_order"):
        bank.predict("y", 0, _feature(0), 0.2, "unknown", "update", "源")
    with pytest.raises(ValueError, match="unknown_prediction"):
        bank.release("absent", 1, 8)
    with pytest.raises(ValueError, match="label_invalid"):
        bank.release("x", True, 8)
    bank.release("x", 1, 8)
    with pytest.raises(ValueError, match="state_config_mismatch"):
        Bank(bank.path, grammar(), "priority", 0.2, 2)
    ledger = bank.state["ledger"]
    for field in ("event_hash", "state_hash"):
        corrupted = json.loads(json.dumps(ledger))
        corrupted[0][field] = "sha256:bad"
        with pytest.raises(ValueError, match="ledger_mismatch"):
            replay_ledger(corrupted)
    state = json.loads(bank.path.read_text())
    state["scalar_count"] = 99
    bank.path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="ledger_mismatch"):
        Bank(bank.path, grammar(), "priority", 0.1, 2)


def test_7705_proposal_and_admission_rejection_paths(tmp_path: Path) -> None:
    """REQ-CL-7705-BOUNDED-BANK: budgets and admission roles cannot be bypassed."""
    bank = _bank(tmp_path)
    with pytest.raises(ValueError, match="no_proposal"):
        bank.simulate_crash_before_commit()
    bank.predict("a", 0, _feature(0), 0.1, "unknown", "update", "s")
    bank.release("a", 1, 8)
    assert bank.propose() is not None
    with pytest.raises(ValueError, match="proposal_pending"):
        bank.propose()
    with pytest.raises(ValueError, match="admission_feedback_invalid"):
        bank.admit(True, "a")
    bank.budget["admission_credits_spent"] = 6
    with pytest.raises(ValueError, match="admission_budget_exhausted"):
        bank.admit(True)
    bank.budget["admission_credits_spent"] = 0
    bank.state["templates"] = [{"pair": ["x", "y"], "weight": 0}] * 36
    with pytest.raises(ValueError, match="template_overflow"):
        bank.admit(True)


def test_7705_priority_threshold_and_admission_once(tmp_path: Path) -> None:
    """REQ-CL-7705-BOUNDED-BANK: frozen threshold and one-use admission apply."""
    bank = Bank(tmp_path / "high.json", grammar(), "priority", 0.95, 2)
    bank.predict("a", 0, _feature(0), 0.2, "unknown", "update", "s")
    bank.release("a", 1, 8)
    assert bank.propose() is None
    bank = _bank(tmp_path / "low")
    bank.predict("a", 0, _feature(0), 0.1, "unknown", "update", "s")
    bank.release("a", 1, 8)
    bank.predict("b", 9, _feature(0), 0.1, "unknown", "admission", "s2")
    bank.release("b", 1, 17)
    assert bank.propose() is not None
    assert bank.admit(True, "b")["accepted"] is True
    assert bank.state["used_admissions"] == ["b"]
    with pytest.raises(ValueError, match="no_proposal"):
        bank.admit(True, "b")


def test_7705_publish_paths_and_reader_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7705-TERMINAL: validator exits determine final verdict."""
    from carnot import experiment_7705_v671_constraint_bank_protocol as exp

    frozen = tmp_path / exp.FROZEN
    frozen.parent.mkdir(parents=True)
    frozen.write_text(json.dumps({"scheduler": {"threshold": 0.1}, "thresholds": [0.0, 0.55]}))
    check = exp._check("fixture", "current", "fixture", "exists", True, True)
    hashes = {"producers": {}, "pre_gate_receipts": {}, "missing_inputs": []}
    monkeypatch.setattr(exp, "preflight", lambda root: ([check], hashes))

    def measured(raw: Path, frozen_grammar: dict, threshold: float) -> dict:
        rows = [
            exp._row(
                case,
                arm,
                case["base_probability"],
                0 if case["partition"] != "retention" else None,
                8 if case["partition"] != "retention" else None,
            )
            for arm in exp.ARMS
            for group in fixture_cases().values()
            for case in group
        ]
        budget = {
            arm: {
                "proposal_credits_spent": 1,
                "admission_credits_spent": 1,
                "gradient_steps_spent": 50,
            }
            for arm in exp.ARMS
        }
        return {
            "rows": rows,
            "lifecycle_rows": [{"kind": "commit"}],
            "budget_accounting": budget,
            "bank_state_paths": {},
            "restart_exact_parity": True,
            "ledger_reduction_valid": True,
        }

    monkeypatch.setattr(exp, "measure_fixture", measured)
    failed = False

    def commands(root: Path, specs: list, **kwargs: object) -> list[dict]:
        for spec in specs:
            for argument in spec.argv:
                if argument.startswith("--basetemp="):
                    assert Path(argument.split("=", 1)[1]).parent.exists()
        return [
            {
                "name": spec.name,
                "passed": not (failed and spec.name == "adversarial_verify"),
                "exit_code": 2 if failed and spec.name == "adversarial_verify" else 0,
                "log_path": str(tmp_path / spec.name),
                "log_sha256": "sha256:fixture",
                "command": "fixture",
            }
            for spec in specs
        ]

    monkeypatch.setattr(exp, "run_commands", commands)
    output = tmp_path / "published.json"
    artifact = run_experiment(tmp_path, "20260926", output)
    assert artifact["verdict_class"] == "circular_positive"
    assert len(artifact["validation_receipts"]["affected"]) == 8
    assert len(artifact["validation_receipts"]["terminal_readers"]) == 3
    assert artifact["acceptance_gate_results"][1]["passed"] is None
    failed = True
    artifact = run_experiment(tmp_path, "20260926", output)
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["flagged_adversarial"] is True
    assert artifact["constraint_bank_ready_score"] == 0


def test_7705_main_dispatch_and_bad_reduction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7705-REPLAY: cold reducer exit reflects row validity."""
    from carnot import experiment_7705_v671_constraint_bank_protocol as exp

    seen: list[str] = []
    monkeypatch.setattr(exp, "run_experiment", lambda root, date, output: seen.append(date))
    assert exp.main(["--date", "20260926", "--output", str(tmp_path / "out.json")]) == 0
    assert seen == ["20260926"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"rows": [], "lifecycle_rows": []}))
    assert exp.main(["--cold-reduce", str(path)]) == 1
    assert '"lifecycle_valid": false' in capsys.readouterr().out


def test_7705_missing_isolated_store_after_primary_custody(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7705-CUSTODY: absent evaluator bytes stay missing."""
    from carnot import experiment_7705_v671_constraint_bank_protocol as exp

    required = [
        Path("results/experiment_7700_v671_record_span_protocol.json"),
        Path("results/experiment_7701_v671_sealed_cohort.json"),
        exp.HEAD,
        exp.COHORT,
        exp.FROZEN,
    ]
    protocol = json.loads((exp.ROOT / exp.COHORT).read_text())
    for role in ("online_update", "online_admission", "retention"):
        required.append(exp.COHORT.parent / protocol["roles"][role]["model_inputs"])
        if role != "retention":
            required.append(exp.COHORT.parent / protocol["evaluator_stores"][role]["path"])
    for relative in required:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(exp.ROOT / relative)
    checks, hashes = preflight(tmp_path)
    missing = exp.COHORT.parent / protocol["evaluator_stores"]["retention"]["path"]
    assert str(missing) in hashes["missing_inputs"]
    assert any(c["artifact_path"] == str(missing) and c["observed"] is None for c in checks)


def test_7705_future_admission_and_reused_label(tmp_path: Path) -> None:
    """SCENARIO-CL-7705-ACQUISITION: an old admission cannot validate a new proposal."""
    from carnot import experiment_7705_v671_constraint_bank_protocol as exp

    bank = _bank(tmp_path)
    bank.predict("update", 0, _feature(0), 0.1, "unknown", "update", "s")
    bank.predict("admission", 1, _feature(0), 0.1, "unknown", "admission", "s")
    bank.release("update", 1, 8)
    proposal = bank.propose()
    assert proposal is not None
    bank.release("admission", 1, 9)
    assert exp._admit_decision(bank, "admission") is False
    bank.admit(True, "admission")
    bank.state["proposal"] = proposal
    with pytest.raises(ValueError, match="admission_used"):
        bank.admit(True, "admission")


def test_7705_end_of_stream_rollback_and_checkpoint_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7705-ACQUISITION: late counterexamples cannot borrow old labels."""
    from carnot import experiment_7705_v671_constraint_bank_protocol as exp

    original = fixture_cases()
    for index, row in enumerate(original["update"]):
        row["label"] = int(index == 63)
        row["base_probability"] = 0.1
    monkeypatch.setattr(exp, "fixture_cases", lambda: original)
    monkeypatch.setattr(exp, "ARMS", ("priority",))
    first = exp.measure_fixture(tmp_path, grammar(), 0.1)
    second = exp.measure_fixture(tmp_path, grammar(), 0.1)
    assert first["budget_accounting"]["priority"]["admission_credits_spent"] == 1
    assert any(event["kind"] == "rollback" for event in second["lifecycle_rows"])


def test_7705_empty_bank_is_circular_mechanics_without_benefit() -> None:
    """REQ-REPORT-7705: fixture success carries measured operands, never benefit."""
    import time

    from carnot import experiment_7705_v671_constraint_bank_protocol as exp

    rows = [
        exp._row(
            case,
            arm,
            case["base_probability"],
            0 if case["partition"] != "retention" else None,
            8 if case["partition"] != "retention" else None,
        )
        for arm in exp.ARMS
        for group in fixture_cases().values()
        for case in group
    ]
    measured = {
        "rows": rows,
        "lifecycle_rows": [],
        "budget_accounting": {
            arm: {
                "proposal_credits_spent": 0,
                "admission_credits_spent": 0,
                "gradient_steps_spent": 0,
            }
            for arm in exp.ARMS
        },
        "restart_exact_parity": True,
        "ledger_reduction_valid": True,
    }
    check = exp._check("fixture", "current", "fixture", "exists", True, True)
    artifact = exp.build_artifact(
        "20260926",
        time.monotonic(),
        [check],
        {"producers": {}, "pre_gate_receipts": {}, "missing_inputs": []},
        measured,
        {},
        [],
    )
    assert artifact["verdict_class"] == "circular_positive"
    assert "no_acquisition" in artifact["honest_verdict"]
    assert artifact["constraint_bank_ready_score"] == 0
    gates = {row["gate"]: row for row in artifact["acceptance_gate_results"]}
    assert gates["probability"]["operands"]["mean_brier_by_arm"]["priority"] > 0
    assert gates["retention"]["operands"]["mean_retention_brier_by_arm"]["priority"] > 0
    assert artifact["prior_failures"][0]["experiment_id"] == "exp7691-constraint-bank-protocol"
