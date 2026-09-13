"""Terminal handoff checks for the ARC transition-witness receipt.

Spec: REQ-ARC-WMTE-7262 and SCENARIO-ARC-WMTE-7262-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import carnot.experiment_7262_v639_arc_witness_receipt as receipt
from carnot.agentic.arc_transition_witness_exp7248 import build_transition_witness
from carnot.experiment_7248_v638_arc_witness import BAD_CODE, _changed_rows, _exec_engine
from carnot.experiment_7262_v639_arc_witness_receipt import (
    RUN_DATE,
    ValidationCommand,
    _array_sha256,
    _blocked_artifact,
    _complete_artifact,
    _payload_from_second_request,
    _phase_span,
    _reduce_raw,
    _sha256_file,
    _write_evidence,
    build_validation_commands,
    check_preconditions,
    main,
    reduce_receipt_rows,
    run_policy_handoff_panel,
    run_validation_commands,
    validate_artifact,
)


def _passing_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"unit": unit, "passed": True}
        for unit in (
            "identity_changed",
            "identity_unchanged",
            "wrong_direction",
            "missing_effect",
            "malformed_output",
        )
    ]
    rows.extend(
        [
            {
                "unit": "runtime_delivery",
                "generated_witness_sha256": "sha256:old",
                "delivered_witness_sha256": "sha256:old",
                "accepted_by_existing_policy_gate": True,
            },
            {"unit": "default_parity", "request_bytes_equal": True, "action_equal": True},
            {"unit": "leakage", "agent_observation_history_only": True},
            {"unit": "disconnect_mutation", "caught": True},
            {
                "unit": "policy_runtime_delivery",
                "generated_witness_sha256": "sha256:new",
                "delivered_witness_sha256": "sha256:new",
                "delivered_to_next_prompt": True,
                "direction_and_state_identity_match": True,
                "accepted_by_existing_policy_gate": True,
            },
            {
                "unit": "policy_default_parity",
                "request_bytes_equal": True,
                "action_equal": True,
            },
            {"unit": "policy_future_leakage", "future_transition_absent": True},
            {"unit": "policy_plan_authority", "policy_plan_unchanged": True},
        ]
    )
    return rows


def _artifact_kwargs(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "started_at": "2026-09-13T12:00:00+00:00",
        "completed_at": "2026-09-13T12:00:01+00:00",
        "duration_s": 1.0,
        "preconditions": {"checks": [], "failed_checks": []},
        "source_hashes": {"CODEX.md": "sha256:abc"},
        "rows": rows,
        "witness_rows": [{"unit": "policy_runtime_delivery"}],
        "terminal_handoff_rows": [
            {"candidate": "partial_checkpoint", "checker_passed": False},
            {"candidate": "complete_null_candidate", "checker_passed": True},
        ],
        "validation_receipts": [
            {"name": "all", "exit_code": 0, "passed": True, "log_sha256": "sha256:ok"}
        ],
        "phase_spans": [],
    }


def test_reducer_requires_actual_policy_handoff_and_controls() -> None:
    """SCENARIO-7262-WITNESS-DELIVERY: readiness needs the scored policy seam."""
    rows = _passing_rows()
    reduced = reduce_receipt_rows(rows)
    assert reduced["arc_witness_ready_score"] == 1
    assert all(reduced["gates"].values())

    disconnected = deepcopy(rows)
    disconnected[-4]["delivered_witness_sha256"] = None
    assert reduce_receipt_rows(disconnected)["arc_witness_ready_score"] == 0

    leaked = deepcopy(rows)
    leaked[-2]["future_transition_absent"] = False
    assert reduce_receipt_rows(leaked)["gates"]["no_future_transition_leakage"] is False


def test_real_policy_handoff_binds_observed_direction_and_next_prompt(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7262: scripted HTTP drives the real E3AgentPolicy wrapper."""
    panel = run_policy_handoff_panel(tmp_path)
    assert panel["reduction"]["arc_witness_ready_score"] == 1
    delivery = next(row for row in panel["rows"] if row["unit"] == "policy_runtime_delivery")
    assert delivery["delivered_to_next_prompt"] is True
    assert delivery["direction_and_state_identity_match"] is True
    assert delivery["accepted_by_existing_policy_gate"] is True
    assert delivery["refinement_action"] == "refactor_tool_loop"
    assert len(panel["witness_rows"]) == 8
    assert all(row["actual_next_call_delivery"] for row in panel["witness_rows"])
    assert all(
        row["observed_successor_sha256"].startswith("sha256:") for row in panel["witness_rows"]
    )
    expected_by_pre_hash = {
        build_transition_witness([transition], _exec_engine(BAD_CODE))["mismatches"][0][
            "pre_frame_hash"
        ]: _array_sha256(transition.next_grid)
        for transition in _changed_rows()
    }
    assert all(
        row["observed_successor_sha256"]
        == expected_by_pre_hash[row["emitted_witness"]["pre_frame_hash"]]
        for row in panel["witness_rows"]
    )
    assert panel["default_parity_receipt"]["request_bytes_equal"] is True
    assert panel["default_parity_receipt"]["action_equal"] is True
    assert panel["plan_installation_authority"]["policy_plan_unchanged"] is True


def test_complete_builder_and_blocked_builder_are_terminal() -> None:
    """SCENARIO-7262-TERMINAL-HANDOFF: terminal states never reuse partial."""
    rows = _passing_rows()
    complete = _complete_artifact(**_artifact_kwargs(rows))
    validate_artifact(complete)
    assert complete["status"] == "complete"
    assert complete["run_date"] == RUN_DATE
    assert complete["arc_witness_ready_score"] == 1
    assert complete["verdict_class"] == "circular_positive"

    broken = deepcopy(complete)
    broken["status"] = "partial"
    broken["honest_verdict"] = "partial_measurement_checkpoint_not_terminal"
    broken["verdict_class"] = "partial"
    try:
        validate_artifact(broken)
    except ValueError as exc:
        assert "terminal status" in str(exc)
    else:  # pragma: no cover - the assertion above is the regression target
        raise AssertionError("partial checkpoint passed the terminal validator")

    failure = {
        "check": "required_readable_input",
        "upstream": "missing.json",
        "field": "exists",
        "observed": False,
        "expected": True,
        "passed": False,
    }
    blocked = _blocked_artifact(
        started_at="2026-09-13T12:00:00+00:00",
        completed_at="2026-09-13T12:00:01+00:00",
        duration_s=1.0,
        preconditions={"failed_checks": [failure]},
        source_hashes={},
        failures=[failure],
    )
    validate_artifact(blocked)
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"] == [failure]
    assert blocked["arc_witness_ready_score"] == 0
    assert "partial" not in blocked["honest_verdict"]


def test_complete_builder_downgrades_failed_validation_to_null() -> None:
    """REQ-ARC-WMTE-7262: a failed scientific check cannot retain readiness."""
    kwargs = _artifact_kwargs(_passing_rows())
    kwargs["validation_receipts"] = [
        {"name": "ruff_check", "exit_code": 1, "passed": False, "log_sha256": "sha256:no"}
    ]
    artifact = _complete_artifact(**kwargs)
    validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert artifact["arc_witness_ready_score"] == 0


def test_validation_plan_is_scoped_and_has_both_terminal_checker_controls(
    tmp_path: Path,
) -> None:
    """SCENARIO-7262-TERMINAL-HANDOFF: no historical full-suite runner is imported."""
    commands = build_validation_commands(
        terminal_candidate=tmp_path / "experiment_terminal.json",
        partial_checkpoint=tmp_path / "experiment_partial.json",
        raw_rows=tmp_path / "rows.json",
    )
    names = [row.name for row in commands]
    assert "full_python_suite" not in names
    assert names.count("focused_exp7262") == 1
    assert names.count("affected_exp7248") == 1
    assert names.count("E2E-009") == 1
    assert names.count("E2E-010") == 1
    assert names.index("partial_checkpoint_rejected") < names.index(
        "complete_candidate_adversarial_verify"
    )
    assert "independent_raw_row_reducer" in names
    assert "offline_e3_smoke" in names
    partial = commands[names.index("partial_checkpoint_rejected")]
    assert partial.expected_exit_code == 1
    assert str(tmp_path / "experiment_partial.json") in partial.command
    assert all("tests/python -q" not in " ".join(row.command) for row in commands)


def test_artifact_checksum_detects_mutation() -> None:
    """REQ-ARC-WMTE-7262: the digest binds terminal raw evidence."""
    artifact = _complete_artifact(**_artifact_kwargs(_passing_rows()))
    artifact["rows"][0]["passed"] = False
    try:
        validate_artifact(artifact)
    except ValueError as exc:
        assert "checksum" in str(exc)
    else:  # pragma: no cover - the assertion above is the regression target
        raise AssertionError("mutated artifact passed validation")


def test_raw_rows_are_json_serializable() -> None:
    """REQ-ARC-WMTE-7262: each raw unit remains independently reducible."""
    encoded = json.dumps(_passing_rows(), sort_keys=True)
    assert reduce_receipt_rows(json.loads(encoded))["arc_witness_ready_score"] == 1


def test_request_parser_rejects_missing_second_call_or_marker() -> None:
    """REQ-ARC-WMTE-7262: delivery needs a marked second prompt."""
    assert _payload_from_second_request([]) is None
    no_marker = json.dumps({"messages": [{"content": "ordinary prompt"}]}).encode()
    assert _payload_from_second_request([b"{}", no_marker]) is None


def test_preconditions_authenticate_success_and_external_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-ARC-WMTE-7262: absence and quarantine become structured failures."""
    spec = tmp_path / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
    spec.parent.mkdir(parents=True)
    spec.write_text("REQ-ARC-WMTE-7262\n")
    exclusion = tmp_path / "ops/exclusion_manifest.yaml"
    exclusion.parent.mkdir()
    exclusion.write_text("retired: []\n")
    source = tmp_path / "source.txt"
    source.write_text("bytes\n")
    for directory in (
        tmp_path / "results",
        tmp_path / "results/checkpoints",
        tmp_path / "results/raw",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(receipt, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        receipt,
        "INPUT_PATHS",
        (
            Path("source.txt"),
            Path("openspec/capabilities/arc-world-model-trust-energy/spec.md"),
            Path("ops/exclusion_manifest.yaml"),
        ),
    )
    monkeypatch.setattr(
        receipt,
        "OUTPUT_PATH",
        Path("results/experiment_7262_v639_arc_witness_receipt.json"),
    )
    preconditions, hashes, failures = check_preconditions()
    assert failures == []
    assert hashes["source.txt"] == _sha256_file(source)
    assert preconditions["failed_checks"] == []

    source.unlink()
    spec.write_text("no requirement\n")
    exclusion.write_text("exp7262: quarantined\n")
    (tmp_path / "results/raw").rmdir()
    preconditions, _hashes, failures = check_preconditions()
    assert {row["check"] for row in failures} == {
        "required_readable_input",
        "driving_requirement",
        "quarantine_state",
        "writable_output_directory",
    }
    assert preconditions["failed_checks"] == failures


def test_validation_runner_records_expected_failure_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-7262: expected checker rejection differs from timeout."""
    results = iter(
        [
            {
                "exit_code": 1,
                "duration_s": 0.1,
                "timed_out": False,
                "stdout": "NONTERMINAL_DECLARED_ARTIFACT",
            },
            {
                "exit_code": 124,
                "duration_s": 1.0,
                "timed_out": True,
                "output": "late",
            },
        ]
    )
    monkeypatch.setattr(
        receipt,
        "run_streaming_command",
        lambda *_args, **_kwargs: next(results),
    )
    commands = [
        ValidationCommand(
            "negative",
            ["checker", "partial.json"],
            expected_exit_code=1,
            expected_output="NONTERMINAL",
        ),
        ValidationCommand("timeout", ["checker", "terminal.json"]),
    ]
    rows = run_validation_commands(commands, started=0.0)
    assert rows[0]["passed"] is True
    assert rows[1]["passed"] is False
    assert rows[1]["timed_out"] is True


def test_write_evidence_and_independent_raw_reduction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-7262: raw rows and fixture history use separate substrates."""
    (tmp_path / "results/raw").mkdir(parents=True)
    prior = tmp_path / "results/experiment_7248_v638_arc_witness.json"
    prior.write_text(json.dumps({"MODEL_SPECS": ["historical"], "model_invoked": True}))
    monkeypatch.setattr(receipt, "REPO_ROOT", tmp_path)

    def temporary_write(path, payload, **_kwargs):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload))
        return target

    monkeypatch.setattr(receipt, "atomic_write_json", temporary_write)
    panel = {"rows": _passing_rows(), "reduction": reduce_receipt_rows(_passing_rows())}
    hashes = _write_evidence(panel)
    assert set(hashes) == {str(receipt.SIDECAR_PATH), str(receipt.RAW_PATH)}
    sidecar = json.loads((tmp_path / receipt.SIDECAR_PATH).read_text())
    raw = json.loads((tmp_path / receipt.RAW_PATH).read_text())
    assert sidecar["historical_model_receipts"] == ["historical"]
    assert sidecar["current_invocation"] is False
    assert raw["inference_substrate_class"] == "aggregation"
    assert _reduce_raw(tmp_path / receipt.RAW_PATH) == 0
    assert "arc_witness_ready_score" in capsys.readouterr().out
    raw["expected_reduction"]["arc_witness_ready_score"] = 0
    mismatch = tmp_path / "mismatch.json"
    mismatch.write_text(json.dumps(raw))
    assert _reduce_raw(mismatch) == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("rows"), "missing required"),
        (lambda value: value.__setitem__("run_date", "20260912"), "run date"),
        (lambda value: value["field_principles"].pop("rows"), "field principle"),
        (lambda value: value.__setitem__("MODEL_SPECS", ["model"]), "model invocation"),
        (
            lambda value: value["invocation_counts"].__setitem__("generation_attempts", 1),
            "counters",
        ),
        (lambda value: value.__setitem__("arc_witness_ready_score", 0), "stored readiness"),
        (lambda value: value.__setitem__("verdict_class", "null"), "verdict class"),
    ],
)
def test_terminal_validator_rejects_inconsistent_fields(mutation, message: str) -> None:
    """REQ-ARC-WMTE-7262: terminal fields cannot contradict measured evidence."""
    artifact = _complete_artifact(**_artifact_kwargs(_passing_rows()))
    mutation(artifact)
    artifact["reproducibility_checksum"] = receipt.reproducibility_checksum(artifact)
    with pytest.raises(ValueError, match=message):
        validate_artifact(artifact)


def test_blocked_validator_requires_summary_and_zero_score() -> None:
    """REQ-ARC-WMTE-7262: an external block must name its failed check."""
    failure = {"check": "missing", "passed": False}
    blocked = _blocked_artifact(
        started_at="2026-09-13T12:00:00+00:00",
        completed_at="2026-09-13T12:00:01+00:00",
        duration_s=1.0,
        preconditions={"failed_checks": [failure]},
        source_hashes={},
        failures=[failure],
    )
    blocked["gate_check_summary"] = []
    blocked["reproducibility_checksum"] = receipt.reproducibility_checksum(blocked)
    with pytest.raises(ValueError, match="gate_check_summary"):
        validate_artifact(blocked)
    blocked["gate_check_summary"] = [failure]
    blocked["arc_witness_ready_score"] = 1
    blocked["reproducibility_checksum"] = receipt.reproducibility_checksum(blocked)
    with pytest.raises(ValueError, match="cannot claim readiness"):
        validate_artifact(blocked)


def test_phase_span_uses_measured_offsets() -> None:
    """REQ-ARC-WMTE-7262: phase time comes from monotonic boundaries."""
    assert _phase_span(10.0, "work", 12.0, 15.0) == {
        "phase": "work",
        "start_offset_s": 2.0,
        "end_offset_s": 5.0,
        "duration_s": 3.0,
    }


def test_main_publishes_only_after_both_checker_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-7262-TERMINAL-HANDOFF: readiness follows terminal checking."""
    panel = {
        "rows": _passing_rows(),
        "reduction": reduce_receipt_rows(_passing_rows()),
        "witness_rows": [{"unit": "policy_runtime_delivery"}],
    }
    old_key = "results/experiment_7248_v638_arc_witness.json"
    monkeypatch.setattr(
        receipt,
        "check_preconditions",
        lambda: ({"checks": [], "failed_checks": []}, {old_key: "sha256:same"}, []),
    )
    monkeypatch.setattr(receipt, "run_policy_handoff_panel", lambda *_args: panel)
    monkeypatch.setattr(
        receipt,
        "_write_evidence",
        lambda _panel: {str(receipt.RAW_PATH): "sha256:raw"},
    )
    monkeypatch.setattr(receipt, "_sha256_file", lambda _path: "sha256:same")
    commands = [
        ValidationCommand("independent_raw_row_reducer", ["raw"]),
        ValidationCommand("partial_checkpoint_rejected", ["partial"], expected_exit_code=1),
        ValidationCommand("complete_candidate_adversarial_verify", ["complete"]),
        ValidationCommand("complete_candidate_row_consistency", ["rows"]),
    ]
    monkeypatch.setattr(receipt, "build_validation_commands", lambda **_kwargs: commands)

    def fake_validation(selected, _started):
        return [
            {
                "name": command.name,
                "expected_exit_code": command.expected_exit_code,
                "exit_code": command.expected_exit_code,
                "passed": True,
                "log_sha256": "sha256:ok",
            }
            for command in selected
        ]

    monkeypatch.setattr(receipt, "run_validation_commands", fake_validation)
    writes: list[tuple[Path, dict[str, object]]] = []
    monkeypatch.setattr(
        receipt,
        "atomic_write_json",
        lambda path, payload, **_kwargs: writes.append((Path(path), deepcopy(payload))),
    )
    assert main(["--date", RUN_DATE]) == 0
    assert [path for path, _payload in writes][-1] == receipt.OUTPUT_PATH
    terminal = writes[-1][1]
    assert terminal["arc_witness_ready_score"] == 1
    assert terminal["terminal_handoff_rows"][0]["checker_passed"] is False
    assert terminal["terminal_handoff_rows"][1]["checker_passed"] is True


def test_main_publishes_terminal_block_for_failed_precondition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-7262: external absence publishes blocked, never partial."""
    failure = {"check": "required_readable_input", "passed": False}
    monkeypatch.setattr(
        receipt,
        "check_preconditions",
        lambda: ({"failed_checks": [failure]}, {}, [failure]),
    )
    writes = []
    monkeypatch.setattr(
        receipt,
        "atomic_write_json",
        lambda path, payload, **_kwargs: writes.append((path, payload)),
    )
    assert main(["--date", RUN_DATE]) == 2
    assert writes[-1][1]["status"] == "blocked"
    assert writes[-1][1]["verdict_class"] == "blocked"


def test_main_reduce_mode_and_frozen_date(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-7262: CLI reduction is independent and the date is frozen."""
    rows = _passing_rows()
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"rows": rows, "expected_reduction": reduce_receipt_rows(rows)}))
    assert main(["--reduce-raw", str(raw)]) == 0
    assert "arc_witness_ready_score" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--date", "20260912"])
