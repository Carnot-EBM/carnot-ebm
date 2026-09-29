"""Current capstone controls. REQ-REPORT-7890-V684."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from carnot.reporting import v684_capstone as cap
from scripts.experiments import experiment_7890_v684_capstone as cli


ROOT = Path(__file__).resolve().parents[2]


def fixture_root(tmp_path: Path) -> Path:
    """Keep a real design contract while isolating mutable producer bytes."""
    for name in (cap.DESIGN, cap.STAGED, cap.ACTIVE):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (tmp_path / "results").mkdir()
    return tmp_path


def test_ledger_keeps_missing_producers_and_gate_operands(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7890-LEDGER: skip receipts never fill absent science."""
    root = fixture_root(tmp_path)
    skip = root / "results/experiment_7882_conductor_gate_blocked.json"
    skip.write_text('{"reason":"upstream"}')
    ledger = cap.collect(root, "20260929")
    assert len(ledger["task_evidence_rows"]) == 12
    assert ledger["task_evidence_rows"][-1]["status"] == "own_administrative_row"
    assert ledger["task_evidence_rows"][3]["status"] == "missing_producer"
    assert ledger["task_evidence_rows"][3]["skip_receipts"][0]["path"] == str(skip)
    assert any(
        x["upstream_id"] == "Exp7882"
        and x["artifact_field"] == "producer_exists"
        and x["observed"] is False
        for x in ledger["gate_check_summary"]
    )
    assert any(
        x["upstream_id"] == "Exp7880"
        and x["artifact_field"] == "source_boundary_ready_score"
        and x["observed"] == "missing_source"
        for x in ledger["gate_check_summary"]
    )


def test_gate_field_failure_differs_from_missing_field(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7890-LEDGER: preserve actual false and absent fields."""
    root = fixture_root(tmp_path)
    path = root / "results/experiment_7880_v684_source_boundary.json"
    path.write_text(json.dumps({"experiment_id": 7880, "source_boundary_ready_score": 0}))
    ledger = cap.collect(root, "20260929")
    failures = [x for x in ledger["gate_check_summary"] if x["path"] == str(path)]
    assert any(
        x["artifact_field"] == "source_boundary_ready_score" and x["observed"] == 0
        for x in failures
    )
    assert any(
        x["artifact_field"] == "flagged_adversarial" and x["observed"] == "missing_field"
        for x in failures
    )
    assert all(x["hash"] for x in failures)


def test_family_reduction_rejects_label_and_denominator_changes() -> None:
    """SCENARIO-REPORT-7890-REDUCTION: seeds cannot inflate paired family N."""
    rows = [
        {
            "family_id": "a",
            "arm": arm,
            "seed": seed,
            "status": "completed",
            "label": 1,
            "original_human_label": 1,
            "probability": probability,
            "cost": cost,
        }
        for arm, probability, cost in (("head", 0.8, 0.2), ("control", 0.5, 0.5))
        for seed in (1, 2)
    ]
    result = cap.reduce_primitive(rows, seed=9)
    assert result["independent_family_count"] == 1
    assert result["seed_count"] == 4
    assert result["paired"]["head:control"]["cost_gain"] == pytest.approx(0.3)
    assert result["paired"]["head:control"]["brier_gain"] == pytest.approx(0.21)
    changed = deepcopy(rows)
    changed[0]["label"] = 0
    with pytest.raises(ValueError, match="human_label_changed"):
        cap.reduce_primitive(changed, seed=9)
    changed = deepcopy(rows)
    changed[0]["eligible"] = 0
    changed[0]["completed"] = 1
    with pytest.raises(ValueError, match="denominator_conflict"):
        cap.reduce_primitive(changed, seed=9)


def test_replay_rejects_changed_source_bytes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7890-REDUCTION: candidate hashes bind current bytes."""
    root = fixture_root(tmp_path)
    candidate = cap.build_candidate(root, "20260929")
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate))
    assert cap.cold_replay(path, root) == []
    source = root / cap.DESIGN
    source.write_text(source.read_text() + "\nchanged\n")
    assert "source_bytes_changed" in cap.cold_replay(path, root)


def test_actual_import_roots_and_real_cli(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7890-TERMINAL: both package roots are legitimate."""
    imports = cap.resolve_imports(ROOT)
    assert imports["carnot.reporting.v684_capstone"].endswith(
        "python/carnot/reporting/v684_capstone.py"
    )
    assert imports["scripts.publication_gate"].endswith("scripts/publication_gate.py")
    output = tmp_path / "candidate.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/experiments/experiment_7890_v684_capstone.py"),
        "--date",
        "20260929",
        "--output",
        str(output),
        "--evidence-only",
    ]
    done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    result = json.loads(output.read_text())
    assert result["experiment_id"] == 7890
    assert result["verdict_class"] == "blocked"
    replay = subprocess.run(
        [sys.executable, cmd[1], "--cold-replay", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert replay.returncode == 0, replay.stderr


def test_private_row_and_import_mutations(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7890-REDUCTION: malformed and relabeled rows fail closed."""
    with pytest.raises(ValueError, match="malformed_primitive_row"):
        cap.reduce_primitive([None])  # type: ignore[list-item]
    rows = [
        {"family_id": "f", "arm": "head", "status": "completed", "label": 0},
        {"family_id": "f", "arm": "head", "status": "completed", "label": 1},
        {"family_id": "f", "arm": "control", "status": "completed"},
    ]
    with pytest.raises(ValueError, match="family_label_conflict"):
        cap.reduce_primitive(rows)
    result = cap.reduce_primitive(rows[0:1] + rows[2:3])
    assert result["paired"]["head:control"]["cost_gain"] is None
    assert result["paired"]["head:control"]["cost_gain_ci95"] is None
    assert cap._read(tmp_path / "absent.json") is None
    wrong = tmp_path / "wrong.json"
    wrong.write_text("[]")
    assert cap._read(wrong) is None
    with pytest.raises(ValueError, match="carnot_import_root_changed"):
        cap.resolve_imports(tmp_path)
    real = cap.importlib.import_module
    monkeypatch.setattr(
        cap.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            __file__=str(tmp_path / "wrong.py")
            if name == "scripts.publication_gate"
            else real(name).__file__
        ),
    )
    with pytest.raises(ValueError, match="scripts_import_root_changed"):
        cap.resolve_imports(ROOT)


def test_private_producer_and_replay_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7890-LEDGER: missing, malformed and failed bytes differ."""
    root = fixture_root(tmp_path)
    with pytest.raises(ValueError, match="v684_date_changed"):
        cap.collect(root, "20260928")
    original = cap.parse_design
    monkeypatch.setattr(cap, "parse_design", lambda *args, **kwargs: ([], []))
    with pytest.raises(ValueError, match="v684_contract_order_changed"):
        cap.collect(root, "20260929")
    monkeypatch.setattr(cap, "parse_design", original)
    invalid = root / "results/experiment_7883_v684_decision_abstention.json"
    invalid.write_text("not json")
    prior_attempt = root / "results/raw/experiment_7890_v684_capstone/attempt1_disqualified.json"
    prior_attempt.parent.mkdir(parents=True)
    prior_attempt.write_text(
        json.dumps(
            {
                "run_date": "20260929",
                "validation_receipts": [
                    {
                        "name": "affected_pytest",
                        "classification": "required",
                        "passed": False,
                        "exit_code": 1,
                        "log_path": "/tmp/old.log",
                        "log_sha256": "sha256:old",
                    }
                ],
            }
        )
    )
    science = root / "results/experiment_7882_v684_energy_fit.json"
    science.write_text(
        json.dumps(
            {
                "experiment_id": 7882,
                "task_id": "exp7882-energy-fit",
                "milestone": "2026.09.684",
                "run_date": "20260929",
                "flagged_adversarial": True,
                "verdict_class": "null",
                "validation_receipts": [
                    {"name": "owned", "classification": "required", "passed": False, "exit_code": 1}
                ],
                "rows": [
                    {
                        "status": "completed",
                        "family_id": "f",
                        "arm": "head",
                        "seed": 1,
                        "original_human_label": 1,
                        "label": 1,
                        "probability": 0.8,
                        "cost": 0.2,
                    }
                ],
            }
        )
    )
    ledger = cap.collect(root, "20260929")
    assert ledger["task_evidence_rows"][4]["status"] == "invalid_json"
    assert ledger["task_evidence_rows"][3]["status"] == "null"
    assert ledger["task_evidence_rows"][3]["eligible"] is False
    assert any(x["name"] == "owned" for x in ledger["historical_required_failures"])
    assert any(
        x["upstream_id"] == "Exp7890-attempt1" for x in ledger["historical_required_failures"]
    )
    assert ledger["recomputed_comparison_rows"][3]["independent_family_count"] == 1
    value = cap.build_candidate(root, "20260929")
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(value))
    assert cap.cold_replay(path, root) == []
    assert cap.cold_replay(tmp_path / "absent.json", root) == ["candidate_unreadable"]
    changed = deepcopy(value)
    changed["rows"] = []
    changed["gate_check_summary"] = []
    changed["validation_receipts"] = [
        {"log_path": str(tmp_path / "absent.log"), "log_sha256": "sha256:0"}
    ]
    path.write_text(json.dumps(changed))
    assert {"rows_changed", "gate_operands_changed", "validation_log_changed"} <= set(
        cap.cold_replay(path, root)
    )


def test_primitive_accounting_counts_independent_families() -> None:
    """SCENARIO-REPORT-7890-REDUCTION: causal and service fields stay primitive."""
    rows = [
        {
            "family_id": "a",
            "arm": "head",
            "status": "completed",
            "seed": 1,
            "prediction_step": 3,
            "release_step": 2,
            "admitted": True,
            "constraint_effect": True,
            "no_write_decision": True,
            "retention": True,
            "syntax_valid": True,
            "source_byte_fidelity": True,
            "semantic_sensitivity": True,
            "new_level_solve": True,
            "current_hardware_execution": False,
            "probability_unsupported": 0.4,
            "label": 1,
        }
    ]
    result = cap.reduce_primitive(rows)
    assert result["release_violations"] == 1
    assert result["admitted_count"] == result["constraint_effect_count"] == 1
    assert result["syntax_valid_count"] == result["source_fidelity_count"] == 1
    assert result["source_sensitivity_count"] == result["retention_count"] == 1
    assert result["new_arc_solve_count"] == 1
    assert result["device_execution_count"] == 0


def test_frozen_manifest_and_log_seals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7890-TERMINAL: exact argv and sealed logs are stable."""
    manifest = cli.commands(tmp_path)
    names = {item.name for item in manifest}
    assert {
        "package_root_imports",
        "affected_pytest",
        "coverage_unit",
        "cli_success",
        "cli_missing_input",
        "cold_replay",
        "changed_coverage",
        "e2e_016_fixture",
        "e2e_017",
    } <= names
    assert all(item.timeout_s > 0 and item.argv for item in manifest)
    bad = (*cli.TESTS, "tests/python/absent_7890.py")
    monkeypatch.setattr(cli, "TESTS", bad)
    with pytest.raises(ValueError, match="affected_scope_missing"):
        cli.commands(tmp_path)
    monkeypatch.setattr(cli, "TESTS", bad[:-1])
    log = tmp_path / "child.log"
    log.write_text("closed")
    receipt = {
        "name": "one",
        "log_path": str(log),
        "log_sha256": cap.sha256_file(log),
        "scope": "required",
    }
    cli.seal_receipts([receipt], tmp_path)
    assert Path(receipt["log_path"]).is_file()
    assert receipt["classification"] == "required"
    Path(receipt["log_path"]).write_text("tampered")
    with pytest.raises(ValueError, match="sealed_log_collision"):
        cli.seal_receipts(
            [
                {
                    "name": "one",
                    "log_path": str(log),
                    "log_sha256": cap.sha256_file(log),
                    "scope": "required",
                }
            ],
            tmp_path,
        )


@pytest.mark.parametrize(
    "mode",
    [
        "passing",
        "required_failure",
        "flagged",
        "terminal_failure",
        "replay_failure",
        "recheck_failure",
    ],
)
def test_current_cli_supervision_with_closed_child_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """SCENARIO-REPORT-7890-TERMINAL: own failures and flags zero readiness."""
    root = fixture_root(tmp_path / "repo")
    private = tmp_path / "private"
    private.mkdir()
    calls = 0

    def fake_run_commands(
        _root: Path, specs: list[object], *, log_dir: Path, heartbeat_s: float
    ) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        log_dir.mkdir(parents=True, exist_ok=True)
        receipts = []
        for spec in specs:
            name = spec.name  # type: ignore[attr-defined]
            content = (
                json.dumps({"flagged_count": int(mode in {"flagged", "recheck_failure"})})
                if name == "adversarial_verify"
                else "FileNotFoundError"
                if name == "cli_missing_input"
                else "okay"
            )
            log = log_dir / f"{name}.log"
            log.write_text(content)
            receipts.append(
                {
                    "name": name,
                    "log_path": str(log),
                    "log_sha256": cap.sha256_file(log),
                    "scope": spec.scope,
                    "duration_s": 0.01,  # type: ignore[attr-defined]
                    "exit_code": 1 if name == "cli_missing_input" else 0,
                    "passed": not (
                        name == "cli_missing_input"
                        or (mode == "required_failure" and name == "affected_pytest")
                        or (mode == "terminal_failure" and name == "strict_rows" and calls == 2)
                        or (mode == "recheck_failure" and name == "strict_rows" and calls == 3)
                    ),
                    "output_tail": content,
                    "command_argv": list(spec.argv),
                }
            )  # type: ignore[attr-defined]
        return receipts

    monkeypatch.setattr(cli, "run_commands", fake_run_commands)
    output = tmp_path / "terminal.json"
    if mode == "replay_failure":
        monkeypatch.setattr(cli, "cold_replay", lambda *_: ["changed"])
        with pytest.raises(ValueError, match="terminal_cold_replay_failed"):
            cli.run_current(root, "20260929", output, private)
        return
    if mode == "recheck_failure":
        with pytest.raises(ValueError, match="terminal_recheck_failed"):
            cli.run_current(root, "20260929", output, private)
        return
    result = cli.run_current(root, "20260929", output, private)
    assert output.is_file()
    assert result == json.loads(output.read_text())
    assert result["verdict_class"] == ("disqualified" if mode != "passing" else "blocked")
    assert result["capstone_execution_ready_score"] == int(mode == "passing")
    assert result["flagged_adversarial"] is (mode == "flagged")
    assert calls == (3 if mode in {"flagged", "terminal_failure"} else 2)
    assert cap.cold_replay(output, root) == []


def test_cli_default_dispatch_and_replay_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7890-TERMINAL: main routes a real default invocation."""
    output = tmp_path / "out.json"
    seen = []
    monkeypatch.setattr(cli, "run_current", lambda *args: seen.append(args))
    monkeypatch.setattr(sys, "argv", ["capstone", "--date", "20260929", "--output", str(output)])
    assert cli.main() == 0
    assert seen[0][2] == output
    monkeypatch.setattr(sys, "argv", ["capstone", "--cold-replay", str(tmp_path / "missing.json")])
    assert cli.main() == 1
