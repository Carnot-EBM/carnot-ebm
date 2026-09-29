"""REQ-REPORT-7865-V683: the current contract binds every task before execution."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting.roadmap_contract import compare_contract, cold_replay


ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md"
STAGED = ROOT / "research-roadmap-next.yaml"
ACTIVE = ROOT / "research-roadmap.yaml"
CLI = ROOT / "scripts/experiments/experiment_7865_v683_contract_methods.py"
KW = {"milestone": "2026.09.683", "first_id": 7865, "count": 14}


def authorities():
    """Read three actual files so a missing staged plan fails visibly."""
    return (
        DESIGN.read_text(),
        yaml.safe_load(STAGED.read_text()),
        yaml.safe_load(ACTIVE.read_text()),
    )


def test_current_contract_binds_all_fourteen():
    """SCENARIO-REPORT-7865-V683-CONTRACT: every authority names the same work."""
    design, staged, active = authorities()
    result = compare_contract(design, staged, active, **KW)
    assert result["passed"], result["errors"]
    assert [row["unit_id"].split("-")[0] for row in result["rows"]] == [
        f"exp{i}" for i in range(7865, 7879)
    ]


@pytest.mark.parametrize(
    "mutation",
    ["count", "order", "id", "title", "phase", "path", "model", "class", "gate"],
)
def test_contract_mutations_fail(mutation):
    """SCENARIO-REPORT-7865-V683-CONTRACT: each contract dimension is falsifiable."""
    design, staged, active = authorities()
    staged = deepcopy(staged)
    tasks = staged["tasks"]
    if mutation == "count":
        tasks.pop()
    elif mutation == "order":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif mutation == "id":
        tasks[0]["id"] = "exp0000-wrong"
    elif mutation == "title":
        tasks[0]["title"] = "changed"
    elif mutation == "phase":
        tasks[0]["phase"] = 4
    elif mutation == "path":
        tasks[0]["deliverable"] = "results/wrong.json"
    elif mutation == "model":
        tasks[0]["MODEL_SPECS"] = ["wrong/model"]
    elif mutation == "class":
        tasks[0]["inference_substrate_class"] = "live_gpu"
    else:
        target = next(task for task in tasks if task["gated_on"])
        target["gated_on"][0]["artifact_field"] = "wrong_field"
    assert not compare_contract(design, staged, active, **KW)["passed"]


def test_private_cli_and_cold_replay(tmp_path):
    """SCENARIO-REPORT-7865-V683-VALIDATION: private bytes replay without science."""
    output, raw = tmp_path / "private.json", tmp_path / "rows.json"
    argv = [
        sys.executable,
        str(CLI),
        "--date",
        "20260929",
        "--output",
        str(output),
        "--raw",
        str(raw),
        "--no-validation",
    ]
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    value = json.loads(output.read_text())
    assert value["experiment_id"] == 7865
    assert value["contract_ready_score"] == 0
    assert value["verdict_class"] == "partial"
    assert value["acceptance_gate_results"]["decision_benefit"] is None
    assert len(value["contract_rows"]) == 14
    assert cold_replay(output, raw, experiment_id=7865)


def test_missing_staged_authority_blocks(tmp_path):
    """SCENARIO-REPORT-7865-V683-VALIDATION: external absence terminates honestly."""
    output = tmp_path / "blocked.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--output",
            str(output),
            "--staged",
            str(tmp_path / "missing.yaml"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert value["contract_ready_score"] == 0
    assert value["gate_check_summary"]


def load_cli():
    """Load the owned entrypoint so coverage includes actual route reduction."""
    spec = importlib.util.spec_from_file_location("v683_contract_cli", CLI)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_in_process_private_and_cold_routes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7865-V683-VALIDATION: owned branches retain exact rows."""
    cli = load_cli()
    monkeypatch.setenv("CARNOT_7865_TMP", str(tmp_path / "owned"))
    out, raw = tmp_path / "result.json", tmp_path / "rows.json"
    args = ["--output", str(out), "--raw", str(raw), "--no-validation"]
    assert cli.main(args) == 0
    assert cli.main(["--cold-replay", str(out), "--raw", str(raw)]) == 0
    value = json.loads(out.read_text())
    assert value["verdict_class"] == "partial"
    assert len(value["mutation_rows"]) == 9
    changed = json.loads(raw.read_text())
    changed["rows"][0]["absolute_metric"] = 0
    raw.write_text(json.dumps(changed))
    assert cli.main(["--cold-replay", str(out), "--raw", str(raw)]) == 1
    with pytest.raises(ValueError, match="checkpoint content differs"):
        cli.main(args)
    with pytest.raises(SystemExit):
        cli.main(["--no-validation"])
    with pytest.raises(SystemExit):
        cli.main(["--cold-replay", str(out)])


def test_in_process_block_and_validation_reduction(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7865-V683-VALIDATION: a missing source and a failed child differ."""
    cli = load_cli()
    monkeypatch.setenv("CARNOT_7865_TMP", str(tmp_path / "owned"))
    blocked_dir = tmp_path / "blocked"
    blocked = blocked_dir / "result.json"
    assert cli.main(["--output", str(blocked), "--staged", str(tmp_path / "absent.yaml")]) == 0
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    report = tmp_path / "clean-report.json"
    report.write_text(json.dumps({"flagged_count": 0, "reports": []}))

    def receipts(commands, scratch):
        return [
            {
                "name": c.name,
                "command_argv": list(c.argv),
                "passed": c.name != "ruff_check",
                "classification": "required",
                "exit_code": int(c.name == "ruff_check"),
                "timed_out": False,
                "log_path": str(report),
            }
            for c in commands
        ]

    monkeypatch.setattr(cli, "run_and_seal", receipts)
    out = tmp_path / "validated" / "result.json"
    assert cli.main(["--output", str(out), "--raw", str(tmp_path / "validated" / "rows.json")]) == 1
    value = json.loads(out.read_text())
    assert value["verdict_class"] == "disqualified"
    assert value["contract_ready_score"] == 0
    assert any(not row["passed"] for row in value["validation_receipts"])


def test_helper_failures_and_sealed_logs(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7865-V683-VALIDATION: immutable bytes and import paths are checked."""
    cli = load_cli()
    path = tmp_path / "frozen.json"
    cli.immutable_json(path, {"a": 1})
    cli.immutable_json(path, {"a": 1})
    with pytest.raises(ValueError, match="immutable manifest"):
        cli.immutable_json(path, {"a": 2})
    rows = cli.source_rows(
        [ROOT / "docs/research-notes/v682-method-map.md", tmp_path / "absent"], "20260929"
    )
    assert rows[0]["source_role"] == "historical_development"
    assert rows[1]["sha256"] is None
    assert cli.label(tmp_path) == str(tmp_path)
    assert cli.gate_operand_failures(
        [
            {
                "id": "exp7865-a",
                "prompt": "REQUIRED ARTIFACT FIELDS:\n- one: x",
                "gated_on": [{"upstream": "missing", "artifact_field": "two"}],
                "prior_failures": [{"experiment_id": "x"}],
            }
        ]
    )

    log = tmp_path / "child.log"
    log.write_text(
        json.dumps(
            {
                "resolved_imports": {
                    "carnot.reporting.roadmap_contract": str(
                        ROOT / "python/carnot/reporting/roadmap_contract.py"
                    )
                }
            }
        )
    )
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "run_commands",
        lambda *args, **kw: [
            {
                "name": "worktree_imports",
                "log_path": str(log),
                "log_sha256": cli.sha256_file(log),
                "resolved_imports": {
                    "carnot.reporting.roadmap_contract": str(
                        ROOT / "python/carnot/reporting/roadmap_contract.py"
                    )
                },
                "passed": True,
            }
        ],
    )
    sealed = cli.run_and_seal([], tmp_path / "scratch")
    assert (tmp_path / sealed[0]["log_path"]).is_file()
    assert sealed[0]["classification"] == "required"
    (tmp_path / sealed[0]["log_path"]).write_text("changed")
    with pytest.raises(ValueError, match="collision"):
        cli.run_and_seal([], tmp_path / "scratch")


def test_relative_child_log(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7865-V683-VALIDATION: relative child logs resolve under root."""
    cli = load_cli()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    log = tmp_path / "child.log"
    log.write_text("child exited")
    monkeypatch.setattr(
        cli,
        "run_commands",
        lambda *args, **kw: [
            {
                "name": "schema",
                "log_path": "child.log",
                "log_sha256": cli.sha256_file(log),
                "passed": True,
            }
        ],
    )
    assert cli.run_and_seal([], tmp_path / "scratch")[0]["passed"]


def test_wrong_milestones_block(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7865-V683-CONTRACT: current and staged dates are checked."""
    cli = load_cli()
    active = yaml.safe_load(ACTIVE.read_text())
    active["milestone"] = "wrong"
    bad_active = tmp_path / "active.yaml"
    bad_active.write_text(yaml.safe_dump(active))
    bad_staged = tmp_path / "staged.yaml"
    bad_staged.write_text(yaml.safe_dump(active))
    monkeypatch.setattr(cli, "ACTIVE", bad_active)
    output = tmp_path / "result.json"
    assert cli.main(["--output", str(output), "--staged", str(bad_staged)]) == 0
    assert {x["artifact_field"] for x in json.loads(output.read_text())["gate_check_summary"]} >= {
        "milestone"
    }


def test_snapshot_and_contract_rejection(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7865-V683-CONTRACT: altered authority cannot qualify."""
    cli = load_cli()
    monkeypatch.setenv("CARNOT_7865_TMP", str(tmp_path / "owned"))
    monkeypatch.setattr(cli, "verify_snapshots", lambda snapshots: False)
    with pytest.raises(ValueError, match="snapshot cold verification"):
        cli.main(["--output", str(tmp_path / "snapshot" / "result.json"), "--no-validation"])
    monkeypatch.setattr(cli, "verify_snapshots", lambda snapshots: True)
    original = cli.compare_contract

    def mismatch(*args, **kwargs):
        value = original(*args, **kwargs)
        value["passed"] = False
        return value

    monkeypatch.setattr(cli, "compare_contract", mismatch)
    output = tmp_path / "mismatch" / "result.json"
    assert cli.main(["--output", str(output), "--no-validation"]) == 1
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"


def test_frozen_source_drift_rejected(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7865-V683-VALIDATION: the frozen closure is binding."""
    cli = load_cli()
    monkeypatch.setenv("CARNOT_7865_TMP", str(tmp_path / "owned"))
    original = cli.immutable_json

    def poison(path, value):
        if path.name == "v683-validation-manifest-v2.json":
            first = next(iter(value["frozen_hashes"]))
            value["frozen_hashes"][first] = "sha256:wrong"
        original(path, value)

    monkeypatch.setattr(cli, "immutable_json", poison)
    with pytest.raises(ValueError, match="frozen source changed"):
        cli.main(["--output", str(tmp_path / "frozen" / "result.json")])


def test_terminal_adversarial_flag_disqualifies(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7865-V683-VALIDATION: a real report flag closes readiness."""
    cli = load_cli()
    monkeypatch.setenv("CARNOT_7865_TMP", str(tmp_path / "owned"))
    report = tmp_path / "adversarial.json"
    report.write_text(json.dumps({"flagged_count": 1, "reports": []}))

    def receipts(commands, scratch):
        return [
            {
                "name": c.name,
                "command_argv": list(c.argv),
                "passed": c.name != "adversarial_verify",
                "classification": "required",
                "exit_code": int(c.name == "adversarial_verify"),
                "timed_out": False,
                "log_path": str(report),
                "log_sha256": cli.sha256_file(report),
            }
            for c in commands
        ]

    monkeypatch.setattr(cli, "run_and_seal", receipts)
    out = tmp_path / "result.json"
    assert cli.main(["--output", str(out), "--raw", str(tmp_path / "rows.json")]) == 1
    value = json.loads(out.read_text())
    assert value["flagged_adversarial"] is True
    assert value["verdict_class"] == "disqualified"
    assert value["contract_ready_score"] == 0
    assert {row["name"] for row in value["validation_receipts"]} >= {
        "adversarial_verify",
        "verdict_row_consistency",
    }


def test_terminal_report_parse_and_drift(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7865-V683-VALIDATION: changed terminal evidence cannot publish."""
    cli = load_cli()
    monkeypatch.setenv("CARNOT_7865_TMP", str(tmp_path / "owned"))
    monkeypatch.setattr(cli, "ROOT", ROOT)
    report = tmp_path / "report.json"
    report.write_text("not json")
    assert cli.report_flagged([{"name": "adversarial_verify", "log_path": str(report)}])
    report.write_text(json.dumps({"flagged_count": 0}))
    assert not cli.report_flagged([{"name": "adversarial_verify", "log_path": str(report)}])
    relative = Path("../") / report.relative_to(tmp_path.parent)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    assert not cli.report_flagged([{"name": "adversarial_verify", "log_path": str(relative)}])
    monkeypatch.setattr(cli, "ROOT", ROOT)
    count = 0

    def receipts(commands, scratch):
        nonlocal count
        count += 1
        return [
            {
                "name": c.name,
                "command_argv": list(c.argv),
                "passed": not (count == 3 and c.name == "verdict_row_consistency"),
                "classification": "required",
                "exit_code": 0,
                "timed_out": False,
                "log_path": str(report),
            }
            for c in commands
        ]

    monkeypatch.setattr(cli, "run_and_seal", receipts)
    with pytest.raises(ValueError, match="terminal candidate verification changed"):
        cli.main(
            [
                "--output",
                str(tmp_path / "drift" / "result.json"),
                "--raw",
                str(tmp_path / "drift" / "rows.json"),
            ]
        )
