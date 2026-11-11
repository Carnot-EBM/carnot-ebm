"""REQ-REPORT-8985 / SCENARIO-REPORT-8985-V768: private authority and custody checks."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

from coverage import Coverage, CoverageData
import pytest
import yaml

from carnot.reporting import v768_branch_contract as c


def tree(root):
    """Copy authority into private directories without activating a queue."""
    for name in [c.DESIGN, c.ACTIVE]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((c.ROOT / name).read_bytes())
    return root


def validation(passed=True):
    """Synthetic checks exercise refusal paths; production uses measured receipts."""
    return dict(
        receipts=[dict(name="fixture", passed=passed)],
        coverage=dict(complete=passed, percent=100 if passed else 0),
        global_test_debt=dict(passed=False, scope="global"),
    )


def test_exact_authority(tmp_path):
    """REQ-REPORT-8985: bind every object and reject independent mutations."""
    root = tree(tmp_path / "root")
    result = c.authority(root)
    assert result["ready"] and len(result["tasks"]) == len(result["rows"]) == 14
    assert c.canonical_hash(result["tasks"]) == c.PIN
    assert all(r["passed"] for r in c.mutations(root, tmp_path / "controls"))
    queue = yaml.safe_load((root / c.ACTIVE).read_bytes())
    (root / c.STAGED).write_text(yaml.safe_dump(queue))
    queue["milestone"] = "2026.11.767"
    (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
    assert c.authority(root)["comparison_mode"] == "staged"
    (root / c.STAGED).unlink()
    assert not c.authority(root)["ready"]
    assert not c.authority(tmp_path / "missing")["ready"]


def test_history(tmp_path):
    """REQ-REPORT-8985: retain five primaries, nine skips, and actual absence."""
    history = c.evidence.history(c.ROOT)
    assert history["authenticated"] and len(history["rows"]) == 14
    assert sum(r["kind"] == "primary" for r in history["rows"]) == 5
    assert sum(r["kind"] == "structured_skip" for r in history["rows"]) == 3
    assert sum(r["kind"] == "missing_primary" for r in history["rows"]) == 6
    assert history["capstone_verdict"] == "complete_blocked_external_science_and_source_access"
    assert not c.evidence.history(tmp_path)["authenticated"]


@pytest.mark.parametrize(
    "mutation", ["design", "queue", "identity", "log", "gate", "work", "archive"]
)
def test_historical_mutations(tmp_path, monkeypatch, mutation):
    """REQ-REPORT-8985: independently pinned malformed scheduler custody fails closed."""
    e = c.evidence
    pins = dict(e.PINS)
    monkeypatch.setattr(e, "PINS", pins)
    for name in pins:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((c.ROOT / name).read_bytes())
    name = {
        "design": e.ARCHIVE,
        "queue": e.QUEUE,
        "identity": "research-complete.yaml",
        "log": "ops/conductor-log.md",
        "gate": "results/experiment_8974_feature_online_methods.json",
        "work": e.CAPSTONE,
        "archive": e.CAPSTONE,
    }[mutation]
    path = tmp_path / name
    if mutation == "design":
        path.write_text(
            path.read_text().replace(
                "Bind fourteen tasks and establish the V767 custody boundary", "forged"
            )
        )
        text = path.read_text()
        table_line = next(line for line in text.splitlines() if "| 1 |" in line)
        path.write_text(text.replace(table_line, ""))
    elif mutation in {"queue", "identity"}:
        value = yaml.load(path.read_bytes(), Loader=yaml.CSafeLoader)
        if mutation == "queue":
            value["milestone"] = "forged"
        else:
            next(m for m in value["milestones"] if m["id"] == "2026.11.767")["tasks"][0]["id"] = (
                "forged"
            )
        path.write_text(yaml.dump(value, Dumper=yaml.CSafeDumper))
    elif mutation == "log":
        path.write_text("")
    else:
        value = json.loads(path.read_bytes())
        if mutation == "gate":
            value["gate_check_summary"] = "forged"
        elif mutation == "work":
            value["work_reference"]["sha256"] = "sha256:forged"
        else:
            work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
            next(
                r
                for r in work["references"]
                if r.get("original_path", "").endswith("/research-complete.yaml")
            )["sha256"] = "sha256:forged"
            c.atomic_json(tmp_path / "work.json", work)
            value["work_reference"] = c.reference(tmp_path / "work.json")
        c.atomic_json(path, value)
    pins[name] = c.reference(path)["sha256"]
    result = e.history(tmp_path)
    assert not result["authenticated"] and result["failures"][0]["check"] == "history_schema"


def test_arc_frontier_and_support(tmp_path, monkeypatch):
    """REQ-REPORT-8985: a changed retained frontier cannot grant ARC support."""
    e = c.evidence
    path = tmp_path / "results/experiment_8971_v767_branch_contract.json"
    value = json.loads((c.ROOT / "results/experiment_8971_v767_branch_contract.json").read_bytes())
    value["arc_frontier"]["frontier_after"]["ledger_receipt_identities"] = ["forged"]
    ledger_name = str(tmp_path.joinpath(*e.previous_arc.arc.DEFAULT_LEDGER_PARTS))
    next(
        r
        for r in value["raw_shard_hashes"]
        if r.get("source", {})
        .get("path", "")
        .endswith("/ops/arc_supervisor_refinement_ledger.json")
    )["source"]["path"] = ledger_name
    c.atomic_json(path, value)
    monkeypatch.setattr(e, "ROOT", tmp_path)
    assert not e.arc_evidence()["ready"]
    rows = [
        dict(
            invocation_id="new",
            game=f"g{i % 3}",
            seed=i,
            arm="drop_goal_bias" if i % 2 else "allow_reinduction",
            status="completed",
            selected=True,
            applied=True,
        )
        for i in range(30)
    ]
    support = e.previous_arc.arc_support
    assert support(rows)["ready"]
    for changed in [
        rows[:29],
        rows[:1] * 30,
        [dict(r, status="censored") for r in rows],
        [dict(r, game="one") for r in rows],
        [dict(r, arm="drop_goal_bias") for r in rows],
    ]:
        assert not support(changed)["ready"]


def test_build_replay_and_refusal(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8985-V768: empty ARC is optional; owned failures disqualify."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261111")
    value = c.build(work, validation())
    assert value["contract_ready_score"] == 1 and value["verdict_class"] == "null"
    assert value["completed_count"] == value["intended_count"] == 14
    assert value["arc_new_outcomes_ready_score"] == 0
    assert value["arc_frontier"]["cheap_skip_count"] == 1
    assert value["MODEL_SPECS"] == value["trained_head_specs"] == []
    assert not any(value["model_invocation_counts"].values())
    assert value["independent_count"] == value["independent_generalization_score"] == 0
    assert all(r["required_field_declared"] for r in value["gate_edge_rows"])
    assert len(value["historical_slot_rows"]) == 14
    candidate = tmp_path / (c.NAME + ".json")
    c.atomic_json(candidate, value)
    assert not c.replay(candidate)["passed"]
    monkeypatch.setattr(c.prior, "verify_measured", lambda *args: True)
    assert c.replay(candidate)["passed"]
    for key in ["contract_ready_score", "completed_count", "full_task_digest"]:
        forged = deepcopy(value)
        forged[key] = "forged"
        forged["reproducibility_checksum"] = c.checksum(forged)
        c.atomic_json(candidate, forged)
        assert not c.replay(candidate)["passed"]
    assert c.build(work, validation(False))["verdict_class"] == "disqualified"
    changed_argv = validation()
    changed_argv["receipts"][0]["argv"] = ["unowned"]
    assert c.build(work, changed_argv)["verdict_class"] == "disqualified"
    absent = c.freeze(tmp_path / "absent", tmp_path / "missing-raw", "20261111")
    blocked = c.build(absent, validation())
    assert blocked["verdict_class"] == "blocked" and blocked["contract_ready_score"] == 0
    assert any(r["observed"] == "missing_file" for r in blocked["gate_check_summary"])
    assert not c.replay(tmp_path / "absent.json")["passed"]
    with pytest.raises(FileExistsError):
        c.freeze(tmp_path, Path(work["raw"]), "20261111")


def test_real_child_and_cli(tmp_path, monkeypatch):
    """E2E-018/021 / REQ-REPORT-8985: outside-checkout CLI and measured child lines."""
    from carnot.reporting import v755_branch_runner as runner

    config = tmp_path / "coverage.ini"
    config.write_text(
        "[run]\ninclude =\n"
        + "".join(f"    {c.ROOT / p}\n" for p in c.OWNED)
        + "[report]\nexclude_lines =\n"
    )
    for args, expected in [
        (["--date", "bad"], 2),
        (["--cold-replay", str(tmp_path / "absent")], 1),
    ]:
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--rcfile",
                str(config),
                "--data-file",
                str(tmp_path / ".coverage"),
                "--parallel-mode",
                str(c.ROOT / c.CLI),
                *args,
            ],
            cwd="/var/tmp",
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert child.returncode == expected, child.stdout + child.stderr
    tracer = Coverage.current()
    if tracer:
        for path in tmp_path.glob(".coverage.*"):
            data = CoverageData(basename=str(path))
            data.read()
            tracer.get_data().update(data)

    def checked(raw, tracer):
        tracer.stop()
        tracer.save()
        return validation()

    monkeypatch.setattr(runner, "validate", checked)
    monkeypatch.setattr(runner, "terminal", lambda candidate, raw: c.replay(candidate))
    monkeypatch.setattr(c.prior, "verify_measured", lambda *args: True)
    output = tmp_path / (c.NAME + ".json")
    for args in [["--date", "20261111", "--output", str(output)], ["--cold-replay", str(output)]]:
        monkeypatch.setattr(sys, "argv", [c.CLI, *args])
        with pytest.raises(SystemExit) as error:
            runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
        assert error.value.code == 0
