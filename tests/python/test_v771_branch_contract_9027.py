"""REQ-REPORT-9027-V771 / REQ-VERIFY-9027: private authority and receipt controls."""

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

from carnot.reporting import v771_branch_contract as c


def tree(root):
    """Copies keep negative controls away from historical and operator evidence."""
    names = [c.DESIGN, c.ACTIVE, c.STAGED, "research-complete.yaml", "ops/conductor-log.md"]
    names += [*c.e.PINS, c.e.LEDGER]
    names += [str(Path(r["path"]).relative_to(c.ROOT)) for r in c.e.operands(c.ROOT)]
    for name in dict.fromkeys(names):
        source, target = c.ROOT / name, root / name
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    return root


def validation(work):
    """Constructed receipts test refusal; production receipts come from real children."""
    raw = Path(work["raw"])
    receipts = []
    for name in c.CHECKS:
        log = raw / (name + ".log")
        log.write_text("private receipt control\n")
        receipts.append(
            dict(
                name=name,
                passed=True,
                normal_exit=True,
                timed_out=False,
                exit_code=0,
                started_monotonic_ns=work["end_ns"],
                scope_sha256=c.canonical_hash(work["validation_scope_manifest"]),
                log_path=str(log),
                log_sha256=c.reference(log)["sha256"],
            )
        )
    return dict(receipts=receipts, coverage=dict(complete=True, percent=100), global_test_debt={})


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """One immutable private snapshot serves the independent receipt and replay controls."""
    scratch = tmp_path_factory.mktemp("v771-custody")
    return c.freeze(tree(scratch / "root"), scratch / "raw", "20261112")


def test_full_authority_lifecycle(tmp_path):
    """SCENARIO-REPORT-9027-CUSTODY: active/staged order and full-object refusals."""
    root = tree(tmp_path / "root")
    actual = c.authority(root)
    assert actual["ready"] and actual["comparison_mode"] == "activated"
    assert [r["unit_id"].split("-")[0] for r in actual["rows"]] == [
        f"exp{i}" for i in range(9027, 9041)
    ]
    queue = yaml.safe_load((root / c.ACTIVE).read_bytes())
    (root / c.STAGED).write_text(yaml.safe_dump(queue))
    prior = deepcopy(queue)
    prior["milestone"] = "2026.11.770"
    (root / c.ACTIVE).write_text(yaml.safe_dump(prior))
    assert c.authority(root)["comparison_mode"] == "staged"
    assert c.authority(root)["ready"]
    (root / c.STAGED).unlink()
    assert not c.authority(root)["ready"]
    assert not c.authority(tmp_path / "missing")["ready"]
    (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
    (root / c.STAGED).write_text(yaml.safe_dump(prior))
    assert c.authority(root)["ready"]
    (root / c.STAGED).unlink()
    assert all(r["passed"] for r in c.mutations(root, tmp_path / "controls"))


def test_freeze_preserves_history_and_independent_science(tmp_path):
    """REQ-REPORT-9027-V771: authenticate original states without fitting or model calls."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261112")
    value = c.build(work, validation(work))
    assert (
        value["experiment_id"] == 9027
        and value["honest_verdict"] == "complete_null_v771_branch_contract"
    )
    assert value["contract_ready_score"] == value["frozen_graph_authority_ready_score"] == 1
    assert value["online_protocol_ready_score"] == 1 and value["arc_new_outcomes_ready_score"] == 0
    assert value["arc_frontier"]["baseline_experiment_id"] == 9013
    assert value["frozen_science"]["role_counts"] == dict(calibration=128, test=128, online=768)
    assert len(value["frozen_science"]["development_admission_rows"]) == 4
    assert value["archive_lag"]["plan_time_archive_end"] == "2026.11.769"
    assert value["archive_lag"]["observed_archive_end"] == "2026.11.770"
    assert not value["repository_health_passed"] and len(value["repository_health"]["rows"]) == 3
    history = {r["experiment_id"]: r for r in value["preserved_v770_outcomes"]["rows"]}
    assert history[9014]["verdict_class"] == "disqualified"
    assert any(not r["passed"] for r in history[9014]["validation_receipts"])
    assert any(r["observed"] == 429 for r in history[9015]["failed_gate_operands"])
    assert history[9026]["verdict_class"] == "blocked"
    assert not any(value["model_invocation_counts"].values()) and value["MODEL_SPECS"] == []
    independent = deepcopy(work)
    independent["science"]["frozen_graph_authority_ready_score"] = 0
    assert c.build(independent, validation(work))["contract_ready_score"] == 1
    with pytest.raises(FileExistsError):
        c.freeze(tmp_path, Path(work["raw"]), "20261112")
    absent = c.freeze(tmp_path / "absent", tmp_path / "blocked", "20261112")
    blocked = c.build(absent, validation(absent))
    assert blocked["verdict_class"] == "blocked"
    assert any(r["observed"] == "missing_file" for r in blocked["gate_check_summary"])


@pytest.mark.parametrize(
    "mode", ["missing", "failed", "timeout", "abnormal", "stale", "scope", "log", "coverage"]
)
def test_current_receipt_refusal(work, mode):
    """SCENARIO-REPORT-9027-CUSTODY: current owned failures zero every readiness field."""
    checked = validation(work)
    row = checked["receipts"][0]
    if mode == "missing":
        checked["receipts"].pop()
    if mode == "failed":
        row["passed"] = False
    if mode == "timeout":
        row["timed_out"] = True
    if mode == "abnormal":
        row["normal_exit"] = False
    if mode == "stale":
        row["started_monotonic_ns"] = 0
    if mode == "scope":
        row["scope_sha256"] = "forged"
    if mode == "log":
        row["log_sha256"] = "forged"
    if mode == "coverage":
        checked["coverage"]["complete"] = False
    value = c.build(work, checked)
    assert value["verdict_class"] == "disqualified"
    assert all(value[k] == 0 for k in c.READINESS)


def test_replay_refuses_rehashed_work(work, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-9027-REPLAY: neither repaired percentages nor hashes invent custody."""
    checked = validation(work)
    value = c.build(work, checked)
    candidate = tmp_path / (c.NAME + ".json")
    c.atomic_json(candidate, value)
    assert not c.replay(candidate)["passed"]
    monkeypatch.setattr(c, "verify_measured", lambda *args: True)
    assert c.replay(candidate)["passed"]
    for mode in ["summary", "history", "archive"]:
        forged = deepcopy(value)
        if mode == "summary":
            forged["contract_ready_score"] = 99
        else:
            changed = deepcopy(work)
            changed.pop("raw_reference")
            if mode == "history":
                changed["preserved_v770_outcomes"]["rows"][1]["verdict_class"] = "null"
            if mode == "archive":
                changed["archive_lag"]["observed_archive_end"] = "2026.11.999"
            path = tmp_path / (mode + ".json")
            c.atomic_json(path, changed)
            forged["raw_shard_hashes"][0] = c.reference(path)
            if mode == "history":
                forged["preserved_v770_outcomes"] = changed["preserved_v770_outcomes"]
            else:
                forged["archive_lag"] = changed["archive_lag"]
        forged["reproducibility_checksum"] = c.checksum(forged)
        c.atomic_json(candidate, forged)
        assert not c.replay(candidate)["passed"], mode
    assert not c.replay(tmp_path / "absent")["passed"]


def test_private_cli_and_runner(work, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-9027-REPLAY: actual CLI errors and private publication stay isolated."""
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
        print("before private CLI", args, flush=True)
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
            cwd=tmp_path,
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        print("after private CLI", child.returncode, flush=True)
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
        return validation(work)

    monkeypatch.setattr(c, "_validate", checked)
    monkeypatch.setattr(c, "_terminal", lambda candidate, raw: c.replay(candidate))
    monkeypatch.setattr(c, "verify_measured", lambda *args: True)
    monkeypatch.setattr(c, "freeze", lambda root, raw, run_date: work)
    output = tmp_path / (c.NAME + ".json")
    for args in [
        ["--date", "20261112", "--output", str(output), "--raw-dir", work["raw"]],
        ["--cold-replay", str(output)],
    ]:
        monkeypatch.setattr(sys, "argv", [c.CLI, *args])
        with pytest.raises(SystemExit) as error:
            runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
        assert error.value.code == 0


def test_bound_checks_and_optional_evidence(tmp_path, monkeypatch):
    """REQ-VERIFY-9027: existing check commands and optional absent evidence stay explicit."""
    monkeypatch.setattr(
        c.old.old, "run_check", lambda name, argv, *args: dict(name=name, argv=argv, passed=True)
    )
    for name in ["owned_tests", "private_e2e018_021", "spec_coverage", "ruff_check"]:
        receipt = c.run_check(name, ["python", "-m", "pytest"], tmp_path)
        assert receipt["scope_sha256"] == c.canonical_hash(c.scope_manifest())
    assert c.e.science(tmp_path)["frozen_graph_authority_ready_score"] == 0
    assert not c.e.frontier(tmp_path)["ready"]
    monkeypatch.setattr(c, "_validate", lambda *args: dict(complete=True))
    assert c.validate(tmp_path, None)["complete"]
    monkeypatch.setattr(c, "_terminal", lambda *args: dict(passed=True))
    assert c.terminal(tmp_path / "candidate", tmp_path)["passed"]
