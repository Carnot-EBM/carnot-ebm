"""REQ-REPORT-8971 / SCENARIO-REPORT-8971-V767: custody is separate from science."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest
import yaml

from carnot.reporting import v767_branch_contract as c


def tree(root):
    """Private authorities prevent tests from activating a research queue."""
    for name in [c.DESIGN, c.ACTIVE]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((c.ROOT / name).read_bytes())
    return root


def validation(passed=True):
    """Synthetic checks exercise reduction but never qualify a production run."""
    return dict(
        receipts=[dict(name="fixture", passed=passed)],
        coverage=dict(complete=passed, percent=100 if passed else 0),
        global_test_debt=dict(passed=False, scope="global"),
    )


def test_authority(tmp_path):
    """REQ-REPORT-8971: pin every full object and reject coordinated authority edits."""
    root = tree(tmp_path / "root")
    result = c.authority(root)
    assert result["ready"] and len(result["tasks"]) == len(result["rows"]) == 14
    assert c.canonical_hash(result["tasks"]) == c.PIN
    controls = c.mutations(root, tmp_path / "controls")
    assert {r["mutation"] for r in controls} == {
        "missing",
        "title",
        "order",
        "prompt",
        "gate",
        "path",
        "rehash",
    }
    assert all(r["passed"] for r in controls)
    queue = yaml.load((root / c.ACTIVE).read_bytes(), Loader=yaml.CSafeLoader)
    (root / c.STAGED).write_text(yaml.dump(queue, Dumper=yaml.CSafeDumper))
    queue["milestone"] = "2026.11.766"
    (root / c.ACTIVE).write_text(yaml.dump(queue, Dumper=yaml.CSafeDumper))
    assert c.authority(root)["comparison_mode"] == "staged"
    (root / c.STAGED).unlink()
    assert not c.authority(root)["ready"]
    assert not c.authority(tmp_path / "missing")["ready"]


def test_history(tmp_path):
    """REQ-REPORT-8971: original bytes and scheduler receipts retain all fourteen slots."""
    history = c.evidence.history(c.ROOT)
    assert history["authenticated"] and len(history["rows"]) == 14
    assert history["rows"][-2]["kind"] == "structured_skip"
    assert history["rows"][-1]["scheduler_outcome"] == "SKIPPED (3-fail)"
    assert history["rows"][-1]["primary_reference"]["sha256"] is None
    assert history["rows"][-1]["kind"] == "missing_primary"
    assert history["quarantine"]["flagged_adversarial"] is True
    assert not c.evidence.history(tmp_path)["authenticated"]


def test_build_and_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8971-V767: optional absence cannot block administrative readiness."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261110")
    value = c.build(work, validation())
    assert value["contract_ready_score"] == 1 and value["arc_new_outcomes_ready_score"] == 0
    assert value["verdict_class"] == "null" and value["honest_verdict"].startswith("complete_null_")
    assert value["intended_count"] == value["completed_count"] == 14
    assert value["independent_count"] == value["numerical_training_calls"] == 0
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert value["arc_frontier"]["cheap_skip_count"] == 1
    assert all(
        r["future_dependency"] and r["required_field_declared"] for r in value["gate_edge_rows"]
    )
    assert len(value["exclusion_scope_rows"]) == 14
    assert not value["flagged_adversarial"]
    candidate = tmp_path / (c.NAME + ".json")
    c.atomic_json(candidate, value)
    assert not c.replay(candidate)["passed"]
    monkeypatch.setattr(c, "verify_measured", lambda *args: True)
    assert c.replay(candidate)["passed"]
    for field in ["contract_ready_score", "arc_new_outcomes_ready_score", "completed_count"]:
        changed = deepcopy(value)
        changed[field] = 1 - changed[field]
        changed["reproducibility_checksum"] = c.checksum(changed)
        c.atomic_json(candidate, changed)
        assert not c.replay(candidate)["passed"]
    assert c.build(work, validation(False))["verdict_class"] == "disqualified"
    changed = deepcopy(work)
    changed["preconditions"]["free_bytes"] = 0
    blocked = c.build(changed, validation())
    assert blocked["verdict_class"] == "blocked" and blocked["contract_ready_score"] == 0
    assert any(
        r["artifact_field"] == "free_bytes" and r["observed"] == 0
        for r in blocked["gate_check_summary"]
    )
    with pytest.raises(FileExistsError):
        c.freeze(tmp_path, Path(work["raw"]), "20261110")
    assert not c.replay(tmp_path / "absent")["passed"]


def test_private_cli(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8971-V767: CLI publication, cold replay and rehashed tamper refusal."""
    from carnot.reporting import v755_branch_runner as runner

    def checked(raw, tracer):
        tracer.stop()
        tracer.save()
        return validation()

    monkeypatch.setattr(runner, "validate", checked)
    monkeypatch.setattr(runner, "terminal", lambda candidate, raw: c.replay(candidate))
    monkeypatch.setattr(c, "verify_measured", lambda *args: True)
    output = tmp_path / (c.NAME + ".json")
    for argv, expected in [
        (["--date", "bad"], 2),
        (
            ["--date", "20261110", "--root", str(tree(tmp_path / "root")), "--output", str(output)],
            0,
        ),
        (["--cold-replay", str(output)], 0),
    ]:
        monkeypatch.setattr(sys, "argv", [c.CLI, *argv])
        with pytest.raises(SystemExit) as error:
            runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
        assert error.value.code == expected
    value = json.loads(output.read_bytes())
    value["full_task_objects"][0]["title"] = "forged"
    value["reproducibility_checksum"] = c.checksum(value)
    c.atomic_json(output, value)
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert error.value.code == 1


def test_arc_denominators():
    """REQ-REPORT-8971: repeated views and censored episodes cannot manufacture support."""
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
    support = c.evidence.arc.arc_support
    assert support(rows)["ready"]
    for changed in [
        rows[:29],
        rows[:1] * 30,
        [dict(r, status="censored") for r in rows],
        [dict(r, game="one") for r in rows],
        [dict(r, arm="drop_goal_bias") for r in rows],
    ]:
        assert not support(changed)["ready"]


def test_real_cli_child(tmp_path):
    """SCENARIO-REPORT-8971-V767: direct child statements are measured in real interpreters."""
    import os
    import subprocess
    from coverage import Coverage, CoverageData

    config = tmp_path / "coverage.ini"
    config.write_text(
        "[run]\ninclude =\n"
        + "".join(f"    {c.ROOT / p}\n" for p in c.OWNED)
        + "context = v767_actual_cli_child\n[report]\nexclude_lines =\n"
    )
    tracer = Coverage.current()
    data_file = str(tmp_path / ".coverage")
    for args, expected in [
        (["--date", "bad"], 2),
        (["--cold-replay", str(tmp_path / "absent")], 1),
    ]:
        child = subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "coverage",
                "run",
                "--rcfile",
                str(config),
                "--data-file",
                data_file,
                "--parallel-mode",
                str(c.ROOT / c.CLI),
                *args,
            ],
            cwd="/var/tmp",
            env=dict(os.environ, PYTHONPATH=f"{c.ROOT / 'python'}:{c.ROOT}"),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert child.returncode == expected, child.stdout + child.stderr
    if tracer:
        for path in tmp_path.glob(".coverage.*"):
            child_data = CoverageData(basename=str(path))
            child_data.read()
            tracer.get_data().update(child_data)


def test_primitive_tamper_and_blocking(tmp_path, monkeypatch):
    """REQ-REPORT-8971: rehashing primitives, controls and claims cannot invent qualification."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261110")
    monkeypatch.setattr(c, "verify_measured", lambda *args: True)
    candidate = tmp_path / (c.NAME + ".json")
    original = c.build(work, validation())
    for section in ["contract", "history", "arc", "exclusion_scope_rows"]:
        forged = deepcopy(work)
        if section == "contract":
            forged[section]["ready"] = False
        elif section == "history":
            forged[section]["quarantine"]["flagged_adversarial"] = False
        elif section == "arc":
            forged[section]["ready"] = True
        else:
            forged[section][0]["passed"] = False
        c.atomic_json(
            Path(work["raw_reference"]["path"]),
            {k: v for k, v in forged.items() if k != "raw_reference"},
        )
        forged["raw_reference"] = c.reference(Path(work["raw_reference"]["path"]))
        c.atomic_json(candidate, c.build(forged, validation()))
        assert not c.replay(candidate)["passed"], section
    c.atomic_json(
        Path(work["raw_reference"]["path"]), {k: v for k, v in work.items() if k != "raw_reference"}
    )
    for field in ["raw_shard_hashes", "code_config_hashes"]:
        changed = deepcopy(original)
        if field == "raw_shard_hashes":
            changed[field][0]["sha256"] = "sha256:forged"
        else:
            changed[field][0]["source"]["sha256"] = "sha256:forged"
        changed["reproducibility_checksum"] = c.checksum(changed)
        c.atomic_json(candidate, changed)
        assert not c.replay(candidate)["passed"]
    control = Path(work["raw"]) / "controls" / "missing" / c.ACTIVE
    control.write_bytes((Path(work["raw"]) / "authority" / c.ACTIVE).read_bytes())
    forged = deepcopy(work)
    forged["raw_operands"] = [c.reference(Path(r["path"])) for r in forged["raw_operands"]]
    c.atomic_json(
        Path(work["raw_reference"]["path"]),
        {k: v for k, v in forged.items() if k != "raw_reference"},
    )
    forged["raw_reference"] = c.reference(Path(work["raw_reference"]["path"]))
    c.atomic_json(candidate, c.build(forged, validation()))
    assert not c.replay(candidate)["passed"]
    absent = c.freeze(tmp_path / "absent", tmp_path / "missing-input-raw", "20261110")
    blocked = c.build(absent, validation())
    assert blocked["verdict_class"] == "blocked" and blocked["contract_ready_score"] == 0
    assert blocked["completed_count"] == 0 and blocked["censored_count"] == 14
    assert any(r["observed"] == "missing_file" for r in blocked["gate_check_summary"])


def test_scheduler_schema_refusal(tmp_path, monkeypatch):
    """REQ-REPORT-8971: malformed or mismatched scheduler evidence is an exact failure."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261110")
    root = Path(work["raw"]) / "authority"
    e = c.evidence
    pins = deepcopy(e.PINS)
    monkeypatch.setattr(e, "PINS", pins)
    path = root / e.RECORDS[0]
    archive = yaml.load(path.read_bytes(), Loader=yaml.CSafeLoader)
    next(m for m in archive["milestones"] if m["id"] == "2026.11.766")["tasks"][0]["id"] = "forged"
    path.write_text(yaml.dump(archive, Dumper=yaml.CSafeDumper))
    pins[e.RECORDS[0]] = c.reference(path)["sha256"]
    result = e.history(root)
    assert not result["authenticated"] and result["failures"][0]["check"] == "scheduler_identity"
    path.write_text("{}")
    pins[e.RECORDS[0]] = c.reference(path)["sha256"]
    result = e.history(root)
    assert result["failures"][0]["observed"] == "missing_field"
    assert result["failures"][0]["artifact_field"] == "milestones"

    def missing_record(root):
        raise FileNotFoundError(2, "missing", str(root / "record"))

    monkeypatch.setattr(e, "_history", missing_record)
    assert e.history(root)["failures"][0]["path"] == str(root / "record")
