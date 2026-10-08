"""REQ-REPORT-8289 / REQ-VERIFY-8289: private evidence and fresh CLI controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v715_capstone as runner
from carnot.reporting import v715_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


def fixture(root):
    """Freeze real upstream bytes privately so fixtures never become live evidence."""
    root.mkdir()
    for name in [e.DESIGN, e.ACTIVE, e.STAGED, e.PROTOCOL, e.EXECUTION, *e.HISTORY]:
        source = e.ROOT / name
        if source.is_file():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    (root / "results").mkdir(exist_ok=True)
    return root


def cli(parent, *args):
    """Run direct imports outside the checkout to test real child statements."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd=parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_missing_outcomes(tmp_path):
    """SCENARIO-REPORT-8289-OUTCOMES: missing work never acquires a producer verdict."""
    root = fixture(tmp_path / "root")
    work = e.measure(root, tmp_path / "raw")
    result = e.reduce(work, [dict(passed=True, scope="owned")])
    assert [r["experiment_id"] for r in result["rows"]] == list(range(8276, 8290))
    assert result["completed_count"] == result["intended_count"] == 14
    assert result["missing_output_count"] == 13
    assert all(r["honest_verdict"] is None for r in result["rows"][:-1])
    assert result["H1"]["statistics"] is result["H2"]["statistics"] is None
    assert result["H1"]["intended_count"] == 128 and result["H2"]["intended_count"] == 96
    assert result["H1"]["alpha"] == result["H2"]["alpha"] == 0.025
    assert result["H2"]["retention_intended_count"] == 32
    assert result["verdict_class"] == "blocked" and result["capstone_execution_ready_score"] == 1
    assert result["science_ready_score"] == result["independent_generalization_score"] == 0
    assert len(result["board_obligations"]) == len(result["three_prd_gaps"]) == 3
    assert not result["polarfire_graduation"]["graduated"]
    failed = dict(
        passed=False,
        scope="owned",
        name="private_owned",
        stdout_path="private/check",
        stdout_sha256="sha256:private",
        expected_exit=0,
        actual_exit=1,
    )
    assert e.reduce(work, [failed])["verdict_class"] == "disqualified"
    assert e.reduce(work, [])["capstone_execution_ready_score"] == 0
    broken = deepcopy(work)
    broken["tasks"][0]["title"] = "rehashed change"
    with pytest.raises(ValueError, match="contract"):
        e.reduce(broken, [dict(passed=True)])


def test_private_cli(tmp_path):
    """SCENARIO-VERIFY-8289-CLI: actual terminal checks precede publication."""
    root = fixture(tmp_path / "root")
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--root", root, "--output", output, "--private-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    assert runner.replay(output)["passed"]
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    for key, observed in [
        ("completed_count", 13),
        ("experiment_id", 9),
        ("MODEL_SPECS", [{}]),
        ("paper_ready", not value["paper_ready"]),
        ("reproducibility_checksum", "rehashed"),
    ]:
        changed = dict(value, **{key: observed})
        path = tmp_path / (key + ".json")
        atomic_json(path, changed)
        assert cli(tmp_path, "--cold-replay", path).returncode == 1
    assert cli(tmp_path, "--date", "20260101").returncode == 2
    assert cli(tmp_path, "--private-fixture").returncode == 1
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1


def test_frozen_commands(tmp_path):
    """REQ-VERIFY-8289: CLI coverage and private E2E checks have frozen argv."""
    plan = runner.commands(tmp_path)
    assert any(p["name"] == "private_E2E021" for p in plan)
    assert any(p["name"] == "full_python_suite" for p in plan)
    assert "patch=subprocess" in (tmp_path / "coverage.ini").read_text()
    assert runner.terminal_plan(tmp_path / "candidate")[0]["argv"][-1].endswith("candidate")
    assert canonical_hash(e.MODEL_SPECS) == canonical_hash([])


def test_source_authentication(tmp_path):
    """REQ-REPORT-8289: real source scopes and conductor receipt identity stay distinct."""
    work = e.measure(e.ROOT, tmp_path / "raw")
    result = e.reduce(work, [dict(passed=True, scope="owned")])
    assert result["polarfire_graduation"]["graduated"]
    assert result["polarfire_graduation"]["scope"] == "board-local Linux CPU"
    assert result["actual_executed_task_count"] == 6 and result["pre_gate_count"] == 1
    assert result["missing_output_count"] == 7
    assert result["rows"][2]["honest_verdict"] is None
    assert result["rows"][10]["verdict_class"] == "disqualified"
    assert result["archive_lag"]["historical_executed_count"] == 7
    assert result["historical_CUDA_block"]["honest_verdict"].endswith("CUDA_runtime_available")
    plan = runner.audit_plan(work, tmp_path)
    assert len(plan) == 10 and all(p["scope"] == "upstream" for p in plan)
    changed = deepcopy(work)
    changed["inputs"][0]["reference"] = dict(
        changed["inputs"][0]["reference"], path="private_unbound"
    )
    with pytest.raises(ValueError, match="input_reference_drift"):
        e.reduce(changed, [dict(passed=True, scope="owned")])
    changed = deepcopy(work)
    next(r for r in changed["references"] if "expected_sha256" in r)["expected_sha256"] = (
        "sha256:changed"
    )
    assert not e.reduce(changed, [dict(passed=True, scope="owned")])["polarfire_graduation"][
        "graduated"
    ]
    changed = deepcopy(work)
    item = changed["inputs"][2]["reference"]
    gate = json.loads(Path(item["snapshot_path"]).read_bytes())
    gate["gates_evaluated"][0]["artifact_sha256"] = "sha256:changed"
    private_gate = tmp_path / "changed_gate.json"
    atomic_json(private_gate, gate)
    item["snapshot_path"] = str(private_gate)
    item["sha256"] = sha256_file(private_gate)
    assert e.reduce(changed, [dict(passed=True, scope="owned")])["rows"][2]["missing"]


def test_private_sidecar(tmp_path):
    """SCENARIO-REPORT-8289-OUTCOMES: wrong-path sidecars and missing bytes fail closed."""
    from carnot.reporting.primary_publication import publish_primary

    path = tmp_path / "results/experiment_8277_private.json"
    terminal = path.parent / "raw" / path.stem / "terminal_validation.json"
    value = dict(
        experiment_id=8277,
        task_id="exp8277-lease-backend-qualification",
        honest_verdict="complete_null_private",
        verdict_class="null",
        required_checks_passed=True,
        flagged_adversarial=False,
        rows=[],
        intended_count=0,
        completed_count=0,
        failed_count=0,
        excluded_count=0,
        censored_count=0,
        terminal_validation_sidecar_path=str(terminal),
    )
    report = publish_primary(path, value, lambda p: dict(passed=True))
    atomic_json(terminal, dict(publication=report))
    item = e.bind(path, tmp_path / "raw", [])
    row, failures, source = e.outcome(dict(id=value["task_id"]), 8277, item)
    assert row["producer_executed"] and source == value and not failures
    side = json.loads(Path(report["sidecar_path"]).read_bytes())
    side["primary_path"] = "foreign"
    atomic_json(Path(report["sidecar_path"]), side)
    assert e.outcome(dict(id=value["task_id"]), 8277, e.bind(path, tmp_path / "foreign", []))[0][
        "missing"
    ]
    path.unlink()
    assert not e.bind(path, tmp_path / "missing", [])["reference"]["exists"]


def test_primitive_and_stream_tampering(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8289-REPLAY: fresh reductions reject rehashed work and stream drift."""
    root = fixture(tmp_path / "root")
    output = tmp_path / (e.NAME + ".json")
    monkeypatch.setattr(runner, "commands", lambda p: [])
    assert runner.main(["--root", str(root), "--output", str(output)]) == 0
    assert (root / "docs/research-notes/v715-outcomes.md").is_file()
    original = json.loads(output.read_bytes())
    variants = []
    changed = deepcopy(original)
    changed["validation_receipts"][0]["stdout_sha256"] = "sha256:changed"
    variants.append((changed, "validation_stream_drift"))
    work = json.loads(Path(original["work_reference"]["path"]).read_bytes())
    work["references"] = work["references"][:-1]
    private_work = tmp_path / "changed_work.json"
    atomic_json(private_work, work)
    changed = dict(
        original, work_reference=dict(path=str(private_work), sha256=sha256_file(private_work))
    )
    variants.append((changed, "source_reference_drift"))
    for i, (value, error) in enumerate(variants):
        path = tmp_path / f"tampered{i}.json"
        atomic_json(path, value)
        with pytest.raises(ValueError, match=error):
            runner.replay(path)
    work = json.loads(Path(original["work_reference"]["path"]).read_bytes())
    work["tasks"][0]["title"] = "rehashed changed primitive"
    atomic_json(private_work, work)
    changed["work_reference"]["sha256"] = sha256_file(private_work)
    atomic_json(tmp_path / "primitive.json", changed)
    assert cli(tmp_path, "--cold-replay", tmp_path / "primitive.json").returncode == 1
    monkeypatch.setattr(
        runner, "publish_primary", lambda p, v, validator: validator(tmp_path / "wrong_operand")
    )
    assert runner.main(["--root", str(root), "--output", str(output), "--private-fixture"]) == 1


def test_authority_roster_failure(tmp_path):
    """REQ-REPORT-8289: full-task activation and exact roster fail independently."""
    root = fixture(tmp_path / "root")
    active = json.loads(json.dumps(__import__("yaml").safe_load((root / e.ACTIVE).read_bytes())))
    active["tasks"][0]["title"] = "changed active prompt"
    import yaml

    (root / e.ACTIVE).write_text(yaml.safe_dump(active))
    assert (
        e.reduce(e.measure(root, tmp_path / "raw"), [dict(passed=True, scope="owned")])[
            "capstone_execution_ready_score"
        ]
        == 0
    )
    design = root / e.DESIGN
    design.write_text(
        design.read_text().replace("exp8276-current-contract-readiness", "exp9999-wrong")
    )
    with pytest.raises(ValueError, match="fourteen"):
        e.measure(root, tmp_path / "wrong-roster")
