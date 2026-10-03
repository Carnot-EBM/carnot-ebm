"""REQ-REPORT-8070: custody must preserve independent branches and negative history."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v699_contract_custody as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def authorities(tmp_path):
    design = tmp_path / "design.md"
    active = tmp_path / "active.yaml"
    design.write_bytes((e.ROOT / e.DESIGN).read_bytes())
    active.write_bytes((e.ROOT / "research-roadmap.yaml").read_bytes())
    return design, tmp_path / "consumed.yaml", active


def test_complete_authority_and_mutations(tmp_path):
    """SCENARIO-REPORT-8070-AUTHORITY: full task bytes and visible rows both bind."""
    design, staged, active = authorities(tmp_path)
    value = e.assess(design, staged, active, tmp_path / "snapshots")
    assert value["activated"] and len(value["contract_rows"]) == 13
    assert not value["authority_snapshots"]["staged"]["exists"]
    original = yaml.safe_load(active.read_text())
    for field in ("prompt", "max_turns", "title"):
        bad = deepcopy(original)
        bad["tasks"][0][field] = "changed"
        active.write_text(yaml.safe_dump(bad))
        assert not e.assess(design, staged, active, tmp_path / "snapshots")["activated"]
    active.write_text(yaml.safe_dump(original))
    text = design.read_text()
    design.write_text(text.replace("Canonical complete-task SHA256:", "removed:"))
    assert not e.assess(design, staged, active, tmp_path / "snapshots")["activated"]
    design.write_text(text.replace("| 1 |", "| 99 |", 1))
    assert not e.assess(design, staged, active, tmp_path / "snapshots")["activated"]
    assert not e.assess(tmp_path / "absent", staged, active, tmp_path / "snapshots")["activated"]


def test_historical_dispositions_and_input_authentication(tmp_path):
    """SCENARIO-REPORT-8070-BRANCHES: actual skip and null/censored outcomes survive."""
    history = e.historical(e.ROOT, tmp_path / "history")
    assert len(history["rows"]) == 13
    indexed = {r["task_id"].split("-")[0]: r for r in history["rows"]}
    assert indexed["exp8060"]["evidence_kind"] == "actual_skip_receipt"
    assert indexed["exp8061"]["primary_present"] is False
    assert indexed["exp8062"]["primary_present"] is False
    assert indexed["exp8064"]["verdict_class"] == "null"
    assert indexed["exp8066"]["verdict_class"] == "disqualified"
    inputs = e.qualify(e.ROOT, tmp_path / "inputs")
    assert inputs["source_ready"] and inputs["learning_ready"], inputs["failures"]
    assert len(inputs["feature_rows"]) == 512
    assert all(r["capture_producer"] != 8059 for r in inputs["feature_rows"])
    assert inputs["head_sha256"] == e.canonical_hash(inputs["qualified_head"])
    rows = e.gate_matrix(tmp_path / "gates")
    assert len(rows) == 72 and all(r["matched"] for r in rows)


def test_binder_missing_changed_and_immutable(tmp_path):
    """SCENARIO-REPORT-8070-BRANCHES: exact operands distinguish missing from changed."""
    binder = e.Binder(tmp_path / "raw")
    missing = tmp_path / "missing"
    with pytest.raises(e.InputFailure):
        binder.bind(missing)
    path = tmp_path / "value.json"
    atomic_json(path, {"value": 1})
    with pytest.raises(e.InputFailure):
        binder.bind(path, "sha256:wrong")
    ref = binder.bind(path)
    assert ref == binder.bind(path)
    Path(ref["snapshot_path"]).write_text("corrupted")
    with pytest.raises(ValueError, match="immutable"):
        binder.bind(path)
    with pytest.raises(e.InputFailure):
        binder.require(path, "ready", 1, 0)


def test_independent_branch_corruption(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8070-BRANCHES: a branch-local failure cannot erase its sibling."""
    original = e.Binder.bind
    for role, expected in [("fit", (False, True)), ("stream", (True, False))]:

        def corrupt(self, path, digest=None):
            if str(path).endswith(f"/roles/{role}.json"):
                self.require(path, "forced_role_hash", "original", "changed")
            return original(self, path, digest)

        monkeypatch.setattr(e.Binder, "bind", corrupt)
        work = e.qualify(e.ROOT, tmp_path / role)
        assert (work["source_ready"], work["learning_ready"]) == expected
    monkeypatch.setattr(e.Binder, "bind", original)
    assert not e.qualify(tmp_path / "absent", tmp_path / "empty")["source_ready"]


def invoke(tmp_path, *args):
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    prefix = [sys.executable]
    if env.get("CARNOT_8070_COVERAGE_CONFIG"):
        prefix += ["-m", "coverage", "run", "--rcfile=" + env["CARNOT_8070_COVERAGE_CONFIG"]]
    return subprocess.run(
        [*prefix, str(e.ROOT / e.CLI), *map(str, args)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_real_cli_success_blocked_mutation_replay(tmp_path):
    """SCENARIO-REPORT-8070-CLI: real external CLI publishes terminal routes."""
    design, staged, active = authorities(tmp_path)
    output = tmp_path / "results" / (e.NAME + ".json")
    args = ["--fixture-output", output, "--design", design, "--staged", staged, "--active", active]
    result = invoke(tmp_path, *args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(output.read_text())["contract_ready_score"] == 1
    assert invoke(tmp_path, "--cold-replay", output).returncode == 0
    assert invoke(tmp_path, *args, "--mutate").returncode == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    assert invoke(tmp_path, *args, "--root", tmp_path / "absent").returncode == 0
    assert json.loads(output.read_text())["verdict_class"] == "blocked"
    assert invoke(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_text())
    value["cached_source_inputs_ready_score"] = 1
    atomic_json(output, value)
    assert invoke(tmp_path, "--cold-replay", output).returncode == 1
    assert invoke(tmp_path, "--cold-replay", tmp_path / "missing").returncode == 1
    assert invoke(tmp_path, "--date", "wrong").returncode == 2


def test_terminal_and_head_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8070-BRANCHES: bad terminals and heads fail exact operands."""
    binder = e.Binder(tmp_path / "raw")
    path = tmp_path / "primary.json"
    atomic_json(path, {})
    with pytest.raises(e.InputFailure, match="clean_terminal"):
        binder.terminal(path)
    original = e.Binder.read

    def bad_head(self, path, digest=None):
        value = original(self, path, digest)
        if path.name == "experiment_8058_v698_sealed_evidence_methods.json":
            value["qualified_head_sha256"] = "sha256:changed"
        return value

    monkeypatch.setattr(e.Binder, "read", bad_head)
    work = e.qualify(e.ROOT, tmp_path / "head")
    assert work["source_ready"] and not work["learning_ready"]

    def unreadable(self, path, digest=None):
        raise KeyError("absent_field")

    monkeypatch.setattr(e.Binder, "read", unreadable)
    assert e.qualify(e.ROOT, tmp_path / "malformed")["failures"]


def test_embedded_digest_is_independent(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8070-AUTHORITY: hashing embedded tasks is a separate check."""
    design, staged, active = authorities(tmp_path)
    original = e.authority.assess_authorities

    def changed(*args, **kwargs):
        value = original(*args, **kwargs)
        value["canonical_tasks_sha256"] = "changed"
        return value

    monkeypatch.setattr(e.authority, "assess_authorities", changed)
    assert not e.assess(design, staged, active, tmp_path / "snapshots")["activated"]


def test_production_orchestration_private_logs_and_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8070-CLI: production checks and immutable replay use exact logs."""
    from carnot.reporting import v699_custody_execution as run

    design, staged, active = authorities(tmp_path)
    output = tmp_path / "results" / (e.NAME + ".json")

    def checked(root, spec, private, durable, **kwargs):
        log = durable / (spec["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("Private orchestration control; no real child claim.\n")
        if spec["name"] == "cold_replay":
            assert run.replay(Path(spec["argv"][-1]))
        if spec["name"] == "coverage_json":
            atomic_json(
                private / "coverage.json",
                {
                    "files": {
                        p: {
                            "summary": {"num_statements": 1, "covered_lines": 1},
                            "missing_lines": [],
                        }
                        for p in run.OWNED
                    }
                },
            )
        return dict(
            name=spec["name"],
            passed=True,
            log_path=str(log),
            log_sha256=sha256_file(log),
            exit_code=0,
            expected_exit=0,
            duration_s=0,
            argv=spec["argv"],
            test_control=True,
        )

    monkeypatch.setattr(run, "run_check", checked)
    args = [
        "--output",
        str(output),
        "--design",
        str(design),
        "--staged",
        str(staged),
        "--active",
        str(active),
    ]
    assert run.main(args) == 0
    assert run.replay(output)
    value = json.loads(output.read_text())
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    work = json.loads((raw / "work.json").read_text())
    bad = deepcopy(value["validation_receipts"])
    bad[0]["passed"] = False
    assert run.build(work, raw, bad)["verdict_class"] == "disqualified"
    for field in ["log", "source", "snapshot", "authority", "code", "primitive"]:
        changed = deepcopy(value)
        if field == "log":
            changed["validation_receipts"][0]["log_sha256"] = "changed"
        elif field in ["source", "snapshot"]:
            changed["source_artifact_hashes"][0]["sha256"] = "changed"
            if field == "snapshot":
                changed["source_artifact_hashes"][0]["sha256"] = value["source_artifact_hashes"][0][
                    "sha256"
                ]
                changed["source_artifact_hashes"][0]["snapshot_path"] = str(output)
        elif field == "authority":
            changed["authority_snapshots"]["active"]["sha256"] = "changed"
        elif field == "code":
            changed["code_config_hashes"][e.MODULE] = "changed"
        else:
            changed["raw_shard_hashes"][0]["sha256"] = "changed"
        atomic_json(output, changed)
        assert not run.replay(output), field
    atomic_json(output, value)
    assert run.main(["--cold-replay", str(output)]) == 0
    monkeypatch.setattr(run, "INPUTS", [])
    monkeypatch.setattr(e, "ROOT", tmp_path / "no_tools")
    result = run.measure(tmp_path, design, staged, active, tmp_path / "missing_tools")
    assert any(".venv/bin" in r["path"] for r in result["failures"])
