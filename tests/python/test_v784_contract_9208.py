"""REQ-REPORT-9208-V784 / SCENARIO-REPORT-9208-V784: immutable replay authority."""

from copy import deepcopy
import json
from pathlib import Path
import os
import subprocess
import sys

import pytest

from carnot.reporting import v782_contract as old
from carnot.reporting import v783_contract as previous
from carnot.reporting import v784_contract as c
from carnot.reporting import v784_contract_evidence as e
from carnot.reporting import protocol_owners as owners
from carnot.reporting.historical_contract_inputs import original_root
from carnot.reporting.current_work_receipt import atomic_json


def child_cli(directory, *args, cli=None):
    """Private interpreters remove PYTHONPATH and retain actual child coverage."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(e.ROOT / (cli or e.CLI)), *args]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9208] child_before", args, flush=True)
    result = subprocess.run(
        argv, cwd=directory, env=env, text=True, capture_output=True, timeout=180
    )
    print("[test9208] child_after", result.returncode, flush=True)
    return result


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Real original sources establish custody, without any new model work."""
    return c.measure(e.ROOT, tmp_path_factory.mktemp("contract9208"), "20261117")


def test_protocol_owners_and_false_rehash(tmp_path):
    """SCENARIO-REPORT-9208-V784: rebound evidence never owns inherited protocols."""
    work = dict(protocol_sidecars={})
    for protocol in owners.OWNERS:
        payload, owner = owners.load(protocol)
        path = tmp_path / (protocol + ".json")
        atomic_json(path, payload)
        work["protocol_sidecars"][protocol + "_reference"] = e.ref(path)
        assert owners.validate(work, protocol)["owning_configuration"] == owner
        changed = dict(payload, primary_gate_rescue=0)
        atomic_json(path, changed)
        work["protocol_sidecars"][protocol + "_reference"] = e.ref(path)
        with pytest.raises(ValueError, match="protocol_drift"):
            owners.validate(work, protocol)
        path.write_text('{"invented": true}')
        work["protocol_sidecars"][protocol + "_reference"] = e.ref(path)
        with pytest.raises(ValueError, match="protocol_drift"):
            owners.validate(work, protocol)
        path.unlink()
        with pytest.raises(ValueError, match="protocol_drift"):
            owners.validate(work, protocol)
    with pytest.raises(ValueError, match="protocol_drift"):
        owners.validate(dict(protocol_sidecars={}), "entropy_reference_protocol")


def test_historical_roots_and_pollution(tmp_path):
    """SCENARIO-REPORT-9208-V784: successor queues fail strict old authority."""
    for generation, module in [(782, old), (783, previous)]:
        root = original_root(tmp_path / str(generation), generation)
        assert module.e.authority(root)["ready"]
        (root / module.e.ACTIVE).chmod(0o600)
        (root / module.e.ACTIVE).write_bytes((e.ROOT / e.ACTIVE).read_bytes())
        assert not module.e.authority(root)["ready"]
        with pytest.raises(ValueError, match="historical_input_drift"):
            original_root(root, generation)


def test_contract_and_frozen_scope(work):
    """REQ-REPORT-9208-V784: exact objects stay separate from qualified readiness."""
    inspection = work["inspection"]
    assert all(r["passed"] for r in inspection["checks"]), [
        r for r in inspection["checks"] if not r["passed"]
    ]
    assert len(inspection["contract"]["tasks"]) == 14
    assert len(inspection["validation_obligations"]["node_bindings"]) == 674
    assert inspection["source"]["roles"] == dict(TRAIN=150, CAL=50, TEST=100)
    assert inspection["source"]["answer_count"] == 900
    assert inspection["preserved_v783"]["readiness_scores"] == [0] * 4
    value = c.build(work, {})
    assert value["replay_authority_ready_score"] == 0
    assert value["MODEL_SPECS"] == []
    assert all(v == 0 for v in value["model_invocation_counts"].values())
    assert value["claim_scope"]["independent_generalization_score"] == 0
    assert value["target_identifiability_rubric"]["cross_generator_substitution"] is False
    assert len(value["protocol_owner_rows"]) == 2


def test_private_cli_and_interleaved_replays(work, tmp_path):
    """SCENARIO-REPORT-9208-V784: E2E-018/021 real private publication and tamper."""
    source = tmp_path / "work.json"
    atomic_json(source, work)
    output = tmp_path / (e.NAME + ".json")
    result = child_cli(tmp_path, "--from-work", str(source), "--output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert child_cli(tmp_path, "--cold-replay", str(output)).returncode == 0
    previous_work = previous.measure(
        original_root(tmp_path / "v783", 783), tmp_path / "previous", "20261117"
    )
    previous_source = tmp_path / "previous.json"
    atomic_json(previous_source, previous_work)
    previous_output = tmp_path / (previous.e.NAME + ".json")
    assert (
        child_cli(
            tmp_path,
            "--from-work",
            str(previous_source),
            "--output",
            str(previous_output),
            cli=previous.e.CLI,
        ).returncode
        == 0
    )
    assert (
        child_cli(tmp_path, "--cold-replay", str(previous_output), cli=previous.e.CLI).returncode
        == 0
    )
    assert child_cli(tmp_path, "--cold-replay", str(output)).returncode == 0
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    code = "\n".join(
        [
            "import sys; from pathlib import Path",
            "sys.path[:0] = " + repr([str(e.ROOT / "python"), str(e.ROOT)]),
            "from carnot.reporting import v784_contract as new, v783_contract as old",
            "assert new.replay(Path(" + repr(str(output)) + "))['passed']",
            "assert old.replay(Path(" + repr(str(previous_output)) + "))['passed']",
            "assert new.replay(Path(" + repr(str(output)) + "))['passed']",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    value["replay_authority_ready_score"] = 99
    value["reproducibility_checksum"] = c.legacy.checksum(value)
    atomic_json(output, value)
    assert child_cli(tmp_path, "--cold-replay", str(output)).returncode == 1
    assert child_cli(tmp_path, "--date", "bad").returncode == 1
    assert child_cli(tmp_path, "--from-work", str(source)).returncode == 1


def test_missing_inputs_and_changed_owner(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9208-V784: missing evidence blocks and owner drift refuses."""
    changed = deepcopy(owners.OWNERS)
    changed["entropy_reference_protocol"]["sha256"] = "sha256:" + "0" * 64
    monkeypatch.setattr(owners, "OWNERS", changed)
    with pytest.raises(ValueError, match="protocol_owner_hash"):
        owners.load("entropy_reference_protocol")
    monkeypatch.undo()
    missing = c.measure(tmp_path / "absent", tmp_path / "raw", "20261117")
    value = c.build(missing, {})
    assert value["verdict_class"] == "blocked"
    assert value["inference_substrate_class"] == "blocked_no_run"
    assert value["censored_count"] == 14


def test_complete_object_mutations_and_collisions(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9208-V784: new pins cannot bless changed gates or occupied IDs."""
    import yaml

    for name in [e.DESIGN, e.ACTIVE, *e.COLLISION_INPUTS]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((e.ROOT / name).read_bytes())
    assert e.authority(tmp_path)["ready"]
    frozen = deepcopy(e.FROZEN)
    original = yaml.safe_load((tmp_path / e.ACTIVE).read_bytes())
    for field in [
        "prompt",
        "id",
        "gated_on",
        "MODEL_SPECS",
        "inference_substrate_class",
        "order",
        "count",
    ]:
        value = deepcopy(original)
        if field == "order":
            value["tasks"].reverse()
        elif field == "count":
            value["tasks"].pop()
        else:
            value["tasks"][1][field] = "forged"
        (tmp_path / e.ACTIVE).write_text(yaml.safe_dump(value))
        frozen["pins"][e.ACTIVE] = e.ref(tmp_path / e.ACTIVE)["sha256"]
        monkeypatch.setattr(e, "FROZEN", frozen)
        assert not e.authority(tmp_path)["ready"]
    monkeypatch.undo()
    (tmp_path / e.ACTIVE).write_bytes((e.ROOT / e.ACTIVE).read_bytes())
    (tmp_path / "results").mkdir()
    (tmp_path / "results/experiment_9209_existing.json").write_text("{}")
    assert not e.authority(tmp_path)["ready"]
    (tmp_path / e.ACTIVE).write_text("{}")
    assert e.authority(tmp_path)["checks"][-1]["observed"] == "missing_field"


def test_orchestration_and_failed_owner_work(work, tmp_path, monkeypatch):
    """REQ-REPORT-9208-V784: actual orchestration retains the shared deadline and refusal."""
    monkeypatch.setattr(c, "measure", lambda *_: work)
    monkeypatch.setattr(c, "qualify", lambda _: {})
    monkeypatch.setattr(previous, "publish_primary", lambda *_: dict(passed=True))
    assert c.main(["--output", str(tmp_path / (e.NAME + ".json"))]) == 0
    monkeypatch.setenv("CARNOT_V783_DEADLINE", "0")
    assert c.main(["--output", str(tmp_path / (e.NAME + ".json"))]) == 1


def test_original_failure_reproduces_in_fresh_process(work, tmp_path):
    """SCENARIO-REPORT-9208-V784: original inherited code still exposes the missing-owner bug."""
    capstone = json.loads((e.ROOT / "results/experiment_9193_v782_capstone.json").read_bytes())
    saved = json.loads(Path(capstone["work_reference"]["path"]).read_bytes())
    original = saved["operands"][str(e.ROOT / "python/carnot/reporting/v782_contract.py")]
    config = work["inspection"]["preserved_v783"]["pre_ces_configuration"]
    payload = dict(protocol_sidecars=work["protocol_sidecars"])
    source = tmp_path / "original-work.json"
    atomic_json(source, payload)
    artifact = tmp_path / "original-candidate.json"
    atomic_json(artifact, dict(work_reference=e.ref(source)))
    code = "\n".join(
        [
            "import json, runpy, sys",
            "from pathlib import Path",
            "from types import SimpleNamespace",
            "sys.path[:0] = " + repr([str(e.ROOT / "python"), str(e.ROOT)]),
            "from carnot.reporting.logit_methods_9126 import ref",
            "module = runpy.run_path(" + repr(original["path"]) + ")",
            "frozen = json.loads(Path(" + repr(config["path"]) + ").read_bytes())",
            "assert 'entropy_reference_protocol' not in frozen",
            "module['replay'].__globals__['e'] = SimpleNamespace(ref=ref, FROZEN=frozen)",
            "try:",
            "    module['replay'](Path(" + repr(str(artifact)) + "))",
            "except KeyError as error:",
            "    assert error.args == ('entropy_reference_protocol',)",
            "    print('reproduced_original_entropy_KeyError', flush=True)",
            "else:",
            "    raise AssertionError('original_failure_disappeared')",
        ]
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reproduced_original_entropy_KeyError" in result.stdout


def test_external_absence_and_protocol_failure_controls(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9208-V784: absence blocks, but forged sidecars always refuse."""

    def unavailable(*args, **kwargs):
        raise ValueError("historical_input_drift:" + str(tmp_path / "missing"))

    monkeypatch.setattr(e, "original_root", unavailable)
    with monkeypatch.context() as context:
        context.setattr(e, "HISTORICAL_BASE", tmp_path, raising=False)
        result = e.inspect(e.ROOT)
    assert not result["source_ready"]
    assert any(r["observed"] == "missing_file" for r in result["checks"])
    changed = deepcopy(work)
    fake = tmp_path / "false.json"
    atomic_json(fake, dict(invented=True))
    changed["protocol_sidecars"]["entropy_reference_protocol_reference"] = e.ref(fake)
    with pytest.raises(ValueError, match="protocol_drift"):
        c.protocol_rows(changed)
    missing = deepcopy(owners.OWNERS)
    missing["entropy_reference_protocol"]["path"] = str(tmp_path / "absent-owner")
    monkeypatch.setattr(owners, "OWNERS", missing)
    monkeypatch.setattr(c, "OWNERS", missing)
    rows = c.protocol_rows(work)
    assert rows[0]["gate"]["observed"] == "missing_file"
    assert not rows[0]["gate"]["passed"]


def test_changed_owner_metadata_and_authenticated_absence(work, tmp_path):
    """SCENARIO-REPORT-9208-V784: rehashed false owner metadata fails in a fresh process."""
    from carnot.reporting.historical_contract_inputs import authenticated

    with pytest.raises(ValueError, match="historical_input_drift"):
        authenticated(dict(path=str(tmp_path / "missing"), sha256="sha256:" + "0" * 64))
    changed = deepcopy(work)
    changed["protocol_owner_rows"][0]["original_hash"] = "sha256:" + "0" * 64
    source = tmp_path / "false-work.json"
    atomic_json(source, changed)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, dict(work_reference=e.ref(source)))
    with pytest.raises(ValueError, match="protocol_owner_drift"):
        c.replay(output)
    result = child_cli(tmp_path, "--cold-replay", str(output))
    assert result.returncode == 1 and "protocol_owner_drift" in result.stdout


def test_qualified_scope_uses_one_parent_supervisor(work, tmp_path, monkeypatch):
    """REQ-REPORT-9208-V784: complete scopes retain supervised deadlines and heartbeats."""
    seen = []

    def child(name, *args, **kwargs):
        seen.append((name, kwargs["deadline"]))
        return dict(passed=True)

    monkeypatch.setattr(c.legacy.lifecycle, "child", child)
    monkeypatch.setattr(c.legacy.lifecycle, "measured", lambda *args: dict(measured=True))
    value = c.qualify(work)
    assert len(value["checks"]) == 6 and value["owned_statement_coverage"]["measured"]
    assert value["repository_health"]["passed"]
    assert dict(seen) == dict(
        owned_tests=1200,
        direct_consumers=1200,
        ruff_check=600,
        ruff_format=600,
        strict_mypy=600,
        spec_coverage=600,
        global_python_suite=600,
    )
    assert all(r["scope_sha256"] == c.canonical_hash(work["scope"]) for r in value["checks"])
