"""REQ-REPORT-9194-V783 / SCENARIO-REPORT-9194-V783: preserve authority without science."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting.historical_contract_inputs import original_root
from carnot.reporting import v783_contract as c
from carnot.reporting import v783_contract_evidence as e


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Real immutable producers qualify custody; fixtures grant no scientific credit."""
    return c.measure(
        original_root(tmp_path_factory.mktemp("inputs9194"), 783),
        tmp_path_factory.mktemp("contract9194"),
        "20261117",
    )


def test_original_authority(work):
    """REQ-REPORT-9194-V783: preserve every role, original bound, failure and obligation."""
    i = work["inspection"]
    assert all(r["passed"] for r in i["checks"]), [r for r in i["checks"] if not r["passed"]]
    assert i["source_ready"] and i["static_ready"] and i["online_ready"]
    assert i["source"]["roles"] == dict(TRAIN=150, CAL=50, TEST=100)
    assert i["source"]["answer_count"] == 900
    assert [len(r["qids"]) for r in i["execution_rosters"]] == [100] * 3
    assert len(i["static_protocol"]["arms"]) == 5
    assert len(i["static_protocol"]["bounds"]) == 2
    assert len(i["online_protocol"]["arms"]) == 6
    assert len(i["online_protocol"]["benefit_bounds"]["bounds"]) == 7
    assert len(i["entropy_reference_protocol"]["comparators"]) == 3
    assert i["diagnostic_protocol"]["brier_moments"]["identity_tolerance"] == 1e-12
    assert len(i["previous_outcome_rows"]) == 14
    assert sum(r["verdict_class"] == "disqualified" for r in i["previous_outcome_rows"]) == 2
    o = i["validation_obligations"]
    assert len(o["original_node_ids"]) == 621 and len(o["v779_addition_ids"]) == 53
    assert len(o["node_bindings"]) == 674 and o["current_qualification_credit"] == 0
    assert any("test_no_artifact_on_disk" in r["node_id"] for r in o["affected_node_assertions"])
    assert any(e.TEST in r["node_id"] for r in o["affected_node_assertions"])
    p = i["label_ambiguity_protocol"]
    assert p["primary_gate_rescue"] is False and p["missing_A2"] == -1
    assert p["repeated_retention"] == "sum weighted response coefficients before endpoint selection"
    assert work["protocol_sidecars"]["label_ambiguity_protocol_reference"]["sha256"]
    resources = {r["resource"]: r for r in work["preconditions_checked"]}
    assert resources["required_tools"]["available"]
    assert resources["host_resources"]["check_receipt"]["private_free_disk_bytes"] > 0


def tree(tmp_path):
    """Copy only declared contract inputs so negative controls cannot alter the checkout."""
    source_root = original_root(tmp_path / "original", 783)
    for name in [e.DESIGN, e.ACTIVE, *e.COLLISION_INPUTS]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((source_root / name).read_bytes())
    (tmp_path / "results").mkdir(exist_ok=True)
    return tmp_path


def test_full_object_mutations(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9194-V783: new pins cannot bless changed prompts or gate objects."""
    root = tree(tmp_path)
    assert e.authority(root)["ready"]
    original = yaml.safe_load((root / e.ACTIVE).read_bytes())
    for field in [
        "prompt",
        "id",
        "gated_on",
        "MODEL_SPECS",
        "inference_substrate_class",
        "order",
        "count",
    ]:
        v = deepcopy(original)
        if field == "order":
            v["tasks"].reverse()
        elif field == "count":
            v["tasks"].pop()
        else:
            v["tasks"][1][field] = "forged"
        (root / e.ACTIVE).write_text(yaml.safe_dump(v))
        frozen = deepcopy(e.FROZEN)
        frozen["pins"][e.ACTIVE] = e.ref(root / e.ACTIVE)["sha256"]
        monkeypatch.setattr(e, "FROZEN", frozen)
        assert not e.authority(root)["ready"]
    (root / e.ACTIVE).unlink()
    assert any(r["observed"] == "missing_file" for r in e.authority(root)["checks"])
    (root / e.ACTIVE).write_text("{}")
    assert any(r["observed"] == "missing_field" for r in e.authority(root)["checks"])


def test_collisions(tmp_path):
    """SCENARIO-REPORT-9194-V783: only this task's canonical output is exempt."""
    root = tree(tmp_path)
    (root / "results" / (e.NAME + ".json")).write_text("{}")
    assert e.authority(root)["ready"]
    (root / "results/experiment_9195_existing.json").write_text("{}")
    assert not e.authority(root)["ready"]


def test_blocked_and_build(work, tmp_path, monkeypatch):
    """REQ-REPORT-9194-V783: missing evidence blocks; failed owned checks disqualify."""
    monkeypatch.setattr(c.legacy, "receipts_pass", lambda *_: True)
    v = c.build(work, {})
    assert v["verdict_class"] == "null" and v["contract_ready_score"] == 1
    assert (
        v["source_ready_score"]
        == v["static_methods_ready_score"]
        == v["online_methods_ready_score"]
        == 1
    )
    assert len(v["full_task_objects"]) == 14 and v["experiment_id"] == 9194
    assert v["MODEL_SPECS"] == [] and v["current_llm_invocations"] == v["generated_tokens"] == 0
    assert not v["activation_authorized"] and v["independent_generalization_score"] == 0
    assert v["label_ambiguity_protocol_reference"]["sha256"]
    monkeypatch.setattr(c.legacy, "receipts_pass", lambda *_: False)
    assert c.build(work, {})["verdict_class"] == "disqualified"
    missing = c.measure(tmp_path / "absent", tmp_path / "raw", "20261117")
    blocked = c.build(missing, {})
    assert blocked["verdict_class"] == "blocked" and blocked["censored_count"] == 14
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert [r["unit_id"] for r in blocked["rows"]] == [f"exp{n}" for n in range(9194, 9208)]
    for observed in ["missing_file", "missing_field", 0, False, 1]:
        assert e.check(tmp_path, "score", 1, observed, "gate")["passed"] == (
            type(observed) is int and observed == 1
        )


def child(tmp_path, *args):
    """Actual private interpreters retain subprocess coverage and remove PYTHONPATH."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9194] child_before", args, flush=True)
    result = subprocess.run(
        argv, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=180
    )
    print("[test9194] child_after", result.returncode, flush=True)
    return result


def test_private_cli(work, tmp_path):
    """SCENARIO-REPORT-9194-V783: E2E-018/021 success, refusal, cold replay and tamper."""
    source, output = tmp_path / "work.json", tmp_path / (e.NAME + ".json")
    c.atomic_json(source, work)
    result = child(tmp_path, "--from-work", source, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert child(tmp_path, "--cold-replay", output).returncode == 0
    v = json.loads(output.read_bytes())
    v["source_ready_score"] = 99
    v["reproducibility_checksum"] = c.legacy.checksum(v)
    c.atomic_json(output, v)
    assert child(tmp_path, "--cold-replay", output).returncode == 1
    assert child(tmp_path, "--date", "bad").returncode == 1
    assert child(tmp_path, "--from-work", source).returncode == 1
    assert (
        child(
            tmp_path, "--checks-only", "--root", tmp_path / "absent", "--output", output
        ).returncode
        == 0
    )


def test_protocol_and_historical_tamper(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9194-V783: changed retained evidence never gains authority."""
    original = e.ref

    def altered(path):
        v = original(path)
        if str(path).endswith("experiment_9180_v782_contract.json"):
            v["sha256"] = None
        return v

    monkeypatch.setattr(e, "ref", altered)
    i = e.inspect(Path(work["root"]))
    assert not i["source_ready"] and any(r["observed"] == "missing_file" for r in i["checks"])
    monkeypatch.setattr(e, "ref", original)
    changed = deepcopy(work)
    p = tmp_path / "forged.json"
    c.atomic_json(p, {"invented": True})
    changed["protocol_sidecars"]["label_ambiguity_protocol_reference"] = e.ref(p)
    c.atomic_json(tmp_path / "work.json", changed)
    v = c.build(changed, {})
    v["work_reference"] = e.ref(tmp_path / "work.json")
    c.atomic_json(tmp_path / (e.NAME + ".json"), v)
    with pytest.raises(ValueError, match="label_ambiguity_protocol_drift"):
        c.replay(tmp_path / (e.NAME + ".json"))


def test_missing_original_work(work, monkeypatch):
    """SCENARIO-REPORT-9194-V783: original producer absence stops dependent imports."""
    original = e.ref
    owner = json.loads((e.ROOT / "results/experiment_9180_v782_contract.json").read_bytes())

    def altered(path):
        v = original(path)
        if str(path) == owner["work_reference"]["path"]:
            v["sha256"] = None
        return v

    monkeypatch.setattr(e, "ref", altered)
    assert not e.inspect(Path(work["root"]))["source_ready"]


def test_normal_main_and_deadlines(work, tmp_path, monkeypatch):
    """REQ-REPORT-9194-V783: normal orchestration, qualification and deadlines are executable."""
    monkeypatch.setattr(c.legacy, "qualify", lambda _: dict(delegated=True))
    assert c.qualify(work) == dict(delegated=True)
    monkeypatch.setattr(c, "measure", lambda *_: work)
    monkeypatch.setattr(c, "qualify", lambda _: {})
    monkeypatch.setattr(c, "publish_primary", lambda *_: dict(passed=True))
    assert c.main(["--output", str(tmp_path / (e.NAME + ".json"))]) == 0
    monkeypatch.setenv("CARNOT_V783_DEADLINE", "0")
    assert c.main(["--output", str(tmp_path / (e.NAME + ".json"))]) == 1
    source = tmp_path / "source.json"
    c.atomic_json(source, work)

    def invoke(output, value, validator):
        return validator(output)

    monkeypatch.setattr(c, "publish_primary", invoke)
    assert c.main(["--from-work", str(source), "--output", str(tmp_path / (e.NAME + ".json"))]) == 1
