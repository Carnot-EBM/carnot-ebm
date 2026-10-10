"""REQ-REPORT-8360 / REQ-VERIFY-8360: freeze deployment without new science."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v721_contract_methods as e
from carnot.reporting import v721_contract_runner as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


def cli(*args):
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.fixture(scope="module")
def natural(tmp_path_factory):
    private = tmp_path_factory.mktemp("v721")
    output = private / (e.NAME + ".json")
    result = cli("--date", "20261010", "--output", output, "--private-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    return output, value, work


def test_natural_contract(natural):
    """SCENARIO-REPORT-8360-AUTHORITY: input readiness survives unrelated failures."""
    output, value, work = natural
    assert e.replay(output)
    assert value["verdict_class"] == "circular_positive"
    assert value["current_contract_ready_score"] == 1
    assert value["frozen_heads_ready_score"] == value["frozen_trajectory_ready_score"] == 1
    assert len(value["rows"]) == 14 and len(value["future_dependencies"]) == 13
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert work["inputs"]["predictions"] and work["kernel"]["ready"]
    history = {row["experiment_id"]: row for row in value["historical_dispositions"]}
    assert len(history) == 14 and history[8354]["pre_gate"]
    assert history[8350]["verdict_class"] == history[8351]["verdict_class"] == "disqualified"
    assert history[8359]["verdict_class"] == "disqualified"
    assert all(k in value["field_principles"] for k in value)


def test_frozen_protocol(natural):
    """SCENARIO-REPORT-8360-PROTOCOL: freeze exact operands, gates and scopes."""
    _, value, work = natural
    p = work["deployment"]
    assert p["scientific_protocol"]["sha256"] == e.inputs.PIN
    assert len(p["head"]["coefficients"]) == 34 and p["head"]["temperature"] == 2
    assert p["table"]["grid_points"] == 65 and p["table"]["fractional_bits"] == 11
    assert p["panels"]["random_vectors"] == 4096
    assert p["panels"]["nextafter_neighbors"] is True
    assert p["panels"]["update_fault_seeds"] == [11, 22, 33]
    assert p["panels"]["stream_missing_slots"] == 22
    assert p["benchmark"]["paired_repeats"] == 5
    assert p["transaction"]["llm_acquisition_included"] is False
    assert p["engineering_gates"]["action_disagreements"] == 0
    assert value["protocol_sha256"] == e.DEPLOYMENT_PIN


@pytest.mark.parametrize("mutation", ["match", "prompt", "delete", "reorder", "lineage"])
def test_private_e2e018(tmp_path, mutation):
    """SCENARIO-REPORT-8360-AUTHORITY: private full-object controls reject drift."""
    root = tmp_path / "root"
    for name in [e.DESIGN, e.ACTIVE]:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((e.ROOT / name).read_bytes())
    plan = yaml.safe_load((root / e.ACTIVE).read_bytes())
    if mutation == "prompt":
        plan["tasks"][0]["prompt"] += " changed"
    elif mutation == "delete":
        plan["tasks"].pop()
    elif mutation == "reorder":
        plan["tasks"].reverse()
    elif mutation == "lineage":
        plan["tasks"][0]["prior_failures"][0]["verdict"] += " changed"
    (root / e.ACTIVE).write_text(yaml.safe_dump(plan))
    (root / e.STAGED).write_text(yaml.safe_dump(plan))
    checked = e.authority(root, tmp_path / "authority")
    assert checked["activated"] == (mutation == "match")


def test_missing_and_owned_failure(natural, tmp_path):
    """SCENARIO-VERIFY-8360-REPLAY: missing authority cannot erase good inputs."""
    _, value, work = natural
    changed = deepcopy(work)
    changed["contract"]["activated"] = False
    changed["failures"].append(e.failure(tmp_path / "upstream", "activation", True, None))
    raw = tmp_path / "raw"
    atomic_json(raw / "measurement.json", changed)
    result = e.build(changed, value["validation_receipts"], raw, tmp_path / (e.NAME + ".json"))
    assert result["verdict_class"] == "blocked" and result["current_contract_ready_score"] == 0
    assert result["frozen_heads_ready_score"] == result["frozen_trajectory_ready_score"] == 1
    failed = e.build(changed, [dict(passed=False)], raw, tmp_path / (e.NAME + ".json"))
    assert failed["verdict_class"] == "disqualified"
    assert failed["frozen_heads_ready_score"] == failed["frozen_trajectory_ready_score"] == 0


@pytest.mark.parametrize(
    "field",
    ["contract", "inputs", "deployment", "methods", "historical_dispositions", "trajectory"],
)
def test_rehashed_primitives(natural, tmp_path, field):
    """SCENARIO-VERIFY-8360-REPLAY: fresh hashes cannot grant altered meaning."""
    _, value, work = natural
    changed = deepcopy(work)
    if field == "contract":
        changed[field]["tasks"][0]["prompt"] += " forged"
    elif field == "inputs":
        changed[field]["heads"]["manifest"]["heads"][0]["coefficients"][0] += 0.125
    elif field == "deployment":
        changed[field]["engineering_gates"]["action_disagreements"] = 3
    elif field == "methods":
        changed[field]["science_changed"] = True
    elif field == "historical_dispositions":
        changed[field][0]["honest_verdict"] = "complete_positive_forged"
    else:
        changed[field]["ready"] = False
    raw = tmp_path / field
    atomic_json(raw / "measurement.json", changed)
    forged = e.build(changed, value["validation_receipts"], raw, tmp_path / (e.NAME + ".json"))
    path = tmp_path / "forged.json"
    atomic_json(path, forged)
    assert not e.replay(path)


def test_cli_and_missing_inputs(tmp_path):
    """SCENARIO-VERIFY-8360-REPLAY: missing external inputs publish exact blocks."""
    output = tmp_path / (e.NAME + ".json")
    result = cli("--root", tmp_path / "absent", "--output", output, "--private-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["gate_check_summary"][0]["observed"] is None
    assert value["frozen_heads_ready_score"] == value["frozen_trajectory_ready_score"] == 0
    assert e.replay(output)
    assert not e.replay(tmp_path / "missing.json")
    assert cli("--date", "wrong").returncode != 0
    assert cli("--date").returncode != 0
    assert cli("--private-fixture").returncode != 0


def test_failed_real_child(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8360-REPLAY: owned failure clears all reusable readiness."""
    from carnot.reporting.v709_execution import child

    def fail(plan, logs):
        return [
            child(
                "deliberate_failure",
                [sys.executable, "-u", "-c", "raise SystemExit(3)"],
                logs,
                deadline=10,
            )
        ]

    monkeypatch.setattr(r.qualified, "execute", fail)
    output = tmp_path / (e.NAME + ".json")
    assert r.main(["--date", "20261010", "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert value["validation_receipts"][0]["actual_exit"] == 3
    assert not value["required_checks_passed"] and value["flagged_adversarial"]
    assert value["frozen_heads_ready_score"] == value["frozen_trajectory_ready_score"] == 0
    assert e.replay(output)


def test_terminal_and_pregate_guards(natural, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8360-PROTOCOL: producer identity and terminal qualify reuse."""
    _, _, work = natural
    operand = next(p for p in work["methods"]["historical_consumers"] if p["experiment_id"] == 8347)
    original = e.legacy.authenticate

    def broken(*args):
        args[-1].append(dict(problem="terminal"))
        return {}

    monkeypatch.setattr(e.legacy, "authenticate", broken)
    with pytest.raises(ValueError, match="terminal"):
        e.reusable(operand, "local_kernel_ready_score", tmp_path, [])
    monkeypatch.setattr(
        e.legacy, "authenticate", lambda *args: dict(original(*args), local_kernel_ready_score=0)
    )
    with pytest.raises(ValueError, match="qualified_terminal"):
        e.reusable(operand, "local_kernel_ready_score", tmp_path, [])
    monkeypatch.setattr(e.legacy, "authenticate", original)
    pregate = dict(path=str(tmp_path / "pregate.json"), experiment_id=8354, pre_gate=True)
    atomic_json(Path(pregate["path"]), dict(experiment=8354, blocked_at_layer="wrong"))
    pregate["sha256"] = sha256_file(Path(pregate["path"]))
    with pytest.raises(ValueError, match="pre_gate_identity"):
        e.disposition(pregate, tmp_path, [], [])


def test_reuse_error_preserved(natural, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8360-AUTHORITY: failed original reuse names the operand."""

    def broken(*args):
        raise ValueError("deliberate_reuse_error")

    monkeypatch.setattr(e, "reusable", broken)
    work = e.measure(e.ROOT, tmp_path / "measure")
    assert not work["kernel"] and not work["trajectory"]
    assert (
        len([g for g in work["failures"] if g["artifact_field"].endswith("authenticated_reuse")])
        == 2
    )


def test_rehashed_refs_and_summaries(natural, tmp_path):
    """SCENARIO-VERIFY-8360-REPLAY: primitive lists and checksums remain bound."""
    _, value, work = natural
    for mutation in ["checksum", "source_refs", "contract_rows", "failure_gates", "receipt"]:
        changed = deepcopy(value)
        if mutation == "checksum":
            changed["reproducibility_checksum"] = "invalid"
        elif mutation == "source_refs":
            changed["source_artifact_hashes"] = []
        elif mutation == "receipt":
            changed["validation_receipts"][0]["argv"] = ["forged"]
            changed["reproducibility_checksum"] = canonical_hash(
                {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
            )
        else:
            changed_work = deepcopy(work)
            if mutation == "failure_gates":
                changed_work["input_failures"].append(dict(forged=True))
            else:
                changed_work["contract"]["contract_rows"][0]["prompt_sha256"] = "wrong"
            raw = tmp_path / "changed"
            atomic_json(raw / "measurement.json", changed_work)
            changed = e.build(
                changed_work, value["validation_receipts"], raw, tmp_path / (e.NAME + ".json")
            )
        candidate = tmp_path / (mutation + ".json")
        atomic_json(candidate, changed)
        assert not e.replay(candidate)


def test_manifest_and_adapter(tmp_path, monkeypatch):
    """REQ-VERIFY-8360: commands are frozen, scoped and bounded before evidence."""
    plan = r.manifest(tmp_path)
    assert all(p["deadline"] <= 600 for p in plan)
    assert all(p["task_cap_s"] == 4800 for p in plan)
    assert any(p["name"] == "private_E2E018_V721" for p in plan)
    assert not any("tests/python" in p["argv"] for p in plan)

    def check_scratch(args):
        import tempfile

        assert Path(tempfile.gettempdir()) == r.SCRATCH
        assert r.SCRATCH.stat().st_mode & 0o777 == 0o700
        return 0

    monkeypatch.setattr(r.qualified, "main", check_scratch)
    monkeypatch.setattr(sys, "argv", ["runner"])
    assert r.main() == 0


def test_declared_digest_tamper(tmp_path):
    """SCENARIO-REPORT-8360-AUTHORITY: a declaration must hash the full task list."""
    root = tmp_path / "root"
    for name in [e.DESIGN, e.ACTIVE]:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((e.ROOT / name).read_bytes())
    text = (root / e.DESIGN).read_text()
    import re

    text = re.sub(r"(Canonical full-task SHA-256: `)[a-f0-9]{64}", r"\g<1>" + "0" * 64, text)
    (root / e.DESIGN).write_text(text)
    with pytest.raises(ValueError, match="complete_task_digest"):
        e.authority(root, tmp_path / "authority")
