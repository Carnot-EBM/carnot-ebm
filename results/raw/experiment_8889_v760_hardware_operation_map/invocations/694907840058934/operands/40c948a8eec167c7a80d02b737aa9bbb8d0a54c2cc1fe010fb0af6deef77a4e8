"""REQ-REPORT-8853; SCENARIO-REPORT-8853-CONTRACT: complete authority custody."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest
import yaml

from carnot.reporting import v758_branch_contract as c


def tree(root):
    """Use supplied complete bytes so a queue cannot manufacture missing authority."""
    path = root / c.DESIGN
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((c.ROOT / c.DESIGN).read_bytes())
    queue = yaml.safe_load((c.ROOT / c.ACTIVE).read_bytes())
    queue["tasks"] = c.parse_design(path.read_text(), milestone=c.MILESTONE)[1]
    (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
    return root


def test_authority_lifecycle(tmp_path):
    """REQ-REPORT-8853: staged and activated objects bind every field."""
    root = tree(tmp_path)
    assert c.authority(root)["ready"]
    queue = yaml.safe_load((root / c.ACTIVE).read_bytes())
    (root / c.STAGED).write_text(yaml.safe_dump(queue))
    (root / c.ACTIVE).write_text(yaml.safe_dump(dict(queue, milestone="2026.11.757", tasks=[])))
    assert c.authority(root)["ready"]
    for task in queue["tasks"]:
        task["prompt"] = c.encode(task["prompt"])
    (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
    assert c.authority(root)["ready"]
    queue["tasks"][0]["prompt"] = queue["tasks"][0]["prompt"].replace("{date}", "{{date}}")
    (root / c.STAGED).write_text(yaml.safe_dump(queue))
    assert not c.authority(root)["ready"]
    assert not c.authority(root / "missing")["ready"]


def test_private_mutations(tmp_path):
    """SCENARIO-REPORT-8853-CONTRACT: each required corruption is rejected."""
    root = tree(tmp_path / "source")
    rows = c.mutations(root, tmp_path / "controls")
    assert len(rows) == 7 and all(row["passed"] for row in rows)
    assert {r["mutation"] for r in rows} == {
        "missing_last_task",
        "truncation",
        "prompt",
        "gate",
        "model",
        "closing_marker",
        "table",
    }
    path = root / c.DESIGN
    path.write_bytes((c.ROOT / c.ARCHIVE).read_bytes())
    assert not c.authority(root)["ready"]


def validation(passed=True):
    """Synthetic failures test reduction; real publication requires SQLite custody."""
    return dict(receipts=[dict(passed=passed)], coverage=dict(complete=passed))


def test_freeze_build_and_replay(tmp_path, monkeypatch):
    """REQ-REPORT-8853: optional absence does not gate independent CPU branches."""
    root = tree(tmp_path / "root")
    work = c.freeze(root, tmp_path / "raw", "20261105")
    value = c.build(work, validation(False))
    assert value["verdict_class"] == "disqualified" and value["contract_ready_score"] == 0
    path = tmp_path / "candidate.json"
    c.atomic_json(path, value)
    assert c.replay(path)["passed"]
    value = c.build(work, validation())
    assert value["verdict_class"] == "null" and value["contract_ready_score"] == 1
    assert value["runtime_repair_ready_score"] == 0
    assert value["completed_count"] == value["intended_count"] == 13
    assert value["canonical_tasks_sha256"] == c.PIN
    assert value["preserved_v757_reference"]["verdict_class"] == "disqualified"
    assert value["source_method_map"] and value["numerical_training_calls"] == 0
    assert value["arc_task_count"] == value["generator_training_task_count"] == 0
    assert all(r["required_field_declared"] for r in value["gate_producer_rows"])
    c.atomic_json(path, value)
    assert not c.replay(path)["passed"]  # no authenticated coverage database
    monkeypatch.setattr(c, "verify_measured", lambda *args: True)
    assert c.replay(path)["passed"]
    value = c.build(work, validation(False))
    value["contract_ready_score"] = 1
    value["reproducibility_checksum"] = c.checksum(value)
    c.atomic_json(path, value)
    assert not c.replay(path)["passed"]
    assert not c.replay(tmp_path / "absent")["passed"]
    work["contract"]["ready"] = False
    assert c.build(work, validation())["verdict_class"] == "blocked"
    work["authority_mutation_rows"][0]["passed"] = False
    assert c.build(work, validation())["verdict_class"] == "disqualified"


def test_raw_custody_rejection(tmp_path):
    """REQ-REPORT-8853: changed operands cannot survive a cold replay."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261105")
    path = tmp_path / "candidate.json"
    c.atomic_json(path, c.build(work, validation(False)))
    Path(work["snapshots"][0]["path"]).write_text("changed")
    assert not c.replay(path)["passed"]


def test_sealed_validation_and_semantic_drift(tmp_path):
    """REQ-REPORT-8853: rehashed raw authority cannot alter the reconstructed contract."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261105")
    v = validation(False)
    c.atomic_json(tmp_path / "raw/validation.json", v)
    value = c.build(work, v)
    value["raw_shard_hashes"].append(c.reference(tmp_path / "raw/validation.json"))
    value["reproducibility_checksum"] = c.checksum(value)
    path = tmp_path / "candidate.json"
    c.atomic_json(path, value)
    assert c.replay(path)["passed"]
    work["contract"]["ready"] = False
    c.atomic_json(
        Path(work["raw_reference"]["path"]), {k: v for k, v in work.items() if k != "raw_reference"}
    )
    work["raw_reference"] = c.reference(Path(work["raw_reference"]["path"]))
    value = c.build(work, v)
    c.atomic_json(path, value)
    assert not c.replay(path)["passed"]
    source = tmp_path / "source.txt"
    source.write_text("original")
    ref = c.seal(source, tmp_path / "seal")
    assert c.seal(source, tmp_path / "seal") == ref
    Path(ref["path"]).write_text("changed")
    with pytest.raises(ValueError, match="immutable_operand_conflict"):
        c.seal(source, tmp_path / "seal")


def test_missing_and_changed_history(tmp_path, monkeypatch):
    """REQ-REPORT-8853: a missing original primary is blocked, changed bytes rejected."""
    root = tree(tmp_path / "root")
    monkeypatch.setattr(c, "PINS", {"absent.json": "0" * 64})
    work = c.freeze(root, tmp_path / "raw", "20261105")
    assert not work["history"]["authenticated"]
    assert c.build(work, validation())["verdict_class"] == "blocked"
    monkeypatch.setattr(c, "PINS", {c.ARCHIVE: "0" * 64})
    assert not c.history(tmp_path / "history")["authenticated"]


def test_history_raw_custody(tmp_path, monkeypatch):
    """REQ-REPORT-8853: a pinned producer cannot authenticate changed raw operands."""
    original = c.reference
    monkeypatch.setattr(
        c,
        "reference",
        lambda p: (
            dict(original(p), sha256="forged") if p.name == "sealed_work.json" else original(p)
        ),
    )
    result = c.history(tmp_path)
    assert not result["authenticated"]
    assert result["failures"][0]["check"] == "preserved_raw_operand"


def test_cli_dispatch(tmp_path, monkeypatch):
    """REQ-REPORT-8853: the entry point reuses the runner and rejects invalid dates."""
    from carnot.reporting import v755_branch_runner as runner

    monkeypatch.setattr(sys, "argv", [c.CLI, "--date", "bad"])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert result.value.code == 2
    monkeypatch.setattr(runner, "run", lambda *args: 0)
    monkeypatch.setattr(sys, "argv", [c.CLI, "--output", str(tmp_path / (c.NAME + ".json"))])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert result.value.code == 0


def test_runtime_rejects_self_reports(tmp_path):
    """REQ-REPORT-8853: reported repair without causal transcript custody is rejected."""
    path = tmp_path / "ops/cuda-runtime-repair-receipt.json"
    c.atomic_json(path, dict(ready=True, runtime_repair_ready_score=1))
    result = c.runtime(tmp_path, tmp_path / "raw")
    assert (
        not result["ready"] and result["gate_check_summary"][0]["observed"] == "rejected_producer"
    )
    assert result["device_probes"] == result["model_loads"] == 0


@pytest.mark.parametrize(
    "mutation",
    [None, "unchanged", "model", "validator", "transcript", "offload", "operand", "date"],
)
def test_supplied_runtime_custody(tmp_path, monkeypatch, mutation):
    """REQ-REPORT-8853: synthetic receipts exercise each admission gate without GPU access."""
    from carnot.reporting import v725_contract_methods as old
    import time

    context = dict(
        boot_identity=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        namespaces={p.name: str(p.readlink()) for p in Path("/proc/self/ns").iterdir()},
    )
    model_path = tmp_path / c.MODEL_REVISION / "model.gguf"
    model_path.parent.mkdir()
    model_path.write_bytes(b"private synthetic GGUF identity control")
    monkeypatch.setattr(c, "MODEL_PIN", c.sha256_file(model_path))
    model = dict(
        c.reference(model_path), hf_id="unsloth/Qwen3.8-27B-GGUF", revision=c.MODEL_REVISION
    )
    if mutation == "model":
        model["revision"] = "0" * 40
    validators = [
        dict(source_path=p, sha256=c.sha256_file(c.ROOT / p)) for p in c.RUNTIME_VALIDATORS
    ]
    if mutation == "validator":
        validators[0]["sha256"] = "sha256:" + "0" * 64
    current = dict(
        libraries=[dict(path="/synthetic/libcuda.so", sha256="after")], environment={}, **context
    )
    previous = dict(current, libraries=[dict(path="/synthetic/libcuda.so", sha256="before")])
    if mutation == "unchanged":
        previous = deepcopy(current)
    transcript = dict(
        api_returns={
            k: 0
            for k in [
                "cuInit",
                "cuCtxCreate",
                "cuMemAlloc",
                "cuMemcpyHtoD",
                "cuMemcpyDtoH",
                "cuMemFree",
                "cuCtxDestroy",
            ]
        },
        execution_context=context,
        byte_copy_parity=True,
        cleanup_passed=True,
        duration_s=0.01,
        model_provenance=model,
        offloaded_layers=0 if mutation == "offload" else 1,
    )
    refs = {}
    for key, value in [("current", current), ("previous", previous), ("transcript", transcript)]:
        path = tmp_path / (key + ".json")
        c.atomic_json(path, value)
        refs[key + "_reference"] = c.reference(path)
    if mutation == "transcript":
        refs["transcript_reference"]["sha256"] = "sha256:" + "0" * 64
    if mutation == "operand":
        refs["current_reference"]["sha256"] = "sha256:" + "0" * 64
    c.atomic_json(
        tmp_path / "ops/cuda-runtime-repair-receipt.json",
        dict(
            authority="operator",
            previous_primary_sha256=old.cuda.CACHED_PIN,
            executed_at_ns=time.time_ns() - 1,
            frontier_wall_ns=0,
            action="driver_repair",
            description="private synthetic control; no device execution",
            execution_context=context,
            run_date="20261104" if mutation == "date" else "20261105",
            mandated_model_provenance=model,
            validator_references=validators,
            **refs,
        ),
    )
    result = c.runtime(tmp_path, tmp_path / "raw")
    assert result["ready"] == (mutation is None)
