"""REQ-REPORT-8866; SCENARIO-REPORT-8866-CONTRACT: preserve missing authority."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest
import yaml

from carnot.reporting import v759_branch_contract as c


def tree(root, complete=True):
    """Synthetic private designs test equality; they are never research evidence."""
    queue = yaml.safe_load((c.ROOT / c.ACTIVE).read_bytes())
    (root / c.ACTIVE).parent.mkdir(parents=True, exist_ok=True)
    (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
    text = (c.ROOT / c.DESIGN).read_text().split("## Exact task contract")[0]
    if complete:
        text += "\n## Exact task contract\n"
        for i, task in enumerate(queue["tasks"], 1):
            text += f"| {i} | {task['id']} | {task['title']} | {task['phase']} | {task['deliverable']} |\n"
        text += f"Canonical task SHA-256: `{c.PIN}`\n"
        text += "<!-- V759_TASK_CONTRACT_START -->\n```json\n"
        text += json.dumps(dict(milestone=c.MILESTONE, tasks=queue["tasks"]))
        text += "\n```\n<!-- V759_TASK_CONTRACT_END -->\n"
    path = root / c.DESIGN
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return root


def validation(passed=True):
    """Synthetic receipts exercise reduction; actual publication checks SQLite custody."""
    return dict(receipts=[dict(passed=passed)], coverage=dict(complete=passed))


def test_authority_and_original_mutations(tmp_path):
    """SCENARIO-REPORT-8866-CONTRACT: all original controls reject private corruption."""
    root = tree(tmp_path / "source")
    assert c.authority(root)["ready"]
    rows = c.mutations(root, tmp_path / "controls")
    assert len(rows) == 7 and all(r["passed"] for r in rows)
    queue = yaml.safe_load((root / c.ACTIVE).read_text())
    (root / c.STAGED).write_text(yaml.safe_dump(queue))
    (root / c.ACTIVE).write_text(yaml.safe_dump(dict(queue, milestone="2026.11.758")))
    assert c.authority(root)["ready"]
    (root / c.STAGED).unlink()
    queue["tasks"][0]["prompt"] = c.encode(queue["tasks"][0]["prompt"])
    (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
    assert c.authority(root)["ready"]


def test_missing_authority_retains_twelve_units(tmp_path):
    """REQ-REPORT-8866: absent external JSON must not be reconstructed from the queue."""
    result = c.authority(tree(tmp_path, False))
    assert not result["ready"] and len(result["rows"]) == 12
    assert all(r["status"] == "censored" for r in result["rows"])
    assert result["failures"][0]["observed"] == "missing_field"
    assert c.authority(tmp_path / "absent")["failures"][0]["observed"] == "missing_file"


def test_build_and_cold_replay(tmp_path, monkeypatch):
    """REQ-REPORT-8866: optional absence stays independent of complete CPU authority."""
    work = c.freeze(tree(tmp_path / "source"), tmp_path / "raw", "20261106")
    value = c.build(work, validation())
    assert value["contract_ready_score"] == 1 and value["verdict_class"] == "null"
    assert value["runtime_repair_ready_score"] == 0
    assert value["intended_count"] == value["completed_count"] == 12
    assert value["arc_task_count"] == value["generator_training_task_count"] == 0
    assert len(value["source_method_map"]) == 6
    assert all(r["required_field_declared"] for r in value["gate_producer_rows"])
    assert value["preserved_v758_reference"]["verdict_class"] == "null"
    path = tmp_path / "candidate.json"
    c.atomic_json(path, value)
    assert not c.replay(path)["passed"]
    monkeypatch.setattr(c, "verify_measured", lambda *args: True)
    assert c.replay(path)["passed"]
    value["contract_ready_score"] = 0
    value["reproducibility_checksum"] = c.checksum(value)
    c.atomic_json(path, value)
    assert not c.replay(path)["passed"]
    assert not c.replay(tmp_path / "absent")["passed"]
    assert c.build(work, validation(False))["verdict_class"] == "disqualified"
    work["authority_mutation_rows"][0]["passed"] = False
    assert c.build(work, validation())["verdict_class"] == "disqualified"


def test_blocked_publication_and_raw_drift(tmp_path):
    """REQ-REPORT-8866: incomplete external authority is terminal blocked with zero benefit."""
    work = c.freeze(tree(tmp_path / "source", False), tmp_path / "raw", "20261106")
    checks = validation(False)
    c.atomic_json(tmp_path / "raw/validation.json", checks)
    value = c.build(work, checks)
    value["raw_shard_hashes"].append(c.reference(tmp_path / "raw/validation.json"))
    value["reproducibility_checksum"] = c.checksum(value)
    path = tmp_path / "candidate.json"
    c.atomic_json(path, value)
    assert c.replay(path)["passed"]
    assert c.build(work, validation())["verdict_class"] == "blocked"
    Path(work["snapshots"][0]["path"]).write_text("drift")
    assert not c.replay(path)["passed"]


def test_cli_dispatch(tmp_path, monkeypatch):
    """REQ-REPORT-8866: actual CLI dispatch retains private paths and rejects bad dates."""
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


def test_history_and_sealed_authority_failures(tmp_path, monkeypatch):
    """REQ-REPORT-8866: external operand absence and rehashed authority are rejected."""
    root = tree(tmp_path / "root")
    work = c.freeze(root, tmp_path / "raw", "20261106")
    work["contract"]["ready"] = False
    c.atomic_json(
        Path(work["raw_reference"]["path"]), {k: v for k, v in work.items() if k != "raw_reference"}
    )
    work["raw_reference"] = c.reference(Path(work["raw_reference"]["path"]))
    value = c.build(work, validation(False))
    path = tmp_path / "candidate.json"
    c.atomic_json(path, value)
    assert not c.replay(path)["passed"]
    monkeypatch.setattr(c, "PINS", {"absent.json": "0" * 64, c.ARCHIVE: "0" * 64})
    assert not c.history(tmp_path / "missing")["authenticated"]
    monkeypatch.setattr(
        c, "PINS", {c.PRIOR: "772046dd8cd218673e9882e232faf9fd5bda0de7ad257f7493564e7591b99c87"}
    )
    original = c.reference
    monkeypatch.setattr(
        c,
        "reference",
        lambda p: (
            dict(original(p), sha256="forged")
            if p.name.endswith("sealed_work.json")
            else original(p)
        ),
    )
    assert not c.history(tmp_path / "changed")["authenticated"]


def test_absent_design_and_runtime_self_report(tmp_path):
    """REQ-REPORT-8866: external absence retains twelve censored observations."""
    work = c.freeze(tmp_path / "absent", tmp_path / "raw", "20261106")
    assert not work["contract"]["ready"] and not work["methods"]
    path = tmp_path / "ops/cuda-runtime-repair-receipt.json"
    c.atomic_json(path, dict(ready=True))
    assert not c.runtime(tmp_path, tmp_path / "runtime")["ready"]


@pytest.mark.parametrize(
    "mutation",
    [None, "unchanged", "model", "validator", "transcript", "offload", "operand", "date"],
)
def test_supplied_runtime_custody(tmp_path, monkeypatch, mutation):
    """REQ-REPORT-8866: synthetic receipts exercise each admission gate without GPU access."""
    from carnot.reporting import v725_contract_methods as old
    import time

    context = dict(
        boot_identity=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        namespaces={p.name: str(p.readlink()) for p in Path("/proc/self/ns").iterdir()},
    )
    model_path = tmp_path / c.prior.MODEL_REVISION / "model.gguf"
    model_path.parent.mkdir()
    model_path.write_bytes(b"private synthetic GGUF identity control")
    monkeypatch.setattr(c.prior, "MODEL_PIN", c.prior.sha256_file(model_path))
    model = dict(
        c.reference(model_path), hf_id="unsloth/Qwen3.8-27B-GGUF", revision=c.prior.MODEL_REVISION
    )
    if mutation == "model":
        model["revision"] = "0" * 40
    validators = [
        dict(source_path=p, sha256=c.prior.sha256_file(c.ROOT / p))
        for p in c.prior.RUNTIME_VALIDATORS
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
            run_date="20261105" if mutation == "date" else "20261106",
            mandated_model_provenance=model,
            validator_references=validators,
            **refs,
        ),
    )
    result = c.runtime(tmp_path, tmp_path / "raw")
    assert result["ready"] == (mutation is None)
