"""REQ-REPORT-8904; SCENARIO-REPORT-8904-CONTRACT: independent authority only."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest
import yaml

from carnot.reporting import v762_branch_contract as c


def tree(root):
    """Copy independently supplied design bytes, never synthesize authority from YAML."""
    for name in [c.DESIGN, c.ACTIVE, c.METHODS]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((c.ROOT / name).read_bytes())
    return root


@pytest.fixture
def history_fixture(tmp_path, monkeypatch):
    """Small original operands keep reducer tests independent of the historical corpus."""
    source = tmp_path / "original.json"
    source.write_text(
        json.dumps(dict(verdict_class="blocked", code_config_hashes=[], raw_shard_hashes=[]))
    )
    monkeypatch.setattr(c, "PINS", {str(source): c.reference(source)["sha256"].split(":")[1]})
    return source


def validation(passed=True):
    """Synthetic receipts test reduction; publication still requires measured SQLite data."""
    return dict(receipts=[dict(passed=passed)], coverage=dict(complete=passed))


def test_authority_lifecycle_and_mutations(tmp_path):
    """REQ-REPORT-8904: bind staged authority before activation and actual authority after."""
    root = tree(tmp_path / "root")
    assert c.authority(root)["ready"]
    controls = c.mutations(root, tmp_path / "controls")
    assert len(controls) == 10 and all(r["passed"] for r in controls)
    queue = yaml.safe_load((root / c.ACTIVE).read_bytes())
    (root / c.STAGED).write_text(yaml.safe_dump(queue))
    (root / c.ACTIVE).write_text(yaml.safe_dump(dict(queue, milestone="2026.11.761", tasks=[])))
    assert c.authority(root)["ready"]
    (root / c.STAGED).unlink()
    assert not c.authority(root)["ready"]
    for task in queue["tasks"]:
        task["prompt"] = c.encode(task["prompt"])
    (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
    assert c.authority(root)["ready"]
    queue["tasks"][0]["prompt"] += "forged"
    (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
    assert not c.authority(root)["ready"]
    (root / c.DESIGN).write_text("truncated")
    assert not c.authority(root)["ready"]
    assert c.authority(tmp_path / "missing")["failures"][0]["observed"] == "missing_file"


def test_freeze_reduction_and_replay(tmp_path, monkeypatch, history_fixture):
    """SCENARIO-REPORT-8904-CONTRACT: null custody and optional absence are separate."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261107")
    value = c.build(work, validation())
    assert value["verdict_class"] == "null" and value["contract_ready_score"] == 1
    assert value["runtime_repair_ready_score"] == 0
    assert value["completed_count"] == value["intended_count"] == 13
    assert len(value["full_task_objects"]) == len(value["visible_task_table"]) == 13
    assert len(value["source_method_map"]) == 14
    assert all(r["required_field_declared"] for r in value["gate_edge_rows"])
    assert value["independent_count"] == value["numerical_training_calls"] == 0
    assert value["preserved_v761_reference"]["prior_verdicts"][str(history_fixture)] == "blocked"
    path = tmp_path / "candidate.json"
    c.atomic_json(path, value)
    assert not c.replay(path)["passed"]
    monkeypatch.setattr(c, "verify_measured", lambda *args: True)
    assert c.replay(path)["passed"]
    value["contract_ready_score"] = 0
    value["reproducibility_checksum"] = c.checksum(value)
    c.atomic_json(path, value)
    assert not c.replay(path)["passed"]
    assert c.build(work, validation(False))["verdict_class"] == "disqualified"
    work["authority_mutation_rows"][0]["passed"] = False
    assert c.build(work, validation())["verdict_class"] == "disqualified"
    assert not c.replay(tmp_path / "missing")["passed"]


def test_blocked_inputs_and_raw_drift(tmp_path, history_fixture):
    """REQ-REPORT-8904: missing independent authority blocks and retains thirteen slots."""
    work = c.freeze(tmp_path / "missing", tmp_path / "raw", "20261107")
    assert not work["contract"]["ready"]
    assert len(work["contract"]["rows"]) == 13
    assert not work["methods"]
    assert c.build(work, validation())["verdict_class"] == "blocked"
    checks = validation(False)
    c.atomic_json(Path(work["raw"]) / "validation.json", checks)
    value = c.build(work, checks)
    value["raw_shard_hashes"].append(c.reference(Path(work["raw"]) / "validation.json"))
    value["reproducibility_checksum"] = c.checksum(value)
    path = tmp_path / "candidate.json"
    c.atomic_json(path, value)
    assert c.replay(path)["passed"]
    Path(work["snapshots"][0]["path"]).write_text("changed raw bytes")
    assert not c.replay(path)["passed"]


def test_history_failures_and_rehashed_authority(tmp_path, monkeypatch, history_fixture):
    """REQ-REPORT-8904: original code/raw drift and rehashed authority remain falsifiable."""
    operand = tmp_path / "operand.json"
    operand.write_text("original")
    history_fixture.write_text(
        json.dumps(
            dict(
                verdict_class="blocked",
                code_config_hashes=[c.reference(operand)],
                raw_shard_hashes=[],
            )
        )
    )
    monkeypatch.setattr(
        c, "PINS", {str(history_fixture): c.reference(history_fixture)["sha256"].split(":")[1]}
    )
    assert c.history(tmp_path / "sealed")["authenticated"]
    operand.write_text("drift")
    assert not c.history(tmp_path / "drift")["authenticated"]
    monkeypatch.setattr(c, "PINS", {"absent.json": "0" * 64})
    assert not c.history(tmp_path / "absent")["authenticated"]
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261107")
    work["contract"]["ready"] = False
    c.atomic_json(
        Path(work["raw_reference"]["path"]), {k: v for k, v in work.items() if k != "raw_reference"}
    )
    work["raw_reference"] = c.reference(Path(work["raw_reference"]["path"]))
    path = tmp_path / "candidate.json"
    c.atomic_json(path, c.build(work, validation(False)))
    assert not c.replay(path)["passed"]


def test_cli_dispatch(tmp_path, monkeypatch):
    """REQ-REPORT-8904: actual script dispatch supports private output and date rejection."""
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


def test_primary_source_hash_and_immutable_invocation(tmp_path, history_fixture, monkeypatch):
    """REQ-REPORT-8904: rejected source bytes block; an invocation cannot overwrite evidence."""
    root = tree(tmp_path / "root")
    sources = json.loads((root / c.METHODS).read_bytes())
    source = tmp_path / "paper.html"
    source.write_text("original primary bytes")
    sources[0]["sources"][0] = dict(path=str(source), sha256="sha256:" + "0" * 64)
    (root / c.METHODS).write_text(json.dumps(sources))
    archive = tmp_path / "archive.md"
    archive.write_text("preserved truncated document")
    monkeypatch.setitem(c.PINS, str(archive), c.reference(archive)["sha256"].split(":")[1])
    work = c.freeze(root, tmp_path / "raw", "20261107")
    assert not work["history"]["authenticated"]
    assert work["history"]["failures"][0]["observed"] == "rejected_producer"
    assert c.build(work, validation())["verdict_class"] == "blocked"
    with pytest.raises(FileExistsError):
        c.freeze(root, tmp_path / "raw", "20261107")


def test_missing_primary_source_retains_blocker(tmp_path, history_fixture):
    """REQ-REPORT-8904: missing external source bytes produce a terminal blocker."""
    root = tree(tmp_path / "root")
    sources = json.loads((root / c.METHODS).read_bytes())
    sources[0]["sources"][0] = dict(
        path=str(tmp_path / "missing.html"), sha256="sha256:" + "0" * 64
    )
    (root / c.METHODS).write_text(json.dumps(sources))
    work = c.freeze(root, tmp_path / "raw", "20261107")
    assert work["history"]["failures"][0]["observed"] == "missing_file"
    assert c.build(work, validation())["verdict_class"] == "blocked"


@pytest.mark.parametrize("content", [None, "truncated json"])
def test_missing_or_truncated_method_map(tmp_path, history_fixture, content):
    """REQ-REPORT-8904: a missing method authority has its own named blocker."""
    root = tree(tmp_path / "root")
    path = root / c.METHODS
    if content is None:
        path.unlink()
    else:
        path.write_text(content)
    work = c.freeze(root, tmp_path / "raw", "20261107")
    value = c.build(work, validation())
    assert value["verdict_class"] == "blocked"
    assert any(r["check"] == "method_map" for r in value["gate_check_summary"])


def test_original_recorded_absence_is_preserved(tmp_path, monkeypatch, history_fixture):
    """REQ-REPORT-8904: an original null hash is an absence assertion, never source bytes."""
    original = dict(
        verdict_class="blocked",
        code_config_hashes=[],
        raw_shard_hashes=[c.reference(tmp_path / "originally_absent.yaml")],
    )
    history_fixture.write_text(json.dumps(original))
    monkeypatch.setattr(
        c, "PINS", {str(history_fixture): c.reference(history_fixture)["sha256"].split(":")[1]}
    )
    result = c.history(tmp_path / "raw")
    assert result["authenticated"]
    assert len(result["asserted_absence_rows"]) == 1
    assert result["asserted_absence_rows"][0]["sha256"] is None


def test_runtime_receipt_rejections(tmp_path):
    """REQ-REPORT-8904: absent, malformed and stale receipts cannot admit runtime."""
    raw = tmp_path / "raw"
    assert not c.runtime(tmp_path, raw)["ready"]
    path = tmp_path / "ops/cuda-runtime-repair-receipt.json"
    path.parent.mkdir()
    for data in ["broken", "{}", '{"run_date":"20261106"}']:
        path.write_text(data)
        result = c.runtime(tmp_path, raw)
        assert not result["ready"] and result["gate_check_summary"]
        assert result["device_probes"] == result["model_loads"] == 0


def test_literature_and_snapshot_fields(tmp_path, history_fixture):
    """REQ-REPORT-8904: method ingestion declares local falsifiers and no paper benefit."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261107")
    result = c.build(work, validation())
    assert result["contract_rows"] == result["rows"]
    assert result["execution_snapshot"] and result["full_task_digest"]
    assert result["literature_method_rows"] == result["source_method_map"]
    for topic in [
        "continual_online_learning",
        "neural_constraint_satisfaction",
        "hallucination_and_sequential_risk",
    ]:
        row = next(r for r in result["literature_method_rows"] if r.get("topic") == topic)
        assert row["falsification"] and row["nontransferable_assumptions"]
        assert row["paper_results_inherited"] is False


def test_formatting_preserves_executed_source(tmp_path, history_fixture):
    """REQ-REPORT-8904: qualify formatting without changing the failed producer's logic."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261107")
    assert work["preconditions"]["formatting_only_repair"]
    assert work["preconditions"]["formatting_repair_receipt"]["passed"]
    work["preconditions"]["formatting_only_repair"] = False
    assert c.build(work, validation())["verdict_class"] == "disqualified"


@pytest.mark.parametrize(
    "mutation",
    [None, "unchanged", "model", "validator", "transcript", "offload", "operand", "date"],
)
def test_supplied_runtime_custody(tmp_path, monkeypatch, mutation):
    """REQ-REPORT-8904: synthetic receipts exercise each admission gate without GPU access."""
    from carnot.reporting import v725_contract_methods as old
    import time

    context = dict(
        boot_identity=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        namespaces={p.name: str(p.readlink()) for p in Path("/proc/self/ns").iterdir()},
    )
    model_path = tmp_path / c.runtime_rules.MODEL_REVISION / "model.gguf"
    model_path.parent.mkdir()
    model_path.write_bytes(b"private synthetic GGUF identity control")
    monkeypatch.setattr(c.runtime_rules, "MODEL_PIN", c.runtime_rules.sha256_file(model_path))
    model = dict(
        c.reference(model_path),
        hf_id="unsloth/Qwen3.8-27B-GGUF",
        revision=c.runtime_rules.MODEL_REVISION,
    )
    if mutation == "model":
        model["revision"] = "0" * 40
    validators = [
        dict(source_path=p, sha256=c.runtime_rules.sha256_file(c.ROOT / p))
        for p in c.runtime_rules.RUNTIME_VALIDATORS
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
            run_date="20261105" if mutation == "date" else "20261107",
            mandated_model_provenance=model,
            validator_references=validators,
            **refs,
        ),
    )
    result = c.runtime(tmp_path, tmp_path / "raw")
    assert result["ready"] == (mutation is None)


def test_authenticated_primary_source_snapshot(tmp_path, history_fixture):
    """REQ-REPORT-8904: an imported primary source is sealed only after byte authentication."""
    root = tree(tmp_path / "root")
    path = root / c.METHODS
    rows = json.loads(path.read_bytes())
    source = tmp_path / "primary.html"
    source.write_text("independent primary bytes")
    rows[0]["sources"][0] = c.reference(source)
    path.write_text(json.dumps(rows))
    work = c.freeze(root, tmp_path / "raw", "20261107")
    assert any(r["source"] == c.reference(source) for r in work["snapshots"])
    assert c.build(work, validation())["contract_ready_score"] == 1
