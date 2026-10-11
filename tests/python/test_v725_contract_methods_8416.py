"""REQ-REPORT-8416 and REQ-VERIFY-8416: private evidence cannot grant missing authority."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import yaml

from carnot.reporting import v725_contract_methods as e
from carnot.reporting import v725_contract_runner as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v685_authority_lifecycle import tasks_digest
from carnot.reporting.v709_execution import child


@pytest.fixture
def full_root(tmp_path):
    """SCENARIO-REPORT-8416-AUTHORITY: synthetic complete objects exist only privately."""
    root = tmp_path / "complete"
    root.mkdir()
    tasks = []
    template = e.protocol()["observed_complete_tasks"][0]
    for i in range(13):
        task = deepcopy(template)
        task.update(
            id=f"exp{8416 + i}-fixture",
            title=f"Private task {i}",
            phase=1,
            deliverable=f"results/experiment_{8416 + i}_fixture.json",
            gated_on=[],
        )
        tasks.append(task)
    plan = dict(milestone=e.MILESTONE, tasks=tasks)
    (root / e.ACTIVE).write_text(yaml.safe_dump(plan))
    (root / e.STAGED).write_text(yaml.safe_dump(plan))
    design = root / e.DESIGN
    design.parent.mkdir(parents=True)
    design.write_text(
        "## Exact task contract\n"
        + "\n".join(
            f"| {i + 1} | {t['id']} | {t['title']} | {t['phase']} | {t['deliverable']} |"
            for i, t in enumerate(tasks)
        )
        + "\n<!-- V725_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(plan)
        + "\n```\nCanonical full-task SHA-256: `"
        + tasks_digest(tasks)
        + "`\n"
    )
    return root


def test_actual_missing_authority_and_frozen_methods(tmp_path):
    """SCENARIO-REPORT-8416-METHODS: preserve observed four objects and every missing slot."""
    p = e.protocol()
    assert len(p["partition"]["roles"]["fit"]) == 150
    assert len(p["partition"]["roles"]["tune"]) == 50
    assert len(p["partition"]["roles"]["test"]) == 100
    assert p["feedback"]["blocks"] == [
        p["partition"]["roles"]["test"][i : i + 20] for i in range(0, 100, 20)
    ]
    assert p["feedback"]["capacities"] == [8, 16]
    assert p["feedback"]["seeds"] == [17, 29, 43, 71, 101]
    assert len(p["cost"]["cells"]) == 7
    assert p["independent_evaluation"]["ready_score"] == 0
    c = e.authority(e.ROOT, tmp_path / "authority")
    assert not c["activated"] and len(c["contract_rows"]) == 13
    assert c["observed_count"] == 4
    assert sum(row["status"] == "unstarted" for row in c["contract_rows"]) == 9
    assert any(g["observed"] is None for g in c["gate_check_summary"])
    assert e.runtime(e.ROOT)["ready"] is False
    assert e.authority(tmp_path / "missing", tmp_path / "missing-raw")["observed_count"] == 0


@pytest.mark.parametrize(
    "mutation",
    ["none", "drop", "reorder", "prompt", "model", "gate", "retirement", "routing", "digest"],
)
def test_full_object_mutations(full_root, tmp_path, mutation):
    """SCENARIO-REPORT-8416-AUTHORITY: the complete object is the authority unit."""
    active = full_root / e.ACTIVE
    plan = yaml.safe_load(active.read_text())
    if mutation == "drop":
        plan["tasks"].pop()
    if mutation == "reorder":
        plan["tasks"][0:2] = plan["tasks"][1::-1]
    if mutation == "prompt":
        plan["tasks"][0]["prompt"] += " changed"
    if mutation == "model":
        plan["tasks"][0]["MODEL_SPECS"] = [{"name": "changed"}]
    if mutation == "gate":
        plan["tasks"][1]["gated_on"] = [
            dict(upstream="missing", artifact_field="ready", op="==", value=1)
        ]
    if mutation == "retirement":
        plan["tasks"][0]["prior_failures"][0]["retire_if_same_verdict"] = False
    if mutation == "routing":
        plan["tasks"][0]["model"] = "changed"
    if mutation == "digest":
        design = full_root / e.DESIGN
        design.write_text(design.read_text().replace(tasks_digest(plan["tasks"]), "0" * 64))
    active.write_text(yaml.safe_dump(plan))
    c = e.authority(full_root, tmp_path / "raw")
    assert c["activated"] == (mutation == "none")
    assert len(c["contract_rows"]) == 13


def test_build_and_replay_controls(tmp_path):
    """SCENARIO-VERIFY-8416-REPLAY: rebuilding hashes cannot forge a scientific or authority claim."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw)
    receipt = child("actual", [sys.executable, "-c", "print('real child')"], raw / "checks")
    value = e.build(work, [receipt], raw, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert value["required_checks_passed"]
    assert value["current_contract_ready_score"] == value["methods_ready_score"] == 0
    assert not any(value["model_invocation_counts"].values())
    assert set(value) <= set(value["field_principles"])
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert e.replay(candidate)
    for field in [
        "completed_count",
        "runtime_repair_ready_score",
        "independent_generalization_score",
    ]:
        bad = deepcopy(value)
        bad[field] += 1
        bad["reproducibility_checksum"] = e.checksum(bad)
        atomic_json(candidate, bad)
        assert not e.replay(candidate)
    altered = deepcopy(work)
    altered["contract"]["contract_rows"][0]["checks"]["independent_design_contract"] = True
    atomic_json(raw / "measurement.json", altered)
    atomic_json(candidate, e.build(altered, [receipt], raw, tmp_path / (e.NAME + ".json")))
    assert not e.replay(candidate)
    assert not e.replay(tmp_path / "absent")
    receipt["passed"] = False
    assert (
        e.build(work, [receipt], raw, tmp_path / (e.NAME + ".json"))["verdict_class"]
        == "disqualified"
    )


def test_real_private_cli_and_consumers(tmp_path):
    """SCENARIO-VERIFY-8416-REPLAY: execute actual publication and fresh child failures privately."""
    output = tmp_path / (e.NAME + ".json")
    base = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    checks = [
        (["--date", "20261011", "--private-control", "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "missing")], 1),
        (["--deliberate-error"], 7),
        (["--date", "20261010"], 2),
        (["--private-control"], 2),
    ]
    for i, (args, expected) in enumerate(checks):
        done = child(f"cli-{i}", base + args, tmp_path / "logs", expected=expected, deadline=90)
        assert done["passed"], Path(done["stderr_path"]).read_text()
    assert e.replay(output)
    value = json.loads(output.read_bytes())
    assert Path(value["terminal_validation_sidecar_path"]).is_file()
    assert r.manifest(tmp_path / "manifest")


@pytest.mark.parametrize(
    "mode", ["success", "no_change", "context", "api", "duration", "hash", "authority"]
)
def test_supplied_runtime_only(tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8416-REPLAY: only causal same-context copy/free evidence can admit the canary."""
    monkeypatch.setattr(e.cuda, "inventory", lambda *a: pytest.fail("no inventory permitted"))
    monkeypatch.setattr(e.cuda, "probe", lambda *a: pytest.fail("no probe permitted"))
    ops = tmp_path / "ops"
    ops.mkdir()
    context = dict(
        boot_identity=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        namespaces={p.name: str(p.readlink()) for p in Path("/proc/self/ns").iterdir()},
    )
    current = dict(kernel_driver="after", libraries=[], **context)
    previous = dict(current, kernel_driver="before" if mode != "no_change" else "after")
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
    )
    if mode == "api":
        transcript["api_returns"]["cuMemFree"] = 999
    if mode == "context":
        transcript["execution_context"] = {}
    if mode == "duration":
        transcript["duration_s"] = 31
    refs = {}
    for name, value in [("current", current), ("previous", previous), ("transcript", transcript)]:
        path = ops / (name + ".json")
        atomic_json(path, value)
        refs[name + "_reference"] = e.reference(path)
    receipt = dict(
        authority="operator",
        previous_primary_sha256=e.cuda.CACHED_PIN,
        executed_at_ns=e.time.time_ns() - 1,
        frontier_wall_ns=0,
        action="driver_repair",
        description="synthetic private receipt control only",
        execution_context=context,
        **refs,
    )
    if mode == "hash":
        receipt["current_reference"]["sha256"] = "sha256:" + "0" * 64
    if mode == "authority":
        receipt["authority"] = "untrusted"
    atomic_json(ops / "cuda-runtime-repair-receipt.json", receipt)
    result = e.runtime(tmp_path)
    assert result["ready"] == (mode == "success")
    assert result["current_device_probe_count"] == 0
    assert result["inspection_duration_s"] < 30


def test_resource_and_seal_failures(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8416-REPLAY: genuine resource observations and fixed seals precede readiness."""
    original = e.resources
    monkeypatch.setattr(
        e, "resources", lambda raw: dict(original(raw), private_disk_backed_scratch=False)
    )
    work = e.measure(tmp_path / "absent", tmp_path / "raw")
    assert work["failures"][0]["artifact_field"] == "private_disk_memory_tools_budget"
    original_hash = e.sha256_file
    frozen_path = Path(e.protocol()["source_references"][0]["path"])
    monkeypatch.setattr(
        e, "sha256_file", lambda p: "changed" if p == e.ROOT / e.METHODS else original_hash(p)
    )
    with pytest.raises(ValueError, match="methods_hash"):
        e.protocol()
    monkeypatch.setattr(
        e,
        "sha256_file",
        lambda p: "changed" if p == frozen_path else original_hash(p),
    )
    with pytest.raises(ValueError, match="frozen_source_hash"):
        e.measure(e.ROOT, tmp_path / "bad-seal")


def test_complete_contract_optional_absence_and_real_failure(full_root, tmp_path):
    """SCENARIO-REPORT-8416-AUTHORITY: optional CUDA absence preserves complete private custody."""
    raw = tmp_path / "qualified"
    work = e.measure(full_root, raw)
    actual = child("pass", [sys.executable, "-c", "print('actual success')"], raw / "logs")
    value = e.build(work, [actual], raw, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "circular_positive"
    assert value["methods_ready_score"] == 1 and value["runtime_repair_ready_score"] == 0
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert e.replay(candidate)
    failed = child("fail", [sys.executable, "-c", "raise SystemExit(3)"], raw / "logs")
    assert (
        e.build(work, [failed], raw, tmp_path / (e.NAME + ".json"))["verdict_class"]
        == "disqualified"
    )


def test_normal_runner_real_coverage_shards(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8416-REPLAY: an actual child writes coverage bytes on the normal runner path."""

    def plan(private):
        code = (
            "import coverage;from carnot.reporting import v725_contract_methods as e; c=coverage.Coverage(config_file=False,data_file="
            + repr(str(private / ".coverage.control"))
            + "); c.start();e.progress('actual_child_coverage');c.stop();c.save();c.json_report(outfile="
            + repr(str(private / "coverage.json"))
            + ")"
        )
        return [
            dict(
                name="actual_coverage",
                argv=[sys.executable, "-u", "-c", code],
                expected=0,
                deadline=30,
                scope="owned",
            )
        ]

    monkeypatch.setattr(r, "manifest", plan)
    output = tmp_path / (e.NAME + ".json")
    assert r.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and e.replay(output)
    raw = Path(value["primitive_reference"]["path"]).parent
    assert list((raw / "coverage_shards").glob(".coverage*"))


@pytest.mark.parametrize("mode", ["protocol", "source_hash", "receipt", "log", "runtime"])
def test_cold_replay_operand_failures(tmp_path, mode):
    """SCENARIO-VERIFY-8416-REPLAY: self-consistent outer hashes cannot authorize changed primitives."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw)
    receipt = child("actual", [sys.executable, "-c", "print('actual primitive log')"], raw / "logs")
    if mode == "protocol":
        work["protocol"]["partition"]["reseeding"] = True
    if mode == "runtime":
        work["runtime"]["ready"] = True
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, [receipt], raw, tmp_path / (e.NAME + ".json"))
    if mode == "source_hash":
        value["code_config_hashes"][0]["sha256"] = "sha256:" + "0" * 64
    if mode == "receipt":
        value["validation_receipts"][0]["duration_s"] += 1
    if mode == "log":
        Path(receipt["stdout_path"]).write_text("forged primitive log")
    value["reproducibility_checksum"] = e.checksum(value)
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert not e.replay(candidate)
