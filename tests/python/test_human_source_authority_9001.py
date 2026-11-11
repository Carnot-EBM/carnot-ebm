"""REQ-REPORT-9001, REQ-VERIFY-9001: original humans cannot make exposed sources unseen."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from carnot.reporting import human_source_authority_9001 as e
from carnot.reporting import human_source_runner_9001 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def records(count: int = 600) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Distinct private contexts provide both classes without becoming measurements."""
    sources, responses = [], []
    for i in range(count):
        sid = str(i)
        sources.append(
            dict(
                source_id=sid,
                task_type="Summary",
                source="private",
                source_info=f"document{i} unique{i}",
                prompt=f"summarize{i}",
            )
        )
        responses.append(
            dict(
                id=sid,
                source_id=sid,
                response="é truth",
                quality="good",
                split="train" if i < 400 else "test",
                labels=[]
                if i % 2
                else [
                    dict(
                        start=2,
                        end=7,
                        text="truth",
                        label_type="Evident Baseless Info",
                        implicit_true=True,
                        meta="private human control",
                    )
                ],
            )
        )
    return sources, responses


def inputs(tmp_path: Path) -> Path:
    """Only explicit private scope may bypass the immutable author hashes."""
    sources, responses = records()
    files = {}
    for name, data in [("sources", sources), ("responses", responses)]:
        p = tmp_path / (name + ".jsonl")
        p.write_text("".join(json.dumps(v) + "\n" for v in data))
        files[name] = e.ref(p)
    policy = tmp_path / "README.md"
    policy.write_text(e.MANUAL_POLICY)
    license_path = tmp_path / "LICENSE"
    license_path.write_text("MIT License")
    files.update(policy=e.ref(policy), license=e.ref(license_path))
    exposure = tmp_path / "prior.jsonl"
    exposure.write_text(json.dumps(dict(source_id="0")) + "\n")
    p = tmp_path / "inputs.json"
    atomic_json(
        p,
        dict(
            scope="private_fixture",
            revision=e.REVISION,
            files=files,
            exposure_witnesses=[e.ref(exposure)],
            source_access_receipts=[],
        ),
    )
    return p


def validation() -> dict[str, Any]:
    """Fake receipts test policy only; the delivered run uses actual child receipts."""
    return dict(
        checks=[dict(name=n, passed=True, normal_exit=True, exit_code=0) for n in e.CHECKS],
        owned_statement_coverage=dict(percent_covered=100, missing_lines=[], excluded_lines=[]),
    )


def cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Exercise the real entry point and collect child coverage from a foreign directory."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI), *args]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    return subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)


def test_original_selection_and_sealed_labels(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9001-ORIGINAL: select before labels and retain every sensitivity."""
    sources, responses = records()
    result = e.audit(sources, responses, set(), True, tmp_path)
    assert len(result["role_manifest"]) == 600
    assert result["natural_pilot_ready_score"] == result["independent_label_ready_score"] == 1
    assert result["class_counts"]["test"]["1"] >= 20
    assert all(
        "labels" not in row and "primary_error" not in row for row in result["role_manifest"]
    )
    before = result["role_manifest"]
    for response in responses:
        response["labels"] = []
    changed = e.audit(sources, responses, set(), True, tmp_path / "changed")
    assert changed["role_manifest"] == before
    assert changed["natural_pilot_ready_score"] == 0
    assert changed["independent_generalization_score"] == 0


def test_exposure_and_overlap(tmp_path: Path) -> None:
    """REQ-REPORT-9001: shared documents transitively merge before exposure subtraction."""
    sources, responses = records(4)
    common = " ".join(f"word{i}" for i in range(40))
    sources[0]["source_info"] = common
    sources[1]["task_type"] = "QA"
    sources[1]["source_info"] = dict(question="why", passages="passage 1:" + common + " extra")
    sources[2]["source_info"] = "other " + common
    sources[3]["task_type"] = "Data2txt"
    result = e.audit(sources, responses, {"0"}, True, tmp_path)
    assert len(result["source_group_rows"]) == 1
    assert set(result["source_group_rows"][0]["source_ids"]) == {"0", "1", "2"}
    assert result["independent_count"] == 0 and not result["role_manifest"]
    assert result["excluded_count"] == 1
    assert result["gate_check_summary"][0]["observed"] == 0
    assert e.audit([], [], set(), False, tmp_path / "empty")["independent_label_ready_score"] == 0


@pytest.mark.parametrize(
    "mutation",
    ["quality", "duplicate_source", "duplicate_response", "offset", "orphan", "split", "empty"],
)
def test_rejections(tmp_path: Path, mutation: str) -> None:
    """SCENARIO-REPORT-9001-ORIGINAL: invalid original records cannot grant readiness."""
    sources, responses = records(2)
    if mutation == "quality":
        responses[0]["quality"] = "truncated"
        result = e.audit(sources, responses, set(), True, tmp_path)
        assert result["quality_exclusions"][0]["quality"] == "truncated"
        return
    if mutation == "duplicate_source":
        sources.append(sources[0])
    if mutation == "duplicate_response":
        responses.append(responses[0])
    if mutation == "offset":
        responses[0]["labels"][0]["start"] = 20
    if mutation == "orphan":
        responses[0]["source_id"] = "missing"
    if mutation == "split":
        responses[0]["split"] = "unknown"
    if mutation == "empty":
        responses[0]["response"] = ""
    with pytest.raises(ValueError):
        e.audit(sources, responses, set(), True, tmp_path)


def test_private_cli_missing_and_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9001-CUSTODY: real publication, missing input and rehashed tamper."""
    manifest = inputs(tmp_path)
    receipts = tmp_path / "receipts.json"
    atomic_json(receipts, validation())
    output = tmp_path / (e.NAME + ".json")
    run = cli(
        tmp_path, "--inputs", str(manifest), "--receipts", str(receipts), "--output", str(output)
    )
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["natural_pilot_ready_score"] == 0
    assert cli(tmp_path, "--cold-replay", str(output)).returncode == 0
    value["completed_count"] += 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    rejected = cli(tmp_path, "--cold-replay", str(output))
    assert rejected.returncode == 1 and "reduction_drift" in rejected.stdout
    missing = cli(
        tmp_path,
        "--inputs",
        str(tmp_path / "missing"),
        "--receipts",
        str(receipts),
        "--output",
        str(output),
    )
    assert missing.returncode == 0, missing.stdout + missing.stderr
    value = json.loads(output.read_bytes())
    assert value["honest_verdict"] == "complete_blocked_authenticated_inputs"
    assert "missing_file" in str(value["gate_check_summary"])
    assert cli(tmp_path, "--date", "20261131").returncode == 1
    assert cli(tmp_path, "--date", "2026111").returncode == 1


def test_custody_and_owned_failure(tmp_path: Path) -> None:
    """REQ-VERIFY-9001: absent authority blocks; private evidence and failed checks cannot qualify."""
    manifest = inputs(tmp_path)
    result = e.inspect(manifest, tmp_path / "audit")
    assert result["input_scope"] == "private_fixture"
    info = json.loads(manifest.read_bytes())
    original = deepcopy(info)
    info["revision"] = "mutable"
    atomic_json(manifest, info)
    with pytest.raises(ValueError, match="revision"):
        e.inspect(manifest, tmp_path / "bad")
    info = deepcopy(original)
    info["scope"] = "author_release"
    atomic_json(manifest, info)
    with pytest.raises(ValueError, match="pinned_release"):
        e.inspect(manifest, tmp_path / "bad")
    atomic_json(manifest, original)
    source = Path(original["files"]["sources"]["path"])
    body = source.read_bytes()
    source.write_bytes(body + b" ")
    with pytest.raises(ValueError, match="changed_bytes"):
        e.inspect(manifest, tmp_path / "bad")
    source.write_bytes(body)
    reference = r.measure(manifest, tmp_path / (e.NAME + ".json"), "20261111")
    failed = r.build(reference, {})
    assert failed["verdict_class"] == "disqualified"
    assert failed["independent_label_ready_score"] == failed["natural_pilot_ready_score"] == 0
    output = tmp_path / (e.NAME + ".json")
    value = r.build(reference, validation())
    atomic_json(output, value)
    assert r.replay(output)["passed"]
    source.write_bytes(body + b" ")
    with pytest.raises(ValueError):
        r.replay(output)
    source.write_bytes(body)
    primitive = r.read_ref(reference)
    primitive["code_config_hashes"][0]["sha256"] = "wrong"
    atomic_json(Path(reference["path"]), primitive)
    value = r.build(e.ref(Path(reference["path"])), validation())
    atomic_json(output, value)
    with pytest.raises(ValueError, match="executed_source_hash"):
        r.replay(output)


def test_replay_refuses_repaired_custody(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-9001-CUSTODY: repair of outer hashes cannot hide changed original summaries."""
    manifest = inputs(tmp_path)
    output = tmp_path / (e.NAME + ".json")
    reference = r.measure(manifest, output, "20261111")
    original = r.read_ref(reference)

    def saved(primitive: dict[str, Any], checks: dict[str, Any] | None = None) -> None:
        atomic_json(Path(reference["path"]), primitive)
        atomic_json(output, r.build(e.ref(Path(reference["path"])), checks or validation()))

    data = deepcopy(original)
    data["reduction"]["completed_count"] += 1
    saved(data)
    with pytest.raises(ValueError, match="original_reduction_drift"):
        r.replay(output)
    data = deepcopy(original)
    data["reduction"]["label_sidecar_hashes"][0]["sha256"] = "wrong"
    saved(data)
    with pytest.raises(ValueError, match="raw_shard_hash"):
        r.replay(output)
    data = deepcopy(original)
    sidecar = Path(data["reduction"]["label_sidecar_hashes"][0]["path"])
    payload = sidecar.read_bytes()
    sidecar.write_text("{}")
    data["reduction"]["label_sidecar_hashes"][0] = e.ref(sidecar)
    saved(data)
    with pytest.raises(ValueError, match="original_sidecar_drift"):
        r.replay(output)
    sidecar.write_bytes(payload)
    saved(original)
    payload = manifest.read_bytes()
    manifest.write_bytes(payload + b" ")
    with pytest.raises(ValueError, match="manifest_hash"):
        r.replay(output)
    manifest.write_bytes(payload)
    log = tmp_path / "stdout.log"
    log.write_text("original")
    checks = validation()
    checks["checks"][0].update(stdout_path=str(log), stdout_sha256=e.ref(log)["sha256"])
    saved(original, checks)
    log.write_text("changed")
    with pytest.raises(ValueError, match="validation_log_hash"):
        r.replay(output)
    monkeypatch.setattr(r.shutil, "disk_usage", lambda p: type("Disk", (), dict(free=0))())
    limited = r.read_ref(r.measure(manifest, output, "20261111"))
    assert limited["reduction"]["input_scope"] == "external_unavailable"
    sources, responses = records(2)
    sources[0]["source_info"] = ""
    with pytest.raises(ValueError, match="missing_context"):
        e.audit(sources, responses, set(), True, tmp_path / "bad")


def test_cited_authority_and_access_receipts_are_authenticated(tmp_path: Path) -> None:
    """REQ-VERIFY-9001: historical references and access logs must retain their actual bytes."""
    manifest = inputs(tmp_path)
    info = json.loads(manifest.read_bytes())
    history = tmp_path / "history.json"
    atomic_json(history, dict(experiment_id=8908, source_id="0"))
    info["cited_upstream_artifacts"] = [
        dict(e.ref(history), experiment_id=8908, fields_imported=["source_id"])
    ]
    info["custody_references"] = [e.ref(history)]
    atomic_json(manifest, info)
    assert e.inspect(manifest, tmp_path / "good")["cited_upstream_artifacts"]
    history.write_text("{}")
    with pytest.raises(ValueError, match="changed_bytes"):
        e.inspect(manifest, tmp_path / "bad")
