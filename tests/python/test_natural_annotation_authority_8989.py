"""REQ-REPORT-8989: public machine labels must not become human authority."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from carnot.reporting import natural_annotation_authority_8989 as e
from carnot.reporting import natural_annotation_runner_8989 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def record() -> dict[str, Any]:
    """Use a small Unicode control to distinguish characters from UTF8 bytes."""
    response = dict(
        text="é truth",
        hallucinations=[dict(start=2, end=7, text="truth", valid=True)],
        details=dict(
            output="é truth",
            token_starts=[0, 1, 2],
            annotations=dict(
                original_output=dict(partially_hallucinated=True),
                result=dict(
                    all_valid=True, hallucinations=[dict(start=2, end=7, text="truth", valid=True)]
                ),
            ),
        ),
    )
    return dict(
        user_prompt_index=1,
        user_prompt="Which?",
        answerable=True,
        documents=[dict(title="Article", text="original context")],
        documents_str="original context",
        responses=dict(model=response, golden_answer="generated"),
        details=dict(
            user_prompt=dict(
                article_title="Article",
                details=dict(
                    suitable_article=dict(
                        url="https://example.invalid/article",
                        revision_id=1,
                        content="original context",
                        passage_data=[],
                    )
                ),
            )
        ),
    )


def manifest(tmp_path: Path) -> Path:
    """Keep synthetic data private and explicitly ineligible for release readiness."""
    receipt = tmp_path / "retrieval.json"
    atomic_json(receipt, {"scope": "private_fixture"})
    source = tmp_path / "split.parquet"
    pq.write_table(pa.Table.from_pylist([record()]), source)
    path = tmp_path / "manifest.json"
    atomic_json(
        path,
        dict(
            scope="private_fixture",
            retrieval_receipt=e.ref(receipt),
            release_revision=e.REVISIONS,
            files=dict(train=e.ref(source)),
            source_urls={},
            license_references={},
        ),
    )
    return path


def validation() -> dict[str, Any]:
    """Synthetic receipts test failure policy; production retains real child logs."""
    return dict(
        checks=[dict(name=n, passed=True, normal_exit=True, exit_code=0) for n in e.CHECKS],
        owned_statement_coverage=dict(percent_covered=100, missing_lines=[], excluded_lines=[]),
    )


def cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the delivered entry point outside the checkout with real child coverage."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI), *args]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    return subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)


def test_exact_spans_grouping_and_machine_authority() -> None:
    """SCENARIO-REPORT-8989-V768: variant count cannot multiply human decisions."""
    first = record()
    other = deepcopy(first)
    other["answerable"] = False
    reduced = e.audit([("train", 0, first), ("test", 0, other)], {})
    assert reduced["intended_count"] == reduced["completed_count"] == 2
    assert reduced["independent_count"] == 1
    assert (
        reduced["independent_label_ready_score"] == reduced["independent_generalization_score"] == 0
    )
    assert reduced["rows"][0]["group_id"] == reduced["rows"][1]["group_id"]
    span = reduced["token_alignment_rows"][0]["spans"][0]
    assert span["character_match"] and span["utf8_match"] and span["byte_start"] == 3
    assert all(row["cross_split"] for row in reduced["group_overlap_rows"])
    assert reduced["annotation_origin_rows"]
    assert reduced["human_review_evidence"] == "not_documented_in_inspected_release"


@pytest.mark.parametrize("mutation", ["offset", "text", "valid", "starts", "output", "empty"])
def test_invalid_alignment_is_retained(mutation: str) -> None:
    """REQ-REPORT-8989: reject alignment claims while retaining rejected evidence."""
    item = record()
    response = item["responses"]["model"]
    if mutation == "offset":
        response["hallucinations"][0]["start"] = None
    if mutation == "text":
        response["hallucinations"][0]["text"] = "wrong"
    if mutation == "valid":
        response["hallucinations"][0]["valid"] = False
    if mutation == "starts":
        response["details"]["token_starts"] = [10, -1]
    if mutation == "output":
        response["details"]["output"] = "drift"
    if mutation == "empty":
        response["text"] = ""
    result = e.audit([("test", 0, item)], {})
    assert result["token_alignment_rows"][0]["alignment_valid"] is False
    assert result["completed_count"] == 1 and result["failed_count"] == 0


def test_private_cli_replay_and_rehashed_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8989-CUSTODY: actual publication, cold replay and aggregate tamper."""
    inputs = manifest(tmp_path)
    receipts = tmp_path / "receipts.json"
    atomic_json(receipts, validation())
    output = tmp_path / (e.NAME + ".json")
    run = cli(
        tmp_path, "--inputs", str(inputs), "--receipts", str(receipts), "--output", str(output)
    )
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["honest_verdict"] == "complete_null_machine_annotation_only"
    assert value["source_access_ready_score"] == value["annotation_audit_ready_score"] == 0
    replay = cli(tmp_path, "--cold-replay", str(output))
    assert replay.returncode == 0, replay.stdout + replay.stderr
    value["completed_count"] += 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    rejected = cli(tmp_path, "--cold-replay", str(output))
    assert rejected.returncode == 1 and "reduction_drift" in rejected.stdout
    assert cli(tmp_path, "--date", "20261131").returncode == 1


def test_missing_inputs_and_owned_failure(tmp_path: Path) -> None:
    """REQ-REPORT-8989: missing external bytes block; failed owned checks disqualify."""
    missing = tmp_path / "missing.json"
    output = tmp_path / (e.NAME + ".json")
    primitive = r.measure(missing, output, "20261111")
    blocked = r.build(primitive, validation())
    assert blocked["verdict_class"] == "blocked"
    assert "missing_file" in str(blocked["gate_check_summary"])
    atomic_json(output, blocked)
    assert r.replay(output)["passed"]
    failed = r.build(primitive, {})
    assert failed["verdict_class"] == "disqualified"
    assert failed["source_access_ready_score"] == failed["annotation_audit_ready_score"] == 0
    receipts = tmp_path / "receipts.json"
    atomic_json(receipts, validation())
    assert (
        cli(
            tmp_path, "--inputs", str(missing), "--receipts", str(receipts), "--output", str(output)
        ).returncode
        == 0
    )


def test_hash_authentication_and_repaired_primitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-8989-CUSTODY: retained-byte and source-code drift fail closed."""
    inputs = manifest(tmp_path)
    info = json.loads(inputs.read_bytes())
    assert e.inspect(inputs)["intended_count"] == 1
    info["release_revision"] = {}
    atomic_json(inputs, info)
    with pytest.raises(ValueError, match="release_revision"):
        e.inspect(inputs)
    info["release_revision"] = e.REVISIONS
    info["scope"] = "author_release"
    atomic_json(inputs, info)
    with pytest.raises(ValueError, match="pinned_release"):
        e.inspect(inputs)
    info["scope"] = "private_fixture"
    atomic_json(inputs, info)
    source = Path(info["files"]["train"]["path"])
    original = source.read_bytes()
    source.write_bytes(b"tamper")
    with pytest.raises(ValueError, match="changed_bytes"):
        e.inspect(inputs)
    source.unlink()
    with pytest.raises(FileNotFoundError, match="missing_file"):
        e.inspect(inputs)
    source.write_bytes(original)
    monkeypatch.setattr(e, "PINNED", {"train": e.ref(source)["sha256"]})
    info["scope"] = "author_release"
    atomic_json(inputs, info)
    assert e.inspect(inputs)["source_access_ready_score"] == 1
    output = tmp_path / (e.NAME + ".json")
    reference = r.measure(inputs, output, "20261111")
    result = r.build(reference, validation())
    atomic_json(output, result)
    assert r.replay(output)["passed"]
    data = r.read_ref(reference)
    data["reduction"]["completed_count"] += 1
    atomic_json(Path(reference["path"]), data)
    result["primitive_reference"] = e.ref(Path(reference["path"]))
    atomic_json(output, result)
    with pytest.raises(ValueError, match="original_reduction_drift"):
        r.replay(output)


def test_resource_custody_and_validation_failure_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-8989-CUSTODY: changed code, manifest and validation logs cannot replay."""
    inputs = manifest(tmp_path)
    output = tmp_path / (e.NAME + ".json")
    reference = r.measure(inputs, output, "20261111")
    original = r.read_ref(reference)
    primitive = deepcopy(original)
    primitive["code_config_hashes"][0]["sha256"] = "wrong"
    atomic_json(Path(reference["path"]), primitive)
    result = r.build(e.ref(Path(reference["path"])), validation())
    atomic_json(output, result)
    with pytest.raises(ValueError, match="executed_source_hash"):
        r.replay(output)
    atomic_json(Path(reference["path"]), original)
    result = r.build(reference, validation())
    atomic_json(output, result)
    body = inputs.read_bytes()
    inputs.write_bytes(body + b" ")
    with pytest.raises(ValueError, match="manifest_hash"):
        r.replay(output)
    inputs.write_bytes(body)
    log = tmp_path / "check.log"
    log.write_text("original")
    checks = validation()
    checks["checks"][0].update(stdout_path=str(log), stdout_sha256=e.ref(log)["sha256"])
    result = r.build(reference, checks)
    atomic_json(output, result)
    assert r.replay(output)["passed"]
    log.write_text("tampered")
    with pytest.raises(ValueError, match="validation_log_hash"):
        r.replay(output)
    monkeypatch.setattr(r.shutil, "disk_usage", lambda p: type("Disk", (), dict(free=0))())
    limited = r.read_ref(r.measure(inputs, output, "20261111"))
    assert "insufficient_disk_or_memory" in str(limited["reduction"]["gate_check_summary"])
    assert cli(tmp_path, "--date", "2026111").returncode == 1
