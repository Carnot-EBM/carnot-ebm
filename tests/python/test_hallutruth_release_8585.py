"""REQ-REPORT-8585 and REQ-VERIFY-8585: audit authority without training a detector."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pytest

from carnot.reporting import hallutruth_release_8585 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def release(tmp_path: Path) -> Path:
    """Synthetic Arabic controls stay private and never stand in for released data."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    row = dict(
        id="a",
        question="سؤال",
        generated_answer="نص خطأ",
        gold_answer="مرجع",
        generator_model="fixture",
        label="hallucination",
        answer="A",
        options={},
        hallucinations=[
            dict(span_start=3, span_end=6, hallucinated_span="خطأ", explanation="fixture")
        ],
        needs_human_review=False,
    )
    files = []
    for split, records in {"train": [row], "test": [dict(row, id="b")]}.items():
        path = tmp_path / (split + ".jsonl")
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
        files.append(dict(e.reference(path), split=split, url="public-fixture"))
    manifest = tmp_path / "manifest.json"
    atomic_json(
        manifest,
        dict(
            commit="fixture",
            files=files,
            declared_counts={"train": 1, "test": 1},
            annotation=dict(
                expert_review_documented=True,
                independent_of_carnot=True,
                adjudication_logs_available=False,
            ),
            rights=dict(license="cc-by-nc-nd-4.0", deployment_authorized=False),
            source_metadata=dict(fixture=True),
        ),
    )
    return manifest


def test_inventory_offsets_overlap_and_exclusions(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8585-RELEASE: released units and independent claims differ."""
    result = e.reduce(e.load(release(tmp_path)))
    assert result["completed_count"] == 2
    assert result["release_audit_ready_score"] == 1
    assert result["human_label_evidence_score"] == 1
    assert result["independent_count"] == 0
    assert result["split_overlap_rows"][0]["exact_question_overlap_count"] == 1
    assert result["span_diagnostics"]["invalid_character_span_count"] == 0
    assert result["annotation_lineage_rows"][0]["row_adjudication_receipt_count"] == 0
    row = e.load(release(tmp_path))["splits"]["train"][0]
    features = e.feature_reader(row, e.EXCLUSION_MANIFEST, ["permitted"])
    assert features == {
        "question": row["question"],
        "generated_answer": row["generated_answer"],
        "evidence": ["permitted"],
    }
    assert set(features).isdisjoint(e.EXCLUSION_MANIFEST["excluded_fields"])
    with pytest.raises(ValueError, match="feature_allowlist"):
        e.feature_reader(row, dict(allowed_fields=["label"]))


def test_missing_and_invalid_instances(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8585-CONTROLS: absence differs from measured limitations."""
    manifest = release(tmp_path)
    data = e.load(manifest)
    changed = copy.deepcopy(data)
    row = changed["splits"]["train"][0]
    row.update(label=None, needs_human_review=True, question="", generator_model=None)
    row["hallucinations"][0].update(span_end=99)
    changed["splits"]["test"].append(copy.deepcopy(row))
    result = e.reduce(changed)
    assert result["release_audit_ready_score"] == 0
    assert result["human_label_evidence_score"] == 0
    assert result["span_diagnostics"]["invalid_character_span_count"] == 2
    assert any(r["missing_reasons"] for r in result["private_rows"])
    assert result["unresolved_review_count"] == 2
    assert e.reduce(e.load(tmp_path / "absent"))["release_audit_ready_score"] == 0
    datafile = tmp_path / "train.jsonl"
    datafile.unlink()
    assert e.reduce(e.load(manifest))["release_audit_ready_score"] == 0
    datafile.write_text("altered")
    with pytest.raises(ValueError, match="source_hash"):
        e.load(manifest)


def test_budgets_and_missing_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8585: complete bounded release and honest review denominators."""
    manifest = release(tmp_path)
    data = e.load(manifest)
    del data["splits"]["train"][0]["needs_human_review"]
    data["splits"]["test"][0]["hallucinations"] = []
    data["splits"]["test"][0]["label"] = "no_hallucination"
    assert e.reduce(data)["missing_review_flag_count"] == 1
    monkeypatch.setattr(e, "MAX_ROWS", 1)
    with pytest.raises(ValueError, match="row_budget"):
        e.load(manifest)
    monkeypatch.setattr(e, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="byte_budget"):
        e.load(manifest)


def test_replay_and_owned_failure(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8585-TAMPER: rehashing cannot replace original operands."""
    output = tmp_path / "results" / (e.NAME + ".json")
    primitive = e.measure(release(tmp_path / "release"), output)
    value = e.build(primitive, [dict(passed=True)], output)
    atomic_json(output, value)
    assert e.replay(output)["passed"]
    value["human_label_evidence_score"] = 0
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        e.replay(output)
    assert e.build(primitive, [dict(passed=False)], output)["verdict_class"] == "disqualified"
    blocked = e.measure(tmp_path / "missing", output)
    assert e.build(blocked, [dict(passed=True)], output)["verdict_class"] == "blocked"


def cli(tmp_path: Path, *args: Any) -> subprocess.CompletedProcess[str]:
    """Actual child imports from outside the checkout and records its complete output."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        command[1:2] = ["-m", "coverage", "run", "--parallel-mode"]
    print("[8585 test] before child", command, flush=True)
    started = time.monotonic()
    result = subprocess.run(
        command, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=90
    )
    print("[8585 test] after child", result.returncode, result.stdout, result.stderr, flush=True)
    print(
        dict(
            argv=command,
            exit_code=result.returncode,
            duration_s=time.monotonic() - started,
            stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
            stderr_sha256=hashlib.sha256(result.stderr.encode()).hexdigest(),
        ),
        flush=True,
    )
    return result


def test_private_cli_publication_and_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8585-TAMPER: genuine standalone CLI and both live consumers."""
    manifest = release(tmp_path / "release")
    receipts = tmp_path / "receipts.json"
    atomic_json(receipts, [dict(passed=True)])
    output = tmp_path / "results" / (e.NAME + ".json")
    done = cli(tmp_path, "--manifest", manifest, "--receipts", receipts, "--output", output)
    assert done.returncode == 0, done.stdout + done.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "null"
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    value["completed_count"] += 1
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    assert cli(tmp_path, "--date", "20000101").returncode == 2
    assert cli(tmp_path, "--deliberate-error").returncode == 1


@pytest.mark.parametrize("mutation", ["code", "private", "summary", "log"])
def test_primitive_custody_rejections(tmp_path: Path, mutation: str) -> None:
    """REQ-REPORT-8585-REPLAY: rehashed primitives still need original row and code evidence."""
    output = tmp_path / "results" / (e.NAME + ".json")
    source = e.measure(release(tmp_path / "release"), output)
    primitive = e.read(source)
    receipts = [dict(passed=True)]
    if mutation == "code":
        primitive["code_config_hashes"][0]["sha256"] = "sha256:bad"
    elif mutation == "private":
        path = Path(primitive["private_rows_reference"]["path"])
        atomic_json(path, [])
        primitive["private_rows_reference"] = e.reference(path)
    elif mutation == "summary":
        primitive["reduction"]["completed_count"] = 999
    else:
        path = tmp_path / "log"
        path.write_text("actual")
        receipts[0].update(stdout_path=str(path), stdout_sha256="sha256:wrong")
    atomic_json(Path(source["path"]), primitive)
    source = e.reference(Path(source["path"]))
    atomic_json(output, e.build(source, receipts, output))
    with pytest.raises(ValueError):
        e.replay(output)


def test_runner_owned_rejections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8585-CONTROLS: failed consumer and final validator remain failures."""
    from carnot.reporting import hallutruth_release_runner_8585 as runner

    manifest = release(tmp_path / "release")
    receipts = tmp_path / "receipts.json"
    atomic_json(receipts, [dict(passed=True)])
    output = tmp_path / "results" / (e.NAME + ".json")
    args = ["--manifest", str(manifest), "--receipts", str(receipts), "--output", str(output)]
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    monkeypatch.setattr(
        runner, "validate", lambda *a: dict(passed=False, checks=[dict(passed=False)])
    )
    assert runner.main(args) == 1
    monkeypatch.setattr(
        runner, "validate", lambda *a: dict(passed=True, checks=[dict(passed=True)])
    )
    monkeypatch.setattr(runner, "reader_receipt", lambda *a, **kw: dict(passed=False))
    assert runner.main(args) == 1
    monkeypatch.setattr(runner, "reader_receipt", lambda *a, **kw: dict(passed=True))
    assert (
        runner.main(
            [
                "--manifest",
                str(tmp_path / "absent"),
                "--receipts",
                str(receipts),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert (
        runner.main(
            [
                "--manifest",
                str(manifest),
                "--receipts",
                str(tmp_path / "absent"),
                "--output",
                str(output),
            ]
        )
        == 1
    )


def test_original_release_authority_and_scope(tmp_path: Path) -> None:
    """REQ-REPORT-8585-REPLAY: a rehashed replacement is not the pinned author release."""
    manifest = release(tmp_path)
    original = json.loads(manifest.read_bytes())
    for update in [dict(commit="other"), dict(commit=e.AUTHOR_COMMIT)]:
        changed = dict(original, **update)
        atomic_json(manifest, changed)
        with pytest.raises(ValueError, match="release_authority"):
            e.load(manifest)
    atomic_json(manifest, original)
    from carnot.reporting import hallutruth_release_runner_8585 as runner

    assert runner.main(["--manifest", str(manifest)]) == 1


def test_binary_natural_scope_gate(tmp_path: Path) -> None:
    """REQ-REPORT-8585: false eligibility must never assert a measured zero error rate."""
    assert e.reduce(e.load(release(tmp_path)))["natural_error_evidence_score"] is False


def test_missing_spans_and_duplicate_instance_ids(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8585-CONTROLS: absence and repeated IDs retain honest denominators."""
    data = e.load(release(tmp_path))
    row = data["splits"]["train"][0]
    row["hallucinations"] = []
    data["splits"]["train"].append(copy.deepcopy(row))
    data["declared_counts"]["train"] = 2
    result = e.reduce(data)
    assert result["release_audit_ready_score"] == 0
    assert any(
        "hallucination_spans_missing" in r["missing_reasons"] for r in result["private_rows"]
    )
    gate = result["gate_check_summary"][0]
    assert gate["expected"] != gate["observed"]
