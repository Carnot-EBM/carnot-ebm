"""REQ-REPORT-8597 / REQ-VERIFY-8597: missing labels cannot become medical truth."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from carnot.reporting import medexpert_evidence_8597 as e
from carnot.reporting import medexpert_evidence_runner_8597 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def receipt(tmp_path: Path, passed: bool = True) -> list[dict[str, Any]]:
    """Use durable command streams so replay can check the evidence behind a verdict."""
    log = tmp_path / "validation.stdout"
    log.write_text("owned validation control\n")
    return [
        dict(
            name="unit_control",
            passed=passed,
            stdout_path=str(log),
            stdout_sha256=e.reference(log)["sha256"],
        )
    ]


def cli(tmp_path: Path, *args: Any) -> subprocess.CompletedProcess[str]:
    """Start the actual script outside the checkout to test independent imports and exits."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before exp8597 private child", flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    print("after exp8597 private child exit=" + str(result.returncode), flush=True)
    return result


def test_original_and_assisted_authority() -> None:
    """SCENARIO-REPORT-8597-LINEAGE: protocol counts never substitute for released rows."""
    reduced = e.inspect(e.MANIFEST)
    assert reduced["release_audit_ready_score"] == 1
    assert reduced["release_available_score"] == 0
    assert reduced["original_annotation_authority_score"] == 1
    assert reduced["assisted_label_independence_score"] == 0
    assert reduced["evaluation_eligible_score"] == 0
    assert reduced["completed_count"] == 0
    assert reduced["intended_count"] is None
    assert reduced["selection_rows"][0]["paper_selected_count"] == 225
    assert reduced["selection_rows"][0]["verifier_votes_shown"] == 6
    assert reduced["count_reconciliation"]["resolved"] is False
    assert all(row["matched_count"] is None for row in reduced["overlap_rows"])
    assert all(row["missing_reason"] for row in reduced["rows"])
    assert all(row["sha256"] is not None for row in reduced["gate_check_summary"])


def test_input_authentication(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8597-REPLAY: a repaired manifest cannot invent new authority."""
    bad = tmp_path / "manifest.json"
    value = json.loads(e.MANIFEST.read_bytes())
    value["original_sentence_bytes"] = []
    atomic_json(bad, value)
    with pytest.raises(ValueError, match="original_authority"):
        e.inspect(bad)
    with pytest.raises(OSError):
        e.inspect(tmp_path / "missing")


def test_build_replay_and_rehashed_controls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8597-REPLAY: fresh reductions reject altered but rehashed summaries."""
    output = tmp_path / (e.NAME + ".json")
    source = e.measure(e.MANIFEST, output, "20261017")
    checks = receipt(tmp_path)
    value = e.build(source, checks, output)
    assert value["verdict_class"] == "blocked"
    assert value["required_checks_passed"] is True
    atomic_json(output, value)
    assert e.replay(output)["replay_passed"]
    bad = deepcopy(value)
    bad["evaluation_eligible_score"] = 1
    bad["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in bad.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, bad)
    with pytest.raises(ValueError, match="reduction_drift"):
        e.replay(output)
    atomic_json(output, value)
    Path(checks[0]["stdout_path"]).write_text("changed")
    with pytest.raises(ValueError, match="validation_log_hash"):
        e.replay(output)
    disqualified = e.build(source, [], output)
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["release_audit_ready_score"] == 0
    assert e.build(source, receipt(tmp_path, False), output)["required_checks_passed"] is False


def test_replay_owned_tampering(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8597-REPLAY: rehashed primitive edits still face original sources."""
    output = tmp_path / (e.NAME + ".json")
    source = e.measure(e.MANIFEST, output, "20261018")
    original = e.read(source)
    for field, expected in [
        ("code_config_hashes", "code_hash"),
        ("reduction", "primitive_reduction_drift"),
    ]:
        changed = deepcopy(original)
        changed[field] = (
            [] if field == "code_config_hashes" else dict(original[field], completed_count=9)
        )
        atomic_json(Path(source["path"]), changed)
        repaired = e.reference(Path(source["path"]))
        atomic_json(output, e.build(repaired, receipt(tmp_path), output))
        with pytest.raises(ValueError, match=expected):
            e.replay(output)
    atomic_json(Path(source["path"]), original)


def test_pinned_source_rejections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-8597: source bytes and retrieval bounds remain authoritative."""
    manifest = json.loads(e.MANIFEST.read_bytes())
    altered = tmp_path / "manifest.json"
    paper = tmp_path / "paper.html"
    paper.write_text("changed source")
    manifest["sources"]["paper"]["path"] = str(paper)
    atomic_json(altered, manifest)
    monkeypatch.setattr(e, "MANIFEST_SHA", e.reference(altered)["sha256"])
    with pytest.raises(ValueError, match="source_hash"):
        e.inspect(altered)
    manifest = json.loads(e.MANIFEST.read_bytes())
    retrievals = e.read(manifest["retrievals"])
    retrievals["total_requests"] = 11
    retrieval_path = tmp_path / "retrievals.json"
    atomic_json(retrieval_path, retrievals)
    manifest["retrievals"] = e.reference(retrieval_path)
    atomic_json(altered, manifest)
    monkeypatch.setattr(e, "MANIFEST_SHA", e.reference(altered)["sha256"])
    with pytest.raises(ValueError, match="retrieval_budget"):
        e.inspect(altered)
    manifest = json.loads(e.MANIFEST.read_bytes())
    manifest["sources"]["paper"] = e.reference(paper)
    atomic_json(altered, manifest)
    monkeypatch.setattr(e, "MANIFEST_SHA", e.reference(altered)["sha256"])
    with pytest.raises(ValueError, match="paper_protocol"):
        e.inspect(altered)


def test_missing_tool_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8597-BLOCK: missing local tools receive explicit gate diagnostics."""
    is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda p: False if p.name == "mypy" else is_file(p))
    source = e.measure(e.MANIFEST, tmp_path / (e.NAME + ".json"), "20261017")
    assert any(g["field"] == "tool_mypy" for g in e.read(source)["reduction"]["gate_check_summary"])


def test_private_cli_and_controls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8597-REPLAY: genuine bounded children check publication and errors."""
    output = tmp_path / (e.NAME + ".json")
    checks = tmp_path / "receipts.json"
    atomic_json(checks, receipt(tmp_path))
    done = cli(tmp_path, "--date", "20261017", "--receipts", checks, "--output", output)
    assert done.returncode == 0, done.stdout + done.stderr
    value = json.loads(output.read_bytes())
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    value["independent_count"] = 50
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    for args in [
        ("--deliberate-error",),
        ("--date", "20260230"),
        ("--date", "2026101"),
        ("--manifest", tmp_path / "missing", "--output", output),
    ]:
        assert cli(tmp_path, *args).returncode == 1
    assert runner.main(["--date", "20261017", "--cold-replay", str(tmp_path / "absent")]) == 1
