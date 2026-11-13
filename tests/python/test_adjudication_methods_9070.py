"""REQ-REPORT-9070-V774 / SCENARIO-VERIFY-9070-V774: protect unknown authority."""

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import adjudication_methods_9070 as d
from carnot.reporting import adjudication_methods_runner_9070 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


def releases() -> dict[str, list[dict[str, Any]]]:
    """Private synthetic judgments exercise custody without opening author targets."""
    record = dict(
        id="cf_vsg_queries-q.1-Qwen/Qwen3-32b-",
        question="Private question?",
        answer="Private section.",
        annotations=dict(
            sections_with_annotations=[
                dict(id="section-1", section="Private section.", adjudication=None)
            ]
        ),
    )
    return {name: [deepcopy(record)] for name in d.ANNOTATION_FILES}


def test_frozen_protocol_and_original_mapping() -> None:
    """REQ-REPORT-9070-V774: original categories and all fixed numerical bounds survive."""
    assert [d.label(v) for v in ["major", "minor", "no_hallucination", "needs_research"]] == [
        1,
        1,
        0,
        None,
    ]
    assert d.label("hallucination") is None
    p = d.protocol()
    assert p["folds"]["salt"] == "v774-mdhj-v1"
    assert p["folds"]["minimum_families"] == 60
    assert p["folds"]["minimum_test_families"] == 12
    assert p["features"]["projection_seed"] == 774070
    assert p["heads"]["ridge"] == 0.01
    assert p["heads"]["temperature_grid"] == [0.5, 1, 2, 4]
    assert p["bound"]["bootstrap_draws"] == 10000
    assert p["bound"]["one_sided_tail"] == 0.05 / 3
    assert p["bound"]["bootstrap_seed"] == 774077


def test_metadata_join_retains_all_sections_without_targets() -> None:
    """REQ-VERIFY-9070-V774: joined sections remain excluded while families are unknown."""
    data = releases()
    data[d.ANNOTATION_FILES[1]][0]["annotations"]["sections_with_annotations"][0][
        "adjudication"
    ] = dict(q1_response="major", aggregate_judgment="hallucination")
    reduced = d.join(data)
    assert len(reduced["rows"]) == 1
    assert reduced["independent_count"] == 0
    assert reduced["rows"][0]["status"] == "excluded"
    assert reduced["targets_opened"] is False
    assert "q1_response" not in json.dumps(reduced)
    assert "major" not in json.dumps(reduced)


@pytest.mark.parametrize(
    "mutation", ["response", "question", "section", "duplicate", "order", "missing"]
)
def test_join_rejects_identity_mutations(mutation: str) -> None:
    """SCENARIO-VERIFY-9070-V774: IDs, exact text and section ordering cannot drift."""
    data = releases()
    row = data[d.ANNOTATION_FILES[-1]][0]
    section = row["annotations"]["sections_with_annotations"][0]
    if mutation == "response":
        row["id"] = "other"
    elif mutation == "question":
        row["question"] = "other"
    elif mutation == "section":
        section["id"] = "other"
    elif mutation == "duplicate":
        row["annotations"]["sections_with_annotations"].append(deepcopy(section))
    elif mutation == "order":
        section["section"] = "other"
    else:
        data.pop(d.ANNOTATION_FILES[-1])
    with pytest.raises(ValueError):
        d.join(data)


@pytest.mark.parametrize(
    "denied", ["label", "generator_id", "reference", "judge_id", "explanation", "evidence"]
)
def test_feature_boundary_rejects_denied_metadata(denied: str) -> None:
    """SCENARIO-REPORT-9070-V774: learner input cannot contain outcome or source shortcuts."""
    view = dict(question="q", response="answer", section_start=0, section_end=6)
    assert d.feature_view(view) == view
    with pytest.raises(ValueError, match="feature_fields"):
        d.feature_view(dict(view, **{denied: "private"}))
    with pytest.raises(ValueError, match="section_offsets"):
        d.feature_view(dict(view, section_end=7))


def test_missing_release_is_blocked_with_operands(tmp_path: Path) -> None:
    """REQ-REPORT-9070-V774: absence is missing_file, never an observed zero."""
    value = d.inspect(tmp_path)
    assert value["blocker"].startswith("release_bytes")
    assert value["gate_check_summary"][0]["observed"] == "missing_file"
    assert value["rows"] == []
    assert value["targets_opened"] is False


def test_audit_binds_pins_and_unknown_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-9070-V774: a valid section join cannot grant family authority."""
    data = releases()
    for name, records in data.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in records))
    pins = {name: d.ref(tmp_path / name)["sha256"] for name in data}
    monkeypatch.setattr(d, "PINS", pins)
    monkeypatch.setattr(d, "HISTORY", {})
    value = d.inspect(tmp_path)
    assert value["blocker"] == "base_question_family_authority"
    assert value["independent_count"] == 0
    assert value["gate_check_summary"][-2]["observed"] == "missing_field"
    (tmp_path / d.ANNOTATION_FILES[0]).write_text("changed")
    assert d.inspect(tmp_path)["blocker"].startswith("release_bytes")


def test_build_failed_validation_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9070-V774: disqualification and changed aggregates cannot imply readiness."""
    work = runner.measure(tmp_path / "missing", tmp_path / "raw", "20261113", {})
    value = runner.build(work, {})
    assert value["verdict_class"] == "disqualified"
    assert value["human_methods_ready_score"] == value["human_source_ready_score"] == 0
    monkeypatch.setattr(runner, "receipts_pass", lambda _scope, _validation: True)
    value = runner.build(work, {})
    assert value["verdict_class"] == "blocked"
    path = tmp_path / (d.NAME + ".json")
    atomic_json(path, value)
    assert runner.replay(path)["passed"]
    value["human_source_ready_score"] = 1
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(path, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        runner.replay(path)


def test_conflicting_sections_response_sets_and_invalid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-9070-V774: malformed originals and global ID aliases are rejected."""
    data = releases()
    duplicate = deepcopy(data[d.ANNOTATION_FILES[0]][0])
    duplicate["id"] = "other-response"
    data[d.ANNOTATION_FILES[0]].append(duplicate)
    with pytest.raises(ValueError, match="duplicate_section"):
        d.join(data)
    data = releases()
    data[d.ANNOTATION_FILES[-1]] = []
    with pytest.raises(ValueError, match="mismatched_response_set"):
        d.join(data)
    for name, records in releases().items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in records))
    witness = tmp_path / "witness.json"
    atomic_json(witness, dict(historical_exposure=True))
    inventory = tmp_path / "exposure_inventory.json"
    atomic_json(inventory, dict(exposed_keys=[], rows=[d.ref(witness)]))
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(d, "HISTORY", {"exposure_inventory.json": d.ref(inventory)["sha256"]})
    pins = {name: d.ref(tmp_path / name)["sha256"] for name in d.ANNOTATION_FILES}
    monkeypatch.setattr(d, "PINS", pins)
    assert d.inspect(tmp_path)["blocker"] == "base_question_family_authority"
    path = tmp_path / d.ANNOTATION_FILES[0]
    path.write_text("invalid-json")
    pins[d.ANNOTATION_FILES[0]] = d.ref(path)["sha256"]
    assert d.inspect(tmp_path)["blocker"] == "release_identity"


def test_private_storage_and_changed_replay_operands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-9070-V774: private storage precedes source reads and every operand is bound."""
    raw = tmp_path / "not-private"
    raw.mkdir(mode=0o755)
    work = runner.measure(tmp_path / "missing", raw, "20261113", {})
    monkeypatch.setattr(runner, "receipts_pass", lambda _scope, _validation: True)
    value = runner.build(work, {})
    assert value["honest_verdict"] == "complete_blocked_private_storage"
    path = tmp_path / (d.NAME + ".json")
    value["source_artifact_hashes"][0]["sha256"] = "changed"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="operand_hash"):
        runner.replay(path)
    value = runner.build(work, {})
    atomic_json(path, value)
    original = d.inspect
    monkeypatch.setattr(
        d, "inspect", lambda directory: dict(original(directory), response_count=99)
    )
    with pytest.raises(ValueError, match="source_reduction_drift"):
        runner.replay(path)


def test_cli_publication_and_refusal_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9070-V774: real publication machinery obeys qualification and replay."""
    assert runner.main(["--date", "invalid"]) == 1
    scope = tmp_path / "scope.json"
    receipt = tmp_path / "receipt.json"
    atomic_json(scope, runner.scope_manifest([]))
    atomic_json(receipt, {})
    output = tmp_path / (d.NAME + ".json")
    args = [
        "--inputs",
        str(tmp_path / "missing"),
        "--output",
        str(output),
        "--scope",
        str(scope),
        "--receipts",
        str(receipt),
    ]
    assert runner.main(args) == 1
    assert not output.exists()
    monkeypatch.setattr(runner, "receipts_pass", lambda _scope, _validation: True)
    monkeypatch.setattr(runner.prior, "validate", lambda _path, _logs: dict(passed=True))
    assert runner.main(args) == 0
    assert runner.main(["--cold-replay", str(output)]) == 0
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    assert (
        runner.main(
            [
                "--inputs",
                str(tmp_path / "missing"),
                "--output",
                str(output),
                "--scope",
                str(tmp_path / "absent"),
                "--receipts",
                str(tmp_path / "absent"),
            ]
        )
        == 0
    )


def test_actual_foreign_cli_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9070-V774: unchanged private children replay without PYTHONPATH."""
    value = runner.build(runner.measure(tmp_path / "missing", tmp_path / "raw", "20261113", {}), {})
    candidate = tmp_path / (d.NAME + ".json")
    atomic_json(candidate, value)
    prefix = ["env", "-u", "PYTHONPATH", "-C", str(tmp_path), str(d.ROOT / ".venv/bin/python")]
    config = os.environ.get("COVERAGE_RCFILE")
    if config:
        prefix += ["-m", "coverage", "run", "--rcfile=" + config]
    prefix += [str(d.ROOT / d.CLI)]
    assert child(
        "actual_cold_replay", prefix + ["--cold-replay", str(candidate)], tmp_path / "logs"
    )["passed"]
    assert child("invalid_date", prefix + ["--date", "invalid"], tmp_path / "logs", expected=1)[
        "passed"
    ]
    value["human_source_ready_score"] = 1
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(candidate, value)
    assert child(
        "actual_rehashed_tamper",
        prefix + ["--cold-replay", str(candidate)],
        tmp_path / "logs",
        expected=1,
    )["passed"]
