"""REQ-REPORT-8975 / SCENARIO-REPORT-8975-V767: source access must precede readiness."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from carnot.reporting import natural_label_eligibility_8975 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def labels() -> list[dict[str, Any]]:
    """Use clearly artificial identities so controls cannot become natural evidence."""
    return [
        dict(
            conversation_id=f"fixture-{i}",
            occurrence1="Hallucination",
            category1="Input-conflicting",
            occurrence2="No Hallucination",
            category2=None,
        )
        for i in range(400)
    ]


def source() -> dict[str, Any]:
    """Permission is a fixture assertion, never a claim about the real corpus."""
    return dict(
        local_use_authorized=True,
        model_terms_verified=True,
        license_reference="fixture-only",
        records=[
            dict(
                conversation_id=f"fixture-{i}",
                model="fixture-model",
                source_offset=i,
                conversation=[
                    dict(role=r, content=f"fixture {i} {j}")
                    for j, r in enumerate(("user", "assistant", "user", "assistant"))
                ],
            )
            for i in range(400)
        ],
    )


def test_missing_source_retains_all_slots() -> None:
    """REQ-REPORT-8975: metadata completion cannot turn absent text into evidence."""
    result = e.audit(labels(), {})
    assert len(result["rows"]) == 800 and len(result["dialogue_join_rows"]) == 400
    assert result["completed_count"] == 0 and result["censored_count"] == 800
    assert result["natural_evidence_ready_score"] == result["independent_count"] == 0
    assert result["eligibility_matrix"] == result["proposed_group_splits"] == []
    assert result["metadata_audited_count"] == 800
    assert all(r["group_id"] == r["conversation_id"] for r in result["rows"])
    assert "model" in result["dialogue_join_rows"][0]["missing_fields"]
    assert all(
        set(("upstream", "check", "path", "sha256", "field", "operator", "expected", "observed"))
        <= set(r)
        for r in result["gate_check_summary"]
    )


def test_qualified_groups_and_feature_boundary() -> None:
    """SCENARIO-REPORT-8975-V767: both turns stay together, with labels outside inference."""
    result = e.audit(labels(), source())
    assert result["natural_evidence_ready_score"] == 1
    assert result["completed_count"] == 800 and result["censored_count"] == 0
    assert result["independent_generalization_score"] == 0
    groups = result["proposed_group_splits"]
    assert [len(r["group_ids"]) for r in groups] == [240, 80, 80]
    assert len({g for r in groups for g in r["group_ids"]}) == 400
    assert e.audit(list(reversed(labels())), source())["proposed_group_splits"] == groups
    assert {"occurrence1", "category2", "explanation", "detector_outputs"} <= set(
        result["excluded_feature_keys"]
    )
    assert result["annotation_criterion_rows"][2]["independent_cited_evidence"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "occurrence",
        "category",
        "missing_label",
        "short",
        "roles",
        "empty",
        "model",
        "offset",
        "permission",
        "terms",
        "license",
        "duplicate_source",
    ],
)
def test_bad_inputs_cannot_qualify(mutation: str) -> None:
    """REQ-REPORT-8975: ambiguous joins, invalid labels and missing permissions fail closed."""
    rows, access = labels(), source()
    if mutation == "duplicate":
        rows[1] = deepcopy(rows[0])
    if mutation == "occurrence":
        rows[0]["occurrence1"] = "unknown"
    if mutation == "category":
        rows[0]["category1"] = None
    if mutation == "missing_label":
        del rows[0]["occurrence2"]
    if mutation == "short":
        access["records"][0]["conversation"].pop()
    if mutation == "roles":
        access["records"][0]["conversation"][0]["role"] = "assistant"
    if mutation == "empty":
        access["records"][0]["conversation"][0]["content"] = ""
    if mutation in {"model", "offset"}:
        del access["records"][0]["model" if mutation == "model" else "source_offset"]
    if mutation == "permission":
        access["local_use_authorized"] = False
    if mutation == "terms":
        access["model_terms_verified"] = False
    if mutation == "license":
        access["license_reference"] = None
    if mutation == "duplicate_source":
        access["records"].append(deepcopy(access["records"][0]))
    result = e.audit(rows, access)
    assert result["natural_evidence_ready_score"] == 0
    assert result["proposed_group_splits"] == []
    assert result["completed_count"] + result["censored_count"] == 800


def cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Execute the real CLI outside the checkout and retain child coverage when requested."""
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(e.ROOT / e.CLI), *args]
    if env.get("COVERAGE_RCFILE"):
        command[1:2] = ["-m", "coverage", "run"]
    return subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
    )


def test_actual_cli_publication_replay_and_rehashed_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8975-V767: cold reconstruction must reject repaired aggregate hashes."""
    manifest = tmp_path / "manifest.json"
    release = tmp_path / "labels.json"
    atomic_json(release, labels())
    atomic_json(
        manifest,
        dict(
            scope="private_fixture",
            release_revision=e.REVISION,
            files=dict(labels=e.ref(release)),
            source_access={},
        ),
    )
    output = tmp_path / (e.NAME + ".json")
    receipts = tmp_path / "receipts.json"
    atomic_json(
        receipts,
        dict(
            checks=[dict(name=n, passed=True) for n in e.CHECKS],
            owned_statement_coverage=dict(percent_covered=100, missing_lines=[], excluded_lines=[]),
        ),
    )
    run = cli(
        tmp_path,
        "--date",
        "20261110",
        "--inputs",
        str(manifest),
        "--receipts",
        str(receipts),
        "--output",
        str(output),
    )
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"] == "complete_blocked_source_access"
    assert value["inference_substrate_class"] == "no_model_load"
    replay = cli(tmp_path, "--cold-replay", str(output))
    assert replay.returncode == 0, replay.stdout + replay.stderr
    value["completed_count"] = 800
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    refused = cli(tmp_path, "--cold-replay", str(output))
    assert refused.returncode == 1 and "reduction_drift" in refused.stdout
    bad_date = cli(tmp_path, "--date", "20261131")
    assert bad_date.returncode == 1


def test_input_authentication_and_missing_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8975: absent files, changed bytes, missing fields and zero joins stay distinct."""
    from carnot.reporting import natural_label_runner_8975 as r

    missing = dict(path=str(tmp_path / "absent"), sha256=None)
    with pytest.raises(FileNotFoundError, match="missing_file"):
        r.read_ref(missing)
    path = tmp_path / "labels.json"
    atomic_json(path, labels())
    original = e.ref(path)
    with pytest.raises(ValueError, match="changed_bytes"):
        r.read_ref(dict(original, sha256="wrong"))
    manifest = tmp_path / "manifest.json"
    value = dict(
        scope="author_release",
        release_revision=e.REVISION,
        files=dict(labels=original),
        source_access={},
    )
    atomic_json(manifest, value)
    with pytest.raises(ValueError, match="pinned_release_hash"):
        r.inspect(manifest)
    monkeypatch.setattr(
        e, "PINNED", dict(labels=original["sha256"].split(":")[1], license="absent")
    )
    with pytest.raises(ValueError, match="missing_release_fields"):
        r.inspect(manifest)
    monkeypatch.setattr(e, "PINNED", dict(labels=original["sha256"].split(":")[1]))
    assert r.inspect(manifest)["natural_evidence_ready_score"] == 0
    access = source()
    records = tmp_path / "source.json"
    atomic_json(records, access.pop("records"))
    access["records_reference"] = e.ref(records)
    value["source_access"] = access
    atomic_json(manifest, value)
    assert r.inspect(manifest)["natural_evidence_ready_score"] == 1
    with records.open("wb") as stream:
        stream.truncate(10_000_001)
    with pytest.raises(ValueError, match="exceeds_budget"):
        r.inspect(manifest)
    value["release_revision"] = "wrong"
    atomic_json(manifest, value)
    with pytest.raises(ValueError, match="release_revision"):
        r.inspect(manifest)
    value["release_revision"] = e.REVISION
    path.write_text("[]")
    atomic_json(manifest, value)
    with pytest.raises(ValueError, match="changed_bytes"):
        r.inspect(manifest)
    path.unlink()
    with pytest.raises(FileNotFoundError, match="missing_file"):
        r.inspect(manifest)


def test_blocked_replay_and_owned_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8975-V767: rehashed primitive drift and failed checks cannot self-qualify."""
    import time
    from carnot.reporting import natural_label_runner_8975 as r

    manifest = tmp_path / "absent.json"
    output = tmp_path / (e.NAME + ".json")
    primitive_ref = r.measure(manifest, output, "20261110", time.monotonic_ns())
    value = r.build(primitive_ref, {})
    assert value["verdict_class"] == "disqualified" and not value["required_checks_passed"]
    atomic_json(output, value)
    assert r.replay(output)["passed"]
    primitive = r.read_ref(primitive_ref)
    primitive["reduction"]["rows"][0]["source_joined"] = 1
    atomic_json(Path(primitive_ref["path"]), primitive)
    value["primitive_reference"] = e.ref(Path(primitive_ref["path"]))
    atomic_json(output, value)
    with pytest.raises(ValueError, match="blocked_reduction_drift"):
        r.replay(output)
    primitive["reduction"]["rows"][0]["source_joined"] = 0
    primitive["code_config_hashes"][0]["sha256"] = "wrong"
    atomic_json(Path(primitive_ref["path"]), primitive)
    value["primitive_reference"] = e.ref(Path(primitive_ref["path"]))
    atomic_json(output, value)
    with pytest.raises(ValueError, match="executed_source_hash"):
        r.replay(output)
    monkeypatch.setattr(r, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(r, "validate", lambda *args: dict(passed=True, checks=[]))
    assert r.main(["--output", str(output), "--date", "20261110"]) == 0
    assert r.main(["--date", "2026111"]) == 1
    monkeypatch.setattr(r.shutil, "disk_usage", lambda p: type("Disk", (), dict(free=0))())
    blocked = r.measure(manifest, output, "20261110", time.monotonic_ns())
    assert "insufficient_disk_or_memory" in str(
        r.read_ref(blocked)["reduction"]["gate_check_summary"]
    )


def test_replay_reauthenticates_labels_receipts_and_invocation(tmp_path: Path) -> None:
    """REQ-REPORT-8975: raw reduction and current work remain bound after checksum repair."""
    import time
    from carnot.reporting import natural_label_runner_8975 as r

    release, manifest = tmp_path / "labels.json", tmp_path / "manifest.json"
    atomic_json(release, labels())
    atomic_json(
        manifest,
        dict(
            scope="private_fixture",
            release_revision=e.REVISION,
            files=dict(labels=e.ref(release)),
            source_access={},
        ),
    )
    output = tmp_path / (e.NAME + ".json")
    primitive_ref = r.measure(manifest, output, "20261110", time.monotonic_ns())
    primitive = r.read_ref(primitive_ref)
    baseline = deepcopy(primitive)
    for mutation, expected in [
        ("rows", "original_reduction_drift"),
        ("receipt", "current_work_receipt"),
        ("manifest", "manifest_hash"),
        ("log", "validation_log_hash"),
    ]:
        p = deepcopy(baseline)
        if mutation == "rows":
            p["reduction"]["completed_count"] = 800
        if mutation == "receipt":
            p["current_work_receipt"]["duration_s"] += 1
        if mutation == "manifest":
            p["manifest"]["sha256"] = "wrong"
        atomic_json(Path(primitive_ref["path"]), p)
        check = dict(name="fake", passed=True)
        if mutation == "log":
            check.update(stdout_path=str(release), stdout_sha256="wrong")
        value = r.build(e.ref(Path(primitive_ref["path"])), dict(checks=[check]))
        atomic_json(output, value)
        with pytest.raises(ValueError, match=expected):
            r.replay(output)
