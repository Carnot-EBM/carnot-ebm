"""REQ-REPORT-9044-V772: private structure controls never substitute for measurements."""

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

from carnot.reporting import annotation_feasibility_9044 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def release(tmp_path: Path) -> Path:
    """Keep synthetic outcomes private; only the author-shaped ID leaf may be read."""
    files = {}
    for split in ["validation", "test"]:
        p = tmp_path / (split + ".parquet")
        rows = [
            dict(
                id=split + str(i),
                lang="en",
                model_output_text="é abc",
                annotations=[dict(annotator_id=str(j), labels=[[j, j + 1]]) for j in range(3)],
            )
            for i in range(2)
        ]
        pq.write_table(pa.Table.from_pylist(rows), p)
        files[split] = e.ref(p)
    for key, value in [
        ("policy", "Raw annotations annotator_id labels"),
        ("guidelines", e.CRITERION_PHRASE),
        ("license", "CC BY 4.0"),
    ]:
        p = tmp_path / key
        p.write_text(value)
        files[key] = e.ref(p)
    meta = tmp_path / "metadata.json"
    atomic_json(meta, dict(sha=e.REVISION, cardData=dict(license="cc-by-4.0")))
    p = tmp_path / "manifest.json"
    atomic_json(
        p, dict(scope="private_fixture", revision=e.REVISION, files=files, metadata=e.ref(meta))
    )
    return p


def test_structural_projection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9044-STRUCTURE: read IDs without ever materializing span values."""
    manifest = release(tmp_path)
    calls = []
    original = pq.ParquetFile.read

    def read(self: Any, columns: list[str], **kwargs: Any) -> Any:
        calls.append(columns)
        assert columns == e.COLUMNS
        return original(self, columns=columns, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read", read)
    result = e.inspect(manifest)
    assert len(calls) == 2 and len(result["annotation_schema_rows"]) == 4
    assert all(r["structure_ready"] for r in result["annotation_schema_rows"])
    assert result["annotation_structure_ready_score"] == 0  # private, not author authority
    assert result["test_outcome_values_accessed"] is False
    text = json.dumps(result)
    assert '"labels":' not in text and "é abc" not in text
    assert result["criterion_mapping"]["y_hard"] == "y_soft >= 0.5"
    assert result["pilot_contract"]["role_counts"] == dict(FIT=160, CAL=40, TEST=200)


@pytest.mark.parametrize(
    "language,count,limit",
    [
        ("en", 12, 12),
        ("EN", 13, 12),
        ("zh", 12, 12),
        ("ZH", 53, 12),
        ("de", 4, 4),
        ("DE", 5, 4),
        ("fi", 3, 3),
        ("FI", 6, 3),
    ],
)
def test_documented_participation_limits(language: str, count: int, limit: int) -> None:
    """SCENARIO-REPORT-9044-STRUCTURE: compare rosters to original language limits."""
    row = dict(
        id="private", lang=language, annotations=[dict(annotator_id=str(i)) for i in range(count)]
    )
    result = e.structural_row(row, "test", True)
    assert result["documented_annotator_limit"] == limit
    assert result["annotator_count_exceeds_documented_limit"] is (count > limit)
    assert result["structure_ready"] is (count <= limit)
    assert (result["original_judgment_participation"] == "unqualified") is (count > limit)


def test_invalid_span_schema_preserves_id_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9044-STRUCTURE: missing span authority cannot hide roster mismatch."""
    manifest = release(tmp_path)
    info = json.loads(manifest.read_bytes())
    path = Path(info["files"]["test"]["path"])
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(
                    id="roster",
                    lang="EN",
                    annotations=[dict(annotator_id=str(i), labels=[]) for i in range(95)],
                )
            ]
        ),
        path,
    )
    info["files"]["test"] = e.ref(path)
    atomic_json(manifest, info)
    original = pq.ParquetFile.read
    calls = []

    def read(self: Any, columns: list[str], **kwargs: Any) -> Any:
        calls.append(columns)
        assert columns == e.COLUMNS
        return original(self, columns=columns, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read", read)
    result = e.inspect(manifest)
    assert len(calls) == 2
    row = result["annotation_schema_rows"][-1]
    assert row["annotator_count"] == 95
    assert row["original_span_schema_present"] is False
    assert row["structure_ready"] is False
    assert result["participation_mismatch_counts"] == dict(validation=0, test=1)
    assert result["missing_authority"][-1]["check"] == "original_participation_authority"
    assert result["annotation_count_histograms"]["test"] == {"95": 1}


@pytest.mark.parametrize(
    "ids,ready",
    [
        (["a", "b", "c"], True),
        (["a", "a", "b"], False),
        (["a", "b"], False),
        (["", "b", "c"], False),
        ([None, "b", "c"], False),
        (None, False),
    ],
)
def test_id_controls(ids: Any, ready: bool) -> None:
    """SCENARIO-REPORT-9044-STRUCTURE: count actual distinct nonempty people, not copies."""
    row = dict(
        id="private",
        lang="en",
        annotations=None if ids is None else [dict(annotator_id=i) for i in ids],
    )
    actual = e.structural_row(row, "test", True)
    assert actual["structure_ready"] is ready
    assert e.structural_row(row, "test", False)["structure_ready"] is False


@pytest.mark.parametrize(
    "kind",
    [
        "revision",
        "metadata",
        "policy",
        "guidelines",
        "license",
        "hash",
        "pin",
        "missing",
        "field",
        "schema",
    ],
)
def test_authority_controls(tmp_path: Path, kind: str) -> None:
    """SCENARIO-REPORT-9044-REPLAY: changed or unavailable authority cannot qualify."""
    manifest = release(tmp_path)
    info = json.loads(manifest.read_bytes())
    if kind == "revision":
        info["revision"] = "main"
    elif kind == "metadata":
        p = Path(info["metadata"]["path"])
        atomic_json(p, dict(sha=e.REVISION, cardData=dict(license="other")))
        info["metadata"] = e.ref(p)
    elif kind in ["policy", "guidelines", "license", "hash"]:
        key = "policy" if kind == "hash" else kind
        p = Path(info["files"][key]["path"])
        p.write_text("changed")
        if kind != "hash":
            info["files"][key] = e.ref(p)
    elif kind == "pin":
        info["scope"] = "author_release"
    elif kind == "missing":
        Path(info["files"]["test"]["path"]).unlink()
    elif kind == "field":
        del info["files"]["test"]
    else:
        p = Path(info["files"]["test"]["path"])
        pq.write_table(pa.table(dict(id=["aggregate"], lang=["en"], soft_labels=[[0.5]])), p)
        info["files"]["test"] = e.ref(p)
    atomic_json(manifest, info)
    result = e.inspect(manifest)
    assert result["annotation_structure_ready_score"] == 0
    assert result["missing_authority"]
    assert all(r["value"] is not None for r in result["rows"])


def test_original_response_target() -> None:
    """SCENARIO-REPORT-9044-STRUCTURE: Unicode and disjoint marks use original votes."""
    from carnot.reporting.human_disagreement_source_9015 import labels

    row = dict(
        id="private",
        model_output_text="é abc",
        annotations=[dict(annotator_id=str(j), labels=[[j, j + 1]]) for j in range(3)],
    )
    assert labels(row)["y_soft"] == 1
    row["annotations"][0]["labels"] = [[0, 99]]
    with pytest.raises(ValueError, match="offset"):
        labels(row)


def cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """A private child must work outside the checkout without an import-path crutch."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI), *args]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    return subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)


def test_private_cli_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9044-REPLAY: missing authority and rehashed tampering are honest."""
    manifest = release(tmp_path)
    output = tmp_path / (e.NAME + ".json")
    run = cli(tmp_path, "--inputs", str(manifest), "--output", str(output))
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["natural_pilot_ready_score"] == 0
    assert cli(tmp_path, "--cold-replay", str(output)).returncode == 0
    value["annotation_structure_ready_score"] = 1
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", str(output)).returncode == 1
    assert cli(tmp_path, "--date", "wrong").returncode == 1
    assert cli(tmp_path, "--cold-replay", str(tmp_path / "absent")).returncode == 1
    missing = tmp_path / "missing" / (e.NAME + ".json")
    assert (
        cli(tmp_path, "--inputs", str(tmp_path / "absent"), "--output", str(missing)).returncode
        == 0
    )
    assert cli(tmp_path, "--cold-replay", str(missing)).returncode == 0


def receipt_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """Exercise receipt policy privately; actual publication uses measured SQLite coverage."""
    from carnot.reporting import annotation_feasibility_runner_9044 as r

    monkeypatch.setattr(r, "verify_measured", lambda report, owned: report.get("complete", False))
    scope = r.scope_manifest([dict(name=n, argv=["private", n]) for n in r.CHECKS])
    log = tmp_path / "log"
    log.write_text("private receipt log")
    receipts = [
        dict(
            name=n,
            argv=["private", n],
            passed=True,
            normal_exit=True,
            timed_out=False,
            exit_code=0,
            started_monotonic_ns=scope["frozen_ns"],
            ended_monotonic_ns=scope["frozen_ns"],
            scope_sha256=canonical_hash(scope),
            stdout_path=str(log),
            stdout_sha256=e.ref(log)["sha256"],
            stderr_path=str(log),
            stderr_sha256=e.ref(log)["sha256"],
        )
        for n in r.CHECKS
    ]
    return scope, dict(checks=receipts, owned_statement_coverage=dict(complete=True))


@pytest.mark.parametrize(
    "kind",
    [
        "valid",
        "missing",
        "stale",
        "failed",
        "timeout",
        "command",
        "log",
        "code",
        "coverage",
        "malformed",
    ],
)
def test_receipt_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    """SCENARIO-REPORT-9044-REPLAY: no changed, stale or fabricated validation passes."""
    from carnot.reporting import annotation_feasibility_runner_9044 as r

    scope, validation = receipt_fixture(tmp_path, monkeypatch)
    if kind == "missing":
        validation["checks"].pop()
    elif kind == "stale":
        validation["checks"][0]["started_monotonic_ns"] = 0
    elif kind == "failed":
        validation["checks"][0]["passed"] = False
    elif kind == "timeout":
        validation["checks"][0]["timed_out"] = True
    elif kind == "command":
        validation["checks"][0]["argv"] = ["other"]
    elif kind == "log":
        Path(validation["checks"][0]["stdout_path"]).write_text("changed")
    elif kind == "code":
        scope["code_test_hashes"][0]["sha256"] = "bad"
    elif kind == "coverage":
        validation["owned_statement_coverage"]["complete"] = False
    elif kind == "malformed":
        validation = {}
    assert r.receipts_pass(scope, validation) is (kind == "valid")


def test_build_and_rehashed_primitive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9044-REPLAY: independently repeat ID reads, not aggregate trust."""
    from carnot.reporting import annotation_feasibility_runner_9044 as r

    manifest = release(tmp_path)
    scope, validation = receipt_fixture(tmp_path, monkeypatch)
    output = tmp_path / (e.NAME + ".json")
    reference = r.measure(manifest, output, "20261112", scope)
    value = r.build(reference, validation)
    assert value["verdict_class"] == "blocked" and value["required_checks_passed"]
    assert all(v == 0 for v in value["model_invocation_counts"].values())
    atomic_json(output, value)
    assert r.replay(output)["replay_passed"]
    primitive = json.loads(Path(reference["path"]).read_bytes())
    primitive["reduction"]["annotation_schema_rows"][0]["annotator_count"] = 99
    atomic_json(Path(reference["path"]), primitive)
    value["primitive_reference"] = e.ref(Path(reference["path"]))
    value["raw_shard_hashes"][0] = value["primitive_reference"]
    atomic_json(output, value)
    with pytest.raises(ValueError, match="reduction"):
        r.replay(output)


def test_preconditions_and_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9044-REPLAY: environment and authority failures remain actionable."""
    from carnot.reporting import annotation_feasibility_runner_9044 as r

    manifest = release(tmp_path)
    scope = r.scope_manifest([])
    monkeypatch.setattr(r.shutil, "disk_usage", lambda p: type("Disk", (), dict(free=0))())
    reference = r.measure(manifest, tmp_path / (e.NAME + ".json"), "20261112", scope)
    value = r.build(reference, {})
    assert any(
        g["check"] == "virtualenv_and_resources" and not g["passed"]
        for g in value["gate_check_summary"]
    )
    monkeypatch.setattr(e, "AUTHORITY", {"missing": "abc"})
    reference = r.measure(manifest, tmp_path / (e.NAME + ".json"), "20261112", scope)
    assert not json.loads(Path(reference["path"]).read_bytes())["authority_gates"][0]["passed"]


def test_author_structure_and_schema_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9044-STRUCTURE: private authority branches cannot certify real bytes."""
    manifest = release(tmp_path)
    info = json.loads(manifest.read_bytes())
    info["scope"] = "author_release"
    license_path = Path(info["files"]["license"]["path"])
    license_path.write_text("Creative Commons")
    info["files"]["license"] = e.ref(license_path)
    monkeypatch.setattr(
        e, "PINNED", {k: v["sha256"].split(":")[1] for k, v in info["files"].items()}
    )
    atomic_json(manifest, info)
    result = e.inspect(manifest)
    assert result["annotation_structure_ready_score"] == 1
    assert len(result["annotation_schema_rows"]) == 4
    p = Path(info["files"]["test"]["path"])
    pq.write_table(pa.table(dict(id=["aggregate"], lang=["en"])), p)
    info["files"]["test"] = e.ref(p)
    monkeypatch.setitem(e.PINNED, "test", info["files"]["test"]["sha256"].split(":")[1])
    atomic_json(manifest, info)
    assert e.inspect(manifest)["missing_authority"][-1]["check"] == "original_individual_judgments"


@pytest.mark.parametrize("kind", ["hash", "authority", "rows", "health"])
def test_rehashed_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    """SCENARIO-REPORT-9044-REPLAY: independently bound primitives reject rehashed tampering."""
    from carnot.reporting import annotation_feasibility_runner_9044 as r

    manifest = release(tmp_path)
    scope, validation = receipt_fixture(tmp_path, monkeypatch)
    output = tmp_path / (e.NAME + ".json")
    reference = r.measure(manifest, output, "20261112", scope)
    value = r.build(reference, validation)
    p = json.loads(Path(reference["path"]).read_bytes())
    if kind == "hash":
        value["code_config_hashes"][0]["sha256"] = "bad"
    elif kind == "authority":
        p["authority_gates"][0]["expected"] = "invented"
    elif kind == "health":
        p["repository_health"]["passed"] = True
    else:
        path = Path(p["raw_rows"][0]["path"])
        atomic_json(path, dict(rows=[]))
        p["raw_rows"][0] = e.ref(path)
        value["raw_shard_hashes"][1] = e.ref(path)
    if kind != "hash":
        atomic_json(Path(reference["path"]), p)
        value["primitive_reference"] = e.ref(Path(reference["path"]))
        value["raw_shard_hashes"][0] = value["primitive_reference"]
    atomic_json(output, value)
    with pytest.raises(ValueError):
        r.replay(output)


def test_health_hash_and_explicit_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9044-REPLAY: missing health bytes and changed receipt scope reject."""
    from carnot.reporting import annotation_feasibility_runner_9044 as r

    manifest = release(tmp_path)
    scope, validation = receipt_fixture(tmp_path, monkeypatch)
    output = tmp_path / (e.NAME + ".json")
    scope_path, checks_path = tmp_path / "scope", tmp_path / "checks"
    atomic_json(scope_path, scope)
    atomic_json(checks_path, validation)
    monkeypatch.setattr(r, "validate", lambda p, logs: dict(passed=True))
    assert (
        r.main(
            [
                "--inputs",
                str(manifest),
                "--output",
                str(output),
                "--scope",
                str(scope_path),
                "--receipts",
                str(checks_path),
            ]
        )
        == 0
    )
    monkeypatch.setattr(
        r.history,
        "health",
        lambda root: dict(references=[dict(path=str(tmp_path / "missing"), sha256="bad")]),
    )
    with pytest.raises(ValueError, match="historical_health_hash"):
        r.measure(manifest, output, "20261112", scope)


def test_roster_is_not_completed_judgments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9044-STRUCTURE: a language roster cannot certify response judgments."""
    manifest = release(tmp_path)
    info = json.loads(manifest.read_bytes())
    p = Path(info["files"]["test"]["path"])
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(
                    id="roster",
                    lang="EN",
                    annotations=[dict(annotator_id=str(j), labels=[]) for j in range(95)],
                )
            ]
        ),
        p,
    )
    info["files"]["test"] = e.ref(p)
    info["scope"] = "author_release"
    license_path = Path(info["files"]["license"]["path"])
    license_path.write_text("Creative Commons")
    info["files"]["license"] = e.ref(license_path)
    monkeypatch.setattr(
        e, "PINNED", {k: v["sha256"].split(":")[1] for k, v in info["files"].items()}
    )
    atomic_json(manifest, info)
    result = e.inspect(manifest)
    assert result["annotation_structure_ready_score"] == 0
    assert result["missing_authority"][-1]["check"] == "original_participation_authority"
    assert result["annotation_schema_rows"][-1]["annotator_count"] == 95
    assert result["annotation_schema_rows"][-1]["original_judgment_participation"] == "unqualified"
    assert result["test_outcome_values_accessed"] is False
