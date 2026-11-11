"""REQ-REPORT-9015-V770: private controls cannot become source measurements."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from carnot.reporting import human_disagreement_source_9015 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

sys.path.insert(0, str(e.ROOT))


def records(count: int = 440) -> list[dict[str, Any]]:
    """Make independent private topics with labels that cannot affect selection."""
    return [
        dict(
            id=str(i),
            lang=e.LANGUAGES[i % 14],
            split="validation" if i < 220 else "test",
            model_input=f"question {i}",
            model_output_text="é abc",
            wikipedia_url=f"https://en.wikipedia.org/wiki/Topic_{i}",
            annotations=[
                dict(annotator_id=str(j), labels=[[2, 3]] if i % 2 else []) for j in range(3)
            ],
        )
        for i in range(count)
    ]


def resolutions(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Identity fixtures stand in for retained metadata, never measured source authority."""
    return {r["wikipedia_url"]: f"Q{i + 1}" for i, r in enumerate(rows)}


def test_response_votes() -> None:
    """SCENARIO-REPORT-9015-LABELS: disjoint spans still give a response majority."""
    row = records(1)[0]
    row["annotations"] = [dict(annotator_id=str(i), labels=[[i, i + 1]]) for i in range(3)]
    result = e.labels(row)
    assert result["y_soft"] == 1 and result["y_hard"] == 1
    assert result["annotator_count"] == 3 and result["original_annotations"] == row["annotations"]
    row["annotations"] = [
        dict(annotator_id=str(i), labels=[[0, 1]] if i < 2 else [], uncertain=i == 3)
        for i in range(4)
    ]
    result = e.labels(row)
    assert result["y_soft"] == 0.5 and result["tie"] and result["uncertain_annotators"] == ["3"]
    assert result["y_hard"] == 1


@pytest.mark.parametrize(
    "kind", ["few", "duplicate", "offset", "empty_id", "automatic", "empty_text"]
)
def test_invalid_annotations(kind: str) -> None:
    """SCENARIO-REPORT-9015-LABELS: unsupported annotators and offsets reject."""
    row = records(1)[0]
    if kind == "few":
        row["annotations"] = row["annotations"][:2]
    if kind == "duplicate":
        row["annotations"][1]["annotator_id"] = "0"
    if kind == "offset":
        row["annotations"][0]["labels"] = [[0, 99]]
    if kind == "empty_id":
        row["annotations"][0]["annotator_id"] = ""
    if kind == "automatic":
        row["annotations"][0]["automatic"] = True
    if kind == "empty_text":
        row["model_output_text"] = ""
    with pytest.raises(ValueError):
        e.labels(row)


def test_roles_and_label_blinding(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9015-GROUPS: fixed roles precede labels and preserve floors."""
    rows = records()
    ids = resolutions(rows)
    result = e.audit(rows, set(), ids, tmp_path)
    assert len(result["role_manifest"]) == 400
    assert result["natural_pilot_ready_score"] == 1
    assert result["class_counts"]["TEST"]["1"] >= 20
    assert all("y_soft" not in r for r in result["role_manifest"])
    before = result["role_manifest"]
    for row in rows:
        for a in row["annotations"]:
            a["labels"] = []
    changed = e.audit(rows, set(), ids, tmp_path / "changed")
    assert changed["role_manifest"] == before and changed["natural_pilot_ready_score"] == 0
    assert json.loads((tmp_path / "role_manifest.json").read_bytes())["labels_accessible"] is False


def test_transitive_exposure_and_unresolved(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9015-GROUPS: multilingual identities merge before exclusion."""
    rows = records(5)
    ids = resolutions(rows)
    ids[rows[1]["wikipedia_url"]] = ids[rows[0]["wikipedia_url"]]
    rows[2]["model_input"] = rows[1]["model_input"]
    rows[3]["topic_id"] = "shared"
    rows[2]["topic_id"] = "shared"
    ids.pop(rows[4]["wikipedia_url"])
    result = e.audit(rows, {"qid:Q1"}, ids, tmp_path)
    assert len(result["source_group_rows"]) == 2
    assert result["independent_count"] == 0 and result["censored_count"] == 400
    assert {r["exclusion_reason"] for r in result["source_group_rows"]} == {
        "prior_exposure",
        "unresolved_source_identity",
    }
    assert len(result["unresolved_source_groups"]) == 1


@pytest.mark.parametrize(
    "kind", ["duplicate", "split", "cross_split", "invalid_labels", "hash_source", "empty_prompt"]
)
def test_group_controls(tmp_path: Path, kind: str) -> None:
    """SCENARIO-REPORT-9015-GROUPS: joins and raw failures cannot silently select replacements."""
    rows = records(3)
    ids = resolutions(rows)
    if kind == "duplicate":
        rows.append(rows[0])
        with pytest.raises(ValueError):
            e.audit(rows, set(), ids, tmp_path)
        return
    if kind == "split":
        rows[0]["split"] = "train_unlabeled"
    if kind == "cross_split":
        rows[1]["split"] = "test"
        rows[1]["model_input"] = rows[0]["model_input"]
    if kind == "invalid_labels":
        rows[0]["annotations"] = []
    if kind == "hash_source":
        rows[0]["source_text"] = "same"
        rows[1]["source_text"] = " SAME "
    if kind == "empty_prompt":
        rows[0]["model_input"] = ""
    result = e.audit(rows, set(), ids, tmp_path)
    assert result["natural_pilot_ready_score"] == 0
    assert result["intended_count"] == sum(
        result[k] for k in ["completed_count", "excluded_count", "censored_count", "failed_count"]
    )
    assert result["independent_generalization_score"] == 0


def inputs(tmp_path: Path, blocked: bool = False) -> Path:
    """Retain a private author-shaped parquet release without granting it live authority."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = records()
    files = {}
    for split in ["validation", "test"]:
        p = tmp_path / (split + ".parquet")
        pq.write_table(pa.Table.from_pylist([r for r in rows if r["split"] == split]), p)
        files[split] = e.ref(p)
    for key, text in [
        ("policy", "Raw annotations annotator_id"),
        ("license", "CC BY 4.0"),
        ("guidelines", "original unsupported span policy"),
    ]:
        p = tmp_path / key
        p.write_text(text)
        files[key] = e.ref(p)
    metadata = tmp_path / "metadata.json"
    atomic_json(metadata, dict(sha=e.REVISION, cardData=dict(license="cc-by-4.0")))
    resolve = tmp_path / "resolve.json"
    atomic_json(resolve, {k: dict(wikidata_id=v) for k, v in resolutions(rows).items()})
    inventory = tmp_path / "inventory.json"
    atomic_json(inventory, dict(exposed_keys=[], cross_language_exposure_resolution_complete=True))
    access = []
    if blocked:
        access = [
            dict(
                **e.ref(files and Path(files["policy"]["path"])),
                status=429,
                error="HTTP Error 429",
                url="https://de.wikipedia.org/w/api.php",
            )
        ]
    manifest = tmp_path / "manifest.json"
    atomic_json(
        manifest,
        dict(
            scope="private_fixture",
            revision=e.REVISION,
            files=files,
            resolutions=e.ref(resolve),
            exposure_inventory=e.ref(inventory),
            metadata=e.ref(metadata),
            resolution_receipts=[],
            source_access_receipts=access,
            network_bytes=100,
            network_duration_s=1,
        ),
    )
    return manifest


def test_inspect_authenticity(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9015-CUSTODY: original hashes, revision, policy and access stay explicit."""
    manifest = inputs(tmp_path)
    result = e.inspect(manifest, tmp_path / "audit")
    assert result["natural_pilot_ready_score"] == 0 and len(result["raw_annotator_rows"]) == 400
    info = json.loads(manifest.read_bytes())
    original = deepcopy(info)
    info["revision"] = "main"
    atomic_json(manifest, info)
    with pytest.raises(ValueError, match="revision"):
        e.inspect(manifest, tmp_path / "bad")
    atomic_json(manifest, original)
    policy = Path(info["files"]["policy"]["path"])
    policy.write_text("tampered")
    with pytest.raises(ValueError, match="changed_bytes"):
        e.inspect(manifest, tmp_path / "bad")
    original["files"]["policy"] = e.ref(policy)
    atomic_json(manifest, original)
    assert e.inspect(manifest, tmp_path / "noauthority")["gate_check_summary"][0]["passed"] is False
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    m = inputs(blocked, True)
    result = e.inspect(m, blocked / "audit")
    assert result["gate_check_summary"][0]["check"] == "source_resolution_access"
    assert result["gate_check_summary"][0]["observed"] == 429
    assert result["completed_count"] == 0 and result["labels_deferred_due_to_external_block"]


def cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """A real child must import from its script path with no checkout working directory."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI), *args]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    return subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)


def test_runner_cli_missing_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9015-CUSTODY: real missing-input publication and cold tamper refusal."""
    output = tmp_path / (e.NAME + ".json")
    run = cli(tmp_path, "--inputs", str(tmp_path / "missing"), "--output", str(output))
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert "missing" in str(value["gate_check_summary"])
    assert cli(tmp_path, "--cold-replay", str(output)).returncode == 0
    value["completed_count"] += 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", str(output)).returncode == 1
    assert cli(tmp_path, "--date", "2026111").returncode == 1
    assert cli(tmp_path, "--cold-replay", str(tmp_path / "missing")).returncode == 1


def receipt_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Private receipt policy controls are separate from actual coverage in the delivered run."""
    from carnot.reporting import human_disagreement_runner_9015 as r

    monkeypatch.setattr(r, "verify_measured", lambda report, owned: report.get("complete", False))
    scope = r.scope_manifest([dict(name=n, argv=["private", n]) for n in r.CHECKS])
    log = tmp_path / "log"
    log.write_text("actual retained private bytes")
    checks = [
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
    return scope, dict(checks=checks, owned_statement_coverage=dict(complete=True))


@pytest.mark.parametrize(
    "kind",
    [
        "valid",
        "missing",
        "stale",
        "failed",
        "timeout",
        "log",
        "command",
        "code",
        "coverage",
        "malformed",
    ],
)
def test_receipt_rejections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    """SCENARIO-REPORT-9015-CUSTODY: retained passing flags cannot hide rejected evidence."""
    from carnot.reporting import human_disagreement_runner_9015 as r

    scope, validation = receipt_fixture(tmp_path, monkeypatch)
    if kind == "missing":
        validation["checks"].pop()
    if kind == "stale":
        validation["checks"][0]["started_monotonic_ns"] = 0
    if kind == "failed":
        validation["checks"][0]["exit_code"] = 1
    if kind == "timeout":
        validation["checks"][0]["timed_out"] = True
    if kind == "log":
        validation["checks"][0]["stdout_sha256"] = "wrong"
    if kind == "command":
        validation["checks"][0]["argv"] = ["wrong"]
    if kind == "code":
        scope["code_test_hashes"][0]["sha256"] = "wrong"
    if kind == "coverage":
        validation["owned_statement_coverage"]["complete"] = False
    if kind == "malformed":
        validation = {}
    assert r.receipts_pass(scope, validation) == (kind == "valid")


def test_runner_replay_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9015-CUSTODY: independently recompute source summaries and sidecars."""
    from carnot.reporting import human_disagreement_runner_9015 as r

    manifest = inputs(tmp_path)
    output = tmp_path / (e.NAME + ".json")
    scope, validation = receipt_fixture(tmp_path, monkeypatch)
    reference = r.measure(manifest, output, "20261111", scope)
    value = r.build(reference, validation)
    atomic_json(output, value)
    assert value["verdict_class"] == "null" and value["natural_pilot_ready_score"] == 0
    assert r.replay(output)["passed"]
    original = json.loads(Path(reference["path"]).read_bytes())
    for kind in ["aggregate", "sidecar", "operand", "health"]:
        data = deepcopy(original)
        if kind == "aggregate":
            data["reduction"]["completed_count"] += 1
        if kind == "sidecar":
            p = Path(data["reduction"]["label_sidecar_hashes"][0]["path"])
            body = p.read_bytes()
            p.write_text("{}")
            data["reduction"]["label_sidecar_hashes"][0] = e.ref(p)
        if kind == "operand":
            data["code_config_hashes"][0]["sha256"] = "wrong"
        if kind == "health":
            data["repository_health"]["references"][0]["sha256"] = "wrong"
        atomic_json(Path(reference["path"]), data)
        atomic_json(output, r.build(e.ref(Path(reference["path"])), validation))
        with pytest.raises(ValueError):
            r.replay(output)
        if kind == "sidecar":
            p.write_bytes(body)
    atomic_json(Path(reference["path"]), original)
    atomic_json(output, r.build(e.ref(Path(reference["path"])), validation))
    bad = e.ref(tmp_path / "nonexistent")
    monkeypatch.setattr(
        r.history, "health", lambda root: dict(references=[bad | dict(sha256="wrong")])
    )
    with pytest.raises(ValueError, match="historical_health_hash"):
        r.measure(manifest, output, "20261111", scope)


def test_external_block_and_resources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9015-CUSTODY: observed external failures block while owned checks stay separate."""
    from carnot.reporting import human_disagreement_runner_9015 as r

    manifest = inputs(tmp_path, True)
    scope, validation = receipt_fixture(tmp_path, monkeypatch)
    output = tmp_path / (e.NAME + ".json")
    reference = r.measure(manifest, output, "20261111", scope)
    value = r.build(reference, validation)
    assert value["honest_verdict"] == "complete_blocked_source_resolution_access"
    assert value["required_checks_passed"] and not value["repository_health_passed"]
    monkeypatch.setattr(r.shutil, "disk_usage", lambda path: type("Disk", (), dict(free=0))())
    value = r.build(r.measure(manifest, output, "20261111", scope), validation)
    assert value["honest_verdict"] == "complete_blocked_authenticated_inputs"


def test_pinned_author_and_invalid_identity(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9015-CUSTODY: relabeling private bytes cannot impersonate author authority."""
    manifest = inputs(tmp_path)
    data = json.loads(manifest.read_bytes())
    data["scope"] = "author_release"
    atomic_json(manifest, data)
    with pytest.raises(ValueError, match="pinned_release"):
        e.inspect(manifest, tmp_path / "audit")
    rows = records(1)
    ids = resolutions(rows)
    ids[rows[0]["wikipedia_url"]] = "not-a-Wikidata-id"
    assert e.audit(rows, set(), ids, tmp_path / "badid")["independent_count"] == 0


def test_label_failure_and_missing_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9015-LABELS: missing original judgments preserve a failed selected slot."""
    from carnot.reporting import human_disagreement_runner_9015 as r

    rows = records()
    for row in rows:
        row["annotations"] = []
    result = e.audit(rows, set(), resolutions(rows), tmp_path / "badlabels")
    assert result["annotation_failures"] and len(result["role_manifest"]) == 400
    manifest = tmp_path / "badmanifest"
    atomic_json(manifest, {})
    scope, validation = receipt_fixture(tmp_path, monkeypatch)
    reference = r.measure(manifest, tmp_path / (e.NAME + ".json"), "20261111", scope)
    assert "missing_field:" in str(r.build(reference, validation)["gate_check_summary"])


def test_source_cli_and_unresolved_exposure(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9015-CUSTODY: original-source CLI replay and exposure denial use private bytes."""
    manifest = inputs(tmp_path)
    output = tmp_path / (e.NAME + ".json")
    run = cli(tmp_path, "--inputs", str(manifest), "--output", str(output))
    assert run.returncode == 0, run.stdout + run.stderr
    assert cli(tmp_path, "--cold-replay", str(output)).returncode == 0
    data = json.loads(manifest.read_bytes())
    inventory = Path(data["exposure_inventory"]["path"])
    atomic_json(inventory, dict(exposed_keys=[], cross_language_exposure_resolution_complete=False))
    data["exposure_inventory"] = e.ref(inventory)
    atomic_json(manifest, data)
    result = e.inspect(manifest, tmp_path / "exposure_block")
    assert result["completed_count"] == 0 and result["labels_deferred_due_to_external_block"]
    assert any(
        g["check"] == "exposure_source_integrity" and not g["passed"]
        for g in result["gate_check_summary"]
    )


@pytest.mark.parametrize("custody", ["missing", "local", "explicit"])
def test_output_local_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, custody: str
) -> None:
    """SCENARIO-REPORT-9015-LOCAL-VALIDATION: checkout receipts cannot qualify private output."""
    from carnot.reporting import human_disagreement_runner_9015 as r

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    foreign_scope, foreign_validation = receipt_fixture(checkout, monkeypatch)
    atomic_json(checkout / "validation_scope_manifest.json", foreign_scope)
    atomic_json(checkout / "validation.json", foreign_validation)
    monkeypatch.setattr(r, "RAW", checkout)
    output = tmp_path / "output" / (e.NAME + ".json")
    args = ["--inputs", str(tmp_path / "missing"), "--output", str(output)]
    expected_scope, expected_validation = None, {}
    if custody != "missing":
        local = output.parent / "raw" / e.NAME
        local.mkdir(parents=True)
        directory = tmp_path / "explicit" if custody == "explicit" else local
        directory.mkdir(exist_ok=True)
        expected_scope, expected_validation = receipt_fixture(directory, monkeypatch)
        atomic_json(directory / "validation_scope_manifest.json", expected_scope)
        atomic_json(directory / "validation.json", expected_validation)
        if custody == "explicit":
            atomic_json(local / "validation_scope_manifest.json", {})
            atomic_json(local / "validation.json", {})
            args += [
                "--scope",
                str(directory / "validation_scope_manifest.json"),
                "--receipts",
                str(directory / "validation.json"),
            ]
    monkeypatch.setattr(r, "validate", lambda candidate, logs: dict(passed=True))
    assert r.main(args) == 0
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"] is (custody != "missing")
    assert value["verdict_class"] == ("disqualified" if custody == "missing" else "blocked")
    assert value["validation_receipts"] == expected_validation.get("checks", [])
    if expected_scope is None:
        assert value["validation_scope_manifest"]["commands"] == []
    else:
        assert value["validation_scope_manifest"] == expected_scope
