"""REQ-REPORT-9097-V776 / SCENARIO-REPORT-9097-V776: original authority only."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import human_authority_9097 as d
from carnot.reporting import human_authority_runner_9097 as runner
from carnot.reporting.current_work_receipt import atomic_json


def test_original_authority_and_shards(tmp_path: Path) -> None:
    """REQ-REPORT-9097-V776: exact old code and author bytes agree for every item."""
    value = d.inspect(d.ROOT, tmp_path / "raw")
    assert value["blocker"] is None
    assert len(value["rows"]) == 900 and value["question_count"] == 300
    assert value["role_counts"] == dict(TRAIN=450, CAL=150, TEST=300)
    assert value["annotator1_observed_count"] == 900
    assert value["annotator2_observed_count"] == 300
    assert value["annotator2_question_count"] == 100
    assert value["historical_producer_snapshot_reference"]["sha256"] == d.SNAPSHOT_PIN
    assert value["replay_receipt"]["compared_rows"] == 900
    a = json.loads(Path(value["shard_a_roster_reference"]["path"]).read_bytes())
    b = json.loads(Path(value["shard_b_roster_reference"]["path"]).read_bytes())
    qids = sorted({r["qid"] for r in value["rows"]})
    assert a["qids"] == qids[:150] and b["qids"] == qids[150:]
    assert len(a["rows"]) == len(b["rows"]) == 450
    for shard in [a, b]:
        features = json.loads(Path(shard["feature_reference"]["path"]).read_bytes())
        assert all(set(view) == {"question", "answer_text"} for view in features["views"])
    assert value["static_protocol"] == d.methods.protocol()


def test_missing_and_substituted_snapshot(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9097-V776: absent closure blocks; rehashing cannot bless current code."""
    missing = d.inspect(tmp_path, tmp_path / "raw")
    assert missing["blocker"]
    assert missing["gate_check_summary"][0]["observed"] == "missing_file"
    snapshot = tmp_path / "snapshot.py"
    snapshot.write_bytes((d.ROOT / d.PRODUCER).read_bytes())
    with pytest.raises(ValueError, match="historical_snapshot"):
        d.read_historical(snapshot)


@pytest.mark.parametrize("mutation", ["label", "qid", "role", "group", "missing"])
def test_projection_rejects_mutations(tmp_path: Path, mutation: str) -> None:
    """SCENARIO-REPORT-9097-V776: independent bytes refute modified replay rows."""
    rows = [
        dict(unit_id="one", qid="q1", group_id="g", annotator1=0, annotator2=None, role="TRAIN")
    ]
    changed = deepcopy(rows)
    if mutation == "missing":
        changed.clear()
    else:
        changed[0][dict(label="annotator1", qid="qid", role="role", group="group_id")[mutation]] = (
            "changed"
        )
    with pytest.raises(ValueError, match="historical_projection"):
        d.compare_projection(rows, changed)


def test_build_and_semantic_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-9097-V776: no receipt means disqualified; summaries cannot invent readiness."""
    reference = runner.measure(tmp_path / "missing", tmp_path / "raw", "20261115", {})
    value = runner.build(reference, {})
    assert value["verdict_class"] == "disqualified" and value["source_ready_score"] == 0
    monkeypatch.setattr(runner, "receipts_pass", lambda _s, _v: True)
    value = runner.build(reference, {})
    assert value["verdict_class"] == "blocked"
    assert value["independent_eval_ready_score"] == value["current_llm_invocations"] == 0
    path = tmp_path / (d.NAME + ".json")
    atomic_json(path, value)
    assert runner.replay(path)["passed"]
    value["source_ready_score"] = 1
    atomic_json(path, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        runner.replay(path)


def test_seal_and_snapshot_failure_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9097-V776: missing Git and substituted closure never become authority."""
    path = tmp_path / "frozen.json"
    d.seal(path, dict(a=1))
    with pytest.raises(ValueError, match="sealed_drift"):
        d.seal(path, dict(a=2))
    monkeypatch.setattr(d, "child", lambda *_a, **_k: dict(passed=False))
    with pytest.raises(ValueError, match="historical_git_blob"):
        d.snapshots(tmp_path, tmp_path / "git-fail", [], [])
    with pytest.raises(ValueError, match="historical_replay"):
        d.replay_mapper(tmp_path, {}, {}, {})


def test_private_mode_and_replay_mutations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-9097-V776: storage, operand, protocol and primitive drift refuse readiness."""
    raw = tmp_path / "public"
    raw.mkdir(mode=0o755)
    reference = runner.measure(tmp_path / "missing", raw, "20261115", {})
    value = runner.build(reference, {})
    path = tmp_path / (d.NAME + ".json")
    atomic_json(path, value)
    assert runner.replay(path)["passed"]
    changed = deepcopy(value)
    changed["source_references"][0]["sha256"] = "changed"
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="operand_hash"):
        runner.replay(path)
    atomic_json(path, value)
    with monkeypatch.context() as m:
        m.setattr(d.methods, "protocol", lambda: {})
        with pytest.raises(ValueError, match="protocol_drift"):
            runner.replay(path)
    actual = d.inspect
    with monkeypatch.context() as m:

        def different(root: Path, raw: Path) -> dict:
            result = actual(root, raw)
            result["question_count"] = 999
            return result

        m.setattr(d, "inspect", different)
        with pytest.raises(ValueError, match="source_reduction_drift"):
            runner.replay(path)


def test_main_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9097-V776: publication is gated; old primaries survive refusal."""
    scope = runner.scope_manifest([])
    assert scope["owned_paths"] == d.OWNED
    assert not runner.receipts_pass(scope, {})
    output = tmp_path / (d.NAME + ".json")
    args = [
        "--root",
        str(tmp_path / "missing"),
        "--output",
        str(output),
        "--scope",
        str(tmp_path / "absent"),
        "--receipts",
        str(tmp_path / "absent"),
    ]
    assert runner.main(["--date", "bad"]) == 1
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    assert runner.main(args) == 1 and not output.exists()
    monkeypatch.setattr(runner, "receipts_pass", lambda _s, _v: True)
    monkeypatch.setattr(runner, "validate", lambda _p, _l: dict(passed=True))
    assert runner.main(args) == 0
    assert runner.main(["--cold-replay", str(output)]) == 0
    atomic_json(tmp_path / "scope.json", scope)
    atomic_json(tmp_path / "validation.json", {})
    assert (
        runner.main(
            args[:-4]
            + [
                "--scope",
                str(tmp_path / "scope.json"),
                "--receipts",
                str(tmp_path / "validation.json"),
            ]
        )
        == 0
    )


def test_foreign_cli_scanners_and_positive_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9097-V776: real foreign CLI and unchanged strict scanners inspect bytes."""
    import os
    from carnot.reporting.v709_execution import child

    reference = runner.measure(d.ROOT, tmp_path / "raw", "20261115", {})
    value = runner.build(reference, {})
    path = tmp_path / (d.NAME + ".json")
    atomic_json(path, value)
    assert runner.validate(path, tmp_path / "terminal")["passed"]
    prefix = ["env", "-u", "PYTHONPATH", "-C", str(tmp_path), str(d.ROOT / ".venv/bin/python")]
    config = os.environ.get("COVERAGE_RCFILE")
    if config:
        prefix += ["-m", "coverage", "run", "--rcfile=" + config]
    prefix += [str(d.ROOT / d.CLI)]
    assert child("cold", prefix + ["--cold-replay", str(path)], tmp_path / "cli")["passed"]
    value["completed_count"] = 999
    atomic_json(path, value)
    assert child("tamper", prefix + ["--cold-replay", str(path)], tmp_path / "cli", expected=1)[
        "passed"
    ]
    assert child("bad_date", prefix + ["--date", "bad"], tmp_path / "cli", expected=1)["passed"]
    assert child(
        "unqualified",
        prefix
        + [
            "--root",
            str(tmp_path / "absent"),
            "--scope",
            str(tmp_path / "absent"),
            "--receipts",
            str(tmp_path / "absent"),
        ],
        tmp_path / "cli",
        expected=1,
    )["passed"]
    monkeypatch.setattr(runner, "receipts_pass", lambda _s, _v: True)
    positive = runner.build(reference, {})
    assert positive["verdict_class"] == "positive" and positive["source_ready_score"] == 1
    assert positive["independent_eval_ready_score"] == 0
    assert positive["current_llm_invocations"] == 0


def private_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Copied primaries isolate negative controls without changing historical pins or bytes."""
    root, raw = tmp_path / "inputs", tmp_path / "raw"
    for name in d.PINS:
        path = root / "results" / (name + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((d.ROOT / "results" / path.name).read_bytes())
    role = root / "openspec/change-proposals/v726-methods-protocol.json"
    role.parent.mkdir(parents=True)
    role.write_bytes((d.ROOT / "openspec/change-proposals/v726-methods-protocol.json").read_bytes())
    d.snapshots(d.ROOT, raw, [], [])
    monkeypatch.setattr(d, "PINS", dict(d.PINS))
    return root, raw


def replace_primary(root: Path, number: str, field: str, value: object) -> None:
    """Fixture-only repinning tests semantic checks below the independently pinned primary boundary."""
    path = next((root / "results").glob("experiment_" + number + "_*.json"))
    document = json.loads(path.read_bytes())
    if value == "missing_field":
        document.pop(field)
    else:
        document[field] = value
    atomic_json(path, document)
    d.PINS[path.stem] = d.ref(path)["sha256"].split(":")[1]


@pytest.mark.parametrize("operand", ["missing_field", 0])
def test_upstream_missing_field_and_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operand: object
) -> None:
    """SCENARIO-REPORT-9097-V776: exact gate diagnostics distinguish missing values and zero."""
    root, raw = private_tree(tmp_path, monkeypatch)
    replace_primary(root, "8403", "release_criterion_ready_score", operand)
    result = d.inspect(root, raw)
    gate = next(
        c
        for c in result["gate_check_summary"]
        if c["artifact_field"] == "release_criterion_ready_score"
    )
    assert gate["observed"] == operand and not gate["passed"]


@pytest.mark.parametrize("mutation", ["lineage", "label", "roles", "static", "closure", "support"])
def test_full_closure_negative_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """SCENARIO-REPORT-9097-V776: rehashed labels, split drift and partial closure all fail."""
    root, raw = private_tree(tmp_path, monkeypatch)
    if mutation in {"lineage", "label"}:
        primary = json.loads(next((root / "results").glob("experiment_8389_*.json")).read_bytes())
        manifest = json.loads(Path(primary["manifest_reference"]["path"]).read_bytes())
        if mutation == "lineage":
            manifest["release_commit"] = "replacement"
        else:
            import csv
            import io

            source = manifest["release_files"][d.methods.old.MASTER]
            records = list(
                csv.DictReader(io.StringIO(Path(source["path"]).read_text(encoding="utf-8-sig")))
            )
            records[0]["annotator1_label"] = str(1 - int(records[0]["annotator1_label"]))
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
            changed = tmp_path / "labels.csv"
            changed.write_text(stream.getvalue())
            manifest["release_files"][d.methods.old.MASTER] = d.ref(changed)
        manifest_path = tmp_path / "manifest.json"
        atomic_json(manifest_path, manifest)
        replace_primary(root, "8389", "manifest_reference", d.ref(manifest_path))
    elif mutation == "roles":
        path = root / "openspec/change-proposals/v726-methods-protocol.json"
        roles = json.loads(path.read_bytes())
        roles["partition"]["roles"]["fit"][0] = roles["partition"]["roles"]["test"][0]
        atomic_json(path, roles)
    elif mutation == "static":
        monkeypatch.setattr(d.methods, "protocol", lambda: {})
    elif mutation == "closure":
        original = d.replay_mapper

        def incomplete(*args: object) -> dict:
            value = original(*args)
            value["imported_modules"] = {}
            return value

        monkeypatch.setattr(d, "replay_mapper", incomplete)
    else:
        original_inventory = d.methods.inventory

        def insufficient(*args: object) -> dict:
            value = original_inventory(*args)
            value["annotator2_observed_count"] = 299
            return value

        monkeypatch.setattr(d.methods, "inventory", insufficient)
    assert d.inspect(root, raw)["blocker"] is not None


def test_worker_rejects_rehashed_closure_and_projection(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9097-V776: isolated child refuses incomplete imports and changed projection."""
    from carnot.reporting import human_authority_replay_9097 as worker
    from carnot.reporting.v709_execution import child
    import os

    value = d.inspect(d.ROOT, tmp_path / "raw")
    assert value["blocker"] is None
    raw = tmp_path / "raw"
    job = json.loads((raw / "replay_job.json").read_bytes())
    with pytest.raises(ValueError, match="worker_operand_hash"):
        worker.checked(dict(path=str(raw / "replay_job.json"), sha256="changed"))
    prefix = [str(d.ROOT / ".venv/bin/python"), "-I"]
    config = os.environ.get("COVERAGE_RCFILE")
    if config:
        prefix += ["-m", "coverage", "run", "--rcfile=" + config]
    prefix += [str(d.ROOT / d.WORKER)]
    job["modules"].pop("external_evidence_8375")
    atomic_json(raw / "bad_job.json", job)
    assert child("closure", prefix + [str(raw / "bad_job.json")], raw / "negative", expected=1)[
        "passed"
    ]
    assert "worker_import_closure" in (raw / "negative/closure.stderr").read_text()
    projection = raw / "historical_projection.json"
    changed = json.loads(projection.read_bytes())
    changed["rows"][0]["role"] = "TEST"
    atomic_json(projection, changed)
    assert child(
        "projection", prefix + [str(raw / "replay_job.json")], raw / "negative", expected=1
    )["passed"]
    assert "worker_projection_drift" in (raw / "negative/projection.stderr").read_text()


def test_snapshot_package_cannot_inject_code(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9097-V776: snapshot package initializers are part of import closure."""
    package = tmp_path / "snapshot/carnot"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('injected')\n")
    with pytest.raises(ValueError, match="snapshot_package"):
        d.snapshots(d.ROOT, tmp_path, [], [])


def test_worker_rejects_package_injection(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9097-V776: the child checks package bytes immediately before import."""
    from carnot.reporting import human_authority_replay_9097 as worker

    root = tmp_path / "snapshot/carnot"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("injected")
    job = tmp_path / "job.json"
    atomic_json(job, dict(modules={}, snapshot_root=str(root.parent)))
    with pytest.raises(ValueError, match="worker_package_closure"):
        worker.main(job)


def test_sealed_feature_and_label_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9097-V776: a new checksum cannot authorize changed learner/evaluator bytes."""
    value = d.inspect(d.ROOT, tmp_path / "raw")
    assert value["blocker"] is None
    labels = Path(value["evaluator_label_reference"]["path"])
    changed = json.loads(labels.read_bytes())
    changed["rows"][0]["annotator1"] = 1 - changed["rows"][0]["annotator1"]
    atomic_json(labels, changed)
    replayed = d.inspect(d.ROOT, tmp_path / "raw")
    assert replayed["blocker"] == "source_closure"
    assert "sealed_drift:evaluator_labels.json" in replayed["gate_check_summary"][-1]["observed"]
