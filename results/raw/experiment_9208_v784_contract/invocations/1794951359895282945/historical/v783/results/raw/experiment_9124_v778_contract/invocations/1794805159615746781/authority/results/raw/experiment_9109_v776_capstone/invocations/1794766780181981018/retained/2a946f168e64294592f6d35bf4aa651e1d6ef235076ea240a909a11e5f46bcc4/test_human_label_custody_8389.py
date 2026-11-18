"""REQ-REPORT-8389 and REQ-VERIFY-8389: private controls cannot become observations."""

from copy import deepcopy
import csv
import io
import json
from pathlib import Path

import pytest

from carnot.reporting import human_label_custody_8389 as e
from carnot.reporting.current_work_receipt import atomic_json


def csv_bytes(rows: list[dict]) -> bytes:
    """Private source bytes let tests change columns without touching real releases."""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def records(count: int = 300) -> list[dict]:
    """Three answers share one question; repeated generators do not add independence."""
    return [
        dict(
            item_id=f"{i}-{model}",
            qid=f"q{i}",
            dataset="hotpotqa",
            model=model,
            question=f"Question {i}?",
            answer_text=f"Answer {i} {model}",
            annotator1_label=str(i % 2),
            annotator2_label=str(i % 2 if i < 100 else -1),
            label_gpt_5_mini_p1=str(1 - i % 2),
            rougeL="0.4",
        )
        for i in range(count)
        for model in e.GENERATORS
    ]


def private_manifest(tmp_path: Path, monkeypatch, count: int = 300) -> dict:
    """Private pins use the production parser but never authorize public release bytes."""
    rows = records(count)
    refs = {}
    for name in e.RELEASE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(csv_bytes(rows) if name in (e.MASTER, e.AUDIT) else b"private docs")
        refs[name] = e.reference(path)
    monkeypatch.setattr(e, "authenticate", lambda name, ref: e.read_reference(ref))
    return dict(
        release_commit=e.COMMIT,
        release_files=refs,
        target_mapping=deepcopy(e.TARGET),
        overlap_inventory=[dict(path="private", comparable=True, rows=[])],
        license_permitted=True,
        acquisition_gates=[],
    )


def test_panel_and_partial_annotations(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8389-CUSTODY: 900 answers remain 300 independent questions."""
    manifest = private_manifest(tmp_path, monkeypatch)
    value = e.validate_manifest(manifest)
    assert value["response_count"] == 900 and value["question_cluster_count"] == 300
    assert value["independent_count"] == 300 and value["released_label_panel_ready_score"] == 1
    assert value["label_authority_counts"]["A1_valid"] == 900
    assert value["label_authority_counts"]["A2_valid"] == 300
    assert value["label_authority_counts"]["A2_missing"] == 600
    assert all(
        r["human_primary"] != r["automatic_judgments"]["label_gpt_5_mini_p1"] for r in value["rows"]
    )
    assert len({r["cluster"] for r in value["rows"]}) == 300


def test_swapped_mapping_and_wrong_generator(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8389-CUSTODY: automatic labels cannot acquire human authority."""
    manifest = private_manifest(tmp_path, monkeypatch)
    changed = deepcopy(manifest)
    changed["target_mapping"]["primary"] = "label_gpt_5_mini_p1"
    with pytest.raises(ValueError, match="target_mapping"):
        e.validate_manifest(changed)
    rows = records()
    rows[0]["model"] = "wrong-generator"
    Path(manifest["release_files"][e.AUDIT]["path"]).write_bytes(csv_bytes(rows))
    manifest["release_files"][e.AUDIT] = e.reference(
        Path(manifest["release_files"][e.AUDIT]["path"])
    )
    assert e.validate_manifest(manifest)["released_label_panel_ready_score"] == 0


def test_duplicates_truncation_and_unknown_overlap(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8389-CUSTODY: unavailable identity blocks instead of measuring null."""
    manifest = private_manifest(tmp_path, monkeypatch, 299)
    value = e.validate_manifest(manifest)
    assert value["intended_count"] == 900 and len(value["missing_slots"]) == 3
    assert value["released_label_panel_ready_score"] == 0
    manifest = private_manifest(tmp_path, monkeypatch)
    manifest["overlap_inventory"][0]["comparable"] = False
    assert e.validate_manifest(manifest)["independent_count"] == 0
    rows = records()
    rows[3]["question"] = rows[0]["question"]
    master = Path(manifest["release_files"][e.MASTER]["path"])
    master.write_bytes(csv_bytes(rows))
    manifest["release_files"][e.MASTER] = e.reference(master)
    assert e.validate_manifest(manifest)["released_label_panel_ready_score"] == 0


def test_pinned_bytes_and_normalization(tmp_path):
    """SCENARIO-REPORT-8389-CUSTODY: self-consistent hashes do not change reviewed git blobs."""
    path = tmp_path / "changed.csv"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="reviewed_blob"):
        e.authenticate(e.MASTER, e.reference(path))
    assert e.normalized("  QUESTION\u00a0A? ") == e.normalized("question a?")


def test_missing_release(tmp_path):
    """SCENARIO-REPORT-8389-CUSTODY: missing targets preserve all advertised slots."""
    value = e.validate_manifest(
        dict(
            release_commit=e.COMMIT,
            release_files={},
            target_mapping=e.TARGET,
            overlap_inventory=[],
            license_permitted=False,
            acquisition_gates=[],
        )
    )
    assert len(value["rows"]) == 900 and value["censored_count"] == 900
    assert value["response_count"] == 0 and value["released_label_panel_ready_score"] == 0


def empty_work(tmp_path, monkeypatch):
    """REQ-VERIFY-8389: a blocked release is valid evidence for private CLI controls."""
    from carnot.reporting import human_label_custody_runner_8389 as runner

    monkeypatch.setattr(
        runner.authority,
        "authority",
        lambda *a: dict(
            activated=True,
            tasks=[dict(id=e.TASK, MODEL_SPECS=[], inference_substrate_class="no_model_load")],
            gate_check_summary=[],
            canonical_tasks_sha256="private",
        ),
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    work = runner.measure(e.ROOT, raw, tmp_path / "absent.json", tmp_path)
    work["work_path"] = str(raw / "work.json")
    atomic_json(Path(work["work_path"]), work)
    return work


def test_build_replay_and_real_children(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8389-REPLAY: real fresh children exercise all four controls."""
    from carnot.reporting import human_label_custody_runner_8389 as runner

    work = empty_work(tmp_path, monkeypatch)
    receipts = [dict(passed=True, scope="owned")]
    value = e.build(work, receipts)
    assert value["verdict_class"] == "blocked" and value["MODEL_SPECS"] == []
    assert set(value) == set(value["field_principles"])
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert e.replay(candidate)
    controls = runner.controls(candidate, tmp_path / "controls")
    assert all(r["passed"] for r in controls)
    assert [r["exit_code"] for r in controls] == [0, 1, 1, 1]
    assert e.build(work, [dict(passed=False)])["verdict_class"] == "disqualified"
    value["response_count"] += 1
    atomic_json(candidate, value)
    assert not e.replay(candidate)


def test_runner_publication_and_failure(tmp_path, monkeypatch):
    """REQ-VERIFY-8389: real publication keeps the terminal primary byte bound."""
    from carnot.reporting import human_label_custody_runner_8389 as runner
    from carnot.reporting.primary_publication import read_bound_sidecar

    work = empty_work(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "measure", lambda *a: work.copy())
    monkeypatch.setattr(runner, "plan", lambda *a: [])
    output = tmp_path / "results" / (e.NAME + ".json")
    assert (
        runner.main(
            ["--date", "20261010", "--manifest", str(tmp_path / "missing"), "--output", str(output)]
        )
        == 0
    )
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"] and value["verdict_class"] == "blocked"
    publication = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert read_bound_sidecar(output, Path(publication["sidecar_path"]))["report"]["passed"]
    assert runner.main(["--cold-replay", str(output)]) == 0
    assert runner.main(["--deliberate-error"]) == 1
    monkeypatch.setattr(runner, "measure", lambda *a: (_ for _ in ()).throw(ValueError("owned")))
    assert runner.main(["--output", str(output)]) == 1


def test_runner_plan(tmp_path):
    """REQ-VERIFY-8389: private command identities freeze before measurement."""
    from carnot.reporting import human_label_custody_runner_8389 as runner

    specs = runner.plan(tmp_path)
    assert any(r["name"] == "coverage_report" and "--fail-under=100" in r["argv"] for r in specs)
    assert any("strict_mypy" == r["name"] for r in specs)
    assert any("private_E2E018" in r["name"] for r in specs)


def test_acquisition_caps_and_failures(tmp_path, monkeypatch):
    """REQ-VERIFY-8389: transport errors and caps retain missing-byte gates."""
    monkeypatch.setattr(e, "authenticate", lambda name, ref: e.read_reference(ref))
    monkeypatch.setattr(e.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"private"))
    manifest = e.acquire(tmp_path / "success")
    assert len(manifest["release_files"]) == 7 and manifest["acquisition_bytes"] == 49
    assert all(
        Path(r["path"]).stat().st_mode & 0o222 == 0 for r in manifest["release_files"].values()
    )
    monkeypatch.setattr(
        e.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("absent"))
    )
    assert len(e.acquire(tmp_path / "missing")["acquisition_gates"]) == 7
    tick = iter([0] + list(range(601, 609)))
    monkeypatch.setattr(e.time, "monotonic", lambda: next(tick))
    assert all(
        "deadline" in g["observed_value"] for g in e.acquire(tmp_path / "late")["acquisition_gates"]
    )
    monkeypatch.undo()
    monkeypatch.setattr(
        e.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"x" * (52428800 + 1))
    )
    assert len(e.acquire(tmp_path / "too_large")["acquisition_gates"]) == 7


def test_prior_source_custody_and_rehashed_contamination(tmp_path):
    """SCENARIO-REPORT-8389-CUSTODY: opaque hashes stay unknown even after rehashing."""
    root = tmp_path / "repo"
    raw = tmp_path / "raw"
    inventory = e.overlap_inventory(root, raw)
    assert all(not r["comparable"] for r in inventory)
    for item in inventory:
        path = Path(item["path"])
        atomic_json(path, dict(source_role_manifest=[dict(qid="prior", question="Question 0?")]))
    inventory = e.overlap_inventory(root, raw)
    manifest = dict(
        release_commit=e.COMMIT,
        release_files={},
        target_mapping=e.TARGET,
        overlap_inventory=inventory,
        license_permitted=False,
        acquisition_gates=[],
    )
    assert e.validate_manifest(manifest)["acceptance_gates"]["known_question_overlap"]
    changed = deepcopy(manifest)
    changed["overlap_inventory"][0]["rows"] = []
    with pytest.raises(ValueError, match="source_overlap_custody"):
        e.validate_manifest(changed)
    copy = Path(inventory[0]["path"])
    atomic_json(copy, dict(source_role_manifest=[dict(question="clean replacement")]))
    changed["overlap_inventory"][0].update(
        e.reference(copy), rows=[dict(question="clean replacement")]
    )
    with pytest.raises(ValueError, match="input_hash"):
        e.validate_manifest(changed)


def test_annotation_absence_disagreement_and_overlap(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8389-CUSTODY: partial humans and duplicates remain explicit."""
    manifest = private_manifest(tmp_path, monkeypatch)
    rows = records()
    rows[0]["annotator1_label"] = "-1"
    rows[1]["annotator2_label"] = "1"
    rows.pop(3)
    master = Path(manifest["release_files"][e.MASTER]["path"])
    master.write_bytes(csv_bytes(rows))
    manifest["release_files"][e.MASTER] = e.reference(master)
    manifest["overlap_inventory"][0]["rows"] = [dict(qid="q2", question="Question 2?")]
    value = e.validate_manifest(manifest)
    assert value["label_authority_counts"]["A1_missing"] == 1
    assert value["label_authority_counts"]["A1_A2_disagreement"] == 1
    assert any(r["status"] == "overlap" for r in value["overlap_rows"])
    assert value["missing_slots"][0]["reason"] == "missing_generator"
    rows[0]["annotator1_label"] = "2"
    master.write_bytes(csv_bytes(rows))
    manifest["release_files"][e.MASTER] = e.reference(master)
    with pytest.raises(ValueError, match="human_label_encoding"):
        e.validate_manifest(manifest)


def test_measure_supplied_and_resource_failure(tmp_path, monkeypatch):
    """REQ-VERIFY-8389: preconditions prevent acquisition when tools are absent."""
    from carnot.reporting import human_label_custody_runner_8389 as runner

    empty_work(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.json"
    atomic_json(
        manifest,
        dict(
            release_commit=e.COMMIT,
            release_files={},
            target_mapping=e.TARGET,
            overlap_inventory=[],
            license_permitted=False,
            acquisition_gates=[],
        ),
    )
    assert runner.measure(e.ROOT, tmp_path / "present", manifest, tmp_path)["primitive_reference"]
    root = tmp_path / "empty_root"
    root.mkdir()
    monkeypatch.setattr(e, "OWNED", [])
    monkeypatch.setattr(runner.authority, "authority", lambda *a: dict(activated=False, tasks=[]))
    work = runner.measure(root, tmp_path / "missing_tools", None, tmp_path)
    assert {g["check"] for g in work["gate_check_summary"]} >= {
        "resource_preconditions",
        "current_task_authority",
        "input_available",
    }
    assert not work["preconditions_checked"]["tools"]["python"]


def test_reviewed_blob_success_and_swapped_columns(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8389-CUSTODY: new hashes cannot authorize swapped target bytes."""
    import hashlib

    path = tmp_path / "private.csv"
    original = csv_bytes(records(1))
    path.write_bytes(original)
    private_pin = hashlib.sha1(
        b"blob " + str(len(original)).encode() + b"\0" + original, usedforsecurity=False
    ).hexdigest()
    monkeypatch.setitem(e.RELEASE_FILES, e.MASTER, private_pin)
    assert e.authenticate(e.MASTER, e.reference(path)) == original
    rows = records(1)
    rows[0]["annotator1_label"], rows[0]["label_gpt_5_mini_p1"] = (
        rows[0]["label_gpt_5_mini_p1"],
        rows[0]["annotator1_label"],
    )
    path.write_bytes(csv_bytes(rows))
    with pytest.raises(ValueError, match="reviewed_blob"):
        e.authenticate(e.MASTER, e.reference(path))


def test_duplicate_response_and_changed_manifest_rows(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8389-CUSTODY: row custody rejects duplicated joins and tampering."""
    manifest = private_manifest(tmp_path, monkeypatch)
    rows = records()
    rows[1]["model"] = rows[0]["model"]
    master = Path(manifest["release_files"][e.MASTER]["path"])
    master.write_bytes(csv_bytes(rows))
    manifest["release_files"][e.MASTER] = e.reference(master)
    assert not e.validate_manifest(manifest)["audit_alignment"]
    manifest["validated_rows"] = []
    with pytest.raises(ValueError, match="manifest_rows_drift"):
        e.validate_manifest(manifest)


def test_measure_acquisition_and_stable_alias(tmp_path, monkeypatch):
    """REQ-VERIFY-8389: the task-level manifest is a checked alias in private scratch."""
    from carnot.reporting import human_label_custody_runner_8389 as runner

    empty_work(tmp_path, monkeypatch)
    root = tmp_path / "private_repo"
    root.mkdir()
    (root / ".venv").symlink_to(e.ROOT / ".venv", target_is_directory=True)
    monkeypatch.setattr(e, "OWNED", [])
    monkeypatch.setattr(runner, "INPUTS", [])
    monkeypatch.setattr(
        e,
        "acquire",
        lambda *a: dict(
            release_commit=e.COMMIT,
            release_files={},
            target_mapping=e.TARGET,
            license_permitted=False,
            acquisition_gates=[],
        ),
    )
    raw = root / "results/raw" / e.NAME / "invocations" / "private"
    work = runner.measure(root, raw, None, tmp_path)
    alias = root / "results/raw" / e.NAME / "source_manifest.json"
    assert alias.read_bytes() == Path(work["manifest_reference"]["path"]).read_bytes()


def test_owned_child_failure_and_consumer_rejection(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8389-REPLAY: a real failed owned child disqualifies the primary."""
    import sys
    from carnot.reporting import human_label_custody_runner_8389 as runner

    work = empty_work(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "measure", lambda *a: work.copy())

    def frozen(private):
        atomic_json(private / "coverage.json", {"private_control": "failure_path"})
        return [
            dict(
                name="owned_failure",
                argv=[sys.executable, "-c", "raise SystemExit(3)"],
                deadline=20,
                expected=0,
                scope="owned",
            )
        ]

    monkeypatch.setattr(runner, "plan", frozen)
    monkeypatch.setattr(runner, "reader_receipt", lambda *a, **k: dict(passed=False))
    output = tmp_path / "results" / (e.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 1
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and not value["required_checks_passed"]
    assert any(r.get("exit_code") == 3 for r in value["validation_receipts"])
