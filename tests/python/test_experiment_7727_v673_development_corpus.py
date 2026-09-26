"""Development corpus custody checks for REQ-REPORT-7727 and REQ-VERIFY-7727."""

import copy
import json
from pathlib import Path

import pytest

from carnot.experiment_7727_v673_development_corpus import (
    ROOT,
    build_candidate,
    cold_replay,
    cold_reduce,
    main,
    run_experiment,
    seal_development,
    select_development,
    validate_public,
)
from carnot.experiment_7423_v651_annotated_protocol import DEFAULT_CACHE_ROOT
from carnot.reporting.current_work_receipt import atomic_json


def _release():
    sources = [
        {"source_id": str(i), "task_type": "Summary", "source_info": f"source {i}"}
        for i in range(8)
    ]
    responses = [
        {
            "id": str(i),
            "source_id": str(i),
            "split": "test" if i == 7 else "train",
            "quality": "good",
            "response": f"response {i}",
            "labels": [{"start": 0, "end": 8, "text": "response", "label_type": "conflicting"}]
            if i % 2
            else [],
        }
        for i in range(8)
    ]
    return sources, responses


def test_req_report_7727_label_blind_pilot_and_capacity():
    """SCENARIO-REPORT-7727-CUSTODY: exclude pilot before fixed role assignment."""
    sources, responses = _release()
    counts = {"fit": 2, "evaluation": 1}
    selected, capacity = select_development(sources, responses, counts, {"0"})
    assert capacity == {"train": 6, "test": 1}
    assert len(selected) == 3
    assert "0" not in {item["view"]["response_id"] for item in selected}
    changed = copy.deepcopy(responses)
    changed[1]["labels"] = [{"injected": "ignored for assignment"}]
    assert [x["family_id"] for x in select_development(sources, changed, counts, {"0"})[0]] == [
        x["family_id"] for x in selected
    ]
    selected, capacity = select_development(sources, responses, {"fit": 7, "evaluation": 1}, {"0"})
    assert selected == [] and capacity["train"] == 6


def test_req_verify_7727_isolation_and_cold_replay(tmp_path: Path):
    """SCENARIO-VERIFY-7727-ISOLATION: labels and corrupt public bytes fail."""
    sources, responses = _release()
    counts = {"fit": 2, "evaluation": 1}
    selected, _ = select_development(sources, responses, counts, set())
    manifest = seal_development(tmp_path, selected, responses, counts)
    assert cold_reduce(tmp_path / "development_manifest.json", counts)["families"] == 3
    public_path = tmp_path / manifest["roles"]["fit"]["public_path"]
    first = json.loads(public_path.read_text().splitlines()[0])
    assert first["previously_exposed"] is True
    assert first["fresh_generalization_eligible"] is False
    for mutation in (
        lambda row: row.update(label=1),
        lambda row: row.update(role="evaluation"),
        lambda row: row.update(complete_source=""),
        lambda row: row.update(source_sha256="sha256:bad"),
    ):
        bad = dict(first)
        mutation(bad)
        with pytest.raises(ValueError):
            validate_public(bad, "fit")
    duplicate = copy.deepcopy(selected)
    duplicate[1] = duplicate[0]
    with pytest.raises(ValueError, match="duplicate_family"):
        seal_development(tmp_path / "duplicate", duplicate, responses, counts)
    public_path.write_text(public_path.read_text() + "\n")
    with pytest.raises(ValueError, match="public_hash_mismatch"):
        cold_reduce(tmp_path / "development_manifest.json", counts)


def test_req_report_7727_authenticated_full_cold_replay(tmp_path: Path):
    """SCENARIO-REPORT-7727-REPLAY: 640 exposed rows reopen from raw bytes."""
    from carnot.experiment_7423_v651_annotated_protocol import authenticate_assets

    try:
        authenticate_assets(DEFAULT_CACHE_ROOT)
    except Exception:
        pytest.skip("pinned external RAGTruth cache unavailable")
    candidate = build_candidate(ROOT, tmp_path, "20260926")
    assert candidate["honest_verdict"] == "complete_null_development_corpus_ready"
    assert candidate["sample_size_budget"]["observed_families"] == 2894
    assert candidate["sample_size_budget"]["effective_independent_families"] == 640
    assert candidate["sample_size_budget"]["eligible_fresh_families"] == 0
    assert candidate["exposure_accounting"]["fitting"] == 0
    assert len(candidate["development_manifest"]["role_counts"]) == 7
    assert all(not row["fresh_generalization_eligible"] for row in candidate["rows"])
    path = tmp_path / "terminal_candidate.json"
    atomic_json(path, candidate)
    assert cold_replay(path) == {"families": 640, "verdict_class": "null"}
    published = tmp_path / "elsewhere" / "published.json"
    atomic_json(published, candidate)
    assert cold_replay(published) == {"families": 640, "verdict_class": "null"}
    damaged = copy.deepcopy(candidate)
    damaged["rows"][0]["role"] = "evaluation"
    atomic_json(path, damaged)
    with pytest.raises(ValueError, match="raw_row_mismatch"):
        cold_replay(path)
    damaged = copy.deepcopy(candidate)
    damaged["rows"].pop()
    (tmp_path / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in damaged["rows"])
    )
    atomic_json(path, damaged)
    with pytest.raises(ValueError, match="family_count_mismatch"):
        cold_replay(path)
    with pytest.raises(ValueError, match="root_or_date_mismatch"):
        build_candidate(ROOT, tmp_path, "20260927")


def test_req_report_7727_runner_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """SCENARIO-REPORT-7727-REPLAY: terminal checks govern readiness."""
    import carnot.experiment_7727_v673_development_corpus as exp

    try:
        from carnot.experiment_7423_v651_annotated_protocol import authenticate_assets

        authenticate_assets(DEFAULT_CACHE_ROOT)
    except Exception:
        pytest.skip("pinned external RAGTruth cache unavailable")
    monkeypatch.setattr(exp, "RAW", tmp_path / "raw")
    monkeypatch.setattr(exp, "OUTPUT", tmp_path / "published.json")
    monkeypatch.setattr(
        exp.validation,
        "run_scoped_validation",
        lambda *a, **k: {
            "validation_receipts": [],
            "required_checks_passed": True,
            "repository_health": {},
        },
    )
    monkeypatch.setattr(
        exp.validation,
        "run_commands",
        lambda *a, **k: [
            {"passed": True, "output_tail": "clean", "exit_code": 0} for _ in range(3)
        ],
    )
    result = run_experiment(ROOT, "20260926")
    assert result["development_cohort_ready_score"] == 1
    assert json.loads((tmp_path / "published.json").read_text())["verdict_class"] == "null"
    assert main(["--cold-replay", str(tmp_path / "raw/terminal_candidate.json")]) == 0


def test_req_verify_7727_rejects_each_public_boundary(tmp_path: Path):
    """SCENARIO-VERIFY-7727-ISOLATION: reject split, answer hash and exposure drift."""
    sources, responses = _release()
    selected, _ = select_development(sources, responses, {"fit": 1, "evaluation": 1}, set())
    counts = {"fit": 1, "evaluation": 1}
    manifest = seal_development(tmp_path, selected, responses, counts)
    row = json.loads((tmp_path / manifest["roles"]["fit"]["public_path"]).read_text())
    for key, value in (
        ("official_split", "test"),
        ("response_sha256", "sha256:bad"),
        ("previously_exposed", False),
        ("fresh_generalization_eligible", True),
    ):
        bad = {**row, key: value}
        with pytest.raises(ValueError):
            validate_public(bad, "fit")
    with pytest.raises(ValueError, match="role_count_mismatch"):
        seal_development(tmp_path / "short", selected[:1], responses, counts)


def test_req_verify_7727_cold_mutations(tmp_path: Path):
    """SCENARIO-VERIFY-7727-ISOLATION: every custody layer detects tampering."""
    from carnot.reporting.current_work_receipt import sha256_file

    sources, responses = _release()
    counts = {"fit": 1, "evaluation": 1}
    selected, _ = select_development(sources, responses, counts, set())

    def fresh():
        return seal_development(tmp_path, selected, responses, counts)

    manifest = fresh()
    path = tmp_path / "development_manifest.json"
    with pytest.raises(ValueError, match="manifest_contract_mismatch"):
        cold_reduce(path, {"fit": 2, "evaluation": 1})
    (tmp_path / "public_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="public_manifest_hash_mismatch"):
        cold_reduce(path, counts)
    manifest = fresh()
    public_manifest = json.loads((tmp_path / "public_manifest.json").read_text())
    public_manifest["counts"]["fit"] = 2
    atomic_json(tmp_path / "public_manifest.json", public_manifest)
    manifest["public_manifest_sha256"] = sha256_file(tmp_path / "public_manifest.json")
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match="public_manifest_mismatch"):
        cold_reduce(path, counts)
    manifest = fresh()
    evaluator_path = tmp_path / manifest["roles"]["fit"]["evaluator_path"]
    evaluator_path.write_text("{}")
    with pytest.raises(ValueError, match="evaluator_hash_mismatch"):
        cold_reduce(path, counts)
    manifest = fresh()
    manifest["roles"]["fit"]["families"] = []
    public_manifest = json.loads((tmp_path / "public_manifest.json").read_text())
    public_manifest["roles"]["fit"]["families"] = []
    atomic_json(tmp_path / "public_manifest.json", public_manifest)
    manifest["public_manifest_sha256"] = sha256_file(tmp_path / "public_manifest.json")
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match="role_count_or_roster_mismatch"):
        cold_reduce(path, counts)
    manifest = fresh()
    evaluator_path = tmp_path / manifest["roles"]["fit"]["evaluator_path"]
    label = json.loads(evaluator_path.read_text())
    label["label"] = 1 - label["label"]
    evaluator_path.write_text(json.dumps(label) + "\n")
    manifest["roles"]["fit"]["evaluator_sha256"] = sha256_file(evaluator_path)
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match="label_join_mismatch"):
        cold_reduce(path, counts)
    manifest = fresh()
    public_path = tmp_path / manifest["roles"]["evaluation"]["public_path"]
    fit_path = tmp_path / manifest["roles"]["fit"]["public_path"]
    public_path.write_bytes(fit_path.read_bytes())
    manifest["roles"]["evaluation"]["public_sha256"] = sha256_file(public_path)
    manifest["roles"]["evaluation"]["families"] = manifest["roles"]["fit"]["families"]
    atomic_json(path, manifest)
    with pytest.raises(ValueError):
        cold_reduce(path, counts)


def test_req_verify_7727_duplicate_family_in_saved_bytes(tmp_path: Path):
    """SCENARIO-VERIFY-7727-ISOLATION: rehashed duplicates still fail."""
    from carnot.reporting.current_work_receipt import sha256_file

    sources, responses = _release()
    counts = {"fit": 2}
    selected, _ = select_development(sources, responses, counts, set())
    manifest = seal_development(tmp_path, selected, responses, counts)
    public = tmp_path / "fit_public.jsonl"
    first = public.read_text().splitlines()[0]
    public.write_text(first + "\n" + first + "\n")
    manifest["roles"]["fit"]["public_sha256"] = sha256_file(public)
    manifest["roles"]["fit"]["families"] = [json.loads(first)["family_id"]] * 2
    public_manifest = json.loads((tmp_path / "public_manifest.json").read_text())
    public_manifest["roles"]["fit"]["public_sha256"] = sha256_file(public)
    public_manifest["roles"]["fit"]["families"] = manifest["roles"]["fit"]["families"]
    atomic_json(tmp_path / "public_manifest.json", public_manifest)
    manifest["public_manifest_sha256"] = sha256_file(tmp_path / "public_manifest.json")
    atomic_json(tmp_path / "development_manifest.json", manifest)
    with pytest.raises(ValueError, match="duplicate_family"):
        cold_reduce(tmp_path / "development_manifest.json", counts)


def test_req_report_7727_blocked_candidate_and_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """SCENARIO-REPORT-7727-REPLAY: shortage carries exact operands."""
    import runpy
    import sys
    import carnot.experiment_7727_v673_development_corpus as exp

    monkeypatch.setattr(exp, "select_development", lambda *a: ([], {"train": 0, "test": 0}))
    candidate = build_candidate(ROOT, tmp_path, "20260926")
    assert candidate["verdict_class"] == "blocked"
    assert candidate["gate_check_summary"]
    path = tmp_path / "terminal_candidate.json"
    atomic_json(path, candidate)
    assert cold_replay(path)["families"] == 0
    wrong = copy.deepcopy(candidate)
    wrong["development_manifest_sha256"] = "sha256:bad"
    atomic_json(path, wrong)
    with pytest.raises(ValueError, match="development_manifest_hash_mismatch"):
        cold_replay(path)
    wrong = copy.deepcopy(candidate)
    wrong["gate_check_summary"] = []
    atomic_json(path, wrong)
    with pytest.raises(ValueError, match="blocked_without_gate"):
        cold_replay(path)
    wrong = copy.deepcopy(candidate)
    wrong["rows"] = [{"fresh_generalization_eligible": True}]
    (tmp_path / "rows.jsonl").write_text(json.dumps(wrong["rows"][0]) + "\n")
    atomic_json(path, wrong)
    with pytest.raises(ValueError, match="fresh_claim_mismatch"):
        cold_replay(path)
    atomic_json(path, candidate)
    (tmp_path / "rows.jsonl").write_text("")
    monkeypatch.setattr(sys, "argv", ["module", "--cold-replay", str(path)])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module(exp.__name__, run_name="__main__")
    assert exit_info.value.code == 0
    monkeypatch.setattr(exp, "run_experiment", lambda *a: {})
    assert main(["--date", "20260926"]) == 0


def test_req_report_7727_failed_checks_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """SCENARIO-REPORT-7727-REPLAY: required and terminal failures deny readiness."""
    import carnot.experiment_7727_v673_development_corpus as exp

    monkeypatch.setattr(exp, "RAW", tmp_path / "raw")
    monkeypatch.setattr(exp, "OUTPUT", tmp_path / "published.json")
    passed = {"validation_receipts": [], "required_checks_passed": False, "repository_health": {}}
    monkeypatch.setattr(exp.validation, "run_scoped_validation", lambda *a, **k: passed)
    monkeypatch.setattr(
        exp.validation,
        "run_commands",
        lambda *a, **k: [
            {"passed": True, "output_tail": "clean", "exit_code": 0} for _ in range(3)
        ],
    )
    result = run_experiment(ROOT, "20260926")
    assert result["honest_verdict"] == "complete_disqualified_required_validation"
    assert result["development_cohort_ready_score"] == 0
    passed["required_checks_passed"] = True
    monkeypatch.setattr(
        exp.validation,
        "run_commands",
        lambda *a, **k: [
            {"passed": True, "output_tail": "clean", "exit_code": 0},
            {"passed": False, "output_tail": "FLAGGED", "exit_code": 1},
            {"passed": True, "output_tail": "clean", "exit_code": 0},
        ],
    )
    result = run_experiment(ROOT, "20260926")
    assert result["honest_verdict"] == "complete_disqualified_terminal_reader"
    assert result["flagged_adversarial"] is True
