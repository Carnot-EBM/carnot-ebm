"""REQ-REPORT-8409 and REQ-VERIFY-8409: released agreement has bounded scope."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import retrospective_label_criterion_8409 as e
from carnot.reporting.current_work_receipt import atomic_json


def panel():
    """Known arm differences test the fixed contrast without a model or real labels."""
    master = [
        dict(
            item_id=f"{i}:{m}",
            dataset="fixture",
            qid=f"q{i}",
            model=m,
            question=f"Question {i}?",
            answer_text=f"Answer {i} {m}",
            annotator1_label=str(i % 2),
            annotator2_label=str(i % 2),
        )
        for i in range(6)
        for m in e.old.GENERATORS
    ]
    audit = deepcopy(master)
    for row in audit:
        for arm in e.ARMS:
            label = int(row["annotator1_label"]) if arm in ("p2", "p2t") else 1
            row.update(
                {
                    f"label_{arm}": str(label),
                    f"conf_{arm}": "0.8",
                    f"raw_{arm}": json.dumps(dict(label=label, confidence=0.8)),
                    f"error_{arm}": "",
                }
            )
    return master, audit


def test_fixed_contrast_and_cluster_bootstrap():
    """SCENARIO-REPORT-8409-FACTORIAL: positive control keeps generator answers paired."""
    master, audit = panel()
    result = e.analyze(master, audit)
    assert result["primary_contrast"] == 1
    assert result["primary_ci95"] == [1, 1]
    assert result["false_positive_rates"]["p1"] == 1
    assert result["false_positive_rates"]["p2"] == 0
    assert result["false_negative_rates"]["p2"] == 0
    assert len(result["question_cluster_rows"]) == 6
    assert result["independent_count"] == 6
    assert result["intended_count"] == len(result["rows"]) == 900
    assert result["annotation_disagreement"]["disagreements"] == 0
    assert result["kappa"]["annotator1_vs_annotator2"] == 1
    swapped = deepcopy(audit)
    for row in swapped:
        for left, right in [("p1", "p2t"), ("p1s", "p2")]:
            for prefix in ("label", "conf", "raw", "error"):
                a, b = f"{prefix}_{left}", f"{prefix}_{right}"
                row[a], row[b] = row[b], row[a]
    assert e.analyze(master, swapped)["primary_contrast"] == -1
    for row in master:
        row["question"] = "Question alias"
    for row in audit:
        row["question"] = "QUESTION  alias"
    # Exact joined question text is required even when cluster aliases normalize.
    assert e.analyze(master, audit)["primary_contrast"] is None
    for row in audit:
        row["question"] = "Question alias"
    assert e.analyze(master, audit)["independent_count"] == 1


def test_missing_pairs_bounds_and_unusable_targets():
    """SCENARIO-REPORT-8409-FACTORIAL: incomplete arms cannot create a confounded substitute."""
    master, audit = panel()
    audit[0]["error_p1"] = "transport"
    result = e.analyze(master, audit)
    assert result["missing_pair_bounds"]["missing_factual_answers"] == 1
    assert result["missing_pair_bounds"]["contrast_bounds"][0] < 1
    assert result["missing_pairs"]
    assert result["paired_factual_count"] == 8
    assert result["per_item_condition_rows"][0]["false_positive_indicator"] is None
    for row in master:
        row["annotator2_label"] = "-1"
    assert e.analyze(master, audit)["reliability_sensitivity"]["primary_contrast"] is None
    for row in master:
        row["annotator1_label"] = "-1"
    assert e.analyze(master, audit)["primary_ci95"] is None
    assert e.kappa_value([]) is None
    assert e.kappa_value([(0, 0)]) is None


def test_original_operands_and_release_gate(tmp_path):
    """SCENARIO-REPORT-8409-CUSTODY: exact upstream pin differs from an absent input."""
    manifest = e.input_manifest(e.ROOT)
    result = e.reduce_manifest(manifest)
    assert result["release_criterion_ready_score"] == 1
    assert result["completed_count"] == 900
    assert result["disjoint_eval_ready_score"] == 0
    assert len(result["unknown_exposure_obligations"]) == 4
    missing = e.input_manifest(tmp_path)
    blocked = e.reduce_manifest(missing)
    assert blocked["primary_contrast"] is None
    assert blocked["gate_check_summary"][0]["observed_value"] is None
    bad = deepcopy(manifest)
    bad["primaries"]["8403"]["sha256"] = "forged"
    with pytest.raises(ValueError, match="primary_pin"):
        e.reduce_manifest(bad)


def work_at(tmp_path, real=False):
    """Private work exercises shipped resource checks and preserves primitive custody."""
    from carnot.reporting import retrospective_label_criterion_runner_8409 as runner

    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    work = runner.measure(e.ROOT if real else tmp_path / "absent", raw, tmp_path)
    work["work_path"] = str(raw / "work.json")
    atomic_json(Path(work["work_path"]), work)
    return work


def test_build_cold_children_and_rehashed_joins(tmp_path):
    """SCENARIO-VERIFY-8409-REPLAY: real positive and all changed primitive paths run in children."""
    from carnot.reporting import retrospective_label_criterion_runner_8409 as runner

    work = work_at(tmp_path, real=True)
    receipts = [dict(passed=True, scope="owned")]
    value = e.build(work, receipts)
    assert value["criterion_audit_ready_score"] == 1
    assert value["independent_generalization_score"] == 0
    candidate = tmp_path / (e.NAME + ".json")
    atomic_json(candidate, value)
    assert e.replay(candidate)
    assert all(r["passed"] for r in runner.controls(candidate, tmp_path / "controls"))
    for mutation in ("join", "condition", "target"):
        changed = json.loads(e.read_reference(work["primitive_reference"]))
        row = changed["rows"][0]
        if mutation == "join":
            row["generator"] = "swapped"
            row["join_hash"] = e.canonical_hash(row)
        elif mutation == "condition":
            row["factorial_outputs"]["p1"], row["factorial_outputs"]["p2"] = (
                row["factorial_outputs"]["p2"],
                row["factorial_outputs"]["p1"],
            )
        else:
            row["human_primary"] = 1 - row["human_primary"]
        path = tmp_path / (mutation + ".json")
        atomic_json(path, changed)
        tampered = deepcopy(work)
        tampered["primitive_reference"] = e.reference(path)
        tampered["work_path"] = str(tmp_path / (mutation + "_work.json"))
        atomic_json(Path(tampered["work_path"]), tampered)
        atomic_json(candidate, e.build(tampered, receipts))
        assert not e.replay(candidate)
    atomic_json(candidate, value)
    assert runner.validate(candidate, tmp_path / "validators")["passed"]
    assert e.build(work, [dict(passed=False)])["verdict_class"] == "disqualified"


def test_private_cli_and_plan(tmp_path, monkeypatch):
    """REQ-VERIFY-8409: direct CLI, date failures and checked publication are real paths."""
    from carnot.reporting import retrospective_label_criterion_runner_8409 as runner
    from carnot.reporting.primary_publication import read_bound_sidecar

    plan = runner.plan(tmp_path)
    assert any(s["name"] == "coverage_report" and "--fail-under=100" in s["argv"] for s in plan)
    assert any(s["name"] == "private_E2E019" for s in plan)
    work = work_at(tmp_path)
    with monkeypatch.context() as local:
        local.setattr(runner.shipped, "measure", lambda *a: work.copy())
        local.setattr(runner.shipped, "plan", lambda *a: [])
        output = tmp_path / "results" / (e.NAME + ".json")
        assert runner.main(["--date", "20261011", "--output", str(output)]) == 0
        value = json.loads(output.read_bytes())
        assert value["verdict_class"] == "blocked"
        assert value["criterion_audit_ready_score"] == 0
        pub = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        assert read_bound_sidecar(output, Path(pub["sidecar_path"]))["report"]["passed"]
        assert runner.main(["--cold-replay", str(output)]) == 0
        assert runner.main(["--deliberate-error"]) == 1
        local.setattr(
            runner.shipped, "measure", lambda *a: (_ for _ in ()).throw(ValueError("owned"))
        )
        assert runner.main(["--output", str(output)]) == 1
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261010"])


def test_changed_human_bytes_rehashed_manifest(tmp_path):
    """SCENARIO-REPORT-8409-CUSTODY: author blob pins reject self-consistently rehashed labels."""
    import csv
    import io

    manifest = e.input_manifest(e.ROOT)
    primary = json.loads(e.read_reference(manifest["primaries"]["8403"]))
    release = e.release_manifest(primary)
    rows = list(
        csv.DictReader(
            io.StringIO(
                e.old.authenticate(e.old.MASTER, release["release_files"][e.old.MASTER]).decode()
            )
        )
    )
    rows[0]["annotator1_label"] = str(1 - int(rows[0]["annotator1_label"]))
    data = io.StringIO()
    writer = csv.DictWriter(data, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    changed = tmp_path / "changed.csv"
    changed.write_text(data.getvalue())
    with pytest.raises(ValueError, match="reviewed_blob"):
        e.old.authenticate(e.old.MASTER, e.reference(changed))


@pytest.mark.parametrize("mutation", ["release_gate", "release_mapping"])
def test_release_gate_stops_measurement(monkeypatch, mutation):
    """SCENARIO-REPORT-8409-CUSTODY: upstream readiness and mapping failures stop target measurement."""
    actual = e.read_reference
    manifest = e.input_manifest(e.ROOT)
    primary_ref = manifest["primaries"]["8403"]
    if mutation == "release_gate":

        def changed(ref):
            data = actual(ref)
            if ref == primary_ref:
                value = json.loads(data)
                value["release_criterion_ready_score"] = 0
                return json.dumps(value).encode()
            return data

        monkeypatch.setattr(e, "read_reference", changed)
        value = e.reduce_manifest(manifest)
        assert value["completed_count"] == 0
        assert value["gate_check_summary"][0]["field"] == "release_criterion_ready_score"
        assert value["gate_check_summary"][0]["observed_value"] == 0
    else:
        original = e.release_manifest(json.loads(actual(primary_ref)))
        original["target_mapping"] = {}
        monkeypatch.setattr(e, "release_manifest", lambda *a: original)
        with pytest.raises(ValueError, match="release_mapping"):
            e.reduce_manifest(manifest)


def test_qualified_null_and_aggregate_drift(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8409-CUSTODY: zero effects qualify audits and changed reports fail."""
    work = work_at(tmp_path, real=True)
    receipts = [dict(passed=True, scope="owned")]
    candidate = tmp_path / (e.NAME + ".json")
    value = e.build(work, receipts)
    value["primary_contrast"] = 0
    atomic_json(candidate, value)
    assert not e.replay(candidate)
    original = e.reduce_manifest(json.loads(e.read_reference(work["manifest_reference"])))
    original["primary_contrast"] = 0
    original["primary_ci95"] = [-0.1, 0.1]
    monkeypatch.setattr(e, "reduce_manifest", lambda *a: deepcopy(original))
    null = e.build(work, receipts)
    assert null["verdict_class"] == "null" and null["criterion_audit_ready_score"] == 1
