"""Tests for the V636 fixed-cardinality rare-event audit.

Spec: REQ-SAMPLER-7229 and SCENARIO-SAMPLER-7229-*.
"""

from __future__ import annotations

import copy
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7229_v636_rare_event_audit as exp


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def upstream() -> dict[str, object]:
    """Load the immutable audit input once so tests do not duplicate file work."""

    return json.loads((REPO / exp.UPSTREAM_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def trace_records(upstream: dict[str, object]) -> dict[str, dict[str, object]]:
    """Authenticate and decode the real trace archive for reducer tests."""

    return exp.read_trace_archive(REPO, upstream)


@pytest.fixture(scope="module")
def observable_rows(
    upstream: dict[str, object], trace_records: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    """Reduce the authenticated archive once for all row-level assertions."""

    return exp.build_observable_rows(upstream, trace_records)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Build one complete artifact for validator and mutation tests."""

    output = tmp_path_factory.mktemp("exp7229-artifact") / "audit.json"
    return exp.build_artifact(REPO, output=output)


def test_req_sampler_7229_freezes_read_only_scope() -> None:
    """REQ-SAMPLER-7229 fixes aggregation literals, roster, and no-model scope."""

    assert exp.RUN_DATE == "20260912"
    assert exp.MODEL_SPECS == []
    assert exp.INFERENCE_SUBSTRATE == "aggregation_from_upstream_artifacts"
    assert exp.INFERENCE_SUBSTRATE_CLASS == "aggregation"
    assert exp.EXPECTED_OBSERVABLE_ROWS == 240
    assert exp.IID_REFERENCE_ONLY is True


def test_req_sampler_7229_preflight_authenticates_exact_inputs(tmp_path: Path) -> None:
    """REQ-SAMPLER-7229-PREFLIGHT verifies bytes, rows, contract, and access."""

    checks, hashes = exp.collect_preconditions(REPO, output=tmp_path / "audit.json")
    by_name = {row["check"]: row for row in checks}

    assert all(row["passed"] is True for row in checks)
    assert by_name["driving_capability_spec"]["observed_value"] == {
        "exists": True,
        "req_present": True,
        "scenarios_present": True,
    }
    assert by_name["upstream_authentication"]["observed_value"]["producer_errors"] == []
    assert by_name["trace_archive_authentication"]["observed_value"]["record_count"] == 720
    assert by_name["upstream_primary_gate"]["observed_value"] is False
    assert set(hashes) == {str(path) for path in exp.REQUIRED_SOURCE_PATHS}

    wrapped = {"principle": "Exact gate only.", "value": 0}
    arbitrary = {"value": 1, "metadata": "not a wrapper"}
    clean = exp.upstream_quarantine_observation({}, manifest_match=False)
    dirty = exp.upstream_quarantine_observation({"flagged_adversarial": True}, manifest_match=False)
    assert exp.unwrap_principled_value(wrapped) == 0
    assert exp.unwrap_principled_value(arbitrary) is arbitrary
    assert exp.gated_upstream_value({"gate": wrapped}, clean, "gate") == 0
    assert exp.gated_upstream_value({"gate": arbitrary}, clean, "gate") is arbitrary
    assert (
        exp.gated_upstream_value({"gate": wrapped}, dirty, "gate")
        == "not_consumed_due_to_quarantine"
    )
    assert exp.upstream_quarantine_observation({}, manifest_match=True)["quarantined"] is True


def test_scenario_sampler_7229_trace_reconstructs_every_probe(
    trace_records: dict[str, dict[str, object]], observable_rows: list[dict[str, object]]
) -> None:
    """SCENARIO-SAMPLER-7229-TRACE reconstructs the complete source matrix."""

    rows = observable_rows

    assert len(trace_records) == 720
    assert len(rows) == exp.EXPECTED_OBSERVABLE_ROWS
    assert {row["observable_kind"] for row in rows} == {"energy", "occupancy"}
    assert all(row["sample_count"] == 4 * 16384 for row in rows)
    assert all(row["row_sha256"] == exp.sha256_json(exp.without_row_hash(row)) for row in rows)

    occupancy_rows = [row for row in rows if row["observable_kind"] == "occupancy"]
    assert len(occupancy_rows) == 180
    assert all(
        row["visit_count"] / row["sample_count"] == row["observed_mean"] for row in occupancy_rows
    )
    assert all(
        row["exact_variance"]
        == pytest.approx(row["exact_probability"] * (1.0 - row["exact_probability"]))
        for row in occupancy_rows
    )
    assert all(
        row["expected_visits"] == pytest.approx(row["sample_count"] * row["exact_probability"])
        for row in occupancy_rows
    )
    assert all(row["iid_reference_only"] is True for row in occupancy_rows)


def test_scenario_sampler_7229_zero_hit_is_not_degeneracy(
    observable_rows: list[dict[str, object]],
) -> None:
    """SCENARIO-SAMPLER-7229-ZERO-HIT keeps null diagnostics unfavorable."""

    rows = observable_rows
    zero_hits = [
        row
        for row in rows
        if row["observable_kind"] == "occupancy"
        and row["exact_probability"] > 0.0
        and row["visit_count"] == 0
    ]

    assert zero_hits
    assert all(row["structurally_degenerate"] is False for row in zero_hits)
    assert all("unobserved_rare_probe" in row["failure_causes"] for row in zero_hits)
    assert all(row["all_chain_ess_computable"] is False for row in zero_hits)
    assert all(row["rhat_computable"] is False for row in zero_hits)
    assert all(
        row["iid_zero_hit_probability"]
        == pytest.approx((1.0 - row["exact_probability"]) ** row["sample_count"])
        for row in zero_hits
    )
    assert all(row["iid_calculation_is_mcmc_confidence_guarantee"] is False for row in zero_hits)


def test_req_sampler_7229_classifies_each_failed_criterion() -> None:
    """REQ-SAMPLER-7229-CHECKS leaves unknown causes unresolved."""

    summary = {
        "mean_tolerance_passed": False,
        "chain_ess_passed": False,
        "split_rhat_passed": False,
    }
    assert exp.classify_failures(summary, observable_kind="occupancy", visits=0) == {
        "mean_tolerance": "actual_bias",
        "ess": "unobserved_rare_probe",
        "rhat": "unobserved_rare_probe",
    }
    assert exp.classify_failures(summary, observable_kind="occupancy", visits=4) == {
        "mean_tolerance": "actual_bias",
        "ess": "insufficient_transitions",
        "rhat": "insufficient_transitions",
    }
    assert exp.classify_failures(summary, observable_kind="energy", visits=None) == {
        "mean_tolerance": "actual_bias",
        "ess": "insufficient_transitions",
        "rhat": "insufficient_transitions",
    }
    assert exp.classify_failures({}, observable_kind="energy", visits=None) == {
        "mean_tolerance": "unresolved",
        "ess": "unresolved",
        "rhat": "unresolved",
    }


def test_req_sampler_7229_checks_cost_construction_and_envelope(
    upstream: dict[str, object],
    trace_records: dict[str, dict[str, object]],
    observable_rows: list[dict[str, object]],
) -> None:
    """REQ-SAMPLER-7229-CHECKS and NEXT fix audit and retirement decisions."""

    checks = exp.build_evidence_checks(upstream, trace_records, observable_rows)
    envelope = exp.build_next_measurement_envelope(upstream)

    assert checks["all_passed"] is True
    assert checks["energy_moments"]["passed"] is True
    assert checks["observable_definitions"]["passed"] is True
    assert checks["transition_cost_accounting"]["passed"] is True
    assert checks["primary_graph_construction"]["observed_state_counts"] == [496]
    assert checks["primary_graph_construction"]["passed"] is True
    assert checks["failure_classification"]["unclassified_failed_criteria"] == 0
    assert envelope["observables_fixed_before_new_chain"] is True
    assert envelope["accuracy_target"]["relative_half_width"] == 0.25
    assert envelope["budget"]["max_retained_transitions_per_chain"] == 10_000_000
    assert envelope["iid_lower_bound_only"] is True
    assert envelope["feasible_at_budget"] is False
    assert envelope["decision"] == "retire_quality_claim_at_fixed_budget"


def test_scenario_sampler_7229_gate_builds_and_validates_terminal_artifact(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-SAMPLER-7229-GATE preserves the null in a complete artifact."""

    assert exp.validate_artifact(artifact, root=REPO) == []
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert artifact["down_up_value_score"] == 0
    assert artifact["rare_event_audit_complete_score"] == 1
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["original_gate_preserved"]["unchanged"] is True
    assert artifact["original_gate_preserved"]["primary_gate"]["passed"] is False
    assert artifact["iid_reference_only"] is True
    assert artifact["reproducibility_checksum"] == exp.artifact_checksum(artifact)


def test_scenario_sampler_7229_artifact_rejects_tampering(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-SAMPLER-7229-ARTIFACT rejects dropped rows and rescued gates."""

    dropped = copy.deepcopy(artifact)
    dropped["observable_rows"].pop()
    dropped["rows"] = dropped["observable_rows"]
    dropped["reproducibility_checksum"] = exp.artifact_checksum(dropped)
    assert "observable_rows_invalid" in exp.validate_artifact(dropped)

    rescued = copy.deepcopy(artifact)
    rescued["down_up_value_score"] = 1
    rescued["reproducibility_checksum"] = exp.artifact_checksum(rescued)
    assert "original_gate_not_preserved" in exp.validate_artifact(rescued)

    iid = copy.deepcopy(artifact)
    iid["iid_reference_only"] = False
    iid["reproducibility_checksum"] = exp.artifact_checksum(iid)
    assert "iid_reference_invalid" in exp.validate_artifact(iid)

    favorable = copy.deepcopy(artifact)
    target = next(row for row in favorable["observable_rows"] if row["visit_count"] == 0)
    target["chain_ess_values"] = [16384.0] * 4
    target["all_chain_ess_computable"] = True
    target["row_sha256"] = exp.sha256_json(exp.without_row_hash(target))
    favorable["rows"] = favorable["observable_rows"]
    favorable["reproducibility_checksum"] = exp.artifact_checksum(favorable)
    assert "observable_rows_invalid" in exp.validate_artifact(favorable)

    wrong_roster = copy.deepcopy(artifact)
    wrong_roster["observable_rows"][0]["unit_id"] = "unknown"
    wrong_roster["observable_rows"][0]["row_sha256"] = exp.sha256_json(
        exp.without_row_hash(wrong_roster["observable_rows"][0])
    )
    wrong_roster["rows"] = wrong_roster["observable_rows"]
    wrong_roster["reproducibility_checksum"] = exp.artifact_checksum(wrong_roster)
    assert "observable_rows_invalid" in exp.validate_artifact(wrong_roster)

    stale_hash = copy.deepcopy(artifact)
    stale_hash["observable_rows"][0]["metric"] = "changed"
    stale_hash["rows"] = stale_hash["observable_rows"]
    stale_hash["reproducibility_checksum"] = exp.artifact_checksum(stale_hash)
    assert "observable_rows_invalid" in exp.validate_artifact(stale_hash)

    missing_common = copy.deepcopy(artifact)
    missing_common["observable_rows"][0].pop("metric")
    missing_common["observable_rows"][0]["row_sha256"] = exp.sha256_json(
        exp.without_row_hash(missing_common["observable_rows"][0])
    )
    missing_common["rows"] = missing_common["observable_rows"]
    missing_common["reproducibility_checksum"] = exp.artifact_checksum(missing_common)
    assert "observable_rows_invalid" in exp.validate_artifact(missing_common)

    wrong_cause = copy.deepcopy(artifact)
    wrong_cause["observable_rows"][0]["failure_classification"] = {}
    wrong_cause["observable_rows"][0]["row_sha256"] = exp.sha256_json(
        exp.without_row_hash(wrong_cause["observable_rows"][0])
    )
    wrong_cause["rows"] = wrong_cause["observable_rows"]
    wrong_cause["reproducibility_checksum"] = exp.artifact_checksum(wrong_cause)
    assert "observable_rows_invalid" in exp.validate_artifact(wrong_cause)

    bad_iid_row = copy.deepcopy(artifact)
    occupancy = next(
        row for row in bad_iid_row["observable_rows"] if row["observable_kind"] == "occupancy"
    )
    occupancy["exact_probability"] = "bad"
    occupancy["row_sha256"] = exp.sha256_json(exp.without_row_hash(occupancy))
    bad_iid_row["rows"] = bad_iid_row["observable_rows"]
    bad_iid_row["reproducibility_checksum"] = exp.artifact_checksum(bad_iid_row)
    assert "observable_rows_invalid" in exp.validate_artifact(bad_iid_row)


def test_scenario_sampler_7229_blocked_and_cli_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-SAMPLER-7229-ARTIFACT publishes blocks and validates CLI bytes."""

    failed = exp._check_row(
        "upstream_authentication", str(exp.UPSTREAM_PATH), "producer validator", True, False, False
    )
    blocked = exp.build_artifact(
        REPO,
        output=tmp_path / "unused.json",
        preconditions=[failed],
        source_hashes={},
    )
    assert exp.validate_artifact(blocked) == []
    assert blocked["status"] == "blocked_external_precondition"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["rare_event_audit_complete_score"] == 0
    assert blocked["gate_check_summary"]["failed_check"] == "upstream_authentication"
    assert blocked["rows"] == []

    output = tmp_path / "terminal.json"
    written = exp.run_experiment(REPO, output)
    assert json.loads(output.read_text(encoding="utf-8")) == written
    assert exp.main(["--validate", str(output)]) == 0
    assert '"valid":true' in capsys.readouterr().out
    assert exp.main(["--date", "20260911"]) == 2
    assert "run date must be 20260912" in capsys.readouterr().out


def test_req_sampler_7229_iid_formula_handles_probability_boundaries() -> None:
    """REQ-SAMPLER-7229-IID-REFERENCE keeps boundary calculations explicit."""

    assert exp.iid_zero_hit_probability(0.0, 100) == 1.0
    assert exp.iid_zero_hit_probability(1.0, 100) == 0.0
    assert exp.iid_zero_hit_probability(0.5, 0) == 1.0
    with pytest.raises(ValueError, match="probability"):
        exp.iid_zero_hit_probability(-0.1, 4)
    with pytest.raises(ValueError, match="sample_count"):
        exp.iid_zero_hit_probability(0.1, -1)
    required = exp.iid_effective_sample_lower_bound(0.01, relative_half_width=0.25)
    assert required == math.ceil(exp.NORMAL_95_SQUARED * 0.99 / (0.01 * 0.25**2))
    with pytest.raises(ValueError, match="strictly between zero and one"):
        exp.iid_effective_sample_lower_bound(0.0, relative_half_width=0.25)
    with pytest.raises(ValueError, match="relative_half_width"):
        exp.iid_effective_sample_lower_bound(0.1, relative_half_width=1.0)


def test_req_sampler_7229_fail_closed_preflight_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-SAMPLER-7229-PREFLIGHT reports malformed files and unavailable outputs."""

    with pytest.raises(ValueError, match="nonfinite"):
        exp.canonical_json({"bad": math.inf})

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    assert exp._read_json_object(invalid_json) == {}
    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("1", encoding="utf-8")
    assert exp._read_json_object(scalar_json) == {}
    assert exp._read_json_object(tmp_path / "missing.json") == {}

    assert exp._task_contract(tmp_path) is None
    (tmp_path / exp.ROADMAP_PATH).write_text("tasks: [", encoding="utf-8")
    assert exp._task_contract(tmp_path) is None
    (tmp_path / exp.ROADMAP_PATH).write_text("tasks: {}", encoding="utf-8")
    assert exp._task_contract(tmp_path) is None
    (tmp_path / exp.ROADMAP_PATH).write_text("tasks: []", encoding="utf-8")
    assert exp._task_contract(tmp_path) is None

    file_path = tmp_path / "not-a-directory"
    file_path.write_text("x", encoding="utf-8")
    assert exp._directory_writable(file_path) is False

    assert exp._trace_receipt(tmp_path, {"trace_archive": []}) == {
        "valid": False,
        "record_count": None,
    }
    assert exp._trace_receipt(tmp_path, {"trace_archive": {}}) == {
        "valid": False,
        "record_count": None,
    }
    missing_receipt = {
        "trace_archive": {
            "path": str(exp.TRACE_PATH),
            "bytes": 1,
            "sha256": "sha256:missing",
            "record_count": 1,
        }
    }
    assert exp._trace_receipt(tmp_path, missing_receipt)["valid"] is False

    monkeypatch.setattr(exp, "_trace_receipt", lambda *_: {"valid": False})
    with pytest.raises(ValueError, match="not authenticated"):
        exp.read_trace_archive(tmp_path, {})


def test_scenario_sampler_7229_trace_rejects_malformed_records(
    upstream: dict[str, object], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-SAMPLER-7229-TRACE rejects each record-boundary corruption."""

    monkeypatch.setattr(exp, "_trace_receipt", lambda *_: {"valid": True})
    with pytest.raises(ValueError, match="roster"):
        exp.read_trace_archive(REPO, {})

    def run_stream(text: str, source_upstream: dict[str, object] = upstream) -> None:
        monkeypatch.setattr(exp.gzip, "open", lambda *args, **kwargs: io.StringIO(text))
        exp.read_trace_archive(REPO, source_upstream)

    with pytest.raises(ValueError, match="invalid JSON"):
        run_stream("{\n")
    with pytest.raises(ValueError, match="not an object"):
        run_stream("[]\n")
    with pytest.raises(ValueError, match="unknown or duplicate"):
        run_stream("{}\n")

    source = upstream["matched_budget_rows"][0]
    base = {
        "unit_id": source["unit_id"],
        "panel": "matched_budget",
        "rng_seed": source["seed"],
        "initial_state_index": 0,
    }
    with pytest.raises(ValueError, match="invalid indices"):
        run_stream(json.dumps({**base, "trace_indices": [True]}) + "\n")
    with pytest.raises(ValueError, match="out-of-range"):
        run_stream(json.dumps({**base, "trace_indices": [math.comb(32, 2)]}) + "\n")
    with pytest.raises(ValueError, match="does not match"):
        run_stream(json.dumps({**base, "panel": "quality", "trace_indices": []}) + "\n")
    with pytest.raises(ValueError, match="complete source roster"):
        run_stream("")

    changed = dict(upstream)
    matched = list(upstream["matched_budget_rows"])
    matched[0] = {**source, "trace_sha256": exp.sha256_json([])}
    changed["matched_budget_rows"] = matched
    monkeypatch.setattr(exp.time, "monotonic", iter((0.0, 61.0)).__next__)
    with pytest.raises(ValueError, match="complete source roster"):
        run_stream(json.dumps({**base, "trace_indices": []}) + "\n", changed)
    assert "[phase 3 progress] completed=1/720 elapsed_s=61.000" in capsys.readouterr().out


def test_scenario_sampler_7229_reducer_rejects_missing_or_changed_values(
    upstream: dict[str, object],
    trace_records: dict[str, dict[str, object]],
    observable_rows: list[dict[str, object]],
) -> None:
    """SCENARIO-SAMPLER-7229-TRACE fails closed on incomplete reducer inputs."""

    quality_id = upstream["quality_rows"][0]["unit_id"]
    missing = dict(trace_records)
    missing[quality_id] = {**missing[quality_id], "trace_indices": None}
    with pytest.raises(ValueError, match="missing quality trace bytes"):
        exp.build_observable_rows(upstream, missing)

    short = dict(trace_records)
    short[quality_id] = {
        **short[quality_id],
        "trace_indices": np.asarray(short[quality_id]["trace_indices"][:10]),
    }
    with pytest.raises(ValueError, match="incomplete retained trace"):
        exp.build_observable_rows(upstream, short)

    changed = copy.deepcopy(upstream)
    first_probe = next(
        value
        for value in changed["quality_summary_rows"][0]["probe_summaries"].values()
        if value["exact_mean"] >= 0.0
    )
    first_probe["observed_mean"] += 1.0
    with pytest.raises(ValueError, match="reconstructed mean differs"):
        exp.build_observable_rows(changed, trace_records)

    bad_upstream = copy.deepcopy(upstream)
    summary = bad_upstream["quality_summary_rows"][0]
    summary["probe_summaries"]["energy"]["exact_mean"] += 1.0
    summary["probe_summaries"].pop(
        next(name for name in summary["probe_summaries"] if name != "energy")
    )
    bad_upstream["matched_budget_rows"][0]["energy_evaluations"] += 1
    bad_rows = copy.deepcopy(observable_rows)
    occupancy = next(row for row in bad_rows if row["observable_kind"] == "occupancy")
    occupancy["observed_mean"] += 1.0
    checks = exp.build_evidence_checks(bad_upstream, trace_records, bad_rows)
    assert checks["all_passed"] is False
    assert checks["energy_moments"]["mismatches"] == 1
    assert checks["observable_definitions"]["mismatches"] >= 2
    assert checks["transition_cost_accounting"]["failed_unit_ids"]


def test_scenario_sampler_7229_validator_reports_contract_failures(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-SAMPLER-7229-ARTIFACT names every terminal contract failure."""

    assert exp.validate_artifact({}) == ["missing_required_fields"]
    stale_checksum = copy.deepcopy(artifact)
    stale_checksum["reproducibility_checksum"] = "sha256:changed"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(stale_checksum)
    broken = copy.deepcopy(artifact)
    broken.update(
        {
            "field_principles": {},
            "run_date": "bad",
            "execution_venue": "bad",
            "duration_s": -1,
            "MODEL_SPECS": [{}],
            "sample_size_budget": {},
            "original_gate_preserved": {},
            "iid_reference_only": False,
            "evidence_checks": {},
            "next_measurement_envelope": {},
            "status": "bad",
            "sampler_rerun_performed": True,
            "source_artifact_hashes": {},
        }
    )
    broken["reproducibility_checksum"] = exp.artifact_checksum(broken)
    errors = set(exp.validate_artifact(broken, root=REPO))
    assert {
        "field_principles_invalid",
        "run_date_invalid",
        "execution_identity_invalid",
        "duration_invalid",
        "model_declaration_invalid",
        "sample_size_budget_invalid",
        "original_gate_not_preserved",
        "source_artifact_hashes_invalid",
        "iid_reference_invalid",
        "evidence_checks_invalid",
        "next_measurement_envelope_invalid",
        "terminal_verdict_invalid",
        "claim_limits_invalid",
    } <= errors

    blocked = exp.build_artifact(
        REPO,
        output=REPO / "results" / "unused.json",
        preconditions=[exp._check_row("x", "u", "f", True, False, False)],
        source_hashes={},
    )
    blocked["rows"] = [{}]
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    assert "blocked_contract_invalid" in exp.validate_artifact(blocked)


def test_scenario_sampler_7229_atomic_and_exceptional_cli_paths(
    artifact: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-SAMPLER-7229-ARTIFACT covers atomic cleanup and CLI failures."""

    real_run_experiment = exp.run_experiment
    target = tmp_path / "kept.json"
    target.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(exp.os, "replace", lambda *_: None)
    exp.atomic_write(target, artifact)
    assert not list(tmp_path.glob(".kept.json.*.tmp"))
    monkeypatch.undo()

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert exp.main(["--validate", str(invalid)]) == 2
    assert "validation_error" in capsys.readouterr().out
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text("{}", encoding="utf-8")
    assert exp.main(["--validate", str(incomplete)]) == 2
    assert '"valid":false' in capsys.readouterr().out

    monkeypatch.setattr(exp, "run_experiment", lambda *_: (_ for _ in ()).throw(ValueError("x")))
    assert exp.main(["--output", str(tmp_path / "failed.json")]) == 2
    assert "experiment_error: x" in capsys.readouterr().out

    observed: list[Path] = []
    monkeypatch.setattr(exp, "run_experiment", lambda root, output: observed.append(output))
    assert exp.main(["--output", "results/relative.json"]) == 0
    assert observed == [REPO / "results/relative.json"]

    monkeypatch.setattr(exp, "run_experiment", real_run_experiment)
    monkeypatch.setattr(exp, "build_artifact", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="invalid Exp7229 artifact"):
        exp.run_experiment(REPO, tmp_path / "never.json")
