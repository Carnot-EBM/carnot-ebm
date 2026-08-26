"""Tests for the frozen twin-prefix verifier map.

Spec refs: REQ-CONSTRAINT-6650, SCENARIO-CONSTRAINT-6650-PAIRABLE-TWIN,
SCENARIO-CONSTRAINT-6650-NON-PAIRABLE,
SCENARIO-CONSTRAINT-6650-EXACT-AUTHORITY, REQ-VERIFY-6650,
SCENARIO-VERIFY-6650-PREREGISTERED-UNITS,
SCENARIO-VERIFY-6650-PAIRED-RATES, SCENARIO-VERIFY-6650-ABSTENTION,
SCENARIO-VERIFY-6650-RECOMMENDATION, REQ-REPORT-6650,
SCENARIO-REPORT-6650-COMPLETE-ROWS,
SCENARIO-REPORT-6650-BLOCKED-GATE, and
SCENARIO-REPORT-6650-ATOMIC-CHECKSUM.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6650_twin_prefix_verifier_map as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_PATH = REPO_ROOT / exp.UPSTREAM_PATH


@pytest.fixture(scope="module")
def upstream() -> dict:
    """Load the frozen source once so all tests use identical source bytes."""

    return json.loads(UPSTREAM_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def transition_model(upstream: dict) -> dict:
    """Freeze the advisory transition support before any twin is scored."""

    return exp.build_transition_model(upstream["frozen_task_manifest"])


@pytest.fixture(scope="module")
def construction(upstream: dict, transition_model: dict) -> dict:
    """Construct all accepted and rejected source pairs once."""

    return exp.construct_twins(upstream, transition_model)


def test_spec_anchors_exist_before_implementation() -> None:
    """REQ-*-6650 anchors name every required behavior before code."""

    surfaces = {
        "openspec/capabilities/constraint-verification/spec.md": (
            "REQ-CONSTRAINT-6650",
            "SCENARIO-CONSTRAINT-6650-PAIRABLE-TWIN",
            "SCENARIO-CONSTRAINT-6650-NON-PAIRABLE",
            "SCENARIO-CONSTRAINT-6650-EXACT-AUTHORITY",
        ),
        "openspec/capabilities/verification/spec.md": (
            "REQ-VERIFY-6650",
            "SCENARIO-VERIFY-6650-PREREGISTERED-UNITS",
            "SCENARIO-VERIFY-6650-PAIRED-RATES",
            "SCENARIO-VERIFY-6650-ABSTENTION",
            "SCENARIO-VERIFY-6650-RECOMMENDATION",
        ),
        "openspec/capabilities/research-reporting/spec.md": (
            "REQ-REPORT-6650",
            "SCENARIO-REPORT-6650-COMPLETE-ROWS",
            "SCENARIO-REPORT-6650-BLOCKED-GATE",
            "SCENARIO-REPORT-6650-ATOMIC-CHECKSUM",
        ),
    }
    for path, anchors in surfaces.items():
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        assert all(anchor in text for anchor in anchors)


def test_upstream_receipt_binds_gate_hash_and_rows(upstream: dict) -> None:
    """REQ-REPORT-6650 binds the exact source field, hash, and row count."""

    receipt = exp.build_upstream_gate_receipt(REPO_ROOT, upstream)
    assert receipt["field"] == "candidate_corpus_complete"
    assert receipt["expected_value"] is True
    assert receipt["observed_value"] is True
    assert receipt["passed"] is True
    assert receipt["sha256"] == exp.sha256_file(UPSTREAM_PATH)
    assert receipt["observed_row_count"] == len(upstream["candidate_rows"]) == 48


def test_preconditions_check_exact_identity_resources_and_no_llm(upstream: dict) -> None:
    """REQ-REPORT-6650 checks every frozen input before scoring."""

    protected = exp.protected_hashes(REPO_ROOT)
    receipt = exp.build_upstream_gate_receipt(REPO_ROOT, upstream)
    preconditions = exp.collect_preconditions(REPO_ROOT, upstream, receipt, protected)
    assert preconditions["all_required_preconditions_available"] is True
    assert all(preconditions["checks"].values())
    assert preconditions["exact_checker_identity"]["matches_current_source"] is True
    assert preconditions["resources"]["cpu_count"] > 0
    assert preconditions["resources"]["disk_free_bytes"] > 0
    assert preconditions["tools"]["python_available"] is True
    assert preconditions["no_llm"]["llm_invoked"] is False
    assert preconditions["no_llm"]["substrate"] == exp.INFERENCE_SUBSTRATE


def test_transition_model_is_frozen_from_manifest(upstream: dict, transition_model: dict) -> None:
    """SCENARIO-VERIFY-6650-PREREGISTERED-UNITS freezes support first."""

    manifest = upstream["frozen_task_manifest"]
    assert transition_model["source_manifest_sha256"] == manifest["manifest_sha256"]
    assert transition_model["task_count"] == 24
    assert transition_model["supported_action_ids"]
    assert transition_model["supported_transitions"]
    assert transition_model["model_sha256"] == exp.transition_model_checksum(transition_model)


def test_pairable_twins_change_one_semantic_step_only(construction: dict) -> None:
    """SCENARIO-CONSTRAINT-6650-PAIRABLE-TWIN enforces byte-local twins."""

    twins = construction["twins"]
    assert len(twins) == 8
    for twin in twins:
        clean = twin["clean_plan"].encode("utf-8")
        error = twin["error_plan"].encode("utf-8")
        changed = twin["localized_step"]
        clean_lines = twin["clean_plan"].splitlines()
        error_lines = twin["error_plan"].splitlines()
        assert len(clean) == len(error) == twin["plan_byte_count"]
        assert len(clean_lines) == len(error_lines)
        assert clean_lines[:changed] == error_lines[:changed]
        assert clean_lines[changed + 1 :] == error_lines[changed + 1 :]
        assert clean_lines[changed] != error_lines[changed]
        assert len(clean_lines[changed].encode()) == len(error_lines[changed].encode())
        assert twin["byte_difference_count"] > 0
        assert twin["clean_exact_label"] is True
        assert twin["error_exact_label"] is False
        assert twin["error_exact_result"]["first_failing_step"] == changed


def test_non_pairable_rows_are_explicit(construction: dict) -> None:
    """SCENARIO-CONSTRAINT-6650-NON-PAIRABLE retains all rejected rows."""

    rejected = construction["rejected_pairs"]
    assert len(rejected) == 40
    assert len({row["source_row_id"] for row in rejected}) == 40
    reasons = {row["rejection_reason"] for row in rejected}
    assert reasons == {
        "source_has_no_exact_boolean_label",
        "source_candidate_not_exact_valid",
    }
    assert (
        sum(row["rejection_reason"] == "source_has_no_exact_boolean_label" for row in rejected)
        == 38
    )
    assert (
        sum(row["rejection_reason"] == "source_candidate_not_exact_valid" for row in rejected) == 2
    )


def test_exact_labels_do_not_come_from_advisory_scorer(construction: dict) -> None:
    """SCENARIO-CONSTRAINT-6650-EXACT-AUTHORITY keeps authority separate."""

    for twin in construction["twins"]:
        assert twin["exact_checker_identity"]["exact_executor_version"].startswith("carnot.")
        assert twin["advisory_scorer_supplies_exact_label"] is False
        assert twin["clean_exact_result"]["exact_final_validity"] is True
        assert twin["error_exact_result"]["exact_final_validity"] is False


def test_unit_preregistration_is_closed_and_hashed() -> None:
    """REQ-VERIFY-6650 freezes units, thresholds, and selection logic."""

    prereg = exp.verifier_unit_preregistration()
    assert list(prereg["units"]) == list(exp.UNIT_ORDER)
    assert prereg["same_scorer_for_all_units"] is True
    assert prereg["thresholds"] == {
        "accept_below": 0.25,
        "abstain_below": 0.75,
        "reject_at_or_above": 0.75,
    }
    assert prereg["frozen_before_scoring"] is True
    assert prereg["preregistration_sha256"] == exp.preregistration_checksum(prereg)


def test_same_scorer_separates_one_and_two_step_context(transition_model: dict) -> None:
    """SCENARIO-VERIFY-6650-PREREGISTERED-UNITS changes only view length."""

    actions = sorted(transition_model["supported_action_ids"])
    supported = {tuple(row) for row in transition_model["supported_transitions"]}
    first, second = next(iter(supported))
    unsupported_second = next(
        action for action in actions if action != second and (first, action) not in supported
    )
    assert exp.score_action_unit([first], transition_model)["score"] == 0.0
    assert exp.score_action_unit([first, second], transition_model)["score"] == 0.0
    assert exp.score_action_unit([first, unsupported_second], transition_model)["score"] == 1.0
    assert exp.score_action_unit(["unknown_action"], transition_model)["score"] == 1.0


@pytest.mark.parametrize(
    ("score", "decision", "abstained"),
    [
        (0.0, "accept", False),
        (0.25, "abstain", True),
        (0.5, "abstain", True),
        (0.75, "reject", False),
        (1.0, "reject", False),
    ],
)
def test_frozen_decision_thresholds(score: float, decision: str, abstained: bool) -> None:
    """SCENARIO-VERIFY-6650-ABSTENTION makes the middle interval explicit."""

    assert exp.decision_from_score(score) == {"decision": decision, "abstained": abstained}


def test_scored_rows_cover_every_twin_unit_and_rejection(
    construction: dict, transition_model: dict
) -> None:
    """SCENARIO-REPORT-6650-COMPLETE-ROWS retains the complete source map."""

    scored = exp.score_twins(construction, transition_model)
    assert len(scored["twin_rows"]) == 8
    assert len(scored["per_unit_rows"]) == 88
    accepted = [row for row in scored["per_unit_rows"] if row["row_type"] == "twin_unit"]
    rejected = [row for row in scored["per_unit_rows"] if row["row_type"] == "rejected_pair"]
    assert len(accepted) == 8 * 3 * 2
    assert len(rejected) == 40
    for twin in scored["twin_rows"]:
        assert set(twin["per_unit_results"]) == set(exp.UNIT_ORDER)
        for unit in twin["per_unit_results"].values():
            assert unit["clean"]["exact_label"] is True
            assert unit["error"]["exact_label"] is False
            assert unit["clean"]["latency_ns"] >= 0
            assert unit["error"]["latency_ns"] >= 0


def test_metrics_separate_catches_false_rejects_and_rejection(
    construction: dict, transition_model: dict
) -> None:
    """SCENARIO-VERIFY-6650-PAIRED-RATES prevents rejection-only promotion."""

    scored = exp.score_twins(construction, transition_model)
    metrics = exp.compute_unit_metrics(scored["per_unit_rows"])
    by_unit = {row["unit_id"]: row for row in metrics}
    one = by_unit["one_step"]
    two = by_unit["two_steps"]
    full = by_unit["full_remaining_suffix"]
    assert one["catch_rate"]["value"] == 0.0
    assert one["false_reject_rate"]["value"] == 0.0
    assert one["balanced_accuracy"]["value"] == 0.5
    assert two["catch_rate"]["value"] == 1.0
    assert two["false_reject_rate"]["value"] == 0.0
    assert two["informedness"]["value"] == 1.0
    assert two["balanced_accuracy"]["value"] == 1.0
    assert full["catch_rate"]["value"] == 1.0
    assert full["false_reject_rate"]["value"] == 0.25
    assert full["rejection_rate"]["value"] > two["rejection_rate"]["value"]
    assert all(row["uncertainty"]["method"] == "deterministic_paired_bootstrap" for row in metrics)


def test_auc_pr_and_calibration_are_defined_from_scores(
    construction: dict, transition_model: dict
) -> None:
    """REQ-VERIFY-6650 reports score discrimination and calibration separately."""

    rows = exp.score_twins(construction, transition_model)["per_unit_rows"]
    metrics = {row["unit_id"]: row for row in exp.compute_unit_metrics(rows)}
    assert metrics["one_step"]["auroc"]["value"] == 0.5
    assert metrics["two_steps"]["auroc"]["value"] == 1.0
    assert metrics["two_steps"]["auprc"]["value"] == 1.0
    assert metrics["two_steps"]["calibration"]["brier_score"]["value"] == 0.0
    assert metrics["two_steps"]["calibration"]["expected_calibration_error"]["value"] == 0.0
    assert metrics["full_remaining_suffix"]["calibration"]["brier_score"]["value"] == 0.125


def test_metrics_keep_undefined_denominators_null() -> None:
    """REQ-VERIFY-6650 never turns a missing class into a zero score."""

    assert exp.auroc([True, True], [0.1, 0.2]) is None
    assert exp.auprc([False, False], [0.1, 0.2]) is None
    assert exp.rate(0, 0) is None
    assert exp.percentile_interval([]) is None


def test_recommendation_selects_two_steps_and_rejects_longer_harm(
    construction: dict, transition_model: dict
) -> None:
    """SCENARIO-VERIFY-6650-RECOMMENDATION blocks a rejection-only gain."""

    rows = exp.score_twins(construction, transition_model)["per_unit_rows"]
    metrics = exp.compute_unit_metrics(rows)
    recommendation = exp.recommend_verifier_unit(metrics)
    assert recommendation["selected_unit"] == "two_steps"
    assert recommendation["selection_made"] is True
    assert recommendation["exact_checker_still_authorizes"] is True
    full = next(
        row
        for row in recommendation["unit_assessments"]
        if row["unit_id"] == "full_remaining_suffix"
    )
    assert full["eligible"] is False
    assert "false_reject" in full["reason"]


def test_equal_metrics_produce_no_selection() -> None:
    """SCENARIO-VERIFY-6650-RECOMMENDATION requires measured improvement."""

    metrics = [
        {
            "unit_id": unit,
            "informedness": {"value": 0.5},
            "balanced_accuracy": {"value": 0.75},
            "false_reject_rate": {"value": 0.0},
        }
        for unit in exp.UNIT_ORDER
    ]
    result = exp.recommend_verifier_unit(metrics)
    assert result["selected_unit"] is None
    assert result["selection_made"] is False


def test_complete_artifact_has_required_fields_and_recomputes(upstream: dict) -> None:
    """REQ-REPORT-6650 builds the required bounded evidence artifact."""

    artifact = exp.build_artifact(REPO_ROOT, upstream=upstream, duration_s=0.25)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["verdict_class"] == "positive"
    assert artifact["recommended_verifier_unit"]["selected_unit"] == "two_steps"
    assert artifact["inference_substrate"] == "frozen_candidate_verifier_unit_replay_no_llm"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["authority_boundary"]["exact_checker_authorizes"] is True
    assert artifact["authority_boundary"]["learned_or_advisory_scorer_authorizes"] is False
    assert artifact["aggregate_row_recomputation"]["all_metrics_match_rows"] is True
    assert artifact["reproducibility_checksum"] == exp.artifact_checksum(artifact)
    assert exp.validate_artifact(artifact, REPO_ROOT) == []


def test_validator_rejects_checksum_rows_authority_and_verdict(upstream: dict) -> None:
    """SCENARIO-REPORT-6650-ATOMIC-CHECKSUM detects material mutations."""

    artifact = exp.build_artifact(REPO_ROOT, upstream=upstream, duration_s=0.25)
    mutations = []
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:bad"
    mutations.append((changed, "reproducibility_checksum_mismatch"))
    changed = deepcopy(artifact)
    changed["verifier_is_oracle"] = True
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    mutations.append((changed, "verifier_is_oracle_mismatch"))
    changed = deepcopy(artifact)
    changed["verdict_class"] = "win"
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    mutations.append((changed, "verdict_class_invalid"))
    changed = deepcopy(artifact)
    changed["per_unit_rows"][0]["score"] = 0.5
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    mutations.append((changed, "aggregate_row_recomputation_mismatch"))
    for payload, expected in mutations:
        assert expected in exp.validate_artifact(payload, REPO_ROOT)


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (lambda value: value.update(candidate_corpus_complete=False), "candidate_corpus_complete"),
        (lambda value: value["candidate_rows"].pop(), "row_count"),
        (
            lambda value: value["frozen_task_manifest"]["compiler_checker_identity"].update(
                exact_executor_version="changed"
            ),
            "exact_checker_identity",
        ),
    ],
)
def test_blocked_gate_names_observed_value(upstream: dict, mutation, failed_check: str) -> None:
    """SCENARIO-REPORT-6650-BLOCKED-GATE retains the failed observation."""

    changed = deepcopy(upstream)
    mutation(changed)
    artifact = exp.build_artifact(REPO_ROOT, upstream=changed, duration_s=0.25)
    assert artifact["status"].startswith("blocked_")
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["first_failed_check"] == failed_check
    assert artifact["twin_rows"] == []
    assert artifact["unit_metric_rows"] == []
    assert exp.validate_artifact(artifact, REPO_ROOT, verify_source_file=False) == []


def test_atomic_write_replaces_complete_json(tmp_path: Path, upstream: dict) -> None:
    """SCENARIO-REPORT-6650-ATOMIC-CHECKSUM leaves one durable document."""

    artifact = exp.build_artifact(REPO_ROOT, upstream=upstream, duration_s=0.25)
    target = tmp_path / "nested" / "exp6650.json"
    receipt = exp.write_artifact_atomic(target, artifact, repo_root=REPO_ROOT)
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert receipt["atomic_replace"] is True
    assert receipt["directory_fsync"] is True
    assert receipt["sha256"] == exp.sha256_file(target)


def test_protected_receipt_detects_change(tmp_path: Path) -> None:
    """REQ-REPORT-6650 reports changed protected bytes instead of hiding them."""

    roadmap = tmp_path / "research-roadmap.yaml"
    conductor = tmp_path / "scripts" / "research_conductor.py"
    conductor.parent.mkdir()
    roadmap.write_text("roadmap", encoding="utf-8")
    conductor.write_text("conductor", encoding="utf-8")
    before = exp.protected_hashes(tmp_path)
    roadmap.write_text("changed", encoding="utf-8")
    receipt = exp.protected_files_receipt(tmp_path, before)
    assert receipt["all_unchanged"] is False
    assert receipt["files"]["research-roadmap.yaml"]["unchanged"] is False


def test_field_provenance_covers_every_required_field(upstream: dict) -> None:
    """REQ-REPORT-6650 gives source, hash, reducer, and schema lineage."""

    artifact = exp.build_artifact(REPO_ROOT, upstream=upstream, duration_s=0.25)
    provenance = artifact["field_provenance"]
    assert set(provenance) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert all(
        {"source", "source_sha256", "reducer", "schema"} <= set(row) for row in provenance.values()
    )


def test_invalid_artifact_shape_fails_closed(upstream: dict) -> None:
    """REQ-REPORT-6650 validates missing fields and invalid duration."""

    artifact = exp.build_artifact(REPO_ROOT, upstream=upstream, duration_s=0.25)
    missing = deepcopy(artifact)
    missing.pop("twin_rows")
    assert exp.validate_artifact(missing, REPO_ROOT)[0].startswith("missing_required_fields:")
    bad_duration = deepcopy(artifact)
    bad_duration["duration_s"] = 0.0
    bad_duration["reproducibility_checksum"] = exp.artifact_checksum(bad_duration)
    assert "duration_s_invalid" in exp.validate_artifact(bad_duration, REPO_ROOT)


def test_helper_error_paths_preserve_missing_information(
    tmp_path: Path, transition_model: dict
) -> None:
    """REQ-CONSTRAINT-6650 and REQ-VERIFY-6650 fail closed on malformed helpers."""

    with pytest.raises(ValueError, match="task_payload_missing"):
        exp._task_payload({})
    with pytest.raises(ValueError, match="byte_length_mismatch"):
        exp._byte_difference_count(b"a", b"bb")
    with pytest.raises(ValueError, match="unknown_unit"):
        exp._unit_slice("unknown", 0, 1)
    assert exp._brier([], []) is None
    assert exp._ece([], []) is None
    assert exp._bootstrap_intervals([], exp.BOOTSTRAP_SEED) == {}
    assert exp._metric_value({"value": 1.0}, "missing") is None
    assert exp._first_failed_check(
        {"checks": {"ok": True}, "all_required_preconditions_available": True}
    ) == ("preconditions", True)
    unreadable = exp._no_llm_receipt(tmp_path)
    assert unreadable["forbidden_llm_imports"] == []
    assert unreadable["module_sha256"] == "missing"
    assert exp.score_action_unit(["unknown"], transition_model)["unknown_action_ids"] == ["unknown"]


def test_candidate_search_skips_unsupported_clean_and_supported_error_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CONSTRAINT-6650-NON-PAIRABLE exercises bounded mutation search."""

    monkeypatch.setattr(exp, "_task_action_map", lambda _task: {"AA": "a", "BB": "b", "CC": "c"})
    unsupported_clean = {
        "supported_transitions": [],
    }
    assert exp._candidate_mutation({}, ["AA", "BB"], unsupported_clean) is None
    supported_error = {
        "supported_transitions": [["a", "b"], ["b", "b"]],
    }
    monkeypatch.setattr(
        exp.exp6649,
        "localize_exact_outcome",
        lambda _task, _plan: {"exact_final_validity": True, "first_failing_step": None},
    )
    assert exp._candidate_mutation({}, ["AA", "BB"], supported_error) is None


def test_construct_twins_names_both_late_rejection_reasons(
    upstream: dict, transition_model: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CONSTRAINT-6650-NON-PAIRABLE distinguishes replay and mutation gaps."""

    valid = next(row for row in upstream["candidate_rows"] if row["exact_final_validity"] is True)
    source = deepcopy(upstream)
    source["candidate_rows"] = [deepcopy(valid)]
    monkeypatch.setattr(exp, "_candidate_mutation", lambda *_args: None)
    result = exp.construct_twins(source, transition_model)
    assert result["rejected_pairs"][0]["rejection_reason"] == "no_byte_matched_semantic_mutation"

    monkeypatch.setattr(
        exp.exp6649,
        "localize_exact_outcome",
        lambda _task, _plan: {"exact_final_validity": False},
    )
    monkeypatch.setattr(exp, "_candidate_mutation", lambda *_args: {"unused": True})
    result = exp.construct_twins(source, transition_model)
    assert result["rejected_pairs"][0]["rejection_reason"] == "source_clean_exact_replay_failed"


def test_twin_validator_reports_each_contract_failure(construction: dict) -> None:
    """REQ-CONSTRAINT-6650 detects every byte and authority contract mutation."""

    base = construction["twins"][0]
    changed = deepcopy(base)
    changed["localized_step"] = -1
    assert exp._validate_twin(changed) == ["twin_localized_step_invalid"]
    changed = deepcopy(base)
    changed["error_plan"] += "X"
    assert "twin_byte_or_line_count_mismatch" in exp._validate_twin(changed)
    changed = deepcopy(base)
    lines = changed["error_plan"].splitlines()
    nonlocalized = 0 if changed["localized_step"] != 0 else 1
    lines[nonlocalized] = lines[nonlocalized][::-1]
    changed["error_plan"] = "\n".join(lines)
    assert "twin_nonlocalized_bytes_changed" in exp._validate_twin(changed)
    changed = deepcopy(base)
    changed["clean_exact_label"] = False
    assert "twin_exact_labels_invalid" in exp._validate_twin(changed)
    changed = deepcopy(base)
    changed["advisory_scorer_supplies_exact_label"] = True
    assert "twin_advisory_authority_invalid" in exp._validate_twin(changed)


def test_validator_reports_remaining_schema_and_blocked_mutations(upstream: dict) -> None:
    """REQ-REPORT-6650 covers every fail-closed artifact branch."""

    artifact = exp.build_artifact(REPO_ROOT, upstream=upstream, duration_s=0.25)
    mutations = []
    for field, value, expected in (
        ("inference_substrate", "live_llm_inference", "inference_substrate_mismatch"),
        ("field_provenance", {}, "field_provenance_mismatch"),
        (
            "protected_files_unchanged",
            {"all_unchanged": False},
            "protected_files_changed",
        ),
        (
            "authority_boundary",
            {"learned_or_advisory_scorer_authorizes": True},
            "authority_boundary_mismatch",
        ),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        mutations.append((changed, expected))
    changed = deepcopy(artifact)
    changed["upstream_gate_receipt"]["sha256"] = "sha256:bad"
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    mutations.append((changed, "upstream_artifact_hash_mismatch"))
    changed = deepcopy(artifact)
    changed["recommended_verifier_unit"]["selected_unit"] = None
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    mutations.append((changed, "recommended_verifier_unit_mismatch"))
    for payload, expected in mutations:
        assert expected in exp.validate_artifact(payload, REPO_ROOT)

    blocked_source = deepcopy(upstream)
    blocked_source["candidate_corpus_complete"] = False
    blocked = exp.build_artifact(REPO_ROOT, upstream=blocked_source, duration_s=0.25)
    for field, value, expected in (
        ("status", "complete", "blocked_status_prefix_missing"),
        ("honest_verdict", "complete", "blocked_verdict_prefix_missing"),
        ("gate_check_summary", {}, "blocked_gate_detail_missing"),
        ("twin_rows", [{}], "blocked_artifact_invented_rows"),
    ):
        changed = deepcopy(blocked)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected in exp.validate_artifact(changed, REPO_ROOT, verify_source_file=False)


def test_atomic_write_rejects_invalid_payload(tmp_path: Path, upstream: dict) -> None:
    """SCENARIO-REPORT-6650-ATOMIC-CHECKSUM validates before replacement."""

    artifact = exp.build_artifact(REPO_ROOT, upstream=upstream, duration_s=0.25)
    artifact["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum_mismatch"):
        exp.write_artifact_atomic(tmp_path / "invalid.json", artifact, repo_root=REPO_ROOT)


def test_changed_protected_receipt_blocks_before_scoring(
    upstream: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6650-BLOCKED-GATE covers a run-time protected change."""

    monkeypatch.setattr(
        exp,
        "protected_files_receipt",
        lambda _root, _before: {"files": {}, "all_unchanged": False},
    )
    artifact = exp.build_artifact(REPO_ROOT, upstream=upstream, duration_s=0.25)
    assert artifact["status"] == "blocked_protected_hashes"


def test_run_writes_default_source_and_argument_parser(tmp_path: Path) -> None:
    """REQ-REPORT-6650 exercises the requested end-to-end module path."""

    target = tmp_path / "exp6650.json"
    artifact = exp.run("20260826", REPO_ROOT, output=target)
    assert target.is_file()
    assert artifact["run_date"] == "20260826"
    assert exp.validate_artifact(artifact, REPO_ROOT) == []
    args = exp._parse_args(
        ["--date", "20260826", "--repo-root", str(REPO_ROOT), "--output", str(target)]
    )
    assert args.date == "20260826"
    assert args.repo_root == REPO_ROOT
    assert args.output == target
