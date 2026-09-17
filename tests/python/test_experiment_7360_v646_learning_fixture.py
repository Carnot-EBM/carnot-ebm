"""Tests for the V646 safety-only learning fixture.

Spec refs: REQ-CL-7360 and SCENARIO-CL-7360-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7360_v646_learning_fixture as exp


@pytest.fixture()
def sealed_evidence(tmp_path: Path) -> tuple[exp.FixturePaths, dict, dict]:
    """Build real fresh manifests once for tests that inspect sealed evidence."""

    paths = exp.FixturePaths.for_raw_dir(tmp_path / "raw")
    public_manifest, private_manifest = exp.build_fixture_manifests(
        paths,
        exp.REPO_ROOT / exp.V645_PUBLIC_MANIFEST_PATH,
    )
    return paths, public_manifest, private_manifest


def test_preconditions_accept_eligible_null_and_reject_bad_producers(tmp_path: Path) -> None:
    """REQ-CL-7360: infrastructure readiness does not require efficacy."""

    producer = {
        "status": "complete_validation_contract_null_science",
        "milestone": exp.MILESTONE,
        "run_date": exp.RUN_DATE,
        "validation_contract_ready_score": 1,
        "verdict_class": "null",
        "flagged_adversarial": False,
    }
    producer_path = tmp_path / "producer.json"
    producer_path.write_text(json.dumps(producer), encoding="utf-8")
    exclusion_path = tmp_path / "exclusions.yaml"
    exclusion_path.write_text("entries: []\n", encoding="utf-8")

    rows, hashes, loaded = exp.collect_preconditions(
        exp.REPO_ROOT,
        producer_path=producer_path,
        exclusion_path=exclusion_path,
    )

    assert all(row["available"] for row in rows)
    assert hashes[str(producer_path)] == exp.sha256_file(producer_path)
    assert loaded["verdict_class"] == "null"

    for field, bad_value in (
        ("status", "blocked_upstream"),
        ("validation_contract_ready_score", 0),
        ("verdict_class", "disqualified"),
        ("flagged_adversarial", True),
    ):
        changed = {**producer, field: bad_value}
        producer_path.write_text(json.dumps(changed), encoding="utf-8")
        bad_rows, _hashes, _producer = exp.collect_preconditions(
            exp.REPO_ROOT,
            producer_path=producer_path,
            exclusion_path=exclusion_path,
        )
        failed = [row for row in bad_rows if not row["available"]]
        assert any(row["artifact_field"] == field for row in failed)

    producer_path.unlink()
    missing, _hashes, _producer = exp.collect_preconditions(
        exp.REPO_ROOT,
        producer_path=producer_path,
        exclusion_path=exclusion_path,
    )
    assert any(row["check"] == "producer_path" and not row["available"] for row in missing)

    producer_path.write_text("[]", encoding="utf-8")
    malformed, _hashes, _producer = exp.collect_preconditions(
        exp.REPO_ROOT,
        producer_path=producer_path,
        exclusion_path=exclusion_path,
    )
    assert any(row["check"] == "producer_path" and not row["available"] for row in malformed)

    with pytest.raises(ValueError, match="expected_object"):
        exp.load_json(producer_path)


def test_fresh_panel_counts_hashes_twins_and_private_separation(
    sealed_evidence: tuple[exp.FixturePaths, dict, dict],
) -> None:
    """SCENARIO-CL-7360-PANEL: the new panel is complete and non-overlapping."""

    paths, public_manifest, private_manifest = sealed_evidence
    assert (
        exp.panel_errors(
            public_manifest,
            private_manifest,
            exp.load_json(exp.REPO_ROOT / exp.V645_PUBLIC_MANIFEST_PATH),
        )
        == []
    )
    assert len(public_manifest["development_streams"]) == 32
    assert {stream["cohort"] for stream in public_manifest["development_streams"]} == set(
        exp.COHORTS
    )
    assert all(len(stream["requests"]) == 12 for stream in public_manifest["development_streams"])
    assert len(public_manifest["public_request_streams"]) == 8
    assert len(public_manifest["live_proposal_panel"]) == 32
    assert len(public_manifest["development_canary"]) == 4
    assert all(
        exp._normalized_request(pair["original"]) == exp._normalized_request(pair["twin"])
        for pair in public_manifest["live_proposal_panel"]
    )
    assert "private_rules" not in paths.public_manifest.read_text(encoding="utf-8")
    assert "acceptance_witness" not in paths.public_manifest.read_text(encoding="utf-8")
    assert private_manifest["public_manifest_sha256"] == exp.sha256_file(paths.public_manifest)


def test_panel_mutations_fail_closed(
    sealed_evidence: tuple[exp.FixturePaths, dict, dict],
) -> None:
    """SCENARIO-CL-7360-PANEL: hidden labels and old request reuse are rejected."""

    _paths, public_manifest, private_manifest = sealed_evidence
    old = exp.load_json(exp.REPO_ROOT / exp.V645_PUBLIC_MANIFEST_PATH)
    leaked = deepcopy(public_manifest)
    leaked["development_streams"][0]["requests"][0]["private_rules"] = {"capacity": 1}
    assert "private_data_in_public_manifest" in exp.panel_errors(leaked, private_manifest, old)

    overlap = deepcopy(public_manifest)
    overlap["development_streams"][0]["requests"][0] = deepcopy(
        old["development_streams"][0]["requests"][0]
    )
    assert "v645_request_overlap" in exp.panel_errors(overlap, private_manifest, old)

    duplicate = deepcopy(public_manifest)
    duplicate["development_streams"][0]["requests"][1]["request_id"] = duplicate[
        "development_streams"
    ][0]["requests"][0]["request_id"]
    assert "request_ids_not_distinct" in exp.panel_errors(duplicate, private_manifest, old)

    mutations = []
    bad = deepcopy(public_manifest)
    bad["development_streams"].pop()
    mutations.append((bad, private_manifest, "development_panel_shape"))
    bad = deepcopy(public_manifest)
    bad["development_streams"][0]["cohort"] = "wrong"
    mutations.append((bad, private_manifest, "development_cohorts"))
    bad = deepcopy(public_manifest)
    bad["development_streams"][0]["requests"][0]["warmup"] = False
    mutations.append((bad, private_manifest, "development_warmup"))
    bad = deepcopy(public_manifest)
    bad["public_request_streams"].pop()
    mutations.append((bad, private_manifest, "public_panel_shape"))
    bad = deepcopy(public_manifest)
    bad["live_proposal_panel"][0]["warmup"] = False
    mutations.append((bad, private_manifest, "public_warmup"))
    bad = deepcopy(public_manifest)
    bad["development_canary"].pop()
    mutations.append((bad, private_manifest, "development_canary_count"))
    bad = deepcopy(public_manifest)
    bad["live_proposal_panel"][1]["original"] = deepcopy(bad["live_proposal_panel"][0]["original"])
    mutations.append((bad, private_manifest, "public_requests_not_distinct"))
    bad = deepcopy(public_manifest)
    bad["live_proposal_panel"][0]["twin"]["horizon"] += 1
    mutations.append((bad, private_manifest, "renamed_twin_mismatch"))
    bad = deepcopy(public_manifest)
    bad["evaluation_seed"] = bad["development_seed"]
    mutations.append((bad, private_manifest, "seeds_not_disjoint"))
    bad = deepcopy(public_manifest)
    bad["manifest_hash"] = "sha256:" + "0" * 64
    mutations.append((bad, private_manifest, "public_manifest_hash"))
    bad_private = deepcopy(private_manifest)
    bad_private["public_manifest_hash"] = "sha256:" + "0" * 64
    mutations.append((public_manifest, bad_private, "private_public_manifest_binding"))
    bad_private = deepcopy(private_manifest)
    bad_private["evaluator_records"].pop(next(iter(bad_private["evaluator_records"])))
    mutations.append((public_manifest, bad_private, "private_record_identity"))
    bad_private = deepcopy(private_manifest)
    bad_private["all_acceptance_witnesses_nonempty"] = False
    mutations.append((public_manifest, bad_private, "empty_acceptance_witness"))
    for changed_public, changed_private, expected in mutations:
        assert expected in exp.panel_errors(changed_public, changed_private, old)


def test_acceptance_manifest_freezes_value_without_making_it_a_fixture_gate() -> None:
    """SCENARIO-CL-7360-ACCEPTANCE: Exp7362 thresholds are preregistered only."""

    manifest = exp.frozen_acceptance_manifest("sha256:" + "1" * 64)

    assert manifest["arms"] == list(exp.ARMS)
    assert manifest["query_budget_per_request"] == 24
    assert manifest["state_cap_bytes"] == 69_632
    assert manifest["same_proposal_list_per_paired_request"] is True
    assert manifest["same_exact_executor_per_paired_request"] is True
    assert manifest["charged_operations"] == [
        "information_query",
        "proposal_check",
        "write",
        "persistence",
        "verification",
    ]
    assert all(gate["fixture_gate"] is False for gate in manifest["value_gates"])
    by_name = {gate["check"]: gate for gate in manifest["value_gates"]}
    assert by_name["query_ratio_vs_reset"]["upper_95_exclusive"] == 0.90
    assert by_name["query_ratio_vs_exact_cache"]["upper_95_exclusive"] == 0.90
    assert by_name["complete_service_cost_ratio"]["upper_95_inclusive"] == 1.0


def test_adapter_controls_cover_drift_lifecycle_and_release_authority(tmp_path: Path) -> None:
    """SCENARIO-CL-7360-SAFETY: actual adapter controls fail closed."""

    rows = exp.run_adapter_safety_controls(tmp_path / "state")
    by_control = {row["control"]: row for row in rows}

    assert all(row["passed"] for row in rows)
    assert by_control["post_request_only_commit"]["observed"]["entry_unchanged"] is True
    assert by_control["contradictory_feedback"]["observed"]["invalidated_count"] >= 1
    assert by_control["announced_drift"]["observed"]["active_atoms"] == 0
    assert by_control["rollback_restart_post_request_commit"]["passed"] is True
    assert by_control["unsafe_cached_release"]["observed"] == {
        "final_external_calls": 1,
        "final_cache_hits": 0,
    }
    assert all(row["query_attempts"] <= 24 for row in rows)
    assert all(row["state_bytes"] <= 69_632 for row in rows)


def test_real_process_boundary_and_safety_rows(
    sealed_evidence: tuple[exp.FixturePaths, dict, dict],
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7360-SAFETY: learner and evaluator remain separate processes."""

    paths, public_manifest, private_manifest = sealed_evidence
    process = exp.run_isolated_fixture(exp.REPO_ROOT, paths)
    adapter_rows = exp.run_adapter_safety_controls(tmp_path / "adapter")
    safety_rows = exp.build_safety_rows(
        public_manifest,
        private_manifest,
        process,
        adapter_rows,
    )

    assert process["learner"]["learner_pid"] != process["evaluator"]["evaluator_pid"]
    assert process["learner"]["forbidden_accesses"] == []
    assert process["evaluator"]["response_keys"] == ["accepted", "query_id"]
    assert all(row["passed"] for row in safety_rows)
    assert any(row["control"] == "higher_order_counterexample" for row in safety_rows)
    assert any(row["control"] == "hidden_rule_access_rejection" for row in safety_rows)


def test_terminal_classification_keeps_readiness_separate_from_value(
    sealed_evidence: tuple[exp.FixturePaths, dict, dict],
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7360-TERMINAL: safe fixture readiness survives an unmeasured value gate."""

    paths, public_manifest, private_manifest = sealed_evidence
    process = exp.run_isolated_fixture(exp.REPO_ROOT, paths)
    adapter_rows = exp.run_adapter_safety_controls(tmp_path / "adapter")
    safety_rows = exp.build_safety_rows(public_manifest, private_manifest, process, adapter_rows)
    exp.write_raw_evidence(paths, safety_rows)
    preconditions, source_hashes, _producer = exp.collect_preconditions(exp.REPO_ROOT)
    source_hashes.update(exp.raw_source_hashes(paths))
    artifact = exp.artifact_from_evidence(
        paths=paths,
        preconditions=preconditions,
        source_hashes=source_hashes,
        safety_rows=safety_rows,
        validation_receipts=exp.passing_test_receipts(),
        duration_s=1.0,
        phase_spans=exp.test_phase_spans(),
    )

    assert artifact["learning_fixture_ready_score"] == 1, artifact["acceptance_gate_results"]
    assert artifact["learning_value_score"] == 0
    assert artifact["promotion_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["frozen_acceptance_manifest"]["value_evaluated"] is False
    assert exp.independent_reduce(artifact) == []
    assert exp.validate_artifact(artifact) == []

    raw_missing = deepcopy(artifact)
    raw_missing["raw_evidence_paths"] = {}
    assert exp.independent_reduce(raw_missing) == ["raw_evidence_unavailable"]

    row_mismatch = deepcopy(artifact)
    row_mismatch["rows"][0]["passed"] = False
    assert "safety_rows_mismatch" in exp.independent_reduce(row_mismatch)
    control_failure = deepcopy(artifact)
    control_failure["safety_rows"][0]["passed"] = False
    control_failure["rows"] = deepcopy(control_failure["safety_rows"])
    original_safety_file = exp.load_json(paths.safety_rows)
    exp._atomic_json(
        paths.safety_rows,
        {"schema": original_safety_file["schema"], "rows": control_failure["safety_rows"]},
    )
    assert "safety_control_failed" in exp.independent_reduce(control_failure)
    exp._atomic_json(paths.safety_rows, original_safety_file)
    fixture_mismatch = deepcopy(artifact)
    fixture_mismatch["fixture_manifest"] = {}
    assert "fixture_manifest_mismatch" in exp.independent_reduce(fixture_mismatch)
    acceptance_mismatch = deepcopy(artifact)
    acceptance_mismatch["frozen_acceptance_manifest"] = {}
    assert "acceptance_manifest_mismatch" in exp.independent_reduce(acceptance_mismatch)

    assert exp.validate_artifact([]) == ["artifact_mapping_required"]
    assert exp.validate_artifact({})[0].startswith("missing_required_field:")

    validation_mutations = [
        ("schema", "wrong", "identity_invalid"),
        ("MODEL_SPECS", [{"hf_id": "unexpected"}], "model_contract_invalid"),
        ("inference_substrate", "wrong", "substrate_invalid"),
        ("verdict_class", "wrong", "verdict_class_invalid"),
        ("verifier_is_oracle", False, "oracle_disclosure_missing"),
        ("learning_value_score", 1, "value_promotion_not_zero"),
    ]
    for field, value, expected in validation_mutations:
        bad = deepcopy(artifact)
        bad[field] = value
        bad["reproducibility_checksum"] = exp.reproducibility_checksum(bad)
        assert expected in exp.validate_artifact(bad, verify_source_hashes=False)

    bad = deepcopy(artifact)
    bad["random_seed"]["evaluation"] = bad["random_seed"]["development"]
    bad["reproducibility_checksum"] = exp.reproducibility_checksum(bad)
    assert "random_seed_invalid" in exp.validate_artifact(bad, verify_source_hashes=False)
    bad = deepcopy(artifact)
    bad["validation_receipts"][0]["command"] = ""
    assert "validation_receipt_invalid" in exp.validate_artifact(bad, verify_source_hashes=False)
    bad = deepcopy(artifact)
    bad["validation_receipts"] = []
    assert "ready_state_invalid" in exp.validate_artifact(bad, verify_source_hashes=False)
    bad = deepcopy(artifact)
    bad["flagged_adversarial"] = True
    assert "ready_adversarial_invalid" in exp.validate_artifact(bad, verify_source_hashes=False)
    bad = deepcopy(artifact)
    bad["phase_spans"][1]["start_elapsed_s"] = 2.0
    bad["phase_spans"][1]["end_elapsed_s"] = 1.0
    assert "phase_spans_invalid" in exp.validate_artifact(bad, verify_source_hashes=False)
    bad = deepcopy(artifact)
    bad["field_principles"] = {}
    assert "field_principles_invalid" in exp.validate_artifact(bad, verify_source_hashes=False)
    bad = deepcopy(artifact)
    bad["source_artifact_hashes"][str(paths.public_manifest)] = "sha256:" + "0" * 64
    assert "source_hash_mismatch" in exp.validate_artifact(bad)

    changed = deepcopy(artifact)
    changed["safety_rows"][0]["passed"] = False
    changed["rows"] = deepcopy(changed["safety_rows"])
    changed = exp.reclassify_artifact(changed)
    assert changed["learning_fixture_ready_score"] == 0
    assert changed["verdict_class"] == "disqualified"


def test_blocked_artifact_names_exact_failed_field(tmp_path: Path) -> None:
    """SCENARIO-CL-7360-PRECONDITION: external absence is blocked, not partial."""

    missing = tmp_path / "missing.json"
    rows, hashes, _producer = exp.collect_preconditions(
        exp.REPO_ROOT,
        producer_path=missing,
    )
    artifact = exp.blocked_artifact(rows, hashes, duration_s=0.01)

    assert artifact["status"].startswith("blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["safety_rows"] == []
    assert artifact["learning_fixture_ready_score"] == 0
    assert artifact["gate_check_summary"] == {
        "passed": False,
        "upstream": str(missing),
        "failed_check": "producer_path",
        "artifact_field": "bytes",
        "expected_value": "readable_nonempty_json",
        "observed_value": "missing",
    }
    assert exp.validate_artifact(artifact, verify_source_hashes=False) == []

    with pytest.raises(ValueError, match="requires failed precondition"):
        exp.blocked_artifact(
            [exp._precondition("ready", "fixture", "ready", True, True, True)],
            {},
            duration_s=0.01,
        )

    bad = deepcopy(artifact)
    bad["rows"] = [{"unexpected": True}]
    bad["learning_fixture_ready_score"] = 1
    bad["gate_check_summary"] = {}
    bad["reproducibility_checksum"] = exp.reproducibility_checksum(bad)
    errors = exp.validate_artifact(bad, verify_source_hashes=False)
    assert "blocked_dependent_work_present" in errors
    assert "blocked_gate_summary_incomplete" in errors
    assert "failed_readiness_not_zero" in errors


def test_scoped_plan_and_atomic_writer_reject_mutation(
    sealed_evidence: tuple[exp.FixturePaths, dict, dict],
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7360-E2E: validation stays explicit and terminal bytes fail closed."""

    commands = exp.scoped_command_plan(exp.REPO_ROOT, tmp_path / "validation")
    pytest_commands = [row for row in commands if "pytest" in " ".join(row.argv)]
    assert pytest_commands
    assert all(str(exp.TEST_PATH) in row.argv for row in pytest_commands)
    assert all("tests/python" not in row.argv for row in commands)

    paths, public_manifest, private_manifest = sealed_evidence
    process = exp.run_isolated_fixture(exp.REPO_ROOT, paths)
    safety_rows = exp.build_safety_rows(
        public_manifest,
        private_manifest,
        process,
        exp.run_adapter_safety_controls(tmp_path / "adapter"),
    )
    exp.write_raw_evidence(paths, safety_rows)
    preconditions, hashes, _producer = exp.collect_preconditions(exp.REPO_ROOT)
    hashes.update(exp.raw_source_hashes(paths))
    artifact = exp.artifact_from_evidence(
        paths=paths,
        preconditions=preconditions,
        source_hashes=hashes,
        safety_rows=safety_rows,
        validation_receipts=exp.passing_test_receipts(),
        duration_s=1.0,
        phase_spans=exp.test_phase_spans(),
    )
    output = tmp_path / "artifact.json"
    receipt = exp.write_artifact(output, artifact)
    assert receipt["sha256"] == exp.sha256_file(output)

    mutated = deepcopy(artifact)
    mutated["random_seed"]["evaluation"] = mutated["random_seed"]["development"]
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        exp.write_artifact(tmp_path / "bad.json", mutated)


def test_build_artifact_covers_scoped_and_blocked_orchestration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-7360-E2E: the orchestrator consumes bounded runner receipts."""

    scoped_receipts = [
        row for row in exp.passing_test_receipts() if row["name"] in exp.REQUIRED_CHECK_NAMES
    ]
    terminal_receipts = [
        row for row in exp.passing_test_receipts() if row["name"] in exp.TERMINAL_CHECK_NAMES
    ]

    def fake_scoped(*_args: object, **_kwargs: object) -> dict:
        return {
            "validation_receipts": scoped_receipts,
            "repository_health": {"status": "healthy", "affects_required_checks": False},
        }

    def fake_terminal(*_args: object, **_kwargs: object) -> list[dict]:
        return terminal_receipts

    monkeypatch.setattr(exp, "run_scoped_validation", fake_scoped)
    monkeypatch.setattr(exp, "run_commands", fake_terminal)
    output = tmp_path / "terminal.json"
    artifact = exp.build_artifact(
        root=exp.REPO_ROOT,
        output_path=output,
        raw_dir=tmp_path / "raw",
    )
    assert output.is_file()
    assert artifact["learning_fixture_ready_score"] == 1
    assert exp.validate_artifact(artifact) == []

    failed = exp._precondition("producer_path", "missing", "bytes", "present", "missing", False)
    monkeypatch.setattr(exp, "collect_preconditions", lambda _root: ([failed], {}, {}))
    blocked_output = tmp_path / "blocked.json"
    blocked = exp.build_artifact(
        root=exp.REPO_ROOT,
        output_path=blocked_output,
        raw_dir=tmp_path / "unused",
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked_output.is_file()
