"""Tests for the bounded Exp6716 seal and attack audit.

Spec: REQ-SAFE-6716, REQ-PIPELINE-6716, REQ-REPORT-6716,
REQ-VERIFY-6716, and their SCENARIO-* anchors.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6716_bounded_seal_attack_audit as exp


def passing_test_rows() -> list[dict[str, object]]:
    """Return complete command receipts for reducer and artifact tests."""

    rows: list[dict[str, object]] = []
    for check_id in exp.REQUIRED_TEST_CHECKS:
        rows.append(
            {
                "check_id": check_id,
                "command": f"check {check_id}",
                "exit_code": 0,
                "passed": True,
                "coverage_percent": 100.0 if check_id == "scoped_coverage" else None,
                "summary": "passed",
                "duration_s": 0.1,
            }
        )
    return rows


def fake_runner(command: str, _root: Path) -> dict[str, object]:
    """Return deterministic process evidence without running nested test suites."""

    stdout = ""
    if "coverage report" in command:
        stdout = "TOTAL 600 0 100%\n"
    elif "adversarial_verify.py" in command:
        stdout = json.dumps({"reports": [{"max_severity": 0}]})
    else:
        stdout = "1 passed\n"
    return {
        "command": command,
        "exit_code": 0,
        "stdout": stdout,
        "stderr": "",
        "duration_s": 0.1,
    }


def load_upstream() -> dict[str, object]:
    """Load the immutable Exp6702 fixture used by the bounded tests."""

    return exp.load_json(exp.REPO_ROOT / exp.UPSTREAM_PATH)


def test_attack_manifest_freezes_only_blind_identities() -> None:
    """REQ-SAFE-6716; SCENARIO-SAFE-6716-METAMORPHIC-MUTATION."""

    blinded = exp.public_attack_inputs(load_upstream())
    assert all(
        set(row).isdisjoint({"expected_invariant", "observed_result", "pass_state"})
        for row in blinded["metamorphic_cases"]
    )
    assert all(
        set(row).isdisjoint({"expected_detection", "observed_detection", "pass_state"})
        for row in blinded["mutation_cases"]
    )

    manifest = exp.freeze_attack_manifest(blinded)
    assert manifest["frozen_before_expected_result_read"] is True
    assert manifest["manifest_hash"] == exp.manifest_checksum(manifest)
    assert {row["family"]: row["count"] for row in manifest["attack_families"]} == {
        family: len(cases) for family, cases in exp.EXPECTED_CASES.items()
    }
    assert all(row["time_budget_s"] > 0 for row in manifest["attack_families"])
    assert sum(row["count"] for row in manifest["attack_families"]) == 40

    duplicate = deepcopy(blinded)
    duplicate["metamorphic_cases"].append(deepcopy(duplicate["metamorphic_cases"][0]))
    with pytest.raises(ValueError, match="attack identity mismatch"):
        exp.freeze_attack_manifest(duplicate)


def test_leakage_scanner_detects_each_shortcut_family() -> None:
    """REQ-SAFE-6716; SCENARIO-SAFE-6716-LEAKAGE."""

    upstream = load_upstream()
    clean = exp.scan_leakage(upstream)
    assert {row["check"] for row in clean} == set(exp.LEAKAGE_CASES)
    assert all(row["pass_state"] for row in clean)

    prompt = deepcopy(upstream)
    prompt["instance_rows"][0]["prompt"] += " Exact optimum: 27."
    assert not exp.row_by_case(exp.scan_leakage(prompt), "prompt_direct_label")["pass_state"]

    metadata = deepcopy(upstream)
    metadata["instance_rows"][0]["typed_spec"]["future_value"] = 27
    assert not exp.row_by_case(exp.scan_leakage(metadata), "metadata_label_encoding")["pass_state"]
    encoded_objective = deepcopy(upstream)
    encoded_objective["instance_rows"][0]["typed_spec"]["objective"] = "optimum: 27"
    assert not exp.row_by_case(exp.scan_leakage(encoded_objective), "metadata_label_encoding")[
        "pass_state"
    ]

    duplicate_id = deepcopy(upstream)
    duplicate_id["instance_rows"][-1]["instance"] = duplicate_id["instance_rows"][0]["instance"]
    assert not exp.row_by_case(exp.scan_leakage(duplicate_id), "instance_id_uniqueness")[
        "pass_state"
    ]

    shortcut = deepcopy(upstream)
    shortcut["instance_rows"][0]["instance"] += "-optimum-27"
    assert not exp.row_by_case(exp.scan_leakage(shortcut), "instance_id_shortcut")["pass_state"]

    contaminated = deepcopy(upstream)
    headline = next(row for row in contaminated["instance_rows"] if row["split"] == "headline")
    development = next(
        row
        for row in contaminated["instance_rows"]
        if row["split"] == "development" and row["family"] == headline["family"]
    )
    development["prompt"] = "  ".join(headline["prompt"].upper().split())
    assert not exp.row_by_case(exp.scan_leakage(contaminated), "development_to_held_contamination")[
        "pass_state"
    ]

    bad_split = deepcopy(upstream)
    bad_split["instance_rows"][0]["split"] = "future"
    assert not exp.row_by_case(exp.scan_leakage(bad_split), "split_membership")["pass_state"]

    family = deepcopy(upstream)
    left = family["instance_rows"][0]
    right = next(row for row in family["instance_rows"] if row["family"] != left["family"])
    right["spec_hash"] = left["spec_hash"]
    assert not exp.row_by_case(exp.scan_leakage(family), "family_isolation")["pass_state"]

    stale = deepcopy(upstream)
    stale["label_seal_rows"][0]["prompt_hash"] = "sha256:stale"
    assert not exp.row_by_case(exp.scan_leakage(stale), "hash_and_seal_freshness")["pass_state"]


def test_dynamic_seal_access_denies_all_invalid_timing_paths() -> None:
    """REQ-SAFE-6716; SCENARIO-SAFE-6716-SEAL-ACCESS."""

    rows = exp.run_seal_access_attacks(load_upstream())
    assert [row["case"] for row in rows] == list(exp.SEAL_ACCESS_CASES)
    assert all(row["pass_state"] for row in rows)
    valid = exp.row_by_case(rows, "valid_post_commit")
    assert valid["expected_access"] is True
    assert valid["observed_access"] is True
    assert valid["observed_label_hash"].startswith("sha256:")
    for case in set(exp.SEAL_ACCESS_CASES) - {"valid_post_commit"}:
        row = exp.row_by_case(rows, case)
        assert row["expected_access"] is False
        assert row["observed_access"] is False

    seal = load_upstream()["label_seal_rows"][0]
    store = exp.AuditLabelSeal(seal)
    with pytest.raises(exp.SealAccessError, match="prompt hash mismatch"):
        store.begin_event("bad", "sha256:wrong", 1)
    store.begin_event("event", seal["prompt_hash"], 1)
    with pytest.raises(exp.SealAccessError, match="event mismatch"):
        store.commit("other", {"plan": []})


def test_metamorphic_rows_replay_without_a_solver() -> None:
    """REQ-SAFE-6716; SCENARIO-SAFE-6716-METAMORPHIC-MUTATION."""

    upstream = load_upstream()
    before = exp.sha256_json(upstream)
    rows = exp.replay_metamorphic_rows(upstream)
    assert len(rows) == 16
    assert {row["case"] for row in rows} == set(exp.METAMORPHIC_CASES)
    assert all(row["pass_state"] for row in rows)
    assert all("raw_expected_result" in row and "observed_result" in row for row in rows)
    assert exp.sha256_json(upstream) == before

    tampered = deepcopy(upstream)
    tampered["metamorphic_rows"][0]["observed_result"] = {"aliases": {}}
    assert not exp.replay_metamorphic_rows(tampered)[0]["pass_state"]

    missing = deepcopy(upstream)
    missing["metamorphic_rows"].pop()
    assert any(not row["pass_state"] for row in exp.replay_metamorphic_rows(missing))

    source = (exp.REPO_ROOT / exp.MODULE_PATH).read_text(encoding="utf-8")
    assert "exhaustive_solve" not in source
    assert "import experiment_6702_exact_planning_fixture_recovery" not in source
    assert "import experiment_6703_exact_planning_fixture_audit" not in source


def test_mutation_and_memory_attacks_detect_without_side_effects() -> None:
    """REQ-SAFE-6716; SCENARIO-SAFE-6716-MEMORY-POISON."""

    upstream = load_upstream()
    fixture_before = exp.sha256_json(upstream["instance_rows"])
    seal_before = exp.sha256_json(upstream["label_seal_rows"])
    rows = exp.run_mutation_and_memory_attacks(upstream)
    assert len(rows) == 10
    assert {row["case"] for row in rows} == set(exp.MUTATION_CASES) | set(exp.MEMORY_POISON_CASES)
    assert all(row["expected_detection"] is True for row in rows)
    assert all(row["observed_detection"] is True for row in rows)
    assert all(row["pass_state"] for row in rows)
    assert exp.sha256_json(upstream["instance_rows"]) == fixture_before
    assert exp.sha256_json(upstream["label_seal_rows"]) == seal_before
    poison = [row for row in rows if row["kind"] == "memory_poison"]
    assert all(row["fixture_truth_unchanged"] and row["seal_state_unchanged"] for row in poison)

    changed_expected = deepcopy(upstream)
    changed_expected["mutation_rows"][0]["expected_detection"] = False
    changed = exp.run_mutation_and_memory_attacks(changed_expected)
    assert exp.row_by_case(changed, "bad_transition")["pass_state"] is False


def test_row_reducer_is_complete_and_fail_closed() -> None:
    """REQ-PIPELINE-6716; SCENARIO-PIPELINE-6716-ROW-REDUCTION."""

    upstream = load_upstream()
    manifest = exp.freeze_attack_manifest(exp.public_attack_inputs(upstream))
    campaign = exp.run_attack_campaign(upstream, manifest, clock=exp.StepClock())
    aggregate = exp.recompute_aggregate(
        manifest=manifest,
        leakage_rows=campaign["leakage_rows"],
        seal_access_rows=campaign["seal_access_rows"],
        metamorphic_rows=campaign["metamorphic_rows"],
        mutation_attack_rows=campaign["mutation_attack_rows"],
        budget_rows=campaign["budget_rows"],
        tests_run=passing_test_rows(),
        preconditions_passed=True,
        protected_files_unchanged=True,
        method_contract=exp.method_fidelity_contract(),
    )
    assert aggregate["seal_attack_audit_passed"] is True
    assert aggregate["failed_checks"] == []

    failed_rows = deepcopy(campaign["seal_access_rows"])
    failed_rows[0]["pass_state"] = False
    failed = exp.recompute_aggregate(
        manifest=manifest,
        leakage_rows=campaign["leakage_rows"],
        seal_access_rows=failed_rows,
        metamorphic_rows=campaign["metamorphic_rows"],
        mutation_attack_rows=campaign["mutation_attack_rows"],
        budget_rows=campaign["budget_rows"],
        tests_run=passing_test_rows(),
        preconditions_passed=True,
        protected_files_unchanged=True,
        method_contract=exp.method_fidelity_contract(),
    )
    assert failed["seal_attack_audit_passed"] is False
    assert "seal_access_rows" in failed["failed_checks"]


def test_complete_artifact_recomputes_provenance_and_checksum() -> None:
    """REQ-REPORT-6716; SCENARIO-REPORT-6716-ATOMIC; REQ-VERIFY-6716."""

    before = exp.protected_hashes(exp.REPO_ROOT)
    artifact = exp.build_artifact(
        date="20260828",
        root=exp.REPO_ROOT,
        tests_run=passing_test_rows(),
        duration_s=1.25,
        protected_before=before,
        clock=exp.StepClock(),
    )
    assert artifact["status"] == "complete_passed"
    assert artifact["honest_verdict"].startswith("passed:")
    assert artifact["verdict_class"] == "positive"
    assert artifact["seal_attack_audit_passed"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["gate_check_summary"] == []
    assert exp.validate_artifact(artifact) == []
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert all(
        set(exp.PROVENANCE_KEYS) <= set(row) for row in artifact["field_provenance"].values()
    )

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(bad_checksum)
    bad_units = deepcopy(artifact)
    bad_units["per_unit_rows"].pop()
    bad_units["reproducibility_checksum"] = exp.artifact_checksum(bad_units)
    assert "per_unit_rows_mismatch" in exp.validate_artifact(bad_units)
    bad_gate = deepcopy(artifact)
    bad_gate["seal_attack_audit_passed"] = False
    bad_gate["reproducibility_checksum"] = exp.artifact_checksum(bad_gate)
    assert "audit_gate_mismatch" in exp.validate_artifact(bad_gate)


def test_blocked_artifact_atomic_writer_run_and_cli_validation(tmp_path: Path) -> None:
    """REQ-REPORT-6716; SCENARIO-REPORT-6716-BLOCKED."""

    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    preconditions = exp.collect_preconditions(missing_root)
    assert any(not row["passed"] for row in preconditions)
    blocked = exp.build_blocked_artifact("20260828", missing_root, preconditions, duration_s=0.2)
    assert blocked["status"] == "blocked_precondition"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["seal_attack_audit_passed"] is False
    assert blocked["gate_check_summary"]
    assert exp.validate_artifact(blocked) == []

    target = tmp_path / "atomic.json"
    receipt = exp.write_json_atomic(target, blocked)
    assert receipt["atomic_replace"] is True
    assert exp.load_json(target) == blocked
    assert exp.main(["--validate", "--output", str(target)]) == 0
    assert exp.main(["--validate", "--output", str(tmp_path / "absent.json")]) == 1
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    assert exp.main(["--validate", "--output", str(malformed)]) == 1

    output = tmp_path / "experiment_6716.json"
    artifact = exp.run(
        date="20260828",
        root=exp.REPO_ROOT,
        output_path=output,
        runner=fake_runner,
        clock=exp.StepClock(),
    )
    assert output.is_file()
    assert artifact["seal_attack_audit_passed"] is True
    assert exp.validate_artifact(exp.load_json(output)) == []


def test_e2e_actual_bounded_attack_audit() -> None:
    """REQ-VERIFY-6716; SCENARIO-VERIFY-6716-AUTHORITY."""

    upstream = load_upstream()
    manifest = exp.freeze_attack_manifest(exp.public_attack_inputs(upstream))
    campaign = exp.run_attack_campaign(upstream, manifest, clock=exp.StepClock())
    assert len(campaign["leakage_rows"]) == 8
    assert len(campaign["seal_access_rows"]) == 6
    assert len(campaign["metamorphic_rows"]) == 16
    assert len(campaign["mutation_attack_rows"]) == 10
    assert all(row["pass_state"] for key, rows in campaign.items() for row in rows)


def test_fail_closed_defensive_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-SAFE-6716; REQ-REPORT-6716; SCENARIO-REPORT-6716-BLOCKED."""

    upstream = load_upstream()
    seal = upstream["label_seal_rows"][0]
    invalid_seal = deepcopy(seal)
    invalid_seal["seal_hash"] = "sha256:invalid"
    with pytest.raises(exp.SealAccessError, match="invalid seal"):
        exp.AuditLabelSeal(invalid_seal)

    unopened = exp.AuditLabelSeal(seal)
    with pytest.raises(exp.SealAccessError, match="event not opened"):
        unopened.commit("event", {"plan": []})
    unopened.begin_event("event", seal["prompt_hash"], 1)
    with pytest.raises(exp.SealAccessError, match="reordered event"):
        unopened.begin_event("event", seal["prompt_hash"], 1)
    token = unopened.commit("event", {"plan": []})
    token["candidate_hash"] = "sha256:tampered"
    with pytest.raises(exp.SealAccessError, match="invalid commit receipt"):
        unopened.read("event", token)

    manifest = exp.freeze_attack_manifest(exp.public_attack_inputs(upstream))
    bad_manifest = deepcopy(manifest)
    bad_manifest["manifest_hash"] = "sha256:bad"
    with pytest.raises(ValueError, match="attack manifest hash mismatch"):
        exp.run_attack_campaign(upstream, bad_manifest)

    with monkeypatch.context() as context:
        context.setattr(exp.Path, "read_text", lambda *_args, **_kwargs: "No memory row")
        assert exp._memory_bytes() == 0

    malformed_root = tmp_path / "malformed-root"
    malformed_path = malformed_root / exp.UPSTREAM_PATH
    malformed_path.parent.mkdir(parents=True)
    malformed_path.write_text("{", encoding="utf-8")
    malformed_preconditions = exp.collect_preconditions(malformed_root)
    upstream_check = next(
        row for row in malformed_preconditions if row["name"] == "upstream_artifact"
    )
    assert upstream_check["passed"] is False

    with monkeypatch.context() as context:

        def missing_package(_name: str) -> str:
            raise exp.metadata.PackageNotFoundError

        context.setattr(exp.metadata, "version", missing_package)
        schema_rows = exp.collect_preconditions(malformed_root)
        schema = next(row for row in schema_rows if row["name"] == "artifact_schema")
        assert schema["passed"] is False

    receipt = exp.default_command_runner(".venv/bin/python -c 'print(1)'", exp.REPO_ROOT)
    assert receipt["exit_code"] == 0
    assert receipt["stdout"].strip() == "1"

    def invalid_json_runner(command: str, _root: Path) -> dict[str, object]:
        return {
            "command": command,
            "exit_code": 0,
            "stdout": "not-json" if "adversarial_verify.py" in command else "passed",
            "stderr": "",
            "duration_s": 0.1,
        }

    artifact_checks = exp.run_artifact_checks(
        exp.REPO_ROOT, tmp_path / "candidate.json", runner=invalid_json_runner
    )
    adversarial = next(
        row for row in artifact_checks if row["check_id"] == "adversarial_verification"
    )
    assert adversarial["passed"] is True


def test_validator_classification_and_run_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-PIPELINE-6716; SCENARIO-PIPELINE-6716-ROW-REDUCTION."""

    artifact = exp.build_artifact(
        date="20260828",
        root=exp.REPO_ROOT,
        tests_run=passing_test_rows(),
        duration_s=1.0,
        protected_before=exp.protected_hashes(exp.REPO_ROOT),
        clock=exp.StepClock(),
    )
    assert exp.validate_artifact({}) == ["missing_required_fields"]

    bad_common = deepcopy(artifact)
    bad_common["inference_substrate"] = "wrong"
    bad_common["verifier_is_oracle"] = True
    bad_common["verdict_class"] = "unknown"
    bad_common["duration_s"] = -1
    bad_common["field_provenance"] = {}
    bad_common["reproducibility_checksum"] = exp.artifact_checksum(bad_common)
    common_errors = exp.validate_artifact(bad_common)
    assert "inference_substrate_mismatch" in common_errors
    assert "verifier_is_oracle_mismatch" in common_errors
    assert "verdict_class_invalid" in common_errors
    assert "duration_invalid" in common_errors
    assert "field_provenance_invalid" in common_errors

    blocked = exp.build_blocked_artifact(
        "20260828", tmp_path, exp.collect_preconditions(tmp_path), 0.1
    )
    bad_blocked = deepcopy(blocked)
    bad_blocked["gate_check_summary"] = []
    bad_blocked["reproducibility_checksum"] = exp.artifact_checksum(bad_blocked)
    assert "blocked_state_mismatch" in exp.validate_artifact(bad_blocked)

    bad_manifest = deepcopy(artifact)
    bad_manifest["frozen_attack_manifest"]["manifest_hash"] = "sha256:bad"
    bad_manifest["reproducibility_checksum"] = exp.artifact_checksum(bad_manifest)
    assert "manifest_hash_mismatch" in exp.validate_artifact(bad_manifest)

    bad_aggregate = deepcopy(artifact)
    bad_aggregate["aggregate_row_recomputation"]["counts"]["leakage_row_count"] = 999
    bad_aggregate["reproducibility_checksum"] = exp.artifact_checksum(bad_aggregate)
    assert "aggregate_row_recomputation_mismatch" in exp.validate_artifact(bad_aggregate)

    bad_summary = deepcopy(artifact)
    bad_summary["gate_check_summary"] = [{"check": "invented"}]
    bad_summary["reproducibility_checksum"] = exp.artifact_checksum(bad_summary)
    assert "passed_gate_summary_mismatch" in exp.validate_artifact(bad_summary)

    status = exp._classification({"failed_checks": ["leakage_rows"]})
    assert status[0] == "disqualified_attack_failure"

    blocked_output = tmp_path / "blocked-run.json"
    blocked_run = exp.run(
        date="20260828",
        root=tmp_path,
        output_path=blocked_output,
        runner=fake_runner,
    )
    assert blocked_run["status"] == "blocked_precondition"

    with monkeypatch.context() as context:
        context.setattr(exp, "validate_artifact", lambda _payload: ["candidate-error"])
        with pytest.raises(ValueError, match="candidate"):
            exp.run(
                date="20260828",
                root=exp.REPO_ROOT,
                output_path=tmp_path / "candidate-error.json",
                runner=fake_runner,
                clock=exp.StepClock(),
            )

    with monkeypatch.context() as context:
        calls = iter([[], ["final-error"]])
        context.setattr(exp, "validate_artifact", lambda _payload: next(calls))
        with pytest.raises(ValueError, match="final-error"):
            exp.run(
                date="20260828",
                root=exp.REPO_ROOT,
                output_path=tmp_path / "final-error.json",
                runner=fake_runner,
                clock=exp.StepClock(),
            )

    called: dict[str, object] = {}
    with monkeypatch.context() as context:

        def fake_run(**kwargs: object) -> dict[str, object]:
            called.update(kwargs)
            return {}

        context.setattr(exp, "run", fake_run)
        assert exp.main(["--date", "20260828", "--output", str(tmp_path / "main.json")]) == 0
    assert called["date"] == "20260828"
