"""Tests for Exp5851 deterministic replay provenance contract.

Spec refs: REQ-REPORT-5851, REQ-LEARN-5851,
SCENARIO-REPORT-5851-POSITIVE, SCENARIO-REPORT-5851-FALSE-MARKER,
SCENARIO-REPORT-5851-AGGREGATE-BLOCK,
SCENARIO-REPORT-5851-IMMUTABILITY, SCENARIO-REPORT-5851-SCHEMA,
SCENARIO-LEARN-5851-POSITIVE, SCENARIO-LEARN-5851-FALSE-MARKER,
SCENARIO-LEARN-5851-IMMUTABLE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5851_deterministic_replay_provenance_contract as mod


REPO = Path(__file__).resolve().parents[2]
REPORT_SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"
LEARN_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5851_deterministic_replay_provenance_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5851_deterministic_replay_provenance_contract.py "
    "-m pytest "
    "tests/python/test_experiment_5851_deterministic_replay_provenance_contract.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5851_deterministic_replay_provenance_contract.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5851_deterministic_replay_provenance_contract.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\""
)
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}
PASSING_VERIFIER_RECEIPT = {
    "artifact": mod.RESULT_RELATIVE_PATH.as_posix(),
    "loaded": True,
    "exp_id": 5851,
    "title": "",
    "honest_verdict": "ready: fixture",
    "flag_count": 0,
    "max_severity": -1,
    "flags": [],
    "exit_code": 0,
}


def _preconditions(tmp_path: Path) -> dict[str, object]:
    return mod.collect_preconditions(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        memory_probe=lambda: {
            "available_mb": 8192,
            "required_mb": mod.RAM_FLOOR_MB,
            "ok": True,
        },
        disk_probe=lambda root: {
            "available_mb": 8192,
            "required_mb": mod.DISK_FLOOR_MB,
            "ok": True,
        },
    )


def test_req_report_5851_and_req_learn_5851_specs_declare_contract() -> None:
    """REQ-REPORT-5851, REQ-LEARN-5851: OpenSpec names the contract."""

    report = REPORT_SPEC.read_text(encoding="utf-8")
    learn = LEARN_SPEC.read_text(encoding="utf-8")
    report_section = report[report.index("### REQ-REPORT-5851") :]
    learn_section = learn[learn.index("## REQ-LEARN-5851") :]
    normalized = " ".join(report_section.split())

    for marker in (
        "REQ-REPORT-5851",
        "SCENARIO-REPORT-5851-POSITIVE",
        "SCENARIO-REPORT-5851-FALSE-MARKER",
        "SCENARIO-REPORT-5851-AGGREGATE-BLOCK",
        "SCENARIO-REPORT-5851-IMMUTABILITY",
        "SCENARIO-REPORT-5851-SCHEMA",
        "REQ-LEARN-5851",
        "SCENARIO-LEARN-5851-POSITIVE",
        "SCENARIO-LEARN-5851-FALSE-MARKER",
        "SCENARIO-LEARN-5851-IMMUTABLE",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`deterministic_replay_contract_ready_score`",
    ):
        assert marker in (report_section + learn_section)
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in report_section
        assert " ".join(principle.split()) in normalized


def test_scenarios_5851_positive_negative_and_aggregate_only_paths() -> None:
    """SCENARIO-REPORT-5851-POSITIVE/FALSE-MARKER/AGGREGATE-BLOCK."""

    positive = mod.corrected_deterministic_fixture()
    negative = mod.exp5828_shaped_false_compute_marker_fixture(positive)
    aggregate_only = mod.aggregate_only_positive_metrics_fixture(positive)
    live_marker = mod.live_marker_without_methodology_fixture(positive)
    credible_live = deepcopy(live_marker)
    credible_live["model_specs"] = [{"model_id": "fixture-live-model", "backend": "fixture"}]
    credible_live["measured_duration_s"] = mod.LIVE_INFERENCE_MIN_DURATION_S + 1.0
    credible_live["monotonic_timestamps"]["end_ns"] = credible_live["monotonic_timestamps"][
        "start_ns"
    ] + int(credible_live["measured_duration_s"] * 1_000_000_000)

    positive_receipt = mod.validate_replay_receipt(positive)
    negative_receipt = mod.validate_replay_receipt(negative)
    aggregate_receipt = mod.validate_replay_receipt(aggregate_only)
    live_receipt = mod.validate_replay_receipt(live_marker)
    credible_live_receipt = mod.validate_replay_receipt(credible_live)

    assert positive_receipt["passed"] is True
    assert positive_receipt["missing_exact_replay_fields"] == []
    assert positive_receipt["false_compute_markers_detected"] == []
    assert negative_receipt["passed"] is False
    assert "forbidden_compute_markers_on_deterministic_substrate" in negative_receipt["reasons"]
    assert "model_specs" in " ".join(negative_receipt["false_compute_markers_detected"])
    assert negative["scientific_row_semantics"] == positive["scientific_row_semantics"]
    assert negative["aggregate_metrics"]["future_validated_lifecycle_ready_score"] == 1.0
    assert aggregate_receipt["passed"] is False
    assert "missing_exact_replay_fields" in aggregate_receipt["reasons"]
    assert aggregate_only["aggregate_metrics"]["future_validated_lifecycle_ready_score"] == 1.0
    assert live_receipt["passed"] is False
    assert "live_compute_requires_model_specs" in live_receipt["reasons"]
    assert "live_compute_duration_too_short" in live_receipt["reasons"]
    assert credible_live_receipt["passed"] is False
    assert "live_compute_requires_model_specs" not in credible_live_receipt["reasons"]
    assert "live_compute_duration_too_short" not in credible_live_receipt["reasons"]
    assert "inference_substrate" in credible_live_receipt["reasons"]


def test_scenarios_5851_artifact_ready_schema_and_immutability(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5851-IMMUTABILITY/SCHEMA: run emits stable JSON."""

    exp5828_before = mod.sha256_file(REPO / mod.EXP5828_ARTIFACT_RELATIVE_PATH)
    exp5839_before = mod.sha256_file(REPO / mod.EXP5839_ARTIFACT_RELATIVE_PATH)
    destination = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = mod.run(
        result_path=destination,
        preconditions_checked=_preconditions(tmp_path),
        duration_s=1.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        adversarial_verifier_receipt=PASSING_VERIFIER_RECEIPT,
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert destination.read_text(encoding="utf-8").endswith("\n")
    assert mod.validate_artifact(artifact) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "ready"
    assert artifact["honest_verdict"].startswith("ready:")
    assert artifact["deterministic_replay_contract_ready_score"] == 1.0
    assert isinstance(artifact["deterministic_replay_contract_ready_score"], float)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["historical_artifacts_mutated"] is False
    assert artifact["adversarial_verifier_receipt"]["flag_count"] == 0
    assert artifact["preconditions_checked"]["preconditions_ready"] is True
    assert artifact["contract_schema"]["schema"] == mod.SCHEMA
    assert artifact["required_exact_replay_fields"] == list(mod.REQUIRED_EXACT_REPLAY_FIELDS)
    assert artifact["forbidden_compute_markers"] == list(mod.FORBIDDEN_COMPUTE_MARKERS)
    assert all(receipt["passed"] for receipt in artifact["positive_fixture_receipts"])
    assert all(
        receipt["passed"] is False
        for receipt in artifact["false_compute_marker_rejection_receipts"]
    )
    assert artifact["exp5828_regression_receipt"]["passed"] is False
    assert artifact["exp5828_regression_receipt"]["historical_flagged_adversarial"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.sha256_file(REPO / mod.EXP5828_ARTIFACT_RELATIVE_PATH) == exp5828_before
    assert mod.sha256_file(REPO / mod.EXP5839_ARTIFACT_RELATIVE_PATH) == exp5839_before


def test_scenarios_5851_fail_closed_validation_errors(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5851-FALSE-MARKER: bad receipts cannot become ready."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(tmp_path),
        duration_s=1.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        adversarial_verifier_receipt=PASSING_VERIFIER_RECEIPT,
        write=False,
    )
    auto_duration = mod.build_artifact(
        preconditions_checked=_preconditions(tmp_path),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        adversarial_verifier_receipt=PASSING_VERIFIER_RECEIPT,
    )
    blocked = mod.run(
        result_path=tmp_path / "blocked.json",
        preconditions_checked={
            "preconditions_ready": False,
            "blocked_reasons": ["missing_fixture"],
        },
        duration_s=1.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        adversarial_verifier_receipt=PASSING_VERIFIER_RECEIPT,
        write=False,
    )
    verifier_failed = mod.run(
        result_path=tmp_path / "failed.json",
        preconditions_checked=_preconditions(tmp_path),
        duration_s=1.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        adversarial_verifier_receipt={
            **PASSING_VERIFIER_RECEIPT,
            "flag_count": 1,
            "flags": [{"kind": "fixture", "severity": "critical", "detail": "bad"}],
        },
        write=False,
    )

    assert auto_duration["duration_s"] >= 0.0
    assert mod.validate_artifact(auto_duration) is True
    assert blocked["status"] == "blocked"
    assert blocked["deterministic_replay_contract_ready_score"] == 0.0
    assert blocked["honest_verdict"].startswith("blocked:")
    assert verifier_failed["status"] == "failed"
    assert verifier_failed["deterministic_replay_contract_ready_score"] == 0.0
    assert verifier_failed["honest_verdict"].startswith("failed:")

    for field, value, error in (
        ("inference_substrate", "wrong", "inference_substrate"),
        ("historical_artifacts_mutated", True, "historical_artifacts_mutated"),
        ("contract_schema", {}, "contract_schema"),
        ("required_exact_replay_fields", [], "required_exact_replay_fields"),
        ("forbidden_compute_markers", [], "forbidden_compute_markers"),
        ("deterministic_replay_contract_ready_score", 0.0, "ready_score"),
        ("status", "failed", "status"),
        ("honest_verdict", "complete: wrong", "honest_verdict"),
        ("reproducibility_checksum", "sha256:bad", "reproducibility_checksum"),
    ):
        mutated = deepcopy(artifact)
        mutated[field] = value
        with pytest.raises(ValueError, match=error):
            mod.validate_artifact(mutated)

    missing = deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(missing)

    provenance = deepcopy(artifact)
    provenance["field_provenance"]["status"]["principle"] = "wrong"
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(provenance)

    provenance_not_mapping = deepcopy(artifact)
    provenance_not_mapping["field_provenance"] = []
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(provenance_not_mapping)

    wrong_terminal_text = deepcopy(artifact)
    wrong_terminal_text["honest_verdict"] = "ready: wrong"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(wrong_terminal_text)

    for field, value, reason in (
        ("source_row_hashes", {"row_count": 0}, "source_row_hashes"),
        (
            "source_row_hashes",
            {"row_count": 1, "row_hash_root": "sha256:" + "0" * 64, "sample_row_hashes": ["bad"]},
            "source_row_hash_samples",
        ),
        ("validator_versions", {"primary": [], "independent": []}, "validator_versions"),
        (
            "deterministic_seeds",
            {"base_seed": None, "seed_manifest_hash": "bad"},
            "deterministic_seeds",
        ),
        ("state_hashes", {}, "state_hashes"),
        ("checkpoint_hashes", {"checkpoint_count": 0}, "checkpoint_hashes"),
        ("monotonic_timestamps", {"start_ns": 2, "end_ns": 1}, "monotonic_timestamps"),
        ("restart_receipts", {"restart_equivalence": 0.0}, "restart_receipts"),
        (
            "rollback_receipts",
            {"rollback_hash_mismatch_count": 1, "receipt_hash": "bad"},
            "rollback_receipts",
        ),
    ):
        bad_fixture = mod.corrected_deterministic_fixture()
        bad_fixture[field] = value
        receipt = mod.validate_replay_receipt(bad_fixture)
        assert receipt["passed"] is False
        assert reason in receipt["reasons"]

    blocked_reasons = mod.blocked_reasons(
        {
            **artifact,
            "inference_substrate": "wrong",
            "test_exit_codes": {TEST_COMMANDS[0]: 1},
            "exp5828_regression_receipt": {"passed": True},
            "historical_artifacts_mutated": True,
        }
    )
    assert "inference_substrate" in blocked_reasons
    assert "failed_test_exit_codes" in blocked_reasons
    assert "exp5828_regression_not_rejected" in blocked_reasons
    assert "historical_artifacts_mutated" in blocked_reasons

    corrupt_root = tmp_path / "corrupt-root"
    for relative in mod.UPSTREAM_PATHS.values():
        path = corrupt_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (corrupt_root / mod.EXP5828_ARTIFACT_RELATIVE_PATH).write_text("[]\n", encoding="utf-8")
    corrupt = mod.collect_preconditions(
        root=corrupt_root,
        result_path=corrupt_root / mod.RESULT_RELATIVE_PATH,
        memory_probe=lambda: {
            "available_mb": 8192,
            "required_mb": mod.RAM_FLOOR_MB,
            "ok": True,
        },
        disk_probe=lambda root: {
            "available_mb": 8192,
            "required_mb": mod.DISK_FLOOR_MB,
            "ok": True,
        },
    )
    assert corrupt["corrupt_historical_errors"] == ["ValueError"]
    assert "corrupt_historical_artifact" in corrupt["blocked_reasons"]
