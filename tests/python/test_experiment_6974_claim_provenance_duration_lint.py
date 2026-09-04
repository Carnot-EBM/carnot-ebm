"""Tests for the Exp6974 derivative duration-lint receipt.

Spec refs: REQ-CONDUCTOR-6974 and SCENARIO-CONDUCTOR-6974-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.experiments import experiment_6974_claim_provenance_duration_lint as exp


ROOT = Path(__file__).resolve().parents[2]


def _passing_receipts() -> list[dict[str, object]]:
    return [
        {
            "name": "claim_provenance_mutation_tests",
            "command": "in_process: evaluate_mutation_corpus",
            "exit_code": 0,
            "outcome": "pass",
            "stdout": "7/7 fixtures matched expected decisions",
            "stderr": "",
            "duration_s": 0.01,
        }
    ]


def test_req_conductor_6974_spec_lists_every_artifact_field() -> None:
    """REQ-CONDUCTOR-6974 declares the complete derivative receipt schema."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-CONDUCTOR-6974") :]
    assert all(f"`{field}`" in section for field in exp.REQUIRED_ARTIFACT_FIELDS)


def test_scenarios_conductor_6974_mutation_corpus_decisions() -> None:
    """SCENARIO-CONDUCTOR-6974-LIVE, NESTED, and AMBIGUOUS all stay effective."""

    rows = exp.evaluate_mutation_corpus()
    by_name = {row["fixture_id"]: row for row in rows}

    assert set(by_name) == {
        "ambiguous_model_context",
        "bibliographic_source_scan",
        "deterministic_source_row_reducer",
        "genuine_live_inference",
        "live_embedding_extraction",
        "nested_input_invocation",
        "pre_gate_block",
    }
    assert by_name["deterministic_source_row_reducer"]["old_decision"] == "critical"
    assert by_name["deterministic_source_row_reducer"]["new_decision"] == "clean"
    assert by_name["genuine_live_inference"]["new_decision"] == "critical"
    assert by_name["nested_input_invocation"]["new_decision"] == "critical"
    assert by_name["ambiguous_model_context"]["new_decision"] == "critical"
    assert by_name["bibliographic_source_scan"]["new_decision"] == "clean"
    assert by_name["live_embedding_extraction"]["new_decision"] == "critical"
    assert by_name["pre_gate_block"]["new_decision"] == "clean"
    assert all(row["passed"] is True for row in rows)


@pytest.mark.parametrize(
    ("substrate", "expected"),
    (
        ("verifier_ensemble_against_cached_candidates", 1.0),
        ("aggregation_from_upstream_artifacts", 0.0001),
        ("deterministic_verifier", 0.0001),
        ("offline_arcade_live_agent_runtime_self_discovery_no_llm", 0.01),
        ("arc_log_analysis_plus_local_timing", 1.0),
        ("artifact_qa_lint_tests", 0.0001),
        ("live_llm_inference_local_gguf_sota", 10.0),
        ("deterministic_smt_hint_validation_no_llm", 0.0001),
        ("local_native_llama_cpp_gguf_backend_bisect", 5.0),
        ("simulation", 0.0001),
        ("plain_cpu_work", None),
    ),
)
def test_req_conductor_6974_legacy_comparison_covers_existing_floor_classes(
    substrate: str,
    expected: float | None,
) -> None:
    """REQ-CONDUCTOR-6974 compares against the full pre-repair floor order."""

    payload = {
        "inference_substrate": substrate,
        "duration_s": 3.0,
        "honest_verdict": "complete_fixture",
    }
    assert exp._legacy_duration_floor(payload) == expected


def test_scenario_conductor_6974_builds_clean_readonly_derivative() -> None:
    """SCENARIO-CONDUCTOR-6974-DERIVATIVE keeps the source bytes unchanged."""

    source = ROOT / exp.EXP6967_PATH
    before = exp.sha256_path(source)
    artifact = exp.build_artifact(
        date="20260904",
        repo_root=ROOT,
        focused_test_receipts=_passing_receipts(),
    )
    after = exp.sha256_path(source)

    assert before == after == exp.EXPECTED_EXP6967_SHA256
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["duration_lint_repair_complete_score"] == 1
    assert artifact["fixture_admissibility_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete_circular_")
    assert artifact["gate_check_summary"] == []
    assert artifact["false_negative_rows"] == []
    assert [row["fixture_id"] for row in artifact["false_positive_rows"]] == [
        "deterministic_source_row_reducer"
    ]
    assert [row["fixture_id"] for row in artifact["ambiguous_provenance_rows"]] == [
        "ambiguous_model_context"
    ]
    recheck = artifact["exp6967_readonly_recheck"]
    assert recheck["source_hash_before"] == recheck["source_hash_after"]
    assert recheck["source_unchanged"] is True
    assert recheck["stored_flagged_adversarial"] is True
    assert recheck["live_duration_critical"] is False
    assert recheck["raw_findings"] == []
    exp.validate_artifact(artifact, repo_root=ROOT)


def test_scenario_conductor_6974_blocked_preconditions_are_schema_complete(
    tmp_path: Path,
) -> None:
    """SCENARIO-CONDUCTOR-6974-ARTIFACT records each absent input and blocks."""

    artifact = exp.build_artifact(date="20260904", repo_root=tmp_path)

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["duration_lint_repair_complete_score"] == 0
    assert artifact["fixture_admissibility_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_claim_provenance_duration_lint")
    assert artifact["gate_check_summary"]
    assert all(
        {"failed_check", "expected_value", "observed_value"} <= set(row)
        for row in artifact["gate_check_summary"]
    )
    exp.validate_artifact(artifact, repo_root=tmp_path)


def test_scenario_conductor_6974_validator_rejects_score_hash_and_verdict_drift() -> None:
    """SCENARIO-CONDUCTOR-6974-ARTIFACT derives claims from rows and frozen bytes."""

    artifact = exp.build_artifact(
        date="20260904",
        repo_root=ROOT,
        focused_test_receipts=_passing_receipts(),
    )
    for field, value, match in (
        ("duration_lint_repair_complete_score", 0, "duration_lint_repair_complete_score"),
        ("fixture_admissibility_ready_score", 0, "fixture_admissibility_ready_score"),
        ("verdict_class", "positive", "verdict_class"),
        ("reproducibility_checksum", "sha256:wrong", "reproducibility_checksum"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        with pytest.raises(ValueError, match=match):
            exp.validate_artifact(changed, repo_root=ROOT)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda item: item.pop("rows"), "missing_required_fields"),
        (lambda item: item["field_principles"].pop("rows"), "field_principles_missing"),
        (lambda item: item.__setitem__("inference_substrate", "wrong"), "inference_substrate"),
        (lambda item: item.__setitem__("verifier_is_oracle", False), "verifier_is_oracle"),
        (lambda item: item.__setitem__("verdict_class", "other"), "verdict_class"),
        (
            lambda item: item.__setitem__("verifier_version_hash", "sha256:wrong"),
            "verifier_version_hash",
        ),
        (
            lambda item: item["preconditions_checked"][0].__setitem__("passed", False),
            "preconditions_checked",
        ),
    ),
)
def test_req_conductor_6974_validator_rejects_structural_drift(mutation, match: str) -> None:
    """REQ-CONDUCTOR-6974 validates declarations before trusting terminal scores."""

    artifact = exp.build_artifact(
        date="20260904",
        repo_root=ROOT,
        focused_test_receipts=_passing_receipts(),
    )
    mutation(artifact)
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    with pytest.raises(ValueError, match=match):
        exp.validate_artifact(artifact, repo_root=ROOT)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda item: item.__setitem__("honest_verdict", "complete_wrong"), "honest_verdict"),
        (lambda item: item.__setitem__("gate_check_summary", []), "gate_check_summary"),
        (
            lambda item: item.__setitem__("duration_lint_repair_complete_score", 1),
            "duration_lint_repair_complete_score",
        ),
        (
            lambda item: item.__setitem__("fixture_admissibility_ready_score", 1),
            "fixture_admissibility_ready_score",
        ),
    ),
)
def test_req_conductor_6974_blocked_validator_rejects_forged_terminal_fields(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    """SCENARIO-CONDUCTOR-6974-ARTIFACT keeps the blocked shape fail closed."""

    artifact = exp.build_artifact(date="20260904", repo_root=tmp_path)
    mutation(artifact)
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    with pytest.raises(ValueError, match=match):
        exp.validate_artifact(artifact, repo_root=tmp_path)


def test_req_conductor_6974_null_receipt_and_source_identity_validation(tmp_path: Path) -> None:
    """REQ-CONDUCTOR-6974 validates terminal nulls and the checked-in source identity."""

    failing_receipt = _passing_receipts()
    failing_receipt[0]["outcome"] = "fail"
    artifact = exp.build_artifact(
        date="20260904",
        repo_root=ROOT,
        focused_test_receipts=failing_receipt,
    )
    assert artifact["verdict_class"] == "null"
    assert artifact["duration_lint_repair_complete_score"] == 0
    exp.validate_artifact(artifact, repo_root=ROOT)

    wrong_verdict = deepcopy(artifact)
    wrong_verdict["honest_verdict"] = "complete_circular_wrong"
    wrong_verdict["reproducibility_checksum"] = exp.reproducibility_checksum(wrong_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(wrong_verdict, repo_root=ROOT)

    fake_source = tmp_path / exp.EXP6967_PATH
    fake_source.parent.mkdir(parents=True)
    fake_source.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="exp6967_source_identity"):
        exp.validate_artifact(artifact, repo_root=tmp_path)


def test_req_conductor_6974_checksum_excludes_only_runtime_receipt_noise() -> None:
    """REQ-CONDUCTOR-6974 keeps scientific decisions reproducible across wall clocks."""

    artifact = exp.build_artifact(
        date="20260904",
        repo_root=ROOT,
        focused_test_receipts=_passing_receipts(),
    )
    changed = deepcopy(artifact)
    changed["duration_s"] = 999.0
    changed["focused_test_receipts"][0]["duration_s"] = 888.0
    changed["focused_test_receipts"][0]["stdout"] = "same assertions, different timing"
    assert exp.reproducibility_checksum(changed) == artifact["reproducibility_checksum"]

    changed["focused_test_receipts"].append("terminal marker")
    assert exp.reproducibility_checksum(changed) != artifact["reproducibility_checksum"]


def test_req_conductor_6974_focused_test_receipt_success_and_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CONDUCTOR-6974 preserves both completed and unavailable test execution."""

    class Completed:
        returncode = 0
        stdout = "tests passed"
        stderr = ""

    monkeypatch.setattr(exp.subprocess, "run", lambda *args, **kwargs: Completed())
    success = exp.run_focused_tests(ROOT)[0]
    assert success["outcome"] == "pass"
    assert success["exit_code"] == 0

    def unavailable(*args, **kwargs):
        raise OSError("pytest unavailable")

    monkeypatch.setattr(exp.subprocess, "run", unavailable)
    blocked = exp.run_focused_tests(ROOT)[0]
    assert blocked["outcome"] == "blocked"
    assert blocked["exit_code"] is None


def test_req_conductor_6974_cli_writes_only_the_requested_output(tmp_path: Path) -> None:
    """REQ-CONDUCTOR-6974 gives tests a temporary output instead of tracked state."""

    output = tmp_path / "nested" / "receipt.json"
    exit_code = exp.main(
        [
            "--date",
            "20260904",
            "--repo-root",
            str(ROOT),
            "--output",
            str(output),
            "--skip-focused-tests",
        ]
    )

    assert exit_code == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    exp.validate_artifact(artifact, repo_root=ROOT)
    assert artifact["fixture_admissibility_ready_score"] == 1


def test_req_conductor_6974_cli_runs_focused_receipt_and_blocks_missing_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CONDUCTOR-6974 runs checks only after preconditions and returns blocked as 2."""

    monkeypatch.setattr(exp, "run_focused_tests", lambda root: _passing_receipts())
    ready_output = tmp_path / "ready.json"
    assert (
        exp.main(
            [
                "--date",
                "20260904",
                "--repo-root",
                str(ROOT),
                "--output",
                str(ready_output),
            ]
        )
        == 0
    )
    assert json.loads(ready_output.read_text(encoding="utf-8"))["focused_test_receipts"]

    blocked_output = tmp_path / "blocked.json"
    missing_root = tmp_path / "missing"
    assert (
        exp.main(
            [
                "--date",
                "20260904",
                "--repo-root",
                str(missing_root),
                "--output",
                str(blocked_output),
            ]
        )
        == 2
    )
    assert json.loads(blocked_output.read_text(encoding="utf-8"))["verdict_class"] == "blocked"
