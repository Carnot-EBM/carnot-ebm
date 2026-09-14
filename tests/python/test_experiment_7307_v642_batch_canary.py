"""Tests for the V642 live batch canary.

Spec refs: REQ-VERIFY-7307 and SCENARIO-VERIFY-7307-*.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7307_v642_batch_canary as canary


ROOT = Path(__file__).resolve().parents[2]


def _public_panel() -> dict:
    """Load the authenticated public fixture without private evaluator labels."""

    return json.loads((ROOT / canary.UPSTREAM_PUBLIC_PATH).read_text(encoding="utf-8"))


def test_schedule_uses_two_versions_sixteen_calls_and_fixed_budgets() -> None:
    """REQ-VERIFY-7307; SCENARIO-VERIFY-7307-CALLS and -IDS."""

    schedule = canary.build_schedule(_public_panel())

    assert len(schedule) == 16
    assert Counter(row["arm"] for row in schedule) == {
        "serial_versioned_verifier": 10,
        "batched_versioned_verifier": 4,
        "batched_warm_prefix_direct": 2,
    }
    assert sum(row["allocated_output_tokens"] for row in schedule) == 7_680
    by_arm_version: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in schedule:
        by_arm_version[(row["arm"], row["source_version"])].append(row)
        assert row["decoding_parameters"] == {
            "temperature": 0.0,
            "seed": canary.DEVELOPMENT_SEED,
            "max_output_tokens": row["allocated_output_tokens"],
        }
        assert row["prompt_template"] == canary.PROMPT_TEMPLATE
    assert {
        key: sum(row["allocated_output_tokens"] for row in rows)
        for key, rows in by_arm_version.items()
    } == {(arm, version): 1_280 for arm in canary.ARMS for version in (1, 2)}
    for arm in canary.ARMS:
        identifiers = {
            claim_id for row in schedule if row["arm"] == arm for claim_id in row["claim_ids"]
        }
        assert len(identifiers) == 8
    assert canary.schedule_errors(schedule) == []


def test_schedule_validator_rejects_retuned_or_duplicate_calls() -> None:
    """REQ-VERIFY-7307; SCENARIO-VERIFY-7307-CALLS."""

    schedule = canary.build_schedule(_public_panel())
    damaged = deepcopy(schedule)
    damaged[0]["decoding_parameters"]["temperature"] = 0.4
    damaged[0]["prompt_template"] = "changed"
    damaged[0]["allocated_output_tokens"] += 1
    damaged.append(deepcopy(damaged[0]))

    errors = canary.schedule_errors(damaged)
    assert "call_count" in errors
    assert "duplicate_call_id" in errors
    assert "decoding_parameters" in errors
    assert "allocated_output_tokens" in errors
    assert "prompt_template" in errors

    wrong_arm = deepcopy(schedule)
    wrong_arm[0]["arm"] = "changed"
    assert "arm_call_counts" in canary.schedule_errors(wrong_arm)
    no_ids = deepcopy(schedule)
    for row in no_ids:
        if row["arm"] == "serial_versioned_verifier":
            row["claim_ids"] = []
    assert "identifier_accounting:serial_versioned_verifier" in canary.schedule_errors(no_ids)

    with pytest.raises(ValueError, match="development_group_denominator"):
        canary.build_schedule({"development_groups": []})


def test_real_disqualified_upstream_blocks_despite_ready_score_one() -> None:
    """REQ-VERIFY-7307; SCENARIO-VERIFY-7307-DEPENDENCY."""

    checks, inputs = canary.authenticate_inputs(ROOT)
    upstream = inputs["upstream"]
    summary = canary.gate_summary(checks)

    assert upstream["batch_fixture_ready_score"] == 1
    assert upstream["verdict_class"] == "disqualified"
    assert summary == {
        "failed_check": "upstream_terminal_class",
        "upstream": "exp7306-batch-fixture",
        "field": "verdict_class",
        "expected_value": "not blocked or disqualified",
        "observed_value": "disqualified",
    }
    assert all(row["passed"] is True for row in checks if row["check"] != "upstream_terminal_class")


def test_dependency_gate_rejects_quarantine_even_with_numeric_readiness() -> None:
    """REQ-VERIFY-7307 rejects quarantined dependencies before live work."""

    upstream = {
        "experiment_id": "exp7306-batch-fixture",
        "status": "complete",
        "verdict_class": "null",
        "batch_fixture_ready_score": 1,
        "quarantined": True,
    }
    panel = _public_panel()
    checks = canary.dependency_gate_rows(upstream, panel, exclusions={})

    assert canary.gate_summary(checks) == {
        "failed_check": "upstream_quarantine",
        "upstream": "exp7306-batch-fixture",
        "field": "quarantined_or_flagged_adversarial",
        "expected_value": False,
        "observed_value": True,
    }


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        (
            canary.ZERO_INVOCATION_COUNTS,
            (False, "blocked_before_qualifying_computation", "blocked_no_run", "not_invoked"),
        ),
        (
            {**canary.ZERO_INVOCATION_COUNTS, "model_loads_attempted": 1},
            (True, "model_load_no_generation", "model_load_no_generation", "not_invoked"),
        ),
        (
            {**canary.ZERO_INVOCATION_COUNTS, "generation_calls_attempted": 1},
            (True, "live_llm_inference", "model_bounded_generation", "live_gpu"),
        ),
    ],
)
def test_invocation_classification_preserves_attempt_boundaries(
    counts: dict, expected: tuple[bool, str, str, str]
) -> None:
    """REQ-VERIFY-7307 records the actual model substrate and attempt class."""

    classified = canary.classify_inference(counts)
    assert (
        classified["model_invoked"],
        classified["inference_substrate"],
        classified["inference_substrate_class"],
        classified["inference_mode"],
    ) == expected


def test_blocked_artifact_is_complete_auditable_and_cold_valid() -> None:
    """REQ-VERIFY-7307; SCENARIO-VERIFY-7307-DEPENDENCY and -E2E."""

    checks, _ = canary.authenticate_inputs(ROOT)
    artifact = canary.base_artifact(canary.RUN_DATE, started_at_utc="2026-09-14T12:00:00Z")
    artifact["source_artifact_hashes"] = canary.source_artifact_hashes(ROOT, checks)
    receipts = [
        {
            "name": "focused_pytest",
            "command": "pytest focused",
            "scope": "new behavior",
            "exit_code": 0,
            "passed": True,
            "duration_s": 0.1,
            "log_path": "results/raw/example.log",
            "log_sha256": "sha256:test",
        }
    ]
    blocked = canary.finalize_blocked_artifact(
        artifact,
        checks,
        validation_receipts=receipts,
        duration_s=0.5,
        completed_at_utc="2026-09-14T12:00:01Z",
    )

    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"] == "blocked_exp7306_upstream_terminal_class"
    assert blocked["batch_canary_ready_score"] == 0
    assert blocked["MODEL_SPECS"] == [
        {"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}
    ]
    assert blocked["invocation_counts"] == canary.ZERO_INVOCATION_COUNTS
    assert blocked["per_call_rows"] == []
    assert blocked["parser_replay_receipt"]["status"] == "not_attempted_external_block"
    assert blocked["validation_receipts"] == receipts
    assert blocked["repository_health"]["preserved_upstream_full_suite_log_sha256"] == (
        "sha256:823b3369a44a241aa4f0c41e6d25cefa1397a3a5169f5eb1d5653097feba7a60"
    )
    assert canary.validate_artifact(blocked) == []


def test_validator_rejects_changed_summary_counts_and_checksum() -> None:
    """REQ-VERIFY-7307 preserves exact blocked fields and invocation counts."""

    checks, _ = canary.authenticate_inputs(ROOT)
    artifact = canary.finalize_blocked_artifact(
        canary.base_artifact(canary.RUN_DATE, started_at_utc="2026-09-14T12:00:00Z"),
        checks,
        validation_receipts=[],
        duration_s=0.5,
        completed_at_utc="2026-09-14T12:00:01Z",
    )
    artifact["gate_check_summary"]["observed_value"] = "null"
    artifact["invocation_counts"]["generation_calls_attempted"] = -1

    errors = canary.validate_artifact(artifact)
    assert "gate_check_summary" in errors
    assert "invocation_counts" in errors
    assert "reproducibility_checksum" in errors

    classification = deepcopy(artifact)
    classification["invocation_counts"] = deepcopy(canary.ZERO_INVOCATION_COUNTS)
    classification["model_invoked"] = True
    classification["reproducibility_checksum"] = canary.artifact_checksum(classification)
    assert "inference_classification" in canary.validate_artifact(classification)

    malformed = deepcopy(artifact)
    passed = canary.gate_row(
        "all_good", "test", "field", True, True, True, "A synthetic passing check."
    )
    malformed["preconditions_checked"] = [passed]
    malformed["gate_check_summary"] = canary.gate_summary([passed])
    malformed["honest_verdict"] = "wrong"
    malformed["rows"] = [{}]
    malformed["parser_replay_receipt"] = {}
    malformed["duration_s"] = True
    malformed["reproducibility_checksum"] = canary.artifact_checksum(malformed)
    malformed_errors = canary.validate_artifact(malformed)
    assert {
        "blocked_without_failure",
        "honest_verdict",
        "blocked_rows",
        "parser_replay_receipt",
        "duration_s",
    }.issubset(malformed_errors)
    assert canary.validate_artifact([]) == ["artifact_mapping"]


def test_authentication_fails_closed_for_missing_and_malformed_inputs(tmp_path: Path) -> None:
    """REQ-VERIFY-7307 preserves missing and corrupt input failures."""

    missing, values = canary.authenticate_inputs(tmp_path)
    assert values == {}
    assert canary.gate_summary(missing)["observed_value"] == "missing"

    required = (
        canary.UPSTREAM_PATH,
        canary.UPSTREAM_PUBLIC_PATH,
        canary.EXCLUSION_PATH,
        canary.RESEARCH_PROGRAM_PATH,
        canary.SPEC_PATH,
        canary.SOTA_MODEL_PATH,
        canary.MODULE_PATH,
        canary.WRAPPER_PATH,
        canary.TEST_PATH,
    )
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (tmp_path / canary.UPSTREAM_PATH).write_text("{", encoding="utf-8")

    malformed, values = canary.authenticate_inputs(tmp_path)
    assert values == {}
    assert malformed[-1]["check"] == "input_serialization"
    assert malformed[-1]["passed"] is False


def test_validation_plan_contains_exact_full_suite_and_terminal_checks() -> None:
    """REQ-VERIFY-7307; SCENARIO-VERIFY-7307-E2E."""

    commands = dict(canary.validation_commands(ROOT, ROOT / canary.CANDIDATE_PATH))

    assert tuple(commands) == canary.REQUIRED_VALIDATION_NAMES
    assert commands["full_python_suite"] == [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"]
    assert commands["verdict_row_consistency_strict"][-2:] == [
        "--strict",
        str(ROOT / canary.CANDIDATE_PATH),
    ]
    assert commands["e2e_terminal_candidate"][-2:] == [
        "--check-artifact",
        str(ROOT / canary.CANDIDATE_PATH),
    ]
    pending = canary._pending_receipts(ROOT, ROOT / canary.CANDIDATE_PATH)
    assert len(pending) == len(canary.REQUIRED_VALIDATION_NAMES)
    assert all(row["exit_code"] is None and row["passed"] is False for row in pending)


def test_date_and_progress_contract(capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-VERIFY-7307 flushes visible phase boundaries and fixes the run date."""

    canary._progress(2, "phase_end", checked=3)
    assert '"event":"phase_end"' in capsys.readouterr().out
    assert "+" in canary._utc_now()
    assert canary._date_argument(canary.RUN_DATE) == canary.RUN_DATE
    with pytest.raises(Exception, match=f"run date must be {canary.RUN_DATE}"):
        canary._date_argument("20260913")
