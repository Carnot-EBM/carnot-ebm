"""Tests for the independent structural-addition audit.

Spec refs: REQ-CL-7325 and SCENARIO-CL-7325-*.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from carnot import experiment_7325_v643_addition_audit as audit


ROOT = Path(__file__).resolve().parents[2]


def _passing_validation() -> dict[str, object]:
    receipts = []
    for name in audit.REQUIRED_CHECK_NAMES:
        receipts.append(
            {
                "name": name,
                "command": f"test {name}",
                "scope": "explicit affected files",
                "exit_code": 0,
                "duration_s": 0.01,
                "log_sha256": "sha256:" + "1" * 64,
                "passed": True,
                "timed_out": False,
            }
        )
    return {
        "validation_receipts": receipts,
        "required_checks_passed": True,
        "missing_required_commands": [],
        "failed_required_commands": [],
        "duplicate_required_commands": [],
        "repository_health": audit.repository_health(),
    }


def test_scenario_cl_7325_preconditions_authenticate_exact_external_values(
    tmp_path: Path,
) -> None:
    """REQ-CL-7325: eligible evidence passes and a missing producer blocks exactly."""

    paths = audit.ExperimentPaths.defaults(ROOT)
    checks, hashes, upstream = audit.collect_preconditions(ROOT, paths)
    assert audit.gate_check_summary(checks)["passed"] is True
    assert upstream["addition_capture_complete_score"] == 1
    assert hashes["producer_artifact"]["sha256"].startswith("sha256:")

    missing = replace(paths, upstream=tmp_path / "missing.json")
    failed_checks, failed_hashes, _ = audit.collect_preconditions(ROOT, missing)
    summary = audit.gate_check_summary(failed_checks)
    assert summary["first_failure"] == {
        "check": "upstream_available",
        "upstream": str(missing.upstream),
        "field": "path",
        "expected_value": True,
        "observed_value": False,
        "passed": False,
        "principle": "A missing producer cannot authorize an audit.",
    }
    blocked = audit.build_blocked_artifact(failed_checks, failed_hashes)
    assert blocked["status"] == "blocked"
    assert blocked["rows"] == []
    assert blocked["addition_audit_complete_score"] == 0
    assert blocked["addition_promotion_score"] == 0
    assert audit.validate_artifact(blocked) == []


def test_scenario_cl_7325_reduction_rebuilds_all_rows_calls_and_strata() -> None:
    """SCENARIO-CL-7325-REDUCTION: raw evidence owns all independent totals."""

    paths = audit.ExperimentPaths.defaults(ROOT)
    evidence = audit.recompute_evidence(paths, progress=False)
    assert evidence["row_count"] == 2304
    assert evidence["stream_count"] == 24
    assert evidence["request_count"] == 576
    assert evidence["primary_request_count"] == 480
    assert evidence["query_accounting"]["actual_invocation_count"] == 25632
    assert evidence["query_accounting"]["primary_invocation_count"] == 22256
    assert evidence["query_accounting"]["primary_row_attempt_count"] == 22256
    assert evidence["query_accounting"]["duplicate_invocation_ids"] == []
    assert evidence["query_accounting"]["failed_localization_count"] > 0
    assert evidence["query_accounting"]["final_check_count"] == 2704
    assert evidence["optimum_censored_count"] == 0
    assert evidence["maximum_sealed_state_bytes"] <= audit.STATE_CAP_BYTES
    assert {row["stratum"] for row in evidence["independent_comparison_rows"]} == {
        "overall",
        "stationary",
        "announced_version_change",
        "return_to_known_version",
    }
    assert all(
        row["independent_unit"] == "stream" and row["bootstrap_draws"] == 10_000
        for row in evidence["independent_comparison_rows"]
    )
    assert evidence["returning_version_used_for_policy_tuning"] is False
    assert evidence["executor_process_receipts"]["all_queries_paired"] is True


def test_scenario_cl_7325_causality_rebuilds_updates_and_cache_control() -> None:
    """SCENARIO-CL-7325-CAUSALITY: feedback effects differ from exact caching."""

    evidence = audit.recompute_evidence(audit.ExperimentPaths.defaults(ROOT), progress=False)
    assert len(evidence["independent_update_rows"]) == 120
    assert all(not row["structural_change_only"] for row in evidence["independent_update_rows"])
    assert sum(row["affected_future_request_count"] for row in evidence["independent_update_rows"])
    causal = {row["intervention"]: row for row in evidence["causal_intervention_rows"]}
    assert set(causal) == {"feedback_withheld", "label_shuffled", "learned_atom_erasure"}
    assert all(row["same_prefix_for_stream"] for row in causal.values())
    assert all(row["later_distinct_request_count"] == 480 for row in causal.values())
    assert causal["feedback_withheld"]["changed_decision_count"] > 0
    assert causal["label_shuffled"]["changed_query_sequence_count"] > 0
    assert evidence["exact_plan_cache_explanation"]["cache_hit_count"] == 0
    assert evidence["exact_plan_cache_explanation"]["can_explain_reduction"] is False


def test_scenario_cl_7325_hostile_controls_reject_each_authority() -> None:
    """SCENARIO-CL-7325-HOSTILE: nine independent attacks fail closed."""

    evidence = audit.recompute_evidence(audit.ExperimentPaths.defaults(ROOT), progress=False)
    controls = audit.run_hostile_controls(evidence)
    assert {row["attack_id"] for row in controls} == {
        "stale_version",
        "fabricated_witness",
        "over_specific_pair_prohibition",
        "hidden_query_counter",
        "delayed_duplicate_response",
        "memory_overflow",
        "corrupted_snapshot",
        "unannounced_change",
    }
    assert all(row["passed"] and row["authority_valid"] is False for row in controls)
    unannounced = next(row for row in controls if row["attack_id"] == "unannounced_change")
    assert unannounced["stable_oracle_assumption"] is False
    assert unannounced["recovery_claim"] is False


def test_scenario_cl_7325_hostile_reduction_rejects_tampered_atom_and_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7325-HOSTILE: malformed evidence cannot acquire authority."""

    paths = audit.ExperimentPaths.defaults(ROOT)
    evidence = audit.recompute_evidence(paths, progress=True)
    source = deepcopy(next(row for row in evidence["rows"] if row.get("new_atoms")))
    atom = source["new_atoms"][0]
    atom["atom_id"] = "sha256:" + "2" * 64
    atom["witness"]["query_id"] = "sha256:" + "3" * 64
    atom["version"] = "stale-version"
    atom["kind"] = "pairwise_separation"
    atom["payload"]["minimum"] = 3
    _updates, errors = audit._independent_updates(  # noqa: SLF001 - tests attack the reducer seam.
        [source], audit.read_jsonl(paths.queries)
    )
    assert {error.split(":", 1)[0] for error in errors} == {
        "atom_hash",
        "fabricated_witness",
        "stale_version",
        "over_specific_pair",
    }

    non_object = tmp_path / "non-object.jsonl"
    non_object.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl_object_required"):
        audit.read_jsonl(non_object)
    assert audit._manifest_excludes(tmp_path / "absent.yaml") is True  # noqa: SLF001
    with pytest.raises(ValueError, match="paired_streams_unavailable"):
        audit._interval([], "empty")  # noqa: SLF001

    target = tmp_path / "atomic.json"
    real_replace = audit.os.replace

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(audit.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected rename failure"):
        audit._atomic_json(target, {"status": "candidate"})  # noqa: SLF001
    assert not list(tmp_path.glob(".atomic.json.*.tmp"))
    monkeypatch.setattr(audit.os, "replace", real_replace)


def test_scenario_cl_7325_lifecycle_accounts_restart_rollback_and_version(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7325-LIFECYCLE: current memory preserves exact state boundaries."""

    lifecycle = audit.run_memory_lifecycle(tmp_path / "memory")
    assert lifecycle["update_admitted"] is True
    assert lifecycle["later_request_used_update"] is True
    assert lifecycle["cold_restart_bytes_equal"] is True
    assert lifecycle["rollback_bytes_equal"] is True
    assert lifecycle["stale_version_invalidated"] is True
    assert lifecycle["initialized_memory_bytes"] > 0
    assert lifecycle["serialized_memory_bytes"] > lifecycle["initialized_memory_bytes"]
    assert lifecycle["transient_rollback_bytes"] == lifecycle["initialized_memory_bytes"]
    assert lifecycle["corrupted_snapshot_rejected"] is True


def test_scenario_cl_7325_lifecycle_detects_a_fail_open_snapshot_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7325-LIFECYCLE: accepting corrupt bytes makes lifecycle fail."""

    real_memory = audit.transaction.TransactionalConstraintMemory

    class LenientMemory(real_memory):
        def __init__(self, state_dir: Path | str) -> None:
            path = Path(state_dir)
            state = path / "state.json"
            if path.name.endswith("-corrupt") and state.is_file() and state.read_bytes() == b"{}":
                state.write_bytes(
                    audit.canonical_bytes(
                        {
                            "schema": audit.transaction.STATE_SCHEMA,
                            "version": 0,
                            "records": [],
                        }
                    )
                )
            super().__init__(state_dir)

    monkeypatch.setattr(audit.transaction, "TransactionalConstraintMemory", LenientMemory)
    lifecycle = audit.run_memory_lifecycle(tmp_path / "lenient")
    assert lifecycle["corrupted_snapshot_rejected"] is False
    assert lifecycle["passed"] is False


@pytest.mark.parametrize(
    ("complete", "promotion", "oracle", "expected"),
    [
        (False, False, True, (0, 0, "partial")),
        (True, False, True, (1, 0, "null")),
        (True, True, True, (1, 1, "circular_positive")),
        (True, True, False, (1, 1, "positive")),
    ],
)
def test_scenario_cl_7325_terminal_keeps_completion_independent(
    complete: bool,
    promotion: bool,
    oracle: bool,
    expected: tuple[int, int, str],
) -> None:
    """SCENARIO-CL-7325-TERMINAL: audit completion is not a favorable verdict."""

    assert audit.derive_terminal_scores(complete, promotion, oracle) == expected


def test_scenario_cl_7325_terminal_builds_valid_complete_artifact(tmp_path: Path) -> None:
    """REQ-CL-7325: the measured candidate seals only after scoped checks pass."""

    paths = audit.ExperimentPaths.defaults(ROOT)
    candidate = audit.build_and_seal(ROOT, paths, progress=False)
    assert candidate["verdict_class"] == "partial"
    complete = audit.attach_validation(candidate, _passing_validation())
    assert complete["status"] == "complete"
    assert complete["addition_audit_complete_score"] == 1
    assert complete["addition_promotion_score"] == 1
    assert complete["verdict_class"] == "circular_positive"
    assert complete["honest_verdict"].startswith("complete:")
    assert audit.validate_artifact(complete, check_files=True) == []

    terminal = tmp_path / "terminal.json"
    receipt = audit.write_artifact(terminal, complete)
    assert receipt["sha256"] == audit.sha256_file(terminal)
    reloaded = json.loads(terminal.read_text(encoding="utf-8"))
    assert reloaded["reproducibility_checksum"] == complete["reproducibility_checksum"]

    tampered = deepcopy(complete)
    tampered["rows"][0]["oracle_attempts"] += 1
    assert "reproducibility_checksum" in audit.validate_artifact(tampered)
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        audit.write_artifact(tmp_path / "bad.json", tampered)

    null_candidate = deepcopy(candidate)
    null_candidate["acceptance_gate_results"]["utility_vs_reset"].update(
        {"pass": False, "passed": False}
    )
    null_artifact = audit.attach_validation(null_candidate, _passing_validation())
    assert null_artifact["verdict_class"] == "null"
    assert null_artifact["honest_verdict"].startswith("complete_null:")

    failed_validation = _passing_validation()
    failed_validation.update(
        {
            "required_checks_passed": False,
            "failed_required_commands": ["focused_pytest"],
        }
    )
    disqualified = audit.attach_validation(candidate, failed_validation)
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["addition_audit_complete_score"] == 0

    missing_raw = deepcopy(complete)
    first_receipt = next(iter(missing_raw["source_artifact_hashes"]["raw_evidence"].values()))
    first_receipt["path"] = str(tmp_path / "missing.jsonl")
    missing_raw["reproducibility_checksum"] = audit.reproducibility_checksum(missing_raw)
    assert "raw_evidence_receipts" in audit.validate_artifact(missing_raw, check_files=True)


def test_scenario_cl_7325_terminal_prints_boundaries_and_blocks_missing_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-CL-7325-TERMINAL: every phase boundary is flushed and truthful."""

    paths = audit.ExperimentPaths.defaults(ROOT)
    candidate = audit.build_and_seal(ROOT, paths, progress=True)
    assert candidate["verdict_class"] == "partial"
    output = capsys.readouterr().out
    assert "phase=preconditions event=start" in output
    assert "phase=raw_reduction event=end" in output
    assert "phase=memory_lifecycle event=end" in output

    missing = replace(paths, upstream=tmp_path / "missing.json")
    blocked = audit.build_and_seal(ROOT, missing, progress=True)
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
