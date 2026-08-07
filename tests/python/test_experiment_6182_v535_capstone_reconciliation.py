"""Tests for Exp6182 V535 capstone reconciliation.

Spec refs: REQ-CAPSTONE-6182, SCENARIO-CAPSTONE-6182,
SCENARIO-CAPSTONE-6182-EXACT-PATH,
SCENARIO-CAPSTONE-6182-TERMINAL-CLASS-PRESERVATION,
SCENARIO-CAPSTONE-6182-ADVERSARIAL-VERIFY-AND-CHECKSUM,
SCENARIO-CAPSTONE-6182-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_6182_v535_capstone_reconciliation as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"


def _write_text(root: Path, rel_path: Path | str, text: str = "fixture\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    _write_text(root, rel_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _artifact(task_id: str) -> JsonDict:
    payloads: dict[str, JsonDict] = {
        "exp6170-v535-task-artifact-isolation-canary": {
            "status": "complete_partial",
            "honest_verdict": (
                "complete_partial: canary ready_score=0, repository-wide claim false"
            ),
            "inference_substrate": "deterministic_task_scoped_repository_test_isolation",
            "scope_boundary_and_repository_wide_closure_claimed": {
                "repository_wide_closure_claimed": False,
                "qualified_task_ids": [task_id for task_id, _title, _path in mod.ACTIVATED_TASKS],
            },
            "v535_task_artifact_isolation_ready_score": 0,
            "isolation_violation_count": 1,
            "test_exit_codes": {"focused": 0, "required_full_python_suite_once": 2},
        },
        "exp6171-v535-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete_null: accepted_count=0; references unchanged",
            "inference_substrate": "dated_primary_secondary_source_ingestion",
            "zero_delta_accepted": True,
            "candidate_and_deduplicated_record_counts": {"accepted_count": 0},
            "reference_hash_before_after_and_append_count": {"append_count": 0},
        },
        "exp6172-current-rule-quarantine-determination": {
            "status": "complete_current_rule_clean_historical_quarantine_preserved",
            "honest_verdict": (
                "complete: current_rule_clean=true; historical quarantine still flagged"
            ),
            "inference_substrate": "deterministic_current_rule_companion_determination",
            "flagged_adversarial": True,
            "current_rule_clean": True,
            "historical_quarantine_preserved": True,
            "headline_promotion_authorized": False,
            "operator_reopen_required": True,
        },
        "exp6173-cctu-item-bank-preregistration": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: item bank frozen",
            "inference_substrate": "deterministic_executable_tool_trace_fixture_and_validators",
            "cctu_item_bank_ready_score": 1.0,
            "held_seal_and_access_log_path_hash": {"held_labels_sealed": True},
            "verifier_is_oracle": True,
        },
        "exp6174-cctu-authentic-k8-pool": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: K8 pool integrity score one",
            "inference_substrate": "llama_cpp_local_gemma4_31b_gguf_native_chat_tool_trace_generation",
            "cctu_candidate_pool_integrity_score": 1.0,
            "exact_label_sidecar_paths_hashes_and_counts": {"label_row_count": 960},
        },
        "exp6175-cctu-headroom-audit": {
            "status": "retired",
            "honest_verdict": "retired: failed preregistered headroom conjuncts",
            "inference_substrate": "deterministic_exact_tool_trace_headroom_audit",
            "phase_d_headroom_ready_score": 0.0,
            "future_rows_allowed_by_this_artifact": False,
            "oracle_minus_consensus_delta_and_clustered_interval": {
                "delta": 0.0,
                "lower_ci_excludes_zero": False,
            },
            "consensus_wrong_oracle_right_group_count": {
                "count": 0,
                "minimum_required": 30,
                "passed": False,
            },
            "verifier_is_oracle": True,
        },
        "exp6177-clue-latent-selector-freeze": {
            "schema": "blocked_gate_check_v1",
            "experiment": 6177,
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "duration_s": 0.0,
            "gates_evaluated": [
                {
                    "upstream": "exp6176-hidden-state-surface-qualification",
                    "artifact_field": "hidden_state_surface_ready_score",
                    "passed": False,
                }
            ],
        },
        "exp6179-retention-safe-continuous-strategy-learning-ab": {
            "status": "complete",
            "honest_verdict": (
                "complete: replay beat controls while preserving retention; "
                "live model generation did not execute"
            ),
            "inference_substrate": "deterministic_exact_verifier_and_versioned_external_state_no_llm",
            "retention_safe_continuous_strategy_learning_ready_score": 1.0,
            "model_weight_immutability_receipt": {"all_unchanged": True},
            "rollback_and_quarantine_receipts": {
                "quarantine_count": 3,
                "poison_propagation_count": 0,
            },
            "exact_post_outcome_write_receipts": {
                "all_commits_after_exact_outcome": True,
            },
            "MODEL_SPECS": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-26B-A4B-it-GGUF",
            ],
        },
        "exp6180-exp6166-reproducibility-adjudication": {
            "status": "complete_positive",
            "honest_verdict": (
                "complete_positive: Exp6166 software stochastic result reproduced; "
                "historical Exp6166 remains blocked"
            ),
            "inference_substrate": "jax_cpu_software_exp6166_artifact_replay",
            "companion_determination": {
                "historical_exp6166_status_preserved": "blocked",
                "adjudicated_result": "software_only_positive_reproducible",
            },
            "no_hardware_promotion_receipt": {
                "hardware_execution_claimed": False,
                "latency_power_energy_and_speedup_claimed": False,
            },
        },
        "exp6181-arc-logo-shortcut-audit": {
            "status": "complete_no_shortcut_detected",
            "honest_verdict": "complete_no_shortcut_detected: no solve, no registry delta",
            "inference_substrate": "live_e3_adapter_disabled_runtime_transitions",
            "shortcut_audit_summary": {"shortcut_detected": False},
            "solve_claimed": False,
            "level_credit_delta": 0,
            "registry_delta": 0,
            "registry_levels_unchanged": True,
            "adapter_disabled_live_path_receipt": {"adapter_disabled": True},
        },
    }
    return payloads[task_id]


def _roadmap_payload() -> JsonDict:
    return {
        "milestone": mod.MILESTONE,
        "tasks": [
            {
                "id": task_id,
                "milestone": mod.MILESTONE,
                "title": title,
                "deliverable": rel_path.as_posix(),
                **(
                    {"gated_on": deepcopy(mod.GATED_ON[task_id])} if task_id in mod.GATED_ON else {}
                ),
            }
            for task_id, title, rel_path in mod.ACTIVATED_TASKS
        ]
        + [
            {
                "id": mod.EXPERIMENT_ID,
                "milestone": mod.MILESTONE,
                "title": "Branch-independent .535 capstone",
                "deliverable": mod.RESULT_RELATIVE_PATH.as_posix(),
            }
        ],
    }


def _conductor_log() -> str:
    return "\n".join(
        [
            "| 2026-08-07 01:49 UTC | Exact terminal-boundary handoff from .534 into .53 | FAIL | Hard wall-clock cap |",
            "| 2026-08-07 02:40 UTC | Task-scoped artifact-isolation compatibility canar | OK | 128 passed |",
            "| 2026-08-07 03:32 UTC | Reliable dated evidence refresh after the V535 pla | OK | 86 passed |",
            "| 2026-08-07 03:49 UTC | Immutable current-rule companion determination for | FLAGGED | DURATION_TOO_SHORT |",
            "| 2026-08-07 04:36 UTC | Frozen executable CCTU-style item bank and Phase-D | OK | 87 passed |",
            "| 2026-08-07 05:37 UTC | Gated on Exp6173 bank readiness: authentic Gemma-4 | OK | 87 passed |",
            "| 2026-08-07 07:42 UTC | Gated on Exp6174 pool integrity: CCTU competence,  | FAIL | hard wall L-CLOSED-RETIR |",
            "| 2026-08-07 07:48 UTC | Matching-base per-layer hidden-state surface quali | GATE_BLOCK | upstream retired |",
            "| 2026-08-07 07:48 UTC | Calibration-only CLUE and latent selector freeze | GATE_BLOCK | upstream failed |",
            "| 2026-08-07 08:31 UTC | One-shot held internal-state selection | GATE_BLOCK | upstream retired |",
            "| 2026-08-07 08:12 UTC | Mandatory retention-safe continuous strategy-learn | OK | 87 passed |",
            "| 2026-08-07 08:29 UTC | Exp6166 evidence-preserving reproducibility adjudi | OK | 85 passed |",
            "| 2026-08-07 08:47 UTC | Single ARC slot leave-one-game-out shortcut audit | OK | 87 passed |",
        ]
    )


def _make_root(root: Path) -> None:
    missing = {
        "exp6169-v535-transition",
        "exp6176-hidden-state-surface-qualification",
        "exp6178-held-internal-state-selection",
    }
    for task_id, _title, rel_path in mod.ACTIVATED_TASKS:
        if task_id not in missing:
            _write_json(root, rel_path, _artifact(task_id))
    _write_json(
        root,
        "results/experiment_6169_same_number_alias.json",
        {"status": "complete", "honest_verdict": "complete: alias must be ignored"},
    )
    _write_json(
        root,
        "results/experiment_6174_cctu_authentic_k8_pool.held_access_log.json",
        {"sidecar": True, "must_not_define_terminal_class": True},
    )
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_roadmap_payload()))
    _write_text(root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones: []\n")
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, _conductor_log())
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired_experiments: []\n")
    for rel_path in mod.PROTECTED_FILE_PATHS + mod.PRECONDITION_CONTEXT_PATHS:
        if not (root / rel_path).exists():
            _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\n")


def _receipt(task_id: str, rel_path: Path, flag_count: int = 0) -> JsonDict:
    report = {
        "artifact": rel_path.as_posix(),
        "loaded": True,
        "flag_count": flag_count,
        "max_severity": 5 if flag_count else -1,
        "flags": [{"rule": "fixture_flag"}] if flag_count else [],
    }
    return {
        "task_id": task_id,
        "artifact_path": rel_path.as_posix(),
        "command": f".venv/bin/python scripts/adversarial_verify.py --json {rel_path.as_posix()}",
        "exit_code": 1 if flag_count else 0,
        "stdout_json": {"reports": [report], "flagged_count": flag_count},
    }


def _receipts() -> dict[str, JsonDict]:
    receipts: dict[str, JsonDict] = {}
    for task_id, _title, rel_path in mod.ACTIVATED_TASKS:
        if task_id in {
            "exp6169-v535-transition",
            "exp6176-hidden-state-surface-qualification",
            "exp6178-held-internal-state-selection",
        }:
            continue
        receipts[task_id] = _receipt(
            task_id,
            rel_path,
            flag_count=1 if task_id == "exp6172-current-rule-quarantine-determination" else 0,
        )
    return receipts


def _build(root: Path) -> JsonDict:
    _make_root(root)
    return mod.build_report(
        root,
        adversarial_receipts=_receipts(),
        tests_run={
            ".venv/bin/pytest tests/python/test_experiment_6182_v535_capstone_reconciliation.py -q --no-cov -n 0": 0,
            ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6182_v535_capstone_reconciliation.py --fail-under=100": 0,
        },
        duration_s=1.25,
    )


def test_req_capstone_6182_spec_declares_exact_path_contract() -> None:
    """REQ-CAPSTONE-6182: OpenSpec names exact-path reconciliation."""

    section = SPEC_PATH.read_text(encoding="utf-8")
    section = section[section.index("REQ-CAPSTONE-6182") :]

    for marker in (
        "REQ-CAPSTONE-6182",
        "SCENARIO-CAPSTONE-6182",
        "SCENARIO-CAPSTONE-6182-EXACT-PATH",
        "SCENARIO-CAPSTONE-6182-TERMINAL-CLASS-PRESERVATION",
        "SCENARIO-CAPSTONE-6182-ADVERSARIAL-VERIFY-AND-CHECKSUM",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "Exp6169 through Exp6181",
        "without modifying `scripts/research_conductor.py`",
        "aggregation_from_upstream_artifacts",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in " ".join(section.split())


def test_scenario_capstone_6182_exact_path_terminal_matrix(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6182-EXACT-PATH: sidecars never replace declared paths."""

    report = _build(tmp_path)

    assert report["status"] == "complete_with_missing_skipped_blocked_retired_flagged_and_null"
    assert report["honest_verdict"].startswith("complete:")
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE

    matrix = report["declared_task_matrix"]
    assert list(matrix) == [task_id for task_id, _title, _path in mod.ACTIVATED_TASKS]
    assert matrix["exp6169-v535-transition"]["present"] is False
    assert matrix["exp6169-v535-transition"]["terminal_class"] == "missing"
    assert matrix["exp6169-v535-transition"]["same_number_alias_used"] is False
    assert matrix["exp6169-v535-transition"]["same_number_alias_candidates_ignored"] == [
        "results/experiment_6169_same_number_alias.json"
    ]
    assert matrix["exp6176-hidden-state-surface-qualification"]["terminal_class"] == "skipped"
    assert matrix["exp6178-held-internal-state-selection"]["terminal_class"] == "skipped"
    assert matrix["exp6177-clue-latent-selector-freeze"]["terminal_class"] == "blocked"
    assert matrix["exp6174-cctu-authentic-k8-pool"]["sidecar_candidates_ignored"] == [
        "results/experiment_6174_cctu_authentic_k8_pool.held_access_log.json"
    ]

    classes = report["terminal_classification"]["terminal_class_by_task_id"]
    assert classes == {
        "exp6169-v535-transition": "missing",
        "exp6170-v535-task-artifact-isolation-canary": "delivered",
        "exp6171-v535-source-delta-ingestion": "null",
        "exp6172-current-rule-quarantine-determination": "flagged",
        "exp6173-cctu-item-bank-preregistration": "positive",
        "exp6174-cctu-authentic-k8-pool": "positive",
        "exp6175-cctu-headroom-audit": "retired",
        "exp6176-hidden-state-surface-qualification": "skipped",
        "exp6177-clue-latent-selector-freeze": "blocked",
        "exp6178-held-internal-state-selection": "skipped",
        "exp6179-retention-safe-continuous-strategy-learning-ab": "positive",
        "exp6180-exp6166-reproducibility-adjudication": "positive",
        "exp6181-arc-logo-shortcut-audit": "positive",
    }
    assert report["terminal_class_counts"] == {
        "blocked": 1,
        "delivered": 1,
        "flagged": 1,
        "missing": 1,
        "null": 1,
        "positive": 5,
        "retired": 1,
        "skipped": 2,
    }


def test_scenario_capstone_6182_quarantine_and_no_claim_strengthening(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-6182-TERMINAL-CLASS-PRESERVATION: claims stay bounded."""

    report = _build(tmp_path)

    assert report["adversarial_verification_receipts"]["verified_present_artifact_count"] == 10
    assert report["adversarial_verification_receipts"]["flagged_task_ids"] == [
        "exp6172-current-rule-quarantine-determination"
    ]
    assert report["quarantine_field_receipts"]["flagged_or_quarantined_task_ids"] == [
        "exp6172-current-rule-quarantine-determination"
    ]
    quarantine = report["raw_field_reconciliation"]["quarantine_determination"]
    assert quarantine["current_rule_clean"] is True
    assert quarantine["historical_quarantine_preserved"] is True
    assert quarantine["headline_promotion_authorized"] is False
    assert quarantine["terminal_class"] == "flagged"

    phase_d = report["branch_decisions"]["phase_d"]
    assert phase_d["headroom_ready_score"] == 0.0
    assert phase_d["final_state"] == "retired_headroom_failed_downstream_skipped"
    assert phase_d["future_rows_allowed"] is False
    assert phase_d["downstream_selector_promoted"] is False

    csl = report["branch_decisions"]["continuous_strategy_learning"]
    assert csl["ready_score"] == 1.0
    assert csl["model_weights_immutable"] is True
    assert csl["live_model_generation_claimed"] is False

    stochastic = report["branch_decisions"]["stochastic"]
    assert stochastic["software_reproducible"] is True
    assert stochastic["hardware_promoted"] is False

    arc = report["branch_decisions"]["arc"]
    assert arc["solve_claimed"] is False
    assert arc["level_credit_delta"] == 0
    assert arc["registry_delta"] == 0
    assert arc["solve_credit_promoted"] is False

    no_strengthen = report["no_claim_strengthening_receipts"]
    assert no_strengthen["sidecars_and_aliases_imported"] is False
    assert no_strengthen["flagged_companion_unflagged"] is False
    assert no_strengthen["retired_phase_d_promoted"] is False
    assert no_strengthen["arc_no_solve_promoted"] is False


def test_scenario_capstone_6182_append_once_checksum_and_run(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6182-ADVERSARIAL-VERIFY-AND-CHECKSUM: output is stable."""

    _make_root(tmp_path)
    report = mod.run(
        tmp_path,
        adversarial_receipts=_receipts(),
        tests_run={
            ".venv/bin/pytest tests/python/test_experiment_6182_v535_capstone_reconciliation.py -q --no-cov -n 0": 0,
        },
        duration_s=1.5,
        append_completion_history=True,
    )

    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert mod.validate_report(report) == []
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    assert report["completion_history_multiplicity"] == {
        "milestone": mod.MILESTONE,
        "count_before": 0,
        "count_after": 1,
        "duplicate_history_amplification_count": 0,
    }
    assert report["completion_history_update"]["append_count"] == 1
    complete = yaml.safe_load((tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_text())
    assert [row["id"] for row in complete["milestones"]] == [mod.MILESTONE]
    assert len(complete["milestones"][0]["tasks"]) == len(mod.ACTIVATED_TASKS) + 1
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert (
        report["protected_files_unchanged"]["files"]["scripts/research_conductor.py"]["unchanged"]
        is True
    )


def test_req_capstone_6182_defensive_helpers_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-6182: malformed evidence cannot become success."""

    payload, meta = mod._read_json_mapping(tmp_path / "missing.json")
    assert payload == {}
    assert meta["error"] == "missing"
    _write_text(tmp_path, "bad.json", "{")
    assert mod._read_json_mapping(tmp_path / "bad.json")[1]["error"].startswith("json_error:")
    _write_text(tmp_path, "array.json", "[]")
    assert mod._read_json_mapping(tmp_path / "array.json")[1]["error"] == "json_not_mapping"
    assert mod._read_yaml_mapping(tmp_path / "missing.yaml") == {}
    _write_text(tmp_path, "not_mapping.yaml", "[]\n")
    assert mod._read_yaml_mapping(tmp_path / "not_mapping.yaml") == {}
    assert mod._latest_conductor_receipt("", "not present") == {
        "present": False,
        "status": None,
        "line": None,
        "detail": None,
    }
    assert (
        mod._ignored_same_number_aliases(
            tmp_path / "no-results", "exp6169-any", Path("results/x.json")
        )
        == []
    )
    assert mod._sidecar_candidates(tmp_path / "no-results", Path("results/x.json")) == []
    assert mod._terminal_marker("unknown") is None
    assert mod._terminal_marker("complete_no_shortcut_detected: ok") == "positive"
    assert mod._terminal_marker("complete_null: ok") == "null"
    assert mod._terminal_marker("retired: ok") == "retired"
    assert mod._terminal_marker("blocked_gate_check_failed") == "blocked"
    assert mod._terminal_class({}, False, {"status": "GATE_BLOCK"}) == "skipped"
    assert mod._terminal_class({}, False, {"status": "FAIL"}) == "missing"
    assert mod._terminal_class({"retirement_triggered": True}, True, {}) == "retired"
    assert mod._terminal_class({"status": "complete_partial"}, True, {}) == "delivered"
    assert mod._terminal_class({"status": "complete", "zero_delta_accepted": True}, True, {}) == (
        "null"
    )
    assert mod._terminal_class({"status": "complete", "x_ready_score": 1.0}, True, {}) == (
        "positive"
    )
    assert mod._normalize_tests(None)[1]
    assert mod._normalize_tests([{"command": "cmd", "exit_code": 7}]) == (
        ["cmd"],
        {"cmd": 7},
    )
    assert mod._receipt_report({}) == {"flag_count": 0, "flags": [], "max_severity": -1}
    assert mod._receipt_report({"stdout_json": {"flagged_count": 3}})["flag_count"] == 3
    assert mod._normalize_adversarial_receipts([{"task_id": "expX"}])["expX"]["task_id"] == ("expX")
    _write_text(tmp_path, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones: {}\n")
    assert mod._completion_history_count(tmp_path) == 0
    assert "complete (capstone)" in mod._format_completion_block(
        {"declared_task_matrix": {"bad": []}}
    )
    empty_history = tmp_path / "empty-history"
    empty_history.mkdir()
    assert (
        mod._append_completion_history_if_needed(empty_history, {"declared_task_matrix": {}}) == 1
    )
    assert (
        yaml.safe_load((empty_history / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_text())[
            "milestones"
        ][0]["id"]
        == mod.MILESTONE
    )
    existing_history = tmp_path / "existing-history"
    existing_history.mkdir()
    _write_text(
        existing_history, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones:\n- id: 2026.08.535\n"
    )
    assert mod._append_completion_history_if_needed(existing_history, {}) == 0
    other_history = tmp_path / "other-history"
    other_history.mkdir()
    _write_text(other_history, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones:\n- id: other\n")
    assert (
        mod._append_completion_history_if_needed(other_history, {"declared_task_matrix": {}}) == 1
    )

    (tmp_path / ".git").mkdir(exist_ok=True)

    class _Proc:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: _Proc(0, stdout=" M file.py\n"),
    )
    assert mod._git_status_short(tmp_path) == [" M file.py"]
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: _Proc(2, stderr="boom"),
    )
    assert mod._git_status_short(tmp_path) == ["git_status_error:boom"]

    report = _build(tmp_path)
    monkeypatch.setattr(mod, "_run_live_adversarial_receipts", lambda _root, _paths: _receipts())
    live_report = mod.build_report(
        tmp_path,
        adversarial_receipts=None,
        tests_run={".venv/bin/pytest tests/python -q": 0},
        duration_s=1.0,
    )
    assert live_report["adversarial_verification_receipts"]["verified_present_artifact_count"] == 10
    broken = deepcopy(report)
    broken["field_provenance"]["status"]["principle"] = "wrong"
    broken["reproducibility_checksum"] = mod.payload_checksum(broken)
    assert "field_provenance:status" in mod.validate_report(broken)

    missing = deepcopy(report)
    missing.pop("status")
    assert "missing:status" in mod.validate_report(missing)

    checksum = deepcopy(report)
    checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum" in mod.validate_report(checksum)

    substrate = deepcopy(report)
    substrate["inference_substrate"] = "model"
    substrate["reproducibility_checksum"] = mod.payload_checksum(substrate)
    assert "inference_substrate" in mod.validate_report(substrate)

    no_provenance = deepcopy(report)
    no_provenance["field_provenance"] = []
    no_provenance["reproducibility_checksum"] = mod.payload_checksum(no_provenance)
    assert "field_provenance:not_mapping" in mod.validate_report(no_provenance)

    missing_provenance_row = deepcopy(report)
    missing_provenance_row["field_provenance"].pop("status")
    missing_provenance_row["reproducibility_checksum"] = mod.payload_checksum(
        missing_provenance_row
    )
    assert "field_provenance:status" in mod.validate_report(missing_provenance_row)

    bad_classes = deepcopy(report)
    bad_classes["terminal_classification"]["terminal_class_by_task_id"] = {}
    bad_classes["reproducibility_checksum"] = mod.payload_checksum(bad_classes)
    assert "terminal_classification" in mod.validate_report(bad_classes)

    bad_claim = deepcopy(report)
    bad_claim["no_claim_strengthening_receipts"]["arc_no_solve_promoted"] = True
    bad_claim["reproducibility_checksum"] = mod.payload_checksum(bad_claim)
    assert "no_claim_strengthening:arc_no_solve_promoted" in mod.validate_report(bad_claim)

    monkeypatch.setattr(mod, "validate_report", lambda _report: ["forced"])
    with pytest.raises(ValueError, match="invalid Exp6182 report"):
        mod.run(
            tmp_path,
            adversarial_receipts=_receipts(),
            tests_run={".venv/bin/pytest tests/python -q": 0},
            duration_s=1.0,
            append_completion_history=False,
        )
