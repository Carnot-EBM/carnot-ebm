"""Tests for the Exp6155 V533 capstone reconciliation.

Spec refs: REQ-REPORT-6155,
SCENARIO-REPORT-6155-EXACT-MATRIX,
SCENARIO-REPORT-6155-TERMINAL-AND-QUARANTINE,
SCENARIO-REPORT-6155-BRANCH-BOUNDARIES,
SCENARIO-REPORT-6155-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml
import pytest

from carnot import experiment_6155_v533_capstone_reconciliation as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_text(root: Path, rel_path: Path | str, text: str = "fixture\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    _write_text(root, rel_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _blocked_gate() -> JsonDict:
    return {
        "schema": "blocked_gate_check_v1",
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "blocked_at_layer": "conductor_pre_gate",
        "gate_check_summary": (
            "2 of 2 gate(s) failed; first failure: "
            "exp6148-shifted-family-admission-held.shifted_family_admission_ready_score"
        ),
        "gates_evaluated": [
            {
                "upstream": "exp6148-shifted-family-admission-held",
                "artifact_field": "shifted_family_admission_ready_score",
                "expected": 1.0,
                "actual": 0.0,
                "passed": False,
            },
            {
                "upstream": "exp6149-certified-strategy-schema-fixture",
                "artifact_field": "certified_strategy_fixture_ready_score",
                "expected": 1.0,
                "actual": 0.0,
                "passed": False,
            },
        ],
    }


def _artifact(task_id: str) -> JsonDict:
    payloads: dict[str, JsonDict] = {
        "exp6142-transition-v533": {
            "status": "complete_with_terminal_receipts",
            "honest_verdict": "complete: archived .532 into .533",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "research_complete_append_count": 0,
            "duplicate_history_amplification_count": 0,
        },
        "exp6143-test-artifact-isolation": {
            "status": "complete_partial",
            "honest_verdict": "complete_partial: tracked results remained immutable",
            "inference_substrate": "deterministic_infrastructure_test_isolation",
            "tracked_result_hash_before_after_matrix": {"all_unchanged": True},
            "quarantine_field_before_after_matrix": {"all_preserved": True},
            "remaining_unredirected_writer_census": {"residual_call_site_rows": 6198},
            "attempted_tracked_write_detection": {"negative_control_observed": True},
        },
        "exp6144-v533-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete_null: no accepted post-V533 source deltas",
            "inference_substrate": "literature_ingestion",
            "accepted_rejected_duplicate_retired_and_abstained_findings": {
                "accepted": [],
                "accepted_count": 0,
            },
            "references_append_receipt": {"appended": False},
        },
        "exp6145-constraint-shift-stream": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: exact stream",
            "inference_substrate": "deterministic_exact_fixture_construction",
            "constraint_shift_stream_ready_score": 1.0,
            "stream_row_split_and_outcome_sidecar_paths_and_hashes": {
                "row_file": {"row_count": 240, "sha256": "sha256:rows"},
                "split_file": {"base_template_count": 48, "sha256": "sha256:splits"},
                "outcome_sidecar": {"row_count": 240, "sha256": "sha256:outcomes"},
            },
            "exact_validator_agreement": {"disagreement_count": 0},
        },
        "exp6146-sota-constraint-event-corpus": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: live corpus",
            "inference_substrate": "live_local_sota_gguf_cuda",
            "sota_constraint_event_corpus_ready_score": 1.0,
            "model_specs": [
                {
                    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "sha256": "sha256:qwen",
                    "gpu": 0,
                    "actual_use_count": 240,
                    "loader": "llama_cpp.Llama",
                },
                {
                    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                    "sha256": "sha256:gemma",
                    "gpu": 1,
                    "actual_use_count": 240,
                    "loader": "llama_cpp.Llama",
                },
            ],
            "per_model_event_row_conservation": {"all_models_conserved": True},
        },
        "exp6147-task-aware-energy-calibration": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: calibration positive",
            "inference_substrate": "deterministic_task_aware_calibration",
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "fixture"}
            ],
            "task_aware_energy_calibration_ready_score": 1.0,
            "selected_score_threshold_abstention_and_memory_budget": {
                "selected_score": "task_aware_energy",
                "threshold": 1.25,
                "selection_uses_held_outcomes": False,
            },
        },
        "exp6148-shifted-family-admission-held": {
            "status": "complete_null",
            "honest_verdict": "complete_null: shifted lower CI not positive",
            "inference_substrate": "deterministic_held_admission",
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "fixture"}
            ],
            "shifted_family_admission_ready_score": 0.0,
            "paired_task_aware_minus_global_intervals": {
                "pooled_summary_after_per_model": {
                    "sealed_shifted_family": {
                        "auroc_delta": {"ci95": [0.0, 0.0], "positive_lower_95": False}
                    }
                }
            },
        },
        "exp6149-certified-strategy-schema-fixture": {
            "status": "complete_partial",
            "honest_verdict": "complete_partial: test_commands_clean",
            "inference_substrate": "deterministic_transactional_csl_fixture",
            "continuous_self_learning_task": True,
            "certified_strategy_fixture_ready_score": 0.0,
            "structured_gate_receipt": {
                "all_gates_passed": False,
                "gates": {"test_commands_clean": False, "model_weights_immutable": True},
            },
            "model_weight_immutability_receipt": {
                "all_unchanged": True,
                "model_weight_update_count": 0,
            },
        },
        "exp6150-frozen-qwen-continuous-self-learning-ab": _blocked_gate(),
        "exp6152-typed-stochastic-constraint-ir": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: exact typed IR",
            "inference_substrate": "jax_cpu_exact_stochastic_program",
            "typed_stochastic_ir_ready_score": 1.0,
            "exact_enumeration_case_counts": {
                "state_space_size": 1536,
                "support_count": 6,
                "kernel_count": 9,
            },
            "torx_compatibility_scope": {"torx_imported": True, "api_smoke_passed": True},
        },
        "exp6153-thermalized-program-error-audit": {
            "status": "blocked",
            "honest_verdict": "blocked: program-level error composition held but nonzero_test_commands",
            "inference_substrate": "jax_cpu_software_thermalization",
            "thermalized_program_ready_score": 0.0,
            "hardware_execution_claimed": False,
            "latency_power_energy_and_speedup_claimed": False,
            "bound_slack_and_violation_counts": {"violation_count": 0},
            "test_exit_codes": {"JAX_PLATFORMS=cpu .venv/bin/pytest tests/python -q": 2},
        },
        "exp6154-arc-task-aware-energy-generalization": {
            "status": "complete_positive",
            "honest_verdict": "complete_positive: task aware improved no solve",
            "inference_substrate": "live_e3_adapter_disabled_runtime_transitions",
            "arc_task_aware_generalization_ready_score": 1.0,
            "solve_claimed": False,
            "offline_reproduced": False,
            "level_credit_delta": 0,
            "llm_invocation_count": 0,
            "used_game_source": False,
            "offline_ground_truth_bfs": False,
            "false_confident_admission_and_abstention_matrices": {
                "totals": {
                    "global": {"false_confident_admissions": 10},
                    "task_aware": {"false_confident_admissions": 0},
                }
            },
        },
    }
    return payloads[task_id]


def _roadmap_payload() -> JsonDict:
    tasks: list[JsonDict] = []
    for task_id, title, rel_path in mod.ACTIVATED_TASKS:
        row: JsonDict = {
            "id": task_id,
            "milestone": mod.MILESTONE,
            "title": title,
            "deliverable": rel_path.as_posix(),
        }
        if task_id in mod.GATED_ON:
            row["gated_on"] = deepcopy(mod.GATED_ON[task_id])
        tasks.append(row)
    tasks.append(
        {
            "id": mod.EXPERIMENT_ID,
            "milestone": mod.MILESTONE,
            "title": "Branch-independent .533 capstone",
            "deliverable": mod.RESULT_RELATIVE_PATH.as_posix(),
        }
    )
    return {"milestone": mod.MILESTONE, "tasks": tasks}


def _conductor_log() -> str:
    return "\n".join(
        [
            "| 2026-08-05 17:53 UTC | Exact terminal-boundary handoff from .532 into .53 | OK | 120 passed |",
            "| 2026-08-05 18:21 UTC | Tracked-result test artifact isolation and quarant | OK | 97 passed |",
            "| 2026-08-05 18:38 UTC | Reliable dated evidence refresh after the V533 pla | OK | 86 passed |",
            "| 2026-08-05 20:09 UTC | Exact chronological constraint-event stream with h | OK | 87 passed |",
            "| 2026-08-05 20:48 UTC | Gated on Exp6145 readiness: flagship-GGUF chronolo | OK | 100 passed |",
            "| 2026-08-05 21:10 UTC | Gated on Exp6146 corpus readiness: TOOD-style task | FLAGGED | adversarial_verify CRITICAL: DURATION_TOO_SHORT |",
            "| 2026-08-05 21:33 UTC | Gated on Exp6147 calibration readiness: one-shot s | FLAGGED | adversarial_verify CRITICAL: DURATION_TOO_SHORT |",
            "| 2026-08-05 22:05 UTC | Gated on Exp6145 stream readiness: certified strat | OK | 92 passed |",
            "| 2026-08-05 22:11 UTC | Gated on Exp6148 and Exp6149 readiness: frozen-Qwe | GATE_BLOCK | 2 of 2 gate(s) failed |",
            "| 2026-08-05 22:13 UTC | Gated on Exp6150 positive utility: default-off tra | GATE_BLOCK | Pre-emptive skip: upstream retired |",
            "| 2026-08-05 22:35 UTC | Gated on Exp6145 stream readiness: typed Torx-comp | OK | 89 passed |",
            "| 2026-08-06 01:18 UTC | Gated on Exp6152 IR readiness: software thermaliza | FAIL | artifact_not_updated_past_bootstrap |",
            "| 2026-08-06 02:51 UTC | ARC live-path adapter-disabled task-aware energy g | OK | 108 passed |",
        ]
    )


def _make_root(root: Path) -> None:
    for task_id, _title, rel_path in mod.ACTIVATED_TASKS:
        if task_id == "exp6151-strategy-memory-shadow-adapter":
            continue
        _write_json(root, rel_path, _artifact(task_id))
    _write_json(
        root,
        "results/experiment_6151_same_number_alias.json",
        {"status": "complete_positive", "honest_verdict": "complete_positive: alias"},
    )
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_roadmap_payload()))
    _write_text(root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones: []\n")
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, _conductor_log())
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired_experiments: []\n")
    for rel_path in mod.PROTECTED_FILE_PATHS + mod.PRECONDITION_CONTEXT_PATHS:
        if not (root / rel_path).exists():
            _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\n")


def _receipt(task_id: str, rel_path: Path) -> JsonDict:
    flagged = task_id in {
        "exp6143-test-artifact-isolation",
        "exp6146-sota-constraint-event-corpus",
        "exp6147-task-aware-energy-calibration",
        "exp6148-shifted-family-admission-held",
    }
    report = {
        "artifact": rel_path.as_posix(),
        "loaded": True,
        "flag_count": 1 if flagged else 0,
        "max_severity": 2 if flagged else -1,
        "flags": (
            [{"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "fixture"}]
            if flagged
            else []
        ),
    }
    stdout_json = {"reports": [report], "flagged_count": int(flagged)}
    return {
        "task_id": task_id,
        "artifact_path": rel_path.as_posix(),
        "command": f".venv/bin/python scripts/adversarial_verify.py --json {rel_path.as_posix()}",
        "exit_code": int(flagged),
        "stdout_json": stdout_json,
    }


def _receipts() -> dict[str, JsonDict]:
    return {
        task_id: _receipt(task_id, rel_path)
        for task_id, _title, rel_path in mod.ACTIVATED_TASKS
        if task_id != "exp6151-strategy-memory-shadow-adapter"
    }


def _build(root: Path) -> JsonDict:
    _make_root(root)
    return mod.build_report(
        root,
        adversarial_receipts=_receipts(),
        tests_run=[
            {
                "command": ".venv/bin/pytest tests/python/test_experiment_6155_v533_capstone_reconciliation.py -q --no-cov -n 0",
                "exit_code": 0,
            },
            {"command": ".venv/bin/pytest tests/python -q", "exit_code": 0},
        ],
        duration_s=1.25,
    )


def test_req_report_6155_spec_declares_capstone_contract() -> None:
    """REQ-REPORT-6155: OpenSpec names exact-path and boundary requirements."""

    section = SPEC_PATH.read_text(encoding="utf-8")
    section = section[section.index("### REQ-REPORT-6155") :]

    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "(milestone, task_id, declared_deliverable)" in section
    assert "SCENARIO-REPORT-6155-EXACT-MATRIX" in section
    assert "SCENARIO-REPORT-6155-TERMINAL-AND-QUARANTINE" in section
    assert "SCENARIO-REPORT-6155-BRANCH-BOUNDARIES" in section
    assert "Exp6142 through Exp6154" in section
    assert mod.INFERENCE_SUBSTRATE in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_report_6155_exact_matrix_and_terminal_classes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6155-EXACT-MATRIX: aliases do not fill missing artifacts."""

    report = _build(tmp_path)

    assert report["status"] == "complete_with_blocks_and_quarantine"
    assert report["honest_verdict"].startswith("complete:")
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE

    matrix = report["activated_task_and_declared_deliverable_matrix"]
    assert list(matrix) == [task_id for task_id, _title, _path in mod.ACTIVATED_TASKS]
    assert matrix["exp6151-strategy-memory-shadow-adapter"]["present"] is False
    assert matrix["exp6151-strategy-memory-shadow-adapter"]["declared_deliverable"] == (
        "results/experiment_6151_strategy_memory_shadow_adapter.json"
    )
    assert matrix["exp6151-strategy-memory-shadow-adapter"]["terminal_evidence_source"] == (
        "conductor_structured_gate_receipt"
    )
    assert matrix["exp6151-strategy-memory-shadow-adapter"]["same_number_alias_used"] is False

    classes = report["exact_terminal_classification"]["terminal_class_by_task_id"]
    assert classes == {
        "exp6142-transition-v533": "complete",
        "exp6143-test-artifact-isolation": "partial",
        "exp6144-v533-source-delta-ingestion": "null",
        "exp6145-constraint-shift-stream": "positive",
        "exp6146-sota-constraint-event-corpus": "positive",
        "exp6147-task-aware-energy-calibration": "positive",
        "exp6148-shifted-family-admission-held": "null",
        "exp6149-certified-strategy-schema-fixture": "partial",
        "exp6150-frozen-qwen-continuous-self-learning-ab": "blocked",
        "exp6151-strategy-memory-shadow-adapter": "structured-skip",
        "exp6152-typed-stochastic-constraint-ir": "positive",
        "exp6153-thermalized-program-error-audit": "blocked",
        "exp6154-arc-task-aware-energy-generalization": "positive",
    }
    assert report["exact_terminal_classification"]["all_tasks_terminal"] is True

    counts = report["present_missing_skipped_blocked_null_retired_and_positive_counts"]
    assert counts["present"] == 12
    assert counts["missing"] == 0
    assert counts["structured_skipped"] == 1
    assert counts["blocked"] == 2
    assert counts["null"] == 2
    assert counts["retired"] == 0
    assert counts["partial"] == 2
    assert counts["complete"] == 1
    assert counts["positive"] == 5
    assert counts["adversarial_quarantined"] == 4
    assert counts["positive_aggregation_eligible"] == 3


def test_scenario_report_6155_quarantine_excludes_flagged_positive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6155-TERMINAL-AND-QUARANTINE: flags stay visible."""

    report = _build(tmp_path)
    quarantine = report["adversarial_verifier_and_quarantine_receipts"]

    assert quarantine["verified_present_artifact_count"] == 12
    assert quarantine["flagged_task_ids"] == [
        "exp6143-test-artifact-isolation",
        "exp6146-sota-constraint-event-corpus",
        "exp6147-task-aware-energy-calibration",
        "exp6148-shifted-family-admission-held",
    ]
    assert quarantine["positive_aggregation_eligible_task_ids"] == [
        "exp6145-constraint-shift-stream",
        "exp6152-typed-stochastic-constraint-ir",
        "exp6154-arc-task-aware-energy-generalization",
    ]
    assert (
        quarantine["receipts_by_task_id"]["exp6146-sota-constraint-event-corpus"][
            "excluded_from_positive_aggregation"
        ]
        is True
    )
    assert (
        quarantine["receipts_by_task_id"]["exp6147-task-aware-energy-calibration"][
            "excluded_from_positive_aggregation"
        ]
        is True
    )
    assert (
        quarantine["receipts_by_task_id"]["exp6148-shifted-family-admission-held"][
            "artifact_quarantine_fields_present"
        ]
        is True
    )


def test_scenario_report_6155_branch_boundaries_and_substrates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6155-BRANCH-BOUNDARIES: summaries keep claims separate."""

    report = _build(tmp_path)

    assert report["test_artifact_isolation_summary"]["tracked_results_unchanged"] is True
    assert report["exact_stream_and_sota_corpus_summary"] == {
        "stream_ready_score": 1.0,
        "stream_row_count": 240,
        "stream_exact_disagreement_count": 0,
        "sota_corpus_ready_score": 1.0,
        "sota_model_count": 2,
        "sota_rows_conserved": True,
    }
    held = report["task_aware_calibration_and_held_summary"]
    assert held["calibration_ready_score_raw"] == 1.0
    assert held["calibration_positive_aggregation_eligible"] is False
    assert held["held_ready_score"] == 0.0
    assert held["held_terminal_class"] == "null"

    csl = report["continuous_self_learning_and_shadow_summary"]
    assert csl["fixture_ready_score"] == 0.0
    assert csl["prospective_csl_terminal_class"] == "blocked"
    assert csl["shadow_adapter_terminal_class"] == "structured-skip"
    assert csl["model_weights_immutable"] is True

    typed = report["typed_ir_and_thermalization_summary"]
    assert typed["typed_ir_ready_score"] == 1.0
    assert typed["thermalized_program_ready_score"] == 0.0
    assert typed["thermalization_terminal_class"] == "blocked"
    assert typed["hardware_execution_claimed"] is False

    arc = report["arc_generalization_no_solve_summary"]
    assert arc["arc_ready_score"] == 1.0
    assert arc["solve_claimed"] is False
    assert arc["offline_reproduced"] is False
    assert arc["level_credit_delta"] == 0

    substrate = report["oracle_distinctness_and_inference_substrate_matrix"]
    assert (
        substrate["rows_by_task_id"]["exp6153-thermalized-program-error-audit"]["hardware_claimed"]
        is False
    )
    assert (
        substrate["rows_by_task_id"]["exp6153-thermalized-program-error-audit"][
            "software_never_hardware"
        ]
        is True
    )
    assert (
        substrate["rows_by_task_id"]["exp6154-arc-task-aware-energy-generalization"][
            "solve_claimed"
        ]
        is False
    )


def test_scenario_report_6155_schema_checksum_and_no_history_append(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6155-SCHEMA: required fields and checksum validate."""

    report = _build(tmp_path)

    assert mod.validate_report(report) == []
    assert report["research_complete_append_count"] == 0
    assert report["duplicate_history_amplification_count"] == 0
    assert (
        report["spec_bmad_ops_reference_and_completion_reconciliation"][
            "research_complete_mutation"
        ]
        == "deferred_to_conductor"
    )
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert report["preexisting_worktree_changes_preserved"]["preserved"] is True
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)

    written = mod.run(
        tmp_path,
        adversarial_receipts=_receipts(),
        tests_run=report["test_exit_codes"],
        duration_s=1.5,
    )
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert yaml.safe_load((tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_text()) == {
        "milestones": []
    }
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)


def test_req_report_6155_defensive_helpers_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6155: defensive helpers do not promote malformed evidence."""

    missing_payload, missing_meta = mod._read_json_mapping(tmp_path / "missing.json")
    assert missing_payload == {}
    assert missing_meta["error"] == "missing"
    _write_text(tmp_path, "bad.json", "{")
    assert mod._read_json_mapping(tmp_path / "bad.json")[1]["error"].startswith("json_error:")
    _write_text(tmp_path, "array.json", "[]")
    assert mod._read_json_mapping(tmp_path / "array.json")[1]["error"] == "json_not_mapping"
    assert mod._read_yaml_mapping(tmp_path / "missing.yaml") == {}

    assert mod._latest_conductor_receipt("", "missing title") == {
        "present": False,
        "status": None,
        "line": None,
        "detail": None,
    }
    assert mod._terminal_class({}, False, {}) == "missing"
    assert mod._terminal_class({"status": "retired"}, True, {}) == "retired"
    assert mod._status_text({"status": "A", "honest_verdict": "B"}) == "a b"
    assert mod._terminal_marker("unknown") is None
    assert mod._terminal_class({"retirement_triggered": True}, True, {}) == "retired"
    assert mod._terminal_class({"honest_verdict": "blocked: gate"}, True, {}) == "blocked"
    assert (
        mod._terminal_class(
            {
                "status": "complete_with_terminal_receipts",
                "honest_verdict": "complete: .533 activation mode=already_active",
            },
            True,
            {},
        )
        == "complete"
    )
    assert (
        mod._terminal_class(
            {
                "status": "complete_partial",
                "honest_verdict": "complete_partial: not claimed because unrelated checks blocked",
            },
            True,
            {},
        )
        == "partial"
    )
    assert mod._terminal_class({}, True, {}) == "missing"
    assert mod._normalize_tests(None)[0] == list(mod.DEFAULT_TEST_COMMANDS)
    assert mod._receipt_report({}) == {"flag_count": 0, "flags": [], "max_severity": -1}
    assert mod._receipt_report({"stdout_json": {"flagged_count": 2}})["flag_count"] == 2
    assert mod._normalize_adversarial_receipts([{"task_id": "expX"}])["expX"]["task_id"] == "expX"
    assert mod._history_duplicate_count(tmp_path) == 0
    _write_text(
        tmp_path,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump({"milestones": [{"id": "a"}, {"id": "a"}]}),
    )
    assert mod._history_duplicate_count(tmp_path) == 1

    (tmp_path / ".git").mkdir()

    class _Proc:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: _Proc(0, stdout=" M changed.py\n"),
    )
    assert mod._git_status_short(tmp_path) == [" M changed.py"]
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: _Proc(2, stderr="boom"),
    )
    assert mod._git_status_short(tmp_path) == ["git_status_error:boom"]

    _make_root(tmp_path)
    corpus_path = tmp_path / "results/experiment_6146_sota_constraint_event_corpus.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["model_specs"] = {"not": "a list"}
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    report = mod.build_report(
        tmp_path,
        adversarial_receipts=_receipts(),
        tests_run={".venv/bin/pytest tests/python -q": 0},
        duration_s=None,
    )
    assert report["exact_stream_and_sota_corpus_summary"]["sota_model_count"] == 0

    assert "field_provenance:not_mapping" in mod.validate_report({})
    broken = dict(report)
    broken["field_provenance"] = dict(report["field_provenance"])
    broken["field_provenance"]["status"] = {"principle": "wrong"}
    broken["reproducibility_checksum"] = mod.payload_checksum(broken)
    assert "field_provenance:status" in mod.validate_report(broken)
