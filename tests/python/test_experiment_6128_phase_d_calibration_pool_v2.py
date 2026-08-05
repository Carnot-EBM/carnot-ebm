"""Tests for Exp6128 Phase D calibration pool v2.

Spec refs: REQ-VERIFY-6128, REQ-VERIFY-6128-1, REQ-VERIFY-6128-2,
REQ-VERIFY-6128-3, REQ-VERIFY-6128-4, REQ-VERIFY-6128-5,
REQ-VERIFY-6128-6, REQ-VERIFY-6128-7, REQ-VERIFY-6128-8,
REQ-VERIFY-6128-9, REQ-VERIFY-6128-10, REQ-VERIFY-6128-11,
SCENARIO-VERIFY-6128-GATE, SCENARIO-VERIFY-6128-CALIBRATION-ONLY,
SCENARIO-VERIFY-6128-INDEPENDENT-K, SCENARIO-VERIFY-6128-GATES,
SCENARIO-VERIFY-6128-POLICY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6103_phase_d_difficulty_ladder_fixture as ladder_mod
from carnot import experiment_6128_phase_d_calibration_pool_v2 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/verifiable-reasoning/spec.md"
LADDER_ROWS = REPO / mod.EXP6103_ROW_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6128_phase_d_calibration_pool_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6128_phase_d_calibration_pool_v2.py "
    "-m pytest tests/python/test_experiment_6128_phase_d_calibration_pool_v2.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6128_phase_d_calibration_pool_v2.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6128_phase_d_calibration_pool_v2.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6128_phase_d_calibration_pool_v2.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_COMMAND = (
    "git status --short -- scripts/research_conductor.py ops/changelog.md "
    "ops/status.md _bmad/traceability.md"
)
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _label_maps() -> tuple[dict[str, str], dict[str, str]]:
    exact: dict[str, str] = {}
    wrong: dict[str, str] = {}
    for row in ladder_mod.read_row_file(LADDER_ROWS):
        if row["split"] != "calibration":
            continue
        labels = [str(item["label"]) for item in row["answer_space"]]
        exact_label = str(row["exact_label"])
        exact[str(row["row_id"])] = exact_label
        wrong[str(row["row_id"])] = next(label for label in labels if label != exact_label)
    return exact, wrong


class CalibrationPoolV2FakeBackend:
    """REQ-VERIFY-6128-4: deterministic stand-in for independent natural draws."""

    def __init__(self, *, mode: str = "qualified") -> None:
        self.mode = mode
        self.calls: list[dict[str, Any]] = []
        self.prompts_seen: list[dict[str, Any]] = []
        self.exact, self.wrong = _label_maps()

    def generate(
        self,
        *,
        model_spec: dict[str, Any],
        selected_gpu: int,
        prompts: list[dict[str, Any]],
        decode_config: dict[str, Any],
        baseline_devices: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """SCENARIO-VERIFY-6128-INDEPENDENT-K: return one auditable row per prompt."""

        self.calls.append(
            {
                "hf_id": model_spec["hf_id"],
                "selected_gpu": selected_gpu,
                "prompt_count": len(prompts),
                "max_new_tokens": decode_config["max_new_tokens"],
                "temperature": decode_config["temperature"],
                "explicit_stop_strings": decode_config["explicit_stop_strings"],
                "grammar": decode_config["grammar"],
                "finite_id_transport": decode_config["finite_id_transport"],
            }
        )
        self.prompts_seen.extend(deepcopy(prompts))
        rows: list[dict[str, Any]] = []
        for prompt in prompts:
            source_id = str(prompt["source_exp6103_row_id"])
            sample_index = int(prompt["sample_index"])
            if self.mode == "all_same_correct":
                label = self.exact[source_id]
            else:
                # Five correct and three wrong per question yields 0.625 accuracy,
                # oracle@K=1.0, tuned-SC=1.0.  The answer clusters are split so
                # exact duplicates stay zero while semantic duplicates are present.
                label = self.exact[source_id] if sample_index < 5 else self.wrong[source_id]
            text = (
                f"Draw {sample_index} uses the visible mod rule, slot, task, item, "
                f"person, weight, risk, score, and feasible constraints. "
                f"Final answer: {label}"
            )
            if self.mode == "oracle_gap":
                # Four correct and four wrong with a lexicographic wrong majority
                # makes tuned SC miss while oracle@K still succeeds.
                label = self.exact[source_id] if sample_index in {0, 1, 2, 3} else self.wrong[source_id]
                text = (
                    f"Draw {sample_index} checks mod, rule, slot, task, item, person, "
                    f"weight, risk, score, and feasible choices. Final answer: {label}"
                )
            rows.append(
                {
                    "candidate_prompt_id": prompt["candidate_prompt_id"],
                    "raw_generation": text,
                    "normalized_generation": text,
                    "generated_token_count": 80 + sample_index,
                    "decode_time_s": round(0.30 + sample_index / 100, 6),
                    "finish_reason": "stop",
                    "seed": prompt["seed"],
                }
            )
        return {
            "server_pid": 612800,
            "server_exit_code": 0,
            "pid_exited": True,
            "worker_exit_observed": True,
            "cuda_sync_method": "fake_native_chat_backend_close",
            "vram_release_observed": True,
            "unrelated_processes_killed": [],
            "timeline": [
                {
                    "phase": "pre_load",
                    "task_pid": 612800,
                    "devices": baseline_devices,
                    "compute_apps": [],
                    "timestamp_monotonic_s": 1.0,
                },
                {
                    "phase": "decode",
                    "task_pid": 612800,
                    "devices": [
                        {
                            "index": selected_gpu,
                            "name": "RTX 3090",
                            "memory_total_mb": 24576,
                            "memory_free_mb": 5900,
                            "memory_used_mb": 18432,
                            "temperature_c": 61,
                            "power_draw_w": 260.0,
                        }
                    ],
                    "compute_apps": [{"pid": 612800, "used_memory_mb": 18432}],
                    "timestamp_monotonic_s": 2.0,
                },
                {
                    "phase": "post_release",
                    "task_pid": 612800,
                    "devices": baseline_devices,
                    "compute_apps": [],
                    "timestamp_monotonic_s": 3.0,
                },
            ],
            "gpu_engagement": {
                "attributable": True,
                "task_pid": 612800,
                "selected_gpu": selected_gpu,
                "selected_gpu_memory_delta_mb": 18000,
            },
            "energy_telemetry": {
                "available": True,
                "power_samples": [{"timestamp_monotonic_s": 2.0, "power_draw_w": 260.0}],
                "estimated_energy_j": 260.0,
            },
            "rows": rows,
        }


def _preconditions(tmp_path: Path, *, ready: bool = True) -> dict[str, Any]:
    before = {
        str(path): mod.sha256_file(REPO / path)
        for path in mod.PROTECTED_FILES
        if (REPO / path).exists()
    }
    return {
        "schema": "fixture.preconditions",
        "run_date": mod.RUN_DATE,
        "preconditions_ready": ready,
        "blocked_reasons": [] if ready else ["fixture_gate_block"],
        "gpu": {
            "gpu_count": 1,
            "ok": True,
            "devices": [
                {
                    "index": 0,
                    "name": "RTX 3090",
                    "memory_total_mb": 24576,
                    "memory_free_mb": 24120,
                    "memory_used_mb": 456,
                    "temperature_c": 48,
                    "power_draw_w": 28.0,
                }
            ],
        },
        "lease_state": {
            "task_owned_pid": 1000,
            "child_pids_before": [],
            "lease_scope": "task_owned_child_worker_only",
        },
        "output_paths": {
            "result_path": str(tmp_path / mod.RESULT_RELATIVE_PATH.name),
            "raw_rows_path": str(tmp_path / mod.RAW_ROWS_RELATIVE_PATH.name),
            "parent_writable": True,
            "existed_before": False,
        },
        "root_clutter": {"root_python_file_count": 0, "ok": True},
        "protected_file_hashes_before": before,
        "inherited_debt": {
            "known_issues_sha256": "sha256:fixture-known",
            "exclusion_manifest_sha256": "sha256:fixture-exclusion",
        },
    }


def test_req_verify_6128_spec_declares_v2_contract() -> None:
    """REQ-VERIFY-6128: OpenSpec names subrequirements, fields, and principles."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6128") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-6128",
        "REQ-VERIFY-6128-1",
        "REQ-VERIFY-6128-2",
        "REQ-VERIFY-6128-3",
        "REQ-VERIFY-6128-4",
        "REQ-VERIFY-6128-5",
        "REQ-VERIFY-6128-6",
        "REQ-VERIFY-6128-7",
        "REQ-VERIFY-6128-8",
        "REQ-VERIFY-6128-9",
        "REQ-VERIFY-6128-10",
        "REQ-VERIFY-6128-11",
        "SCENARIO-VERIFY-6128-GATE",
        "SCENARIO-VERIFY-6128-CALIBRATION-ONLY",
        "SCENARIO-VERIFY-6128-INDEPENDENT-K",
        "SCENARIO-VERIFY-6128-GATES",
        "SCENARIO-VERIFY-6128-POLICY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.RAW_ROWS_RELATIVE_PATH.as_posix(),
        mod.MODEL_HF_ID,
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_6128_calibration_only_selection_is_balanced() -> None:
    """SCENARIO-VERIFY-6128-CALIBRATION-ONLY: selected IDs never include held rows."""

    rows = mod.read_jsonl(REPO / mod.EXP6103_ROW_RELATIVE_PATH)
    selected = mod.select_calibration_questions(rows)
    counts = mod.calibration_question_counts(selected)

    assert len(selected) == 90
    assert counts["selected_question_count"] == 90
    assert counts["family_counts"] == {family: 30 for family in ladder_mod.FAMILIES}
    assert counts["held_test_access_count"] == 0
    assert counts["semantic_group_duplicate_count"] == 0
    assert counts["difficulty_strata_preregistered_count"] >= 3
    assert all(row["split"] == "calibration" for row in selected)


def test_scenario_verify_6128_complete_ready_pool_freezes_policy(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6128-GATES/POLICY: conjunctive gates freeze one policy."""

    backend = CalibrationPoolV2FakeBackend(mode="oracle_gap")
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        raw_rows_path=tmp_path / mod.RAW_ROWS_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(tmp_path),
        generation_backend=backend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.128,
        write=True,
    )

    assert len(backend.calls) == 1
    assert backend.calls[0]["hf_id"] == mod.MODEL_HF_ID
    assert backend.calls[0]["prompt_count"] == 90 * 8
    assert backend.calls[0]["max_new_tokens"] == mod.FROZEN_DECODE_POLICY["max_new_tokens"]
    assert backend.calls[0]["explicit_stop_strings"] == []
    assert backend.calls[0]["grammar"] is None
    assert backend.calls[0]["finite_id_transport"] is False
    assert all("exact_label" not in json.dumps(prompt) for prompt in backend.prompts_seen)
    assert all("validator" not in prompt["messages"][-1]["content"].lower() for prompt in backend.prompts_seen)

    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["phase_d_calibration_ready_score"] == pytest.approx(1.0)
    assert artifact["retirement_triggered"] is False
    assert artifact["duration_s"] == pytest.approx(6.128)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact

    counts = artifact["calibration_question_family_stratum_and_semantic_group_counts"]
    assert counts["family_counts"] == {family: 30 for family in ladder_mod.FAMILIES}
    assert counts["held_test_access_count"] == 0
    assert counts["semantic_group_duplicate_count"] == 0

    attempts = artifact["attempted_expected_present_missing_and_duplicate_row_counts"]
    assert attempts["expected_row_count"] == 720
    assert attempts["attempted_row_count"] == 720
    assert attempts["present_row_count"] == 720
    assert attempts["missing_row_count"] == 0
    assert attempts["duplicate_row_count"] == 0
    assert attempts["candidate_rows_per_question_min"] == 8

    raw = artifact["raw_prompt_completion_stop_token_method_answer_and_exact_label_receipts"]
    assert raw["raw_row_count"] == 720
    assert raw["raw_rows_preserved"] is True
    assert raw["terminal_prefix_hash"].startswith("sha256:")
    assert len((tmp_path / mod.RAW_ROWS_RELATIVE_PATH.name).read_text().splitlines()) == 720

    metrics = artifact["per_candidate_accuracy_clustered_intervals_parseability_method_validity"]
    diversity = artifact[
        "effective_k_exact_semantic_duplicate_all_wrong_oracle_and_tuned_sc_metrics"
    ]
    gate = artifact["qualification_gate_matrix"]
    policy = artifact["frozen_policy_receipt"]
    assert metrics["overall"]["accuracy"] == pytest.approx(0.5)
    assert metrics["overall"]["parseability"] == pytest.approx(1.0)
    assert metrics["overall"]["method_validity"] == pytest.approx(1.0)
    assert diversity["overall"]["mean_effective_k"] == pytest.approx(8.0)
    assert diversity["overall"]["oracle_at_k"] == pytest.approx(1.0)
    assert diversity["overall"]["tuned_sc_accuracy"] == pytest.approx(0.0)
    assert diversity["overall"]["oracle_minus_tuned_sc"] == pytest.approx(1.0)
    assert all(row["pass"] is True for row in gate["gates"].values())
    assert gate["all_conjunctive_gates_pass"] is True
    assert policy["policy_frozen"] is True
    assert policy["held_generation_policy"]["decode_policy_id"] == mod.FROZEN_DECODE_POLICY["policy_id"]

    hidden = artifact["hidden_label_retry_and_deterministic_builder_counts"]
    assert hidden["hidden_label_retry_count"] == 0
    assert hidden["deterministic_answer_builder_count"] == 0
    assert hidden["grammar_count"] == 0
    assert hidden["finite_id_transport_count"] == 0
    assert hidden["parser_repair_count"] == 0
    assert hidden["held_label_conditioned_retry_count"] == 0
    lifecycle = artifact["task_owned_gpu_server_pid_engagement_and_release_timeline"]
    assert lifecycle["gpu_engagement_attributable"] is True
    assert lifecycle["release_ready"] is True
    assert lifecycle["unrelated_processes_killed"] == []
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert artifact["field_provenance"][field]["principle"] == principle


def test_scenario_verify_6128_complete_null_does_not_relax_or_retry(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6128-POLICY: a saturated pool stops honestly."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        raw_rows_path=tmp_path / mod.RAW_ROWS_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(tmp_path),
        generation_backend=CalibrationPoolV2FakeBackend(mode="all_same_correct"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=2.0,
        write=False,
    )

    assert artifact["status"] == "complete_null"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["phase_d_calibration_ready_score"] == pytest.approx(0.0)
    assert artifact["frozen_policy_receipt"]["policy_frozen"] is False
    assert artifact["frozen_policy_receipt"]["threshold_relaxation_used"] is False
    assert artifact["hidden_label_retry_and_deterministic_builder_counts"][
        "hidden_label_retry_count"
    ] == 0
    assert artifact["qualification_gate_matrix"]["all_conjunctive_gates_pass"] is False
    assert artifact["qualification_gate_matrix"]["gates"]["accuracy_band"]["pass"] is False
    assert artifact["qualification_gate_matrix"]["gates"]["oracle_minus_tuned_sc"]["pass"] is False
    assert artifact["retirement_triggered"] is False
    assert mod.validate_artifact(artifact) is True


def test_scenario_verify_6128_gate_blocks_before_backend(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6128-GATE: failed preconditions prevent generation."""

    backend = CalibrationPoolV2FakeBackend()
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        raw_rows_path=tmp_path / mod.RAW_ROWS_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(tmp_path, ready=False),
        generation_backend=backend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=0.5,
        write=False,
    )

    assert backend.calls == []
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["structured_gate_receipt"]["model_load_permitted"] is False
    assert artifact["structured_gate_receipt"]["backend_call_count"] == 0
    assert artifact["attempted_expected_present_missing_and_duplicate_row_counts"][
        "attempted_row_count"
    ] == 0
    assert artifact["preconditions_checked"]["blocked_reasons"] == ["fixture_gate_block"]
    assert artifact["phase_d_calibration_ready_score"] == pytest.approx(0.0)
    assert artifact["frozen_policy_receipt"]["policy_frozen"] is False
    assert mod.validate_artifact(artifact) is True


def test_req_verify_6128_backend_and_protected_failures_block(tmp_path: Path) -> None:
    """REQ-VERIFY-6128-1/10: incomplete backend or protected drift blocks."""

    class IncompleteBackend(CalibrationPoolV2FakeBackend):
        def generate(self, **kwargs: Any) -> dict[str, Any]:
            receipt = super().generate(**kwargs)
            receipt["server_exit_code"] = 7
            receipt["rows"] = receipt["rows"][:-1]
            return receipt

    preconditions = _preconditions(tmp_path)
    preconditions["protected_file_hashes_before"] = {
        "scripts/research_conductor.py": "sha256:not-current"
    }
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        raw_rows_path=tmp_path / mod.RAW_ROWS_RELATIVE_PATH.name,
        preconditions_checked=preconditions,
        generation_backend=IncompleteBackend(mode="oracle_gap"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=1.0,
        write=False,
    )

    blockers = artifact["preconditions_checked"]["blocked_reasons"]
    assert artifact["status"] == "blocked"
    assert "backend_nonzero_exit" in blockers
    assert "candidate_row_count_incomplete" in blockers
    assert "protected_files_changed" in blockers
    assert artifact["structured_gate_receipt"]["backend_call_count"] == 1
    assert artifact["protected_files_unchanged"]["unchanged"] is False
    assert mod.validate_artifact(artifact) is True


def test_req_verify_6128_helper_edge_receipts(tmp_path: Path) -> None:
    """REQ-VERIFY-6128-5/11: helper edge receipts are deterministic."""

    jsonl = tmp_path / "blank_rows.jsonl"
    jsonl.write_text('\n{"ok": true}\n', encoding="utf-8")
    no_attempt_artifact = {
        "phase_d_calibration_ready_score": 0.0,
        "attempted_expected_present_missing_and_duplicate_row_counts": {
            "attempted_row_count": 0
        },
    }

    assert mod.read_jsonl(jsonl) == [{"ok": True}]
    assert mod._entropy([]) == 0.0
    assert mod._majority_label([{"answer_cluster": "UNPARSEABLE"}]) == ""
    assert mod._question_summary([]) == {
        "question_count": 0,
        "all_wrong_rate": 0.0,
        "oracle_at_k": 0.0,
        "tuned_sc_accuracy": 0.0,
    }
    assert mod.protected_files_unchanged(root=tmp_path, before_hashes={})["unchanged"] is True
    assert mod._status_and_verdict(no_attempt_artifact, []) == (
        "blocked",
        "blocked: no_generation_attempted",
    )
