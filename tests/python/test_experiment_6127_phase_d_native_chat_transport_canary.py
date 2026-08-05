"""Tests for Exp6127 native-chat transport canary.

Spec refs: REQ-VERIFY-6127, REQ-VERIFY-6127-1, REQ-VERIFY-6127-2,
REQ-VERIFY-6127-3, REQ-VERIFY-6127-4, REQ-VERIFY-6127-5,
REQ-VERIFY-6127-6, REQ-VERIFY-6127-7, REQ-VERIFY-6127-8,
REQ-VERIFY-6127-9, REQ-VERIFY-6127-10,
SCENARIO-VERIFY-6127-GATE, SCENARIO-VERIFY-6127-SLICE,
SCENARIO-VERIFY-6127-NATIVE-CHAT, SCENARIO-VERIFY-6127-THRESHOLDS,
SCENARIO-VERIFY-6127-LIFECYCLE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6103_phase_d_difficulty_ladder_fixture as ladder_mod
from carnot import experiment_6127_phase_d_native_chat_transport_canary as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/verifiable-reasoning/spec.md"
LADDER_ROWS = REPO / ladder_mod.ROW_FILE_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6127_phase_d_native_chat_transport_canary.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6127_phase_d_native_chat_transport_canary.py "
    "-m pytest tests/python/test_experiment_6127_phase_d_native_chat_transport_canary.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6127_phase_d_native_chat_transport_canary.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6127_phase_d_native_chat_transport_canary.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6127_phase_d_native_chat_transport_canary.json"
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


def _exact_label_maps() -> tuple[dict[str, str], dict[str, str]]:
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


class NativeChatFakeBackend:
    """REQ-VERIFY-6127-8: deterministic task-owned treatment backend."""

    def __init__(self, *, mode: str = "ready") -> None:
        self.mode = mode
        self.calls: list[dict[str, Any]] = []
        self.exact, self.wrong = _exact_label_maps()

    def generate(
        self,
        *,
        model_spec: dict[str, Any],
        selected_gpu: int,
        prompts: list[dict[str, Any]],
        decode_config: dict[str, Any],
        baseline_devices: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """SCENARIO-VERIFY-6127-NATIVE-CHAT: return one row per paired prompt."""

        self.calls.append(
            {
                "hf_id": model_spec["hf_id"],
                "selected_gpu": selected_gpu,
                "prompt_count": len(prompts),
                "max_new_tokens": decode_config["max_new_tokens"],
                "explicit_stop_strings": decode_config["explicit_stop_strings"],
                "grammar": decode_config["grammar"],
                "finite_id_transport": decode_config["finite_id_transport"],
            }
        )
        rows: list[dict[str, Any]] = []
        for index, prompt in enumerate(prompts):
            payload = json.dumps(prompt, sort_keys=True)
            assert "exact_label" not in payload
            assert "is_exact" not in payload
            source_id = str(prompt["source_exp6103_row_id"])
            label = self.exact[source_id] if self.mode == "ready" else self.wrong[source_id]
            text = (
                "I compare the visible public facts, rule, slot, task, weight, "
                "risk, score, and feasible choices before committing. "
                f"Final answer: {label}"
            )
            rows.append(
                {
                    "treatment_row_id": prompt["treatment_row_id"],
                    "raw_generation": text,
                    "normalized_generation": text,
                    "generated_token_count": 48 + index,
                    "decode_time_s": round(0.25 + index / 100, 6),
                    "finish_reason": "stop",
                    "seed": prompt["seed"],
                }
            )
        return {
            "server_pid": 612700,
            "server_exit_code": 0,
            "pid_exited": True,
            "worker_exit_observed": True,
            "cuda_sync_method": "fake_native_chat_backend_close",
            "vram_release_observed": True,
            "unrelated_processes_killed": [],
            "timeline": [
                {
                    "phase": "pre_load",
                    "task_pid": 612700,
                    "devices": baseline_devices,
                    "compute_apps": [],
                    "timestamp_monotonic_s": 1.0,
                },
                {
                    "phase": "load_or_decode",
                    "task_pid": 612700,
                    "devices": [
                        {
                            "index": selected_gpu,
                            "name": "RTX 3090",
                            "memory_total_mb": 24576,
                            "memory_free_mb": 6100,
                            "memory_used_mb": 18432,
                            "temperature_c": 61,
                            "power_draw_w": 260.0,
                        }
                    ],
                    "compute_apps": [{"pid": 612700, "used_memory_mb": 18432}],
                    "timestamp_monotonic_s": 2.0,
                },
                {
                    "phase": "post_release",
                    "task_pid": 612700,
                    "devices": baseline_devices,
                    "compute_apps": [],
                    "timestamp_monotonic_s": 3.0,
                },
            ],
            "gpu_engagement": {
                "attributable": True,
                "task_pid": 612700,
                "selected_gpu": selected_gpu,
                "selected_gpu_memory_delta_mb": 18000,
                "attribution_method": "nvidia_smi_compute_app_pid_and_memory_delta",
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


def test_req_verify_6127_spec_declares_native_chat_contract() -> None:
    """REQ-VERIFY-6127: OpenSpec names subrequirements and artifact fields."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6127") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-6127",
        "REQ-VERIFY-6127-1",
        "REQ-VERIFY-6127-2",
        "REQ-VERIFY-6127-3",
        "REQ-VERIFY-6127-4",
        "REQ-VERIFY-6127-5",
        "REQ-VERIFY-6127-6",
        "REQ-VERIFY-6127-7",
        "REQ-VERIFY-6127-8",
        "REQ-VERIFY-6127-9",
        "REQ-VERIFY-6127-10",
        "SCENARIO-VERIFY-6127-GATE",
        "SCENARIO-VERIFY-6127-SLICE",
        "SCENARIO-VERIFY-6127-NATIVE-CHAT",
        "SCENARIO-VERIFY-6127-THRESHOLDS",
        "SCENARIO-VERIFY-6127-LIFECYCLE",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.MODEL_HF_ID,
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_6127_slice_is_family_and_difficulty_balanced() -> None:
    """REQ-VERIFY-6127-2: frozen slice is calibration-only and paired."""

    exp6115_rows = mod.read_jsonl(REPO / mod.EXP6115_ROWS_RELATIVE_PATH)
    source_rows = mod.read_jsonl(REPO / mod.EXP6103_ROW_RELATIVE_PATH)
    frozen = mod.freeze_paired_slice(exp6115_rows, source_rows)

    assert frozen["question_count"] == 18
    assert frozen["baseline_sample_index"] == 0
    assert frozen["held_test_access_count"] == 0
    assert frozen["family_counts"] == {
        "finite_domain_scheduling": 6,
        "logic_grid": 6,
        "typed_finite_choice": 6,
    }
    difficulty_counts = frozen["difficulty_stratum_counts"]
    assert max(difficulty_counts.values()) - min(difficulty_counts.values()) <= 1
    assert len(frozen["pairs"]) == 18
    assert len({row["source_exp6103_row_id"] for row in frozen["pairs"]}) == 18
    for pair in frozen["pairs"]:
        assert pair["source_split"] == "calibration"
        assert pair["baseline_seed"] == pair["treatment_seed"]
        assert pair["baseline_candidate_row_id"].endswith("sample-00")
        assert pair["baseline_prompt_hash"] == mod.sha256_text(pair["baseline_prompt_text"])
        assert pair["treatment_message_hash"].startswith("sha256:")


def test_scenario_verify_6127_complete_ready_with_native_chat_contract(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6127-THRESHOLDS: all preregistered thresholds pass."""

    backend = NativeChatFakeBackend()
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(tmp_path),
        generation_backend=backend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.127,
        write=True,
    )

    assert len(backend.calls) == 1
    assert backend.calls[0]["hf_id"] == mod.MODEL_HF_ID
    assert backend.calls[0]["prompt_count"] == 18
    assert backend.calls[0]["max_new_tokens"] == mod.TREATMENT_MAX_NEW_TOKENS
    assert backend.calls[0]["explicit_stop_strings"] == []
    assert backend.calls[0]["grammar"] is None
    assert backend.calls[0]["finite_id_transport"] is False

    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["model_native_transport_ready_score"] == 1
    assert artifact["retirement_triggered"] is False
    assert artifact["duration_s"] == pytest.approx(6.127)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["structured_gate_receipt"]["backend_call_count"] == 1
    assert mod.validate_artifact(artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact

    contract = artifact["paired_baseline_treatment_prompt_seed_and_budget_contract"]
    assert contract["treatment_contract"]["serialization_api"] == (
        "llama_cpp.Llama.create_chat_completion"
    )
    assert contract["treatment_contract"]["explicit_stop_strings"] == []
    assert contract["treatment_contract"]["newline_stop_removed"] is True
    assert contract["treatment_contract"]["terminal_answer_field"] == "Final answer: <A|B|C|D>"
    assert contract["paired_question_count"] == 18
    assert contract["all_questions_and_seeds_paired"] is True

    rows = artifact["raw_completion_stop_reason_token_and_terminal_field_receipts"]
    assert rows["baseline"]["candidate_count"] == 18
    assert rows["treatment"]["candidate_count"] == 18
    assert rows["treatment"]["terminal_field_reach_count"] == 18
    assert rows["treatment"]["length_finish_reason_count"] == 0
    assert rows["treatment"]["raw_completions_preserved"] is True

    metrics = artifact["nonempty_terminal_parse_channel_method_and_accuracy_arm_metrics"]
    assert metrics["treatment"]["nonempty_rate"] == pytest.approx(1.0)
    assert metrics["treatment"]["terminal_field_reach_rate"] == pytest.approx(1.0)
    assert metrics["treatment"]["parseability"] == pytest.approx(1.0)
    assert metrics["treatment"]["channel_leakage_rate"] == pytest.approx(0.0)
    assert metrics["treatment"]["method_validity"] == pytest.approx(1.0)
    assert metrics["transport_primary_fields"] == [
        "nonempty_rate",
        "terminal_field_reach_rate",
        "parseability",
        "channel_leakage_rate",
        "length_finish_reason_count",
    ]
    assert metrics["accuracy_reported_not_transport_primary"] is True

    thresholds = artifact["paired_deltas_intervals_and_threshold_matrix"]
    assert thresholds["all_preregistered_transport_thresholds_pass"] is True
    assert thresholds["all_conjunctive_readiness_thresholds_pass"] is True
    assert thresholds["method_validity_delta"]["pass"] is True
    assert thresholds["exact_accuracy_delta"]["pass"] is True
    assert thresholds["parseability_cannot_substitute_for_method_validity"] is True

    disabled = artifact["hidden_label_retry_grammar_finite_id_and_parser_repair_counts"]
    assert disabled == {
        "hidden_label_retry_count": 0,
        "grammar_count": 0,
        "finite_id_transport_count": 0,
        "parser_repair_count": 0,
        "deterministic_answer_builder_count": 0,
        "principle": mod.REQUIRED_FIELD_PRINCIPLES[
            "hidden_label_retry_grammar_finite_id_and_parser_repair_counts"
        ],
    }
    lifecycle = artifact["task_owned_gpu_server_pid_engagement_and_release_timeline"]
    assert lifecycle["gpu_engagement_attributable"] is True
    assert lifecycle["release_ready"] is True
    assert lifecycle["unrelated_processes_killed"] == []
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert artifact["field_provenance"][field]["principle"] == principle


def test_scenario_verify_6127_parseability_cannot_substitute_for_method_validity(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6127-9/10: parseable wrong answers retire the attempt."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(tmp_path),
        generation_backend=NativeChatFakeBackend(mode="parseable_wrong"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=2.0,
        write=False,
    )

    metrics = artifact["nonempty_terminal_parse_channel_method_and_accuracy_arm_metrics"]
    thresholds = artifact["paired_deltas_intervals_and_threshold_matrix"]

    assert metrics["treatment"]["parseability"] == pytest.approx(1.0)
    assert metrics["treatment"]["method_validity"] == pytest.approx(0.0)
    assert thresholds["thresholds"]["treatment_parseability_at_least_0_90"]["pass"] is True
    assert thresholds["method_validity_delta"]["pass"] is False
    assert thresholds["parseability_cannot_substitute_for_method_validity"] is True
    assert thresholds["all_conjunctive_readiness_thresholds_pass"] is False
    assert artifact["model_native_transport_ready_score"] == 0
    assert artifact["status"] == "retired"
    assert artifact["honest_verdict"].startswith("retired:")
    assert artifact["retirement_triggered"] is True
    assert mod.validate_artifact(artifact) is True


def test_scenario_verify_6127_gate_blocks_before_backend(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6127-GATE: failed preconditions prevent model calls."""

    backend = NativeChatFakeBackend()
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
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
    assert artifact["preconditions_checked"]["blocked_reasons"] == ["fixture_gate_block"]
    assert artifact["model_native_transport_ready_score"] == 0
    assert artifact["retirement_triggered"] is False
    assert mod.validate_artifact(artifact) is True


def test_req_verify_6127_incomplete_backend_and_protected_change_block(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6127-1/8: backend and protected-file failures block."""

    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('\n{"ok": true}\n', encoding="utf-8")

    class IncompleteBackend(NativeChatFakeBackend):
        def generate(self, **kwargs: Any) -> dict[str, Any]:
            receipt = super().generate(**kwargs)
            receipt["server_exit_code"] = 7
            receipt["rows"] = receipt["rows"][:-1]
            return receipt

    preconditions = _preconditions(tmp_path)
    preconditions["protected_file_hashes_before"] = {
        "scripts/research_conductor.py": "sha256:not-the-current-hash"
    }
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        preconditions_checked=preconditions,
        generation_backend=IncompleteBackend(),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=1.0,
        write=False,
    )

    assert mod.read_jsonl(jsonl) == [{"ok": True}]
    assert mod._status_and_verdict(blockers=[], ready_score=0, treatment_attempted=False) == (
        "complete_null",
        "complete_null: no_treatment_attempt_after_nonblocking_null",
    )
    assert artifact["status"] == "blocked"
    assert "treatment_backend_nonzero_exit" in artifact["preconditions_checked"][
        "blocked_reasons"
    ]
    assert "treatment_row_count_incomplete" in artifact["preconditions_checked"][
        "blocked_reasons"
    ]
    assert "protected_files_changed" in artifact["preconditions_checked"]["blocked_reasons"]
    assert artifact["structured_gate_receipt"]["backend_call_count"] == 1
    assert artifact["protected_files_unchanged"]["unchanged"] is False
    assert mod.validate_artifact(artifact) is True
