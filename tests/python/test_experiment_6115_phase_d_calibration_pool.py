"""Tests for Exp6115 Phase D calibration pool.

Spec refs: REQ-VERIFY-6115, SCENARIO-VERIFY-6115-GATE,
SCENARIO-VERIFY-6115-CALIBRATION-ONLY, SCENARIO-VERIFY-6115-NATURAL-K,
SCENARIO-VERIFY-6115-REPLAY, SCENARIO-VERIFY-6115-POLICY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6103_phase_d_difficulty_ladder_fixture as ladder_mod
from carnot import experiment_6114_phase_d_gpu_ladder_canary as canary_mod
from carnot import experiment_6115_phase_d_calibration_pool as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/verifiable-reasoning/spec.md"
LADDER_ARTIFACT = REPO / ladder_mod.RESULT_RELATIVE_PATH
LADDER_ROWS = REPO / ladder_mod.ROW_FILE_RELATIVE_PATH
LADDER_SPLITS = REPO / ladder_mod.SPLIT_MANIFEST_RELATIVE_PATH
CANARY_ARTIFACT = REPO / canary_mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6115_phase_d_calibration_pool.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6115_phase_d_calibration_pool.py "
    "-m pytest tests/python/test_experiment_6115_phase_d_calibration_pool.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6115_phase_d_calibration_pool.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6115_phase_d_calibration_pool.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6115_phase_d_calibration_pool.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_COMMAND = "git status --short -- scripts/research_conductor.py"
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


class CalibrationFakeBackend:
    """Deterministic natural-generation backend for REQ-VERIFY-6115 tests."""

    def __init__(self, *, mode: str = "select_dense") -> None:
        self.mode = mode
        self.calls: list[dict[str, Any]] = []
        self.prompts_seen: list[dict[str, Any]] = []
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
        """SCENARIO-VERIFY-6115-NATURAL-K: return raw natural rows."""

        self.calls.append(
            {
                "hf_id": model_spec["hf_id"],
                "selected_gpu": selected_gpu,
                "prompt_count": len(prompts),
                "max_new_tokens": decode_config["max_new_tokens"],
                "temperature": decode_config["temperature"],
                "baseline_devices": baseline_devices,
            }
        )
        self.prompts_seen.extend(deepcopy(prompts))
        rows: list[dict[str, Any]] = []
        for prompt in prompts:
            source_id = str(prompt["source_exp6103_row_id"])
            sample_index = int(prompt["sample_index"])
            stratum = str(prompt["difficulty_stratum"])
            if self.mode == "all_too_easy":
                label = self.exact[source_id]
            elif stratum == "dense":
                label = self.exact[source_id] if sample_index < 4 else self.wrong[source_id]
            elif stratum == "compact":
                label = self.exact[source_id] if sample_index < 7 else self.wrong[source_id]
            else:
                label = self.exact[source_id] if sample_index < 2 else self.wrong[source_id]
            text = (
                f"Sample {sample_index} checks the visible rule, public constraints, "
                f"mod arithmetic, weight, risk, score, and feasible choices before "
                f"selecting a choice. Final answer: {label}"
            )
            rows.append(
                {
                    "candidate_prompt_id": prompt["candidate_prompt_id"],
                    "raw_generation": text,
                    "normalized_generation": text,
                    "generated_token_count": 64 + sample_index,
                    "decode_time_s": round(0.40 + sample_index / 100, 6),
                    "finish_reason": "stop",
                    "seed": prompt["seed"],
                }
            )
        return {
            "server_pid": 611500,
            "server_exit_code": 0,
            "pid_exited": True,
            "cuda_sync_method": "fake_backend_close",
            "worker_exit_observed": True,
            "vram_release_observed": True,
            "timeline": [
                {
                    "phase": "pre_load",
                    "task_pid": None,
                    "devices": baseline_devices,
                    "timestamp_monotonic_s": 1.0,
                },
                {
                    "phase": "decode",
                    "task_pid": 611500,
                    "devices": [
                        {
                            "index": selected_gpu,
                            "memory_total_mb": 24576,
                            "memory_free_mb": 5600,
                            "memory_used_mb": 18432,
                            "temperature_c": 61,
                            "power_draw_w": 260.0,
                        }
                    ],
                    "timestamp_monotonic_s": 2.0,
                },
                {
                    "phase": "post_release",
                    "task_pid": None,
                    "devices": baseline_devices,
                    "timestamp_monotonic_s": 3.0,
                },
            ],
            "gpu_engagement": {
                "attributable": True,
                "task_pid": 611500,
                "selected_gpu": selected_gpu,
                "selected_gpu_memory_delta_mb": 18432,
            },
            "energy_telemetry": {
                "power_samples": [{"timestamp_monotonic_s": 2.0, "power_draw_w": 260.0}],
                "estimated_energy_j": 260.0,
                "available": True,
            },
            "rows": rows,
        }


def _preconditions(tmp_path: Path) -> dict[str, Any]:
    before = {
        str(path): mod.sha256_file(REPO / path)
        for path in mod.PROTECTED_FILES
        if (REPO / path).exists()
    }
    return {
        "schema": "fixture.preconditions",
        "run_date": mod.RUN_DATE,
        "preconditions_ready": True,
        "blocked_reasons": [],
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
        "resources": {
            "memory": {"available_mb": 96000, "required_mb": 16384, "ok": True},
            "disk": {"available_mb": 512000, "required_mb": 10240, "ok": True},
        },
        "runtime": {
            "task_owned_pid_leases": {
                "current_pid": 1000,
                "child_pids": [],
                "lease_scope": "task_owned_processes_only",
            },
        },
        "output_paths": {
            "result_path": str(tmp_path / mod.RESULT_RELATIVE_PATH.name),
            "raw_rows_path": str(tmp_path / mod.RAW_ROWS_RELATIVE_PATH.name),
            "parent_writable": True,
        },
        "protected_file_hashes_before": before,
    }


def test_req_verify_6115_spec_declares_calibration_pool_contract() -> None:
    """REQ-VERIFY-6115: OpenSpec names fields, scenarios, and principles."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6115") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-6115",
        "SCENARIO-VERIFY-6115-GATE",
        "SCENARIO-VERIFY-6115-CALIBRATION-ONLY",
        "SCENARIO-VERIFY-6115-NATURAL-K",
        "SCENARIO-VERIFY-6115-REPLAY",
        "SCENARIO-VERIFY-6115-POLICY",
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


def test_scenario_verify_6115_gate_blocks_before_backend(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6115-GATE: false Exp6114 readiness prevents model calls."""

    blocked_canary = json.loads(CANARY_ARTIFACT.read_text(encoding="utf-8"))
    blocked_canary["phase_d_compute_and_ladder_ready_score"] = 0.0
    blocked_canary["status"] = "blocked"
    blocked_canary["honest_verdict"] = "blocked: fixture_gate"
    blocked_path = tmp_path / "blocked_canary.json"
    blocked_path.write_text(json.dumps(blocked_canary, sort_keys=True), encoding="utf-8")
    backend = CalibrationFakeBackend()

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        raw_rows_path=tmp_path / mod.RAW_ROWS_RELATIVE_PATH.name,
        ladder_artifact_path=LADDER_ARTIFACT,
        ladder_rows_path=LADDER_ROWS,
        ladder_split_manifest_path=LADDER_SPLITS,
        canary_artifact_path=blocked_path,
        preconditions_checked=_preconditions(tmp_path),
        generation_backend=backend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=1.0,
        write=True,
    )

    assert backend.calls == []
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["structured_gate_receipt"]["model_call_permitted"] is False
    assert artifact["structured_gate_receipt"]["backend_call_count"] == 0
    assert artifact["held_test_access_count"] == 0
    assert artifact["phase_d_calibration_ready_score"] == pytest.approx(0.0)
    assert mod.validate_artifact(artifact) is True


def test_scenario_verify_6115_replay_helper_edges(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6115-REPLAY: parser and gate edge branches are fixed."""

    answer_space = [
        {"label": "A", "candidate": "11"},
        {"label": "B", "candidate": "22"},
    ]
    parsed_candidate = mod._parse_final_answer("Reasoning.\nFinal answer: 22", answer_space)
    parsed_failure = mod._parse_final_answer("Reasoning only.", answer_space)
    gate = mod._structured_gate(
        {
            "phase_d_compute_and_ladder_ready_score": 1.0,
            "status": "complete_ready",
            "honest_verdict": "complete_ready: fixture",
            "target_model": "wrong/model",
            "server_exit_cuda_sync_pid_exit_and_vram_release_receipts": {"ready": False},
        }
    )
    fallback_model = mod._model_receipt(
        {
            "model_specs_and_exact_file_hashes": {},
            "model_specs": [{"hf_id": mod.MODEL_HF_ID, "model_sha256": "sha256:fixture"}],
        }
    )
    no_before = mod.protected_files_unchanged(root=tmp_path, before_hashes={})
    partial_status = mod._status_and_verdict(
        {
            "phase_d_calibration_ready_score": 0.0,
            "raw_candidate_row_paths_hashes_and_prefix_chain": {"candidate_row_count": 1},
            "selected_stratum_and_fixed_decode_policy": {"selected": {"policy_id": "fixture"}},
        },
        [],
    )
    empty_status = mod._status_and_verdict(
        {
            "phase_d_calibration_ready_score": 0.0,
            "raw_candidate_row_paths_hashes_and_prefix_chain": {"candidate_row_count": 0},
        },
        [],
    )

    assert parsed_candidate["parseable"] is True
    assert parsed_candidate["parsed_label"] == "B"
    assert parsed_failure["parseable"] is False
    assert mod._entropy([]) == 0.0
    assert mod._majority_label([{"answer_cluster": "UNPARSEABLE"}]) == ""
    assert gate["model_call_permitted"] is False
    assert "exp6114_model_mismatch" in gate["blocked_reasons"]
    assert "exp6114_release_not_ready" in gate["blocked_reasons"]
    assert fallback_model["records"][mod.MODEL_HF_ID]["model_sha256"] == "sha256:fixture"
    assert no_before["all_unchanged"] is True
    assert partial_status[0] == "complete_partial"
    assert empty_status == ("blocked", "blocked: no_generation_rows")


def test_scenario_verify_6115_complete_ready_pool_and_policy(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6115-CALIBRATION-ONLY/POLICY: select dense policy."""

    backend = CalibrationFakeBackend()
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        raw_rows_path=tmp_path / mod.RAW_ROWS_RELATIVE_PATH.name,
        ladder_artifact_path=LADDER_ARTIFACT,
        ladder_rows_path=LADDER_ROWS,
        ladder_split_manifest_path=LADDER_SPLITS,
        canary_artifact_path=CANARY_ARTIFACT,
        preconditions_checked=_preconditions(tmp_path),
        generation_backend=backend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.115,
        write=True,
    )

    assert len(backend.calls) == 1
    assert backend.calls[0]["hf_id"] == mod.MODEL_HF_ID
    assert backend.calls[0]["prompt_count"] == 90 * 8
    assert backend.calls[0]["max_new_tokens"] >= 512
    assert backend.calls[0]["temperature"] > 0
    assert all("exact_label" not in json.dumps(prompt) for prompt in backend.prompts_seen)
    assert all("validator" not in prompt["prompt_text"].lower() for prompt in backend.prompts_seen)
    assert all("answer with one label only" not in prompt["prompt_text"].lower() for prompt in backend.prompts_seen)

    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["phase_d_calibration_ready_score"] == pytest.approx(1.0)
    assert artifact["duration_s"] == pytest.approx(6.115)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["held_test_access_count"] == 0
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert mod.validate_artifact(artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8")) == artifact

    counts = artifact["calibration_question_family_stratum_and_semantic_group_counts"]
    assert counts["selected_question_count"] == 90
    assert counts["minimum_total_questions"] == 90
    assert counts["family_counts"] == {family: 30 for family in ladder_mod.FAMILIES}
    assert counts["difficulty_strata_preregistered_count"] >= 3
    assert len(counts["difficulty_stratum_counts"]) >= 3
    assert counts["semantic_group_duplicate_count"] == 0
    assert counts["semantic_siblings_cross_calibration_folds"] == 0

    raw = artifact["raw_candidate_row_paths_hashes_and_prefix_chain"]
    assert raw["candidate_row_count"] == 720
    assert raw["candidate_rows_per_question_min"] == 8
    assert raw["raw_generation_preserved"] is True
    assert raw["terminal_prefix_hash"].startswith("sha256:")
    raw_lines = (tmp_path / mod.RAW_ROWS_RELATIVE_PATH.name).read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 720

    parser = artifact["frozen_parser_and_parseability"]
    replay = artifact["python_z3_correctness_and_method_validity_replay"]
    intervals = artifact["per_candidate_accuracy_intervals"]
    diversity = artifact["duplicate_effective_k_answer_cluster_and_entropy_metrics"]
    sc = artifact["all_wrong_oracle_tuned_sc_and_solver_strata"]
    selected = artifact["selected_stratum_and_fixed_decode_policy"]["selected"]
    assert parser["parseability"] == pytest.approx(1.0)
    assert replay["python_z3_disagreement_count"] == 0
    assert replay["parser_failure_count_counted_as_failure"] == 0
    assert intervals["overall"]["candidate_count"] == 720
    assert intervals["overall"]["wilson_interval_95"][0] < intervals["overall"]["accuracy"] < intervals["overall"]["wilson_interval_95"][1]
    assert diversity["overall"]["mean_effective_k"] == pytest.approx(8.0)
    assert diversity["overall"]["duplicate_rate"] == pytest.approx(0.0)
    assert sc["overall"]["oracle_at_k"] > sc["overall"]["tuned_sc_accuracy"]
    assert selected["difficulty_stratum"] == "dense"
    assert selected["decode_policy"]["policy_id"] == mod.DEFAULT_DECODE_POLICY["policy_id"]
    assert 0.40 <= selected["metrics"]["accuracy"] <= 0.70
    assert selected["metrics"]["parseability"] >= 0.95
    assert selected["metrics"]["mean_effective_k"] >= 7.5
    assert selected["metrics"]["all_wrong_rate"] <= 0.10
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert artifact["field_provenance"][field]["principle"] == principle


def test_scenario_verify_6115_complete_null_when_no_policy_qualifies(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6115-POLICY: do not relax gates when calibration misses."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        raw_rows_path=tmp_path / mod.RAW_ROWS_RELATIVE_PATH.name,
        ladder_artifact_path=LADDER_ARTIFACT,
        ladder_rows_path=LADDER_ROWS,
        ladder_split_manifest_path=LADDER_SPLITS,
        canary_artifact_path=CANARY_ARTIFACT,
        preconditions_checked=_preconditions(tmp_path),
        generation_backend=CalibrationFakeBackend(mode="all_too_easy"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=2.0,
        write=False,
    )

    assert artifact["status"] == "complete_null"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["phase_d_calibration_ready_score"] == pytest.approx(0.0)
    assert artifact["selected_stratum_and_fixed_decode_policy"]["selected"] is None
    assert artifact["selected_stratum_and_fixed_decode_policy"]["held_relaxation_used"] is False
    assert artifact["held_test_access_count"] == 0
    assert mod.validate_artifact(artifact) is True
