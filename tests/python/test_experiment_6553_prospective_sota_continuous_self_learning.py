"""Tests for Exp6553 prospective SOTA chronological CSL.

Spec refs: REQ-CL-6553, SCENARIO-CL-6553-FAIL-CLOSED-PRECONDITIONS,
SCENARIO-CL-6553-CHRONOLOGY-FREEZE, SCENARIO-CL-6553-MATCHED-ARMS,
SCENARIO-CL-6553-SUPPORT-RETENTION, SCENARIO-CL-6553-RESTART-ROLLBACK-SAFETY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6553_prospective_sota_continuous_self_learning as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6553_prospective_sota_continuous_self_learning.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6553_prospective_sota_continuous_self_learning.py "
    "-m pytest tests/python/test_experiment_6553_prospective_sota_continuous_self_learning.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6553_prospective_sota_continuous_self_learning.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6553_prospective_sota_continuous_self_learning.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6553_prospective_sota_continuous_self_learning.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6553_prospective_sota_continuous_self_learning.json"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6553_prospective_sota_continuous_self_learning "
    "--date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6553_prospective_sota_continuous_self_learning "
    "--validate"
)
E2E_COMMAND = "manual e2e-plan check: ops/e2e-test-plan.md GGUF/Z3 pipeline receipts inspected"
GIT_STATUS_COMMAND = "git status --short"

TESTS_RUN = [
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": E2E_COMMAND, "exit_code": 0},
    {"command": GIT_STATUS_COMMAND, "exit_code": 0},
]


class FakeInferenceBackend:
    """Small llama.cpp-shaped backend for REQ-CL-6553 unit coverage."""

    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.calls = 0
        self.closed = False

    def load_model(self, spec: dict[str, Any]) -> dict[str, Any]:
        self.loaded.append(spec["hf_id"])
        return {
            "hf_id": spec["hf_id"],
            "model_path": spec["model_path"],
            "loader": "llama_cpp.Llama",
            "load_ok": True,
            "smoke_ok": True,
            "embedded_tokenizer_ok": True,
            "process_id": 655300 + len(self.loaded),
            "load_s": 0.2,
            "smoke_s": 0.01,
            "error": "",
        }

    def infer(
        self,
        *,
        spec: dict[str, Any],
        query: dict[str, Any],
        arm_id: str,
        seed: int,
        timeout_s: float,
    ) -> dict[str, Any]:
        self.calls += 1
        token_bonus = {"frozen": 8, "hysteretic": 2, "same_query_mutation": 1}.get(arm_id, 4)
        prompt_tokens = 24 + int(query["query_index"])
        output_tokens = token_bonus + (seed % 3)
        return {
            "terminal_status": "terminal",
            "exit_status": "ok",
            "timeout": False,
            "censored": False,
            "request_text": f"{spec['hf_id']} {query['query_id']} {arm_id}",
            "response_text": f"FINAL: exact-satisfying {arm_id} {query['query_id']}",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "model_wall_time_s": round(0.18 + output_tokens * 0.01, 6),
            "first_token_time_s": 0.03,
            "gpu_samples": [
                {
                    "gpu": spec["gpu"],
                    "memory_used_mb": 12000 + output_tokens,
                    "utilization_pct": 82,
                }
            ],
            "timeout_s": timeout_s,
        }

    def close(self) -> None:
        self.closed = True


def _runtime_state(*, low_vram: bool = False) -> dict[str, Any]:
    free_1 = 2048 if low_vram else 24576
    return {
        "platform": "test",
        "python": "3.12",
        "gpu": {
            "available": True,
            "driver_version": "610.43.03",
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 3090",
                    "vram_total_mb": 24576,
                    "vram_free_mb": 24576,
                    "driver_version": "610.43.03",
                },
                {
                    "index": 1,
                    "name": "NVIDIA GeForce RTX 3090",
                    "vram_total_mb": 24576,
                    "vram_free_mb": free_1,
                    "driver_version": "610.43.03",
                },
            ],
        },
        "llama_cpp": {
            "available": True,
            "version": "0.3.33",
            "cuda_backend_available": True,
            "gpu_offload_supported": True,
            "system_info": "CUDA",
            "error": "",
        },
        "llama_cpp_binary": {
            "path": "/tmp/llama-cli",
            "exists": True,
            "executable": True,
            "version": "version: 9606",
        },
        "z3": {"available": True, "version": "4.16.0"},
        "disk": {"checkpoint_dir_writable": True, "disk_free_bytes": 2_000_000_000},
    }


@pytest.fixture()
def fake_model_specs(tmp_path: Path) -> list[dict[str, Any]]:
    """REQ-CL-6553: mandated GGUF paths are local files."""

    rows = []
    for index, hf_id in enumerate(mod.MANDATED_HF_IDS):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(f"fake gguf {hf_id}".encode())
        rows.append(
            {
                "name": mod.MODEL_NAMES_BY_HF_ID[hf_id],
                "hf_id": hf_id,
                "role": mod.MODEL_ROLES_BY_HF_ID[hf_id],
                "gpu": index % 2,
                "quantization": "Q4_K_M",
                "model_path": str(path),
            }
        )
    return rows


@pytest.fixture()
def artifact(tmp_path: Path, fake_model_specs: list[dict[str, Any]]) -> dict[str, Any]:
    """REQ-CL-6553: build a positive artifact with injected local receipts."""

    return mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        work_root=tmp_path / "work",
        write=True,
        duration_s=120.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
        model_specs_override=fake_model_specs,
        runtime_state_override=_runtime_state(),
        tokenizer_probe=lambda _path: (True, "embedded tokenizer ok"),
        inference_backend=FakeInferenceBackend(),
    )


def test_req_cl_6553_spec_declares_prospective_contract() -> None:
    """REQ-CL-6553: OpenSpec owns the Exp6553 prospective CSL contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CL-6553") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-CL-6553-FAIL-CLOSED-PRECONDITIONS",
        "SCENARIO-CL-6553-CHRONOLOGY-FREEZE",
        "SCENARIO-CL-6553-MATCHED-ARMS",
        "SCENARIO-CL-6553-SUPPORT-RETENTION",
        "SCENARIO-CL-6553-RESTART-ROLLBACK-SAFETY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "prospective_csl_ready_score",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_cl_6553_chronology_freeze_and_matched_arms(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-CL-6553-CHRONOLOGY-FREEZE/MATCHED-ARMS: rows are sealed."""

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_positive_prospective_csl_ready"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["verdict_class"] == "positive"
    assert artifact["prospective_csl_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert mod.validate_artifact(artifact) == []

    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(mod.MANDATED_HF_IDS)
    assert all(row["loader"] == "llama_cpp.Llama" for row in artifact["MODEL_SPECS"])
    assert all(row["model_path"].endswith(".gguf") for row in artifact["MODEL_SPECS"])
    assert all(row["gguf_sha256"].startswith("sha256:") for row in artifact["MODEL_SPECS"])

    contract = artifact["frozen_chronology_and_arm_contract"]
    assert contract["query_boundaries_per_model"] == mod.QUERY_BOUNDARY_COUNT
    assert contract["domain_count"] == 3
    assert contract["regime_transition_count"] >= 3
    assert contract["arms"] == list(mod.ARM_IDS)
    assert contract["same_query_arm_adoptable"] is False
    assert contract["held_outcomes_seen_before_freeze"] is False

    rows = artifact["per_unit_rows"]
    expected = len(mod.MANDATED_HF_IDS) * mod.QUERY_BOUNDARY_COUNT * len(mod.ARM_IDS)
    assert len(rows) == expected
    assert {row["arm_id"] for row in rows} == set(mod.ARM_IDS)
    assert {row["model_hf_id"] for row in rows} == set(mod.MANDATED_HF_IDS)
    assert all(row["pre_memory_hash"] == row["frozen_query_snapshot_hash"] for row in rows)
    assert all(row["decision_time_write_count"] == 0 for row in rows)
    assert all(row["post_query_memory_hash"].startswith("sha256:") for row in rows)
    assert all(row["request_hash"].startswith("sha256:") for row in rows)
    assert all(row["response_hash"].startswith("sha256:") for row in rows)
    assert all(row["row_hash"] == mod.row_hash(row) for row in rows)

    transition_rows = artifact["memory_transition_rows"]
    assert transition_rows
    assert all(row["witness_hash"].startswith("sha256:") for row in transition_rows)
    assert all(row["row_hash"] == mod.row_hash(row) for row in transition_rows)
    assert all(
        row["commit_after_exact_verification"] is True
        for row in transition_rows
        if row["arm_id"] != "same_query_mutation"
    )
    assert all(
        row["commit_decision"] == "diagnostic_not_adopted"
        for row in transition_rows
        if row["arm_id"] == "same_query_mutation"
    )


def test_scenario_cl_6553_support_safety_restart_and_recomputation(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-CL-6553-SUPPORT-RETENTION/RESTART-ROLLBACK-SAFETY: gates derive."""

    aggregate = artifact["aggregate_row_recomputation"]
    assert aggregate == mod.aggregate_row_recomputation(artifact)
    assert aggregate["ready_score_from_rows"] == 1.0
    assert aggregate["safe_positive_arm_id"] == "hysteretic"
    assert aggregate["current_value_positive"] is True
    assert aggregate["retained_family_noninferior"] is True
    assert aggregate["future_support_noninferior"] is True
    assert aggregate["multi_model_support"] is True
    assert aggregate["exact_output_equality"] is True

    current = artifact["current_cost_and_success_rows"]
    retained = artifact["retained_family_rows"]
    future = artifact["future_support_rows"]
    assert {row["arm_id"] for row in current} == set(mod.ARM_IDS)
    assert any(row["arm_id"] == "hysteretic" and row["charged_value_delta"] > 0 for row in current)
    assert all(row["noninferior"] for row in retained if row["arm_id"] == "hysteretic")
    assert all(row["noninferior"] for row in future if row["arm_id"] == "hysteretic")

    dose = artifact["coobservation_and_dose_receipt"]
    assert dose["matched_update_dose"] is True
    assert dose["coobservation_arm"] == "matched_dose_coobservation"
    assert dose["replay_benefit_separated_from_extra_update_exposure"] is True

    unsafe = artifact["unsafe_write_and_use_ledger"]
    assert unsafe["safe_arm_unsafe_write_count"] == 0
    assert unsafe["safe_arm_unsafe_use_count"] == 0
    assert unsafe["diagnostic_same_query_unsafe_write_count"] > 0
    assert unsafe["same_query_arm_adopted"] is False

    lifecycle = artifact["restart_and_rollback_receipts"]
    assert lifecycle["all_restarts_exact_output_equal"] is True
    assert lifecycle["all_rollbacks_restored"] is True
    assert lifecycle["corrupt_write_challenge_fail_closed"] is True

    receipts = artifact["live_model_and_gpu_receipts"]
    assert receipts["all_mandated_models_loaded"] is True
    assert receipts["fresh_local_inference_performed"] is True
    assert receipts["unsupported_fallback_count"] == 0
    assert receipts["hidden_legacy_substitution_count"] == 0
    assert artifact["charged_cost_recomputation"]["all_costs_match_rows"] is True
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)


def test_scenario_cl_6553_fail_closed_preconditions(
    tmp_path: Path,
    fake_model_specs: list[dict[str, Any]],
) -> None:
    """SCENARIO-CL-6553-FAIL-CLOSED-PRECONDITIONS: failed VRAM blocks rows."""

    blocked = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked.json",
        work_root=tmp_path / "work",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
        model_specs_override=fake_model_specs,
        runtime_state_override=_runtime_state(low_vram=True),
        tokenizer_probe=lambda _path: (True, "embedded tokenizer ok"),
        inference_backend=FakeInferenceBackend(),
    )
    assert blocked["status"] == "blocked_prospective_csl_preconditions"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["prospective_csl_ready_score"] == 0.0
    assert blocked["per_unit_rows"] == []
    assert blocked["memory_transition_rows"] == []
    assert "gpu_vram_contract" in blocked["gate_check_summary"]["failed_checks"]
    assert blocked["live_model_and_gpu_receipts"]["fresh_local_inference_performed"] is False
    assert blocked["live_model_and_gpu_receipts"]["generated_token_invocation_count"] == 0
    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == blocked
    assert mod.validate_artifact(blocked) == []

    missing_model = deepcopy(fake_model_specs)
    missing_model[0]["model_path"] = str(tmp_path / "missing.gguf")
    blocked_model = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked-model.json",
        work_root=tmp_path / "work-model",
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
        model_specs_override=missing_model,
        runtime_state_override=_runtime_state(),
        tokenizer_probe=lambda _path: (True, "embedded tokenizer ok"),
        inference_backend=FakeInferenceBackend(),
    )
    assert "all_required_gguf_files_resolved" in blocked_model["gate_check_summary"]["failed_checks"]
    assert blocked_model["MODEL_SPECS"][0]["model_path_exists"] is False
    assert blocked_model["verdict_class"] == "blocked"
    assert mod.validate_artifact(blocked_model) == []

    tokenizer_block = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "tokenizer-block.json",
        work_root=tmp_path / "work-tokenizer",
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
        model_specs_override=fake_model_specs,
        runtime_state_override=_runtime_state(),
        tokenizer_probe=lambda _path: (False, "tokenizer failed"),
        inference_backend=FakeInferenceBackend(),
    )
    assert tokenizer_block["verdict_class"] == "partial"
    assert tokenizer_block["live_model_and_gpu_receipts"]["all_mandated_models_loaded"] is False
    assert tokenizer_block["live_model_and_gpu_receipts"]["model_load_rows"][0]["error"] == "tokenizer failed"
    assert mod.validate_artifact(tokenizer_block) == []

    assert mod._load_json(tmp_path / "missing.json") == {}
    missing_jsonl = tmp_path / "missing.jsonl"
    assert mod._load_jsonl(missing_jsonl) == []
    mixed_jsonl = tmp_path / "mixed.jsonl"
    mixed_jsonl.write_text('\n{"ok": true}\n[]\n', encoding="utf-8")
    assert mod._load_jsonl(mixed_jsonl) == [{"ok": True}, {"value": []}]


def test_scenario_cl_6553_validation_edges_and_cli(
    tmp_path: Path,
    artifact: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CL-6553: validator catches drift and CLI validates artifacts."""

    assert (
        mod.main(
            [
                "--validate",
                "--result-path",
                str(tmp_path / "missing.json"),
            ]
        )
        == 1
    )
    assert "artifact not found" in capsys.readouterr().out

    result_path = tmp_path / "artifact.json"
    result_path.write_text(json.dumps(artifact), encoding="utf-8")
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    assert "validated" in capsys.readouterr().out

    validations = [
        ("delete", "status", "required field set mismatch"),
        ("set", ("status", "running"), "status lacks terminal prefix"),
        ("set", ("honest_verdict", "ready"), "honest_verdict lacks terminal prefix"),
        ("set", ("verdict_class", "circular_positive"), "verdict_class must be closed"),
        ("set", ("inference_substrate", "wrong"), "inference_substrate mismatch"),
        ("set", ("verifier_is_oracle", True), "verifier_is_oracle must be false"),
        ("set", ("prospective_csl_ready_score", 0.5), "prospective_csl_ready_score mismatch"),
        ("set", ("field_provenance", {}), "field_provenance must cover required fields"),
        ("set", ("reproducibility_checksum", "sha256:bad"), "reproducibility_checksum mismatch"),
    ]
    for mode, spec, expected in validations:
        bad = deepcopy(artifact)
        if mode == "delete":
            del bad[spec]
        else:
            key, value = spec
            bad[key] = value
        if expected != "reproducibility_checksum mismatch":
            bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        assert any(expected in error for error in mod.validate_artifact(bad))

    bad_row = deepcopy(artifact)
    bad_row["per_unit_rows"][0]["row_hash"] = "sha256:bad"
    bad_row["reproducibility_checksum"] = mod.reproducibility_checksum(bad_row)
    assert "per_unit_rows row_hash mismatch" in mod.validate_artifact(bad_row)

    bad_order = deepcopy(artifact)
    bad_order["MODEL_SPECS"] = list(reversed(bad_order["MODEL_SPECS"]))
    bad_order["reproducibility_checksum"] = mod.reproducibility_checksum(bad_order)
    assert "MODEL_SPECS mandated order mismatch" in mod.validate_artifact(bad_order)

    bad_aggregate = deepcopy(artifact)
    bad_aggregate["aggregate_row_recomputation"]["row_count"] = -1
    bad_aggregate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_aggregate)
    assert "aggregate_row_recomputation mismatch" in mod.validate_artifact(bad_aggregate)

    bad_protected = deepcopy(artifact)
    bad_protected["protected_files_unchanged"]["all_protected_files_unchanged"] = False
    bad_protected["aggregate_row_recomputation"] = mod.aggregate_row_recomputation(bad_protected)
    bad_protected["prospective_csl_ready_score"] = bad_protected["aggregate_row_recomputation"][
        "ready_score_from_rows"
    ]
    bad_protected["reproducibility_checksum"] = mod.reproducibility_checksum(bad_protected)
    assert "protected files changed" in mod.validate_artifact(bad_protected)

    unsafe_positive = deepcopy(artifact)
    unsafe_positive["unsafe_write_and_use_ledger"]["safe_arm_unsafe_write_count"] = 1
    unsafe_positive["aggregate_row_recomputation"] = mod.aggregate_row_recomputation(
        unsafe_positive
    )
    unsafe_positive["prospective_csl_ready_score"] = unsafe_positive[
        "aggregate_row_recomputation"
    ]["ready_score_from_rows"]
    unsafe_positive["reproducibility_checksum"] = mod.reproducibility_checksum(unsafe_positive)
    assert any("positive verdict requires ready score 1.0" in err for err in mod.validate_artifact(unsafe_positive))

    real_validate = mod.validate_artifact
    monkeypatch.setattr(mod, "validate_artifact", lambda _payload: ["forced build error"])
    with pytest.raises(ValueError, match="forced build error"):
        mod.build_artifact(
            repo_root=REPO,
            result_path=tmp_path / "forced-build.json",
            work_root=tmp_path / "forced-build-work",
            write=False,
            duration_s=1.0,
            tests_run=TESTS_RUN,
            run_date="20260823",
            model_specs_override=artifact["MODEL_SPECS"],
            runtime_state_override=_runtime_state(),
            tokenizer_probe=lambda _path: (True, "embedded tokenizer ok"),
            inference_backend=FakeInferenceBackend(),
        )
    monkeypatch.setattr(mod, "validate_artifact", real_validate)

    forced_validate_path = tmp_path / "forced-validate.json"
    forced_validate_path.write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setattr(mod, "validate_artifact", lambda _payload: ["forced validate error"])
    assert mod.main(["--validate", "--result-path", str(forced_validate_path)]) == 1
    assert "forced validate error" in capsys.readouterr().out
    monkeypatch.setattr(mod, "validate_artifact", real_validate)

    monkeypatch.setattr(
        mod,
        "build_artifact",
        lambda **kwargs: {
            **artifact,
            "duration_s": kwargs.get("duration_s") or artifact["duration_s"],
        },
    )
    assert (
        mod.main(
            [
                "--date",
                "20260823",
                "--result-path",
                str(tmp_path / "main.json"),
                "--work-root",
                str(tmp_path / "main-work"),
            ]
        )
        == 0
    )
    assert "wrote" in capsys.readouterr().out

    monkeypatch.setattr(mod, "validate_artifact", lambda _payload: ["forced error"])
    assert (
        mod.main(
            [
                "--date",
                "20260823",
                "--result-path",
                str(tmp_path / "forced.json"),
                "--work-root",
                str(tmp_path / "forced-work"),
            ]
        )
        == 1
    )
    assert "forced error" in capsys.readouterr().out
