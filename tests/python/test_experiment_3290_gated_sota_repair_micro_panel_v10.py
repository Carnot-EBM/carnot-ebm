"""Tests for Exp 3290 gated SOTA repair micro-panel v10.

Spec refs: REQ-VERIFY-3290, SCENARIO-VERIFY-3290.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import gated_sota_repair_micro_panel_v10 as mod


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"

REQUIRED_FIELDS = {
    "sota_repair_micro_panel_v10_ready",
    "repair_panel_ran",
    "model_specs",
    "models_used",
    "missing_model_specs",
    "preconditions_checked",
    "panel_case_count",
    "repair_success_rate",
    "verified_success_count",
    "false_accept_count",
    "abstention_count",
    "localized_failure_feedback",
    "headline_claim_allowed",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_fixtures(root: Path) -> None:
    rows = [
        {
            "fixture_id": "ctx3290-symbolic-01",
            "family": "symbolic_aliases",
            "context": "For this fixture only, mercury means banana.",
            "question": "What does mercury mean?",
            "expected_answer": "banana",
            "exact_checker_type": "exact_alias_string",
            "prior_bait_answer": "planet",
            "minimal_counterexample": {
                "candidate_answer": "planet",
                "failure_mode": "parametric_prior_shortcut",
            },
        },
        {
            "fixture_id": "ctx3290-symbolic-02",
            "family": "symbolic_aliases",
            "context": "For this fixture only, python means blue screwdriver.",
            "question": "What does python mean?",
            "expected_answer": "blue screwdriver",
            "exact_checker_type": "exact_alias_string",
            "prior_bait_answer": "snake",
            "minimal_counterexample": {
                "candidate_answer": "snake",
                "failure_mode": "parametric_prior_shortcut",
            },
        },
        {
            "fixture_id": "ctx3290-symbolic-03",
            "family": "symbolic_aliases",
            "context": "For this fixture only, mars means teacup.",
            "question": "What does mars mean?",
            "expected_answer": "teacup",
            "exact_checker_type": "exact_alias_string",
            "prior_bait_answer": "planet",
            "minimal_counterexample": {
                "candidate_answer": "planet",
                "failure_mode": "parametric_prior_shortcut",
            },
        },
        {
            "fixture_id": "ctx3290-arithmetic-01",
            "family": "local_arithmetic_rules",
            "context": "Use only this worksheet rule: plus means multiply.",
            "question": "Under the worksheet rule, what is 3 plus 4?",
            "expected_answer": "12",
            "exact_checker_type": "exact_integer_string",
            "prior_bait_answer": "7",
            "minimal_counterexample": {
                "candidate_answer": "7",
                "failure_mode": "deterministic_integer_constraint_failure",
            },
        },
    ]
    path = root / mod.CONTEXT_FIXTURE_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _gate_payload(*, open_gate: bool = True) -> dict[str, Any]:
    scope: dict[str, Any] = {
        "repair_task_id": "exp3290-gated-sota-repair-micro-panel-v10",
        "scope_label": "bounded_exact_fixture_code_repair_micro_panel",
        "repair_generation_allowed": open_gate,
        "max_panel_cases": 8,
        "sample_size": {"min_cases": 4, "max_cases": 8},
        "selected_model_ids": [GEMMA26],
        "model_specs": {
            "runtime": "llama_cpp_local_gguf_only",
            "mandated_model_ids": [QWEN, GEMMA31, GEMMA26],
            "selected_model_ids": [GEMMA26],
            "missing_model_ids": [QWEN, GEMMA31],
        },
        "permitted_case_families": [
            "exact_context_fixture_counterexamples",
            "localized_code_or_json_fragment_failures",
            "deterministic_integer_constraint_failures",
        ],
        "exact_verification_requirements": {
            "accepted_repairs_require_exact_pass": True,
            "abstentions_recorded_separately": True,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
            "coverage_rate_floor": 1.0,
            "authority": [
                "calibrated_clean_verifier_v15",
                "exact_context_checker",
            ],
        },
        "claim_boundary": {
            "headline_claim_allowed": False,
            "no_generalization_beyond_panel": True,
            "panel_claim": "diagnostic_micro_panel_only",
        },
    }
    return {
        "schema_version": "carnot.repair_gate_decision.v9",
        "experiment_id": "exp3289",
        "repair_gate_decision_v9_ready": True,
        "repair_gate_open": open_gate,
        "permitted_repair_scope": scope if open_gate else {},
        "blocked_reasons": [] if open_gate else ["clean_verifier_false_accept_relaxation"],
        "reproducibility_checksum": "a" * 64,
        "honest_verdict": "complete: repair gate fixture",
    }


def _patch_one_cached_model(monkeypatch: pytest.MonkeyPatch, model_path: Path) -> None:
    def fake_resolver(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        if hf_id == GEMMA26:
            return str(model_path)
        return None

    monkeypatch.setattr(mod, "cached_sota_pair", lambda gpu_indices=(0, 1): None)
    monkeypatch.setattr(mod, "resolve_cached_gguf", fake_resolver)


def _passed_probe(name: str) -> dict[str, Any]:
    return {"name": name, "passed": True, "detail": "test probe"}


def test_req_verify_3290_spec_anchor_declares_artifact_schema() -> None:
    """REQ-VERIFY-3290: OpenSpec declares the micro-panel schema fields."""

    spec = (mod.REPO_ROOT / mod.SPEC_REL_PATH).read_text(encoding="utf-8")

    assert "REQ-VERIFY-3290" in spec
    assert "SCENARIO-VERIFY-3290" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    for field in REQUIRED_FIELDS:
        assert field in spec


def test_scenario_verify_3290_runs_single_cached_model_and_counts_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-3290: exact checks bound successes, false accepts, and abstentions."""

    _write_json(tmp_path, mod.EXP3289_REL_PATH, _gate_payload())
    _write_fixtures(tmp_path)
    model_path = tmp_path / "models/gemma26.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("gguf", encoding="utf-8")
    _patch_one_cached_model(monkeypatch, model_path)

    def fake_runner(cases: list[dict[str, Any]], model: dict[str, Any], seed: int) -> dict[str, Any]:
        return {
            "gpu_mem_used_mib": 1024,
            "rows": [
                {
                    "case_id": cases[0]["case_id"],
                    "candidate_answer": "banana",
                    "verifier_output_text": "ACCEPT",
                },
                {
                    "case_id": cases[1]["case_id"],
                    "candidate_answer": "snake",
                    "verifier_output_text": "ACCEPT",
                },
                {
                    "case_id": cases[2]["case_id"],
                    "candidate_answer": "teacup",
                    "verifier_output_text": "ABSTAIN",
                },
                {
                    "case_id": cases[3]["case_id"],
                    "candidate_answer": "8",
                    "verifier_output_text": "REJECT",
                },
            ],
        }

    artifact = mod.build_artifact(
        tmp_path,
        nvidia_probe=lambda: _passed_probe("nvidia_smi"),
        python_cuda_probe=lambda: _passed_probe("selected_python_cuda"),
        candidate_runner=fake_runner,
        started_s=1.0,
        now_s=3.5,
        tests_run=["SCENARIO-VERIFY-3290"],
    )

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["repair_panel_ran"] is True
    assert artifact["sota_repair_micro_panel_v10_ready"] is False
    assert artifact["panel_case_count"] == 4
    assert artifact["verified_success_count"] == 1
    assert artifact["false_accept_count"] == 1
    assert artifact["abstention_count"] == 1
    assert artifact["repair_success_rate"] == pytest.approx(0.25)
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["headline_claim_allowed"] is False
    assert artifact["models_used"][0]["model_id"] == GEMMA26
    assert {row["model_id"] for row in artifact["missing_model_specs"]} == {QWEN, GEMMA31}
    assert artifact["model_specs"]["cached_sota_pair_attempted"] is True
    assert artifact["model_specs"]["cached_sota_pair_available"] is False
    assert [row["name"] for row in artifact["preconditions_checked"]] == [
        "exp3289_repair_gate_open",
        "nvidia_smi",
        "selected_python_cuda",
        "mandated_sota_gguf_cache",
        "exact_context_fixture_panel",
    ]
    assert len(artifact["localized_failure_feedback"]) == 3
    assert artifact["localized_failure_feedback"][0]["failure_class"] == "exact_mismatch_false_accept"
    assert artifact["localized_failure_feedback"][1]["failure_class"] == "clean_verifier_abstained"
    assert artifact["localized_failure_feedback"][2]["expected_answer"] == "12"
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_req_verify_3290_ready_true_when_all_cases_are_verified_successes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3290: ready status means the executed panel had no false accepts."""

    _write_json(tmp_path, mod.EXP3289_REL_PATH, _gate_payload())
    _write_fixtures(tmp_path)
    model_path = tmp_path / "models/gemma26.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("gguf", encoding="utf-8")
    _patch_one_cached_model(monkeypatch, model_path)

    def all_success(cases: list[dict[str, Any]], model: dict[str, Any], seed: int) -> dict[str, Any]:
        return {
            "gpu_mem_used_mib": 2048,
            "rows": [
                {
                    "case_id": case["case_id"],
                    "candidate_answer": case["expected_answer"],
                    "verifier_output_text": "ACCEPT",
                }
                for case in cases
            ],
        }

    artifact = mod.build_artifact(
        tmp_path,
        nvidia_probe=lambda: _passed_probe("nvidia_smi"),
        python_cuda_probe=lambda: _passed_probe("selected_python_cuda"),
        candidate_runner=all_success,
    )

    assert artifact["sota_repair_micro_panel_v10_ready"] is True
    assert artifact["verified_success_count"] == 4
    assert artifact["repair_success_rate"] == 1.0
    assert artifact["false_accept_count"] == 0
    assert artifact["abstention_count"] == 0
    assert artifact["localized_failure_feedback"] == []
    mod.validate_artifact(artifact)


def test_scenario_verify_3290_closed_gate_writes_skipped_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3290: a closed Exp 3289 gate never runs repair generation."""

    _write_json(tmp_path, mod.EXP3289_REL_PATH, _gate_payload(open_gate=False))

    artifact = mod.build_artifact(tmp_path, started_s=4.0, now_s=4.5)

    assert artifact["repair_panel_ran"] is False
    assert artifact["sota_repair_micro_panel_v10_ready"] is False
    assert artifact["panel_case_count"] == 0
    assert artifact["models_used"] == []
    assert artifact["missing_model_specs"] == []
    assert artifact["repair_success_rate"] == 0.0
    assert artifact["preconditions_checked"][0]["name"] == "exp3289_repair_gate_open"
    assert artifact["preconditions_checked"][0]["passed"] is False
    assert "gate_blocked" in artifact["blocked_reasons"]
    mod.validate_artifact(artifact)


def test_req_verify_3290_missing_model_blocks_open_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3290: no mandated cached GGUF means no repair rows are fabricated."""

    _write_json(tmp_path, mod.EXP3289_REL_PATH, _gate_payload())
    _write_fixtures(tmp_path)
    monkeypatch.setattr(mod, "cached_sota_pair", lambda gpu_indices=(0, 1): None)
    monkeypatch.setattr(mod, "resolve_cached_gguf", lambda hf_id, preferred_quant="Q4_K_M": None)

    artifact = mod.build_artifact(
        tmp_path,
        nvidia_probe=lambda: _passed_probe("nvidia_smi"),
        python_cuda_probe=lambda: _passed_probe("selected_python_cuda"),
    )

    assert artifact["repair_panel_ran"] is False
    assert artifact["sota_repair_micro_panel_v10_ready"] is False
    assert artifact["models_used"] == []
    assert {row["model_id"] for row in artifact["missing_model_specs"]} == {QWEN, GEMMA31, GEMMA26}
    assert "mandated_sota_gguf_unavailable" in artifact["blocked_reasons"]
    mod.validate_artifact(artifact)


def test_req_verify_3290_writer_and_validator_enforce_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3290: writer persists validated JSON and validator catches bad fields."""

    _write_json(tmp_path, mod.EXP3289_REL_PATH, _gate_payload())
    _write_fixtures(tmp_path)
    model_path = tmp_path / "models/gemma26.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("gguf", encoding="utf-8")
    _patch_one_cached_model(monkeypatch, model_path)

    def all_success(cases: list[dict[str, Any]], model: dict[str, Any], seed: int) -> dict[str, Any]:
        return {
            "gpu_mem_used_mib": 4096,
            "rows": [
                {
                    "case_id": case["case_id"],
                    "candidate_answer": case["expected_answer"],
                    "verifier_output_text": "ACCEPT",
                }
                for case in cases
            ],
        }

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        nvidia_probe=lambda: _passed_probe("nvidia_smi"),
        python_cuda_probe=lambda: _passed_probe("selected_python_cuda"),
        candidate_runner=all_success,
        started_s=2.0,
        now_s=5.0,
        tests_run=["writer"],
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "results/out.json"
    assert saved["tests_run"] == ["writer"]
    assert saved["duration_s"] == pytest.approx(3.0)
    assert len(saved["reproducibility_checksum"]) == 64
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({key: value for key, value in saved.items() if key != "model_specs"})
    with pytest.raises(ValueError, match="repair_success_rate"):
        mod.validate_artifact(saved | {"repair_success_rate": 2.0})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(saved | {"honest_verdict": "blocked: no"})
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(saved | {"reproducibility_checksum": "bad"})
    with pytest.raises(ValueError, match="repair_panel_ran"):
        mod.validate_artifact(saved | {"repair_panel_ran": "yes"})
    with pytest.raises(ValueError, match="panel_case_count"):
        mod.validate_artifact(saved | {"panel_case_count": True})
    with pytest.raises(ValueError, match="duration_s"):
        mod.validate_artifact(saved | {"duration_s": -1.0})
    with pytest.raises(ValueError, match="models_used"):
        mod.validate_artifact(saved | {"models_used": {}})
    with pytest.raises(ValueError, match="headline"):
        mod.validate_artifact(saved | {"headline_claim_allowed": True})
    with pytest.raises(ValueError, match="repair_panel_ran"):
        mod.validate_artifact(saved | {"sota_repair_micro_panel_v10_ready": True, "repair_panel_ran": False})
    with pytest.raises(ValueError, match="models_used"):
        mod.validate_artifact(saved | {"sota_repair_micro_panel_v10_ready": True, "models_used": []})
    with pytest.raises(ValueError, match="false accepts"):
        mod.validate_artifact(
            saved | {"sota_repair_micro_panel_v10_ready": True, "false_accept_count": 1}
        )


def test_req_verify_3290_failed_preconditions_and_small_panel_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3290: failed GPU/Python/panel preconditions block execution."""

    _write_json(tmp_path, mod.EXP3289_REL_PATH, _gate_payload())
    model_path = tmp_path / "models/gemma26.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("gguf", encoding="utf-8")
    _patch_one_cached_model(monkeypatch, model_path)

    artifact = mod.build_artifact(
        tmp_path,
        nvidia_probe=lambda: {"name": "nvidia_smi", "passed": False},
        python_cuda_probe=lambda: {"name": "selected_python_cuda", "passed": False},
    )

    assert artifact["repair_panel_ran"] is False
    assert artifact["panel_case_count"] == 0
    assert "nvidia_smi_unavailable" in artifact["blocked_reasons"]
    assert "selected_python_cuda_unavailable" in artifact["blocked_reasons"]
    assert "exact_context_fixture_panel_too_small" in artifact["blocked_reasons"]
    assert artifact["preconditions_checked"][-1]["passed"] is False
    mod.validate_artifact(artifact)


def test_req_verify_3290_source_read_and_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3290: malformed sources and helper edge cases fail closed."""

    assert mod.read_json_object(tmp_path / "missing.json").error == "missing"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert "Expecting" in str(mod.read_json_object(malformed).error)
    non_object = tmp_path / "non_object.json"
    non_object.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(non_object).error == "json root is not an object"

    assert mod.models_used(None, {}, 0) == []
    assert mod.build_micro_panel(tmp_path, {"permitted_case_families": []}) == []

    fixture_path = tmp_path / mod.CONTEXT_FIXTURE_REL_PATH
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text("\n{bad\n[]\n{}\n", encoding="utf-8")
    assert mod.read_jsonl_objects(fixture_path) == [{}]
    assert mod.repair_case_from_fixture({}) == {}

    cases = [
        {
            "case_id": "int:repair",
            "fixture_id": "int",
            "family": "local_arithmetic_rules",
            "failing_candidate": "4",
            "expected_answer": "5",
            "exact_checker_type": "exact_integer_string",
            "localized_repair_feedback": "fix integer",
        },
        {
            "case_id": "string:repair",
            "fixture_id": "string",
            "family": "symbolic_aliases",
            "failing_candidate": "snake",
            "expected_answer": "north door",
            "exact_checker_type": "exact_alias_string",
            "localized_repair_feedback": "fix string",
        },
    ]
    rows = mod.evaluate_runner_rows(
        cases,
        [
            {"case_id": "int:repair", "candidate_answer": "", "verifier_output_text": ""},
            {
                "case_id": "string:repair",
                "candidate_answer": "Final answer: north door",
                "verifier_output_text": "REJECT",
            },
        ],
        {"model_id": GEMMA26, "model_path": "/models/gemma26.gguf"},
    )

    assert rows[0]["failure_class"] == "missing_candidate_output"
    assert rows[0]["abstained"] is True
    assert rows[1]["failure_class"] == "clean_verifier_rejected_exact_success"
    assert rows[1]["exact_check_passed"] is True
    assert mod.parse_int_string("+12") == 12
    assert mod.parse_int_string("-3") == -3
    assert mod.parse_int_string("-x") is None
    assert mod.normalize_verifier_decision("maybe") == "abstain"
    assert mod.source_artifact_row(fixture_path, "fixture")["present"] is True

    original_read_bytes = Path.read_bytes

    def fake_read_bytes(path: Path) -> bytes:
        if path.name == "boom":
            raise OSError("boom")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    assert mod.sha256_file(tmp_path / "boom") is None
    assert mod.safe_int(True, default=7) == 7
    assert mod.safe_int("not-an-int", default=7) == 7
