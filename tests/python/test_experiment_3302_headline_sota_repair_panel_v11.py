"""Tests for Exp 3302 headline SOTA repair panel v11.

Spec refs: REQ-VERIFY-3302, SCENARIO-VERIFY-3302.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import headline_sota_repair_panel_v11 as mod


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"

REQUIRED_FIELDS = {
    "headline_repair_panel_ready",
    "repair_panel_ran",
    "headline_claim_allowed",
    "model_specs",
    "models_used",
    "missing_model_specs",
    "preconditions_checked",
    "panel_case_count",
    "verified_success_count",
    "repair_success_rate",
    "repair_success_ci95",
    "false_accept_count",
    "false_accept_rate_ci95",
    "abstention_count",
    "per_family_metrics",
    "candidate_results",
    "gpu_mem_used_mib",
    "tokens_generated",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(root: Path, rel_path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _panel_cases(count: int = 30) -> list[dict[str, Any]]:
    families = [
        "symbolic_aliases",
        "arithmetic_exact_rows",
        "context_shortcuts",
        "code_output_checks",
        "bounded_logical_consistency",
    ]
    cases: list[dict[str, Any]] = []
    for index in range(count):
        family = families[index % len(families)]
        expected = str(index + 100) if family == "arithmetic_exact_rows" else f"answer {index}"
        failing = str(index) if family == "arithmetic_exact_rows" else f"wrong {index}"
        case = {
            "case_id": f"case-{index:02d}",
            "family": family,
            "context": f"Local context {index} says the answer is {expected}.",
            "question": f"What is the row {index} answer?",
            "failing_candidate": failing,
            "expected_answer": expected,
            "exact_checker_type": (
                "exact_integer_string" if family == "arithmetic_exact_rows" else "exact_alias_string"
            ),
            "localized_repair_feedback": f"Replace {failing!r} with {expected!r}.",
            "llm_judge_required": False,
        }
        case["case_hash"] = mod.case_hash(case)
        cases.append(case)
    return cases


def _stage_inputs(
    root: Path,
    *,
    garak_passed: bool = True,
    clean_verifier_ready: bool = True,
    manifest_ready: bool = True,
    flagged: bool = False,
    count: int = 30,
) -> list[dict[str, Any]]:
    cases = _panel_cases(count)
    _write_json(
        root,
        mod.EXP3300_REL_PATH,
        {
            "experiment_id": "exp3300",
            "garak_redteam_eval_v3_ready": True,
            "garak_gate_passed": garak_passed,
            "dataflip_gate_passed": False,
            "models_used": [{"model_id": GEMMA26}],
            "flagged_adversarial": flagged,
            "corrigendum_pending": ["pending"] if flagged else [],
            "honest_verdict": "complete: garak fixture",
        },
    )
    _write_json(
        root,
        mod.EXP3287_REL_PATH,
        {
            "experiment_id": "exp3287",
            "abstention_calibrated_clean_verifier_v15_ready": clean_verifier_ready,
            "clean_verifier_rerun_ready": clean_verifier_ready,
            "repair_gate_input_clean_enough": clean_verifier_ready,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
            "abstention_count": 0,
            "coverage_rate": 1.0,
            "calibrated_abstention_policy": {
                "policy_name": "exp3286_strict_leading_token_calibrated_v1",
                "grammar": "ACCEPT|REJECT|ABSTAIN",
                "strict_leading_token": True,
            },
            "flagged_adversarial": flagged,
            "corrigendum_pending": ["pending"] if flagged else [],
            "honest_verdict": "complete: clean verifier fixture",
        },
    )
    _write_jsonl(root, mod.PANEL_CASES_REL_PATH, cases)
    _write_json(
        root,
        mod.EXP3301_REL_PATH,
        {
            "experiment_id": "exp3301",
            "repair_panel_manifest_ready": manifest_ready,
            "panel_case_count": len(cases),
            "panel_cases_path": mod.PANEL_CASES_REL_PATH.as_posix(),
            "case_hashes": [case["case_hash"] for case in cases],
            "panel_cases_sha256": mod.sha256_text(
                "".join(json.dumps(case, sort_keys=True) + "\n" for case in cases)
            ),
            "llm_judge_required_count": 0,
            "known_failing_candidate_count": len(cases),
            "localized_feedback_coverage": 1.0,
            "honest_verdict": "complete: manifest fixture",
        },
    )
    return cases


def _patch_one_cached_model(monkeypatch: pytest.MonkeyPatch, model_path: Path) -> None:
    def fake_resolver(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        del preferred_quant
        return str(model_path) if hf_id == GEMMA26 else None

    monkeypatch.setattr(mod, "cached_sota_pair", lambda gpu_indices=(0, 1): None)
    monkeypatch.setattr(mod, "resolve_cached_gguf", fake_resolver)


def _passed_probe(name: str) -> dict[str, Any]:
    return {"name": name, "passed": True, "detail": "test probe"}


def test_req_verify_3302_spec_anchor_declares_required_schema() -> None:
    """REQ-VERIFY-3302: OpenSpec declares the headline repair panel contract."""

    spec = (mod.REPO_ROOT / mod.SPEC_REL_PATH).read_text(encoding="utf-8")

    assert "REQ-VERIFY-3302" in spec
    assert "SCENARIO-VERIFY-3302" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "scripts/research_conductor.py" not in mod.OUTPUT_REL_PATH.as_posix()
    for field in REQUIRED_FIELDS:
        assert field in spec


def test_scenario_verify_3302_counts_success_false_accept_and_abstention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-3302: exact and clean-verifier authorities both gate success."""

    cases = _stage_inputs(tmp_path, flagged=True)
    model_path = tmp_path / "models/gemma26.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("gguf", encoding="utf-8")
    _patch_one_cached_model(monkeypatch, model_path)

    def mixed_runner(panel: list[dict[str, Any]], model: dict[str, Any], seed: int) -> dict[str, Any]:
        assert [case["case_id"] for case in panel] == [case["case_id"] for case in cases]
        assert model["model_id"] == GEMMA26
        assert seed == mod.RANDOM_SEED
        rows = []
        for index, case in enumerate(panel):
            candidate = case["expected_answer"]
            decision = "ACCEPT"
            if index == 1:
                candidate = case["failing_candidate"]
            if index == 2:
                decision = "ABSTAIN"
            if index == 3:
                decision = "REJECT"
            rows.append(
                {
                    "case_id": case["case_id"],
                    "candidate_answer": candidate,
                    "verifier_output_text": decision,
                    "token_counts": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
                }
            )
        return {"rows": rows, "gpu_mem_used_mib": 2048}

    artifact = mod.build_artifact(
        tmp_path,
        nvidia_probe=lambda: _passed_probe("nvidia_smi"),
        python_cuda_probe=lambda: _passed_probe("selected_python_cuda"),
        candidate_runner=mixed_runner,
        started_s=1.0,
        now_s=4.25,
        tests_run=["SCENARIO-VERIFY-3302"],
    )

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["repair_panel_ran"] is True
    assert artifact["headline_repair_panel_ready"] is True
    assert artifact["headline_claim_allowed"] is False
    assert artifact["panel_case_count"] == 30
    assert artifact["verified_success_count"] == 27
    assert artifact["repair_success_rate"] == pytest.approx(0.9)
    assert artifact["false_accept_count"] == 1
    assert artifact["abstention_count"] == 1
    assert artifact["tokens_generated"] == 60
    assert artifact["gpu_mem_used_mib"] == 2048
    assert len(artifact["candidate_results"]) == 30
    assert artifact["candidate_results"][1]["failure_class"] == "exact_mismatch_false_accept"
    assert artifact["candidate_results"][2]["abstained"] is True
    assert artifact["candidate_results"][3]["failure_class"] == "clean_verifier_rejected_exact_success"
    assert artifact["per_family_metrics"]["symbolic_aliases"]["case_count"] == 6
    assert len(artifact["repair_success_ci95"]) == 2
    assert len(artifact["false_accept_rate_ci95"]) == 2
    assert artifact["provenance_clean"] is False
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_req_verify_3302_all_successes_allow_headline_claim_when_provenance_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3302: headline claim requires N>=30, zero false accepts, and clean provenance."""

    _stage_inputs(tmp_path, flagged=False)
    model_path = tmp_path / "models/gemma26.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("gguf", encoding="utf-8")
    _patch_one_cached_model(monkeypatch, model_path)

    def all_success(panel: list[dict[str, Any]], model: dict[str, Any], seed: int) -> dict[str, Any]:
        del model, seed
        return {
            "gpu_mem_used_mib": 4096,
            "rows": [
                {
                    "case_id": case["case_id"],
                    "candidate_answer": case["expected_answer"],
                    "verifier_output_text": "ACCEPT",
                    "token_counts": {"completion_tokens": 1, "total_tokens": 3},
                }
                for case in panel
            ],
        }

    artifact = mod.build_artifact(
        tmp_path,
        nvidia_probe=lambda: _passed_probe("nvidia_smi"),
        python_cuda_probe=lambda: _passed_probe("selected_python_cuda"),
        candidate_runner=all_success,
    )

    assert artifact["headline_repair_panel_ready"] is True
    assert artifact["headline_claim_allowed"] is True
    assert artifact["verified_success_count"] == 30
    assert artifact["repair_success_rate"] == 1.0
    assert artifact["false_accept_count"] == 0
    assert artifact["abstention_count"] == 0
    assert artifact["models_used"][0]["model_id"] == GEMMA26
    assert {row["model_id"] for row in artifact["missing_model_specs"]} == {QWEN, GEMMA31}
    assert artifact["tokens_generated"] == 30
    mod.validate_artifact(artifact)


def test_scenario_verify_3302_blocked_preconditions_do_not_fabricate_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-3302: failed gates and missing models write a skipped artifact."""

    _stage_inputs(tmp_path, garak_passed=False, manifest_ready=False)
    monkeypatch.setattr(mod, "cached_sota_pair", lambda gpu_indices=(0, 1): None)
    monkeypatch.setattr(mod, "resolve_cached_gguf", lambda hf_id, preferred_quant="Q4_K_M": None)

    artifact = mod.build_artifact(
        tmp_path,
        nvidia_probe=lambda: {"name": "nvidia_smi", "passed": False},
        python_cuda_probe=lambda: {"name": "selected_python_cuda", "passed": False},
        started_s=5.0,
        now_s=5.5,
    )

    assert artifact["repair_panel_ran"] is False
    assert artifact["headline_repair_panel_ready"] is False
    assert artifact["headline_claim_allowed"] is False
    assert artifact["panel_case_count"] == 0
    assert artifact["candidate_results"] == []
    assert artifact["models_used"] == []
    assert artifact["tokens_generated"] == 0
    assert "exp3300_garak_gate_not_passed" in artifact["blocked_reasons"]
    assert "exp3301_fixed_manifest_unavailable" in artifact["blocked_reasons"]
    assert {row["model_id"] for row in artifact["missing_model_specs"]} == {QWEN, GEMMA31, GEMMA26}
    assert artifact["duration_s"] == pytest.approx(0.5)
    mod.validate_artifact(artifact)


def test_req_verify_3302_writer_validator_and_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3302: writer persists JSON and validation rejects overclaims."""

    _stage_inputs(tmp_path)
    model_path = tmp_path / "models/gemma26.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("gguf", encoding="utf-8")
    _patch_one_cached_model(monkeypatch, model_path)

    def all_success(panel: list[dict[str, Any]], model: dict[str, Any], seed: int) -> dict[str, Any]:
        del model, seed
        return {
            "gpu_mem_used_mib": 8192,
            "rows": [
                {
                    "case_id": case["case_id"],
                    "candidate_answer": "Final answer: " + case["expected_answer"],
                    "verifier_output_text": "ACCEPT",
                    "token_counts": {"completion_tokens": 1},
                }
                for case in panel
            ],
        }

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        nvidia_probe=lambda: _passed_probe("nvidia_smi"),
        python_cuda_probe=lambda: _passed_probe("selected_python_cuda"),
        candidate_runner=all_success,
        started_s=10.0,
        now_s=12.0,
        tests_run=["writer"],
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "results/out.json"
    assert saved["tests_run"] == ["writer"]
    assert saved["duration_s"] == pytest.approx(2.0)
    assert len(saved["reproducibility_checksum"]) == 64
    assert mod.clean_candidate_answer("Repaired answer: value") == "value"
    assert mod.clean_candidate_answer("Answer: 12\nextra") == "12"
    assert mod.clean_candidate_answer("") == ""
    assert mod.models_used(None, {}, 0) == []
    assert mod.normalize_verifier_decision("maybe") == "abstain"
    assert mod.normalize_verifier_decision("") == "abstain"
    assert mod.parse_int_string("+12") == 12
    assert mod.parse_int_string("-x") is None
    assert mod.normalize_bool_string("YES") == "true"
    assert mod.normalize_bool_string("0") == "false"
    assert mod.normalize_bool_string("maybe") == "maybe"
    assert mod.exact_check({"exact_checker_type": "exact_bool_string", "expected_answer": "true"}, "yes")
    assert mod.exact_check({"exact_checker_type": "exact_stdout_string", "expected_answer": "x-y"}, "x-y")
    assert mod.wilson_ci95(0, 0) == [0.0, 0.0]
    assert mod.source_clean({"flagged_adversarial": False}, {"corrigendum_pending": []}) is True
    assert mod.source_clean({"flagged_adversarial": True}, {}) is False
    assert mod.source_clean({"corrigendum_pending": ["pending"]}) is False
    assert mod.source_clean({"corrigendum_pending": "pending"}) is False
    assert mod.mapping_list("bad") == []
    assert mod.sequence("bad") == []
    assert mod.safe_int(True, default=7) == 7
    assert mod.duration(5.0, 4.0) == 0.0
    assert mod.read_json_object(tmp_path / "missing.json").error == "missing"

    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(malformed).readable is False
    non_object = tmp_path / "array.json"
    non_object.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(non_object).error == "json root is not an object"
    missing_jsonl = mod.read_jsonl_objects(tmp_path / "missing.jsonl")
    assert missing_jsonl == []
    noisy_jsonl = tmp_path / "noisy.jsonl"
    noisy_jsonl.write_text("\n{bad\n[]\n{\"ok\": true}\n", encoding="utf-8")
    assert mod.read_jsonl_objects(noisy_jsonl) == [{"ok": True}]

    edge_rows = mod.evaluate_runner_rows(
        [
            {
                "case_id": "missing",
                "family": "edge",
                "expected_answer": "done",
                "failing_candidate": "bad",
                "exact_checker_type": "exact_alias_string",
            },
            {
                "case_id": "reject-bad",
                "family": "edge",
                "expected_answer": "done",
                "failing_candidate": "bad",
                "exact_checker_type": "exact_alias_string",
            },
        ],
        [
            {"case_id": "missing", "candidate_answer": "", "verifier_output_text": ""},
            {"case_id": "reject-bad", "candidate_answer": "bad", "verifier_output_text": "REJECT"},
        ],
        {"model_id": GEMMA26, "model_path": str(model_path)},
    )
    assert edge_rows[0]["failure_class"] == "missing_candidate_output"
    assert edge_rows[1]["failure_class"] == "exact_mismatch_rejected"

    original_read_bytes = Path.read_bytes

    def fake_read_bytes(path: Path) -> bytes:
        if path.name == "boom":
            raise OSError("boom")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    assert mod.sha256_file(tmp_path / "boom") is None

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({key: value for key, value in saved.items() if key != "model_specs"})
    with pytest.raises(ValueError, match="headline_claim_allowed"):
        mod.validate_artifact(saved | {"headline_claim_allowed": "yes"})
    with pytest.raises(ValueError, match="panel_case_count"):
        mod.validate_artifact(saved | {"panel_case_count": True})
    with pytest.raises(ValueError, match="repair_success_rate"):
        mod.validate_artifact(saved | {"repair_success_rate": 2.0})
    with pytest.raises(ValueError, match="repair_success_ci95"):
        mod.validate_artifact(saved | {"repair_success_ci95": [0.0]})
    with pytest.raises(ValueError, match="duration_s"):
        mod.validate_artifact(saved | {"duration_s": -1.0})
    with pytest.raises(ValueError, match="candidate_results"):
        mod.validate_artifact(saved | {"candidate_results": {}})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(saved | {"reproducibility_checksum": "bad"})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(saved | {"honest_verdict": "blocked"})
    with pytest.raises(ValueError, match="models_used"):
        mod.validate_artifact(saved | {"repair_panel_ran": True, "models_used": []})
    with pytest.raises(ValueError, match="false accepts"):
        mod.validate_artifact(saved | {"headline_claim_allowed": True, "false_accept_count": 1})
    with pytest.raises(ValueError, match="panel_case_count"):
        mod.validate_artifact(saved | {"headline_claim_allowed": True, "panel_case_count": 29})
    with pytest.raises(ValueError, match="clean provenance"):
        mod.validate_artifact(saved | {"headline_claim_allowed": True, "provenance_clean": False})
    with pytest.raises(ValueError, match="headline_repair_panel_ready"):
        mod.validate_artifact(
            saved | {"headline_claim_allowed": True, "headline_repair_panel_ready": False}
        )
