"""Tests for Exp 3316 SOTA repair rerun v12 runtime-clean.

Spec refs: REQ-VERIFY-3316, SCENARIO-VERIFY-3316.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.verify import sota_repair_rerun_v12_runtime_clean as mod


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"

REQUIRED_FIELDS = {
    "repair_rerun_v12_ready",
    "repair_panel_ran",
    "headline_repair_panel_ready",
    "headline_claim_allowed",
    "runtime_provenance_clean",
    "duration_contract_passed",
    "substrate_consistency_passed",
    "panel_case_count",
    "verified_success_count",
    "false_accept_count",
    "abstention_count",
    "repair_success_rate",
    "confidence_interval",
    "model_specs_used",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(root: Path, rel_path: Path, rows: list[dict[str, Any]]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _cases(count: int = 30) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    families = ["arithmetic_exact_rows", "symbolic_aliases", "code_output_checks"]
    for index in range(count):
        family = families[index % len(families)]
        expected = str(index + 40) if family == "arithmetic_exact_rows" else f"answer {index}"
        failing = str(index) if family == "arithmetic_exact_rows" else f"wrong {index}"
        row = {
            "case_id": f"repair-v12-{index:02d}",
            "family": family,
            "context": f"Fixture context {index} gives {expected}.",
            "question": f"What is fixture answer {index}?",
            "failing_candidate": failing,
            "expected_answer": expected,
            "exact_checker_type": "exact_integer_string"
            if family == "arithmetic_exact_rows"
            else "exact_alias_string",
            "localized_repair_feedback": f"Use {expected} instead of {failing}.",
            "llm_judge_required": False,
        }
        row["case_hash"] = mod.case_hash(row)
        rows.append(row)
    return rows


def _stage_sources(root: Path, *, clean_policy: bool = True, count: int = 30) -> list[dict[str, Any]]:
    cases = _cases(count)
    hashes = [row["case_hash"] for row in cases]
    _write_jsonl(root, mod.PANEL_CASES_REL_PATH, cases)
    _write_json(
        root,
        mod.EXP3301_REL_PATH,
        {
            "experiment_id": "exp3301",
            "repair_panel_manifest_ready": True,
            "panel_case_count": len(cases),
            "panel_cases_path": mod.PANEL_CASES_REL_PATH.as_posix(),
            "case_hashes": hashes,
            "panel_cases_sha256": mod.sha256_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in cases)
            ),
            "llm_judge_required_count": 0,
            "known_failing_candidate_count": len(cases),
            "localized_feedback_coverage": 1.0,
        },
    )
    _write_json(
        root,
        mod.EXP3287_REL_PATH,
        {
            "experiment_id": "exp3287",
            "abstention_calibrated_clean_verifier_v15_ready": True,
            "clean_verifier_rerun_ready": True,
            "repair_gate_input_clean_enough": True,
            "false_accept_count": 0,
            "calibrated_abstention_policy": {
                "grammar": "ACCEPT|REJECT|ABSTAIN",
                "strict_leading_token": True,
            },
        },
    )
    _write_json(
        root,
        mod.EXP3312_REL_PATH,
        {
            "experiment_id": "exp3312",
            "dataflip_gate_passed": True,
            "quality_flags_cleared": True,
            "runtime_provenance_clean": True,
            "duration_contract_passed": True,
            "headline_claim_allowed": True,
            "adversarial_verify_flags": [{"kind": "INFO_ONLY", "severity": "info"}],
            "model_specs_used": {
                "used_model_ids": [GEMMA26],
                "missing_model_ids": [QWEN, GEMMA31],
                "legacy_small_model_substituted": False,
            },
            "checker_versions": _checker_versions(),
        },
    )
    _write_json(
        root,
        mod.EXP3309_REL_PATH,
        {
            "experiment_id": "exp3309",
            "runtime_contract_ready": True,
            "contract_version": mod.RUNTIME_CONTRACT_VERSION,
            "minimum_live_duration_s": 60.0,
        },
    )
    _write_json(
        root,
        mod.EXP3302_REL_PATH,
        {
            "experiment_id": "exp3302",
            "repair_panel_ran": True,
            "headline_repair_panel_ready": True,
            "panel_case_count": len(cases),
            "manifest_case_hashes": hashes,
            "candidate_results": [
                {"case_id": row["case_id"], "case_hash": row["case_hash"], "exact_checker_type": row["exact_checker_type"]}
                for row in cases
            ],
            "repair_success_ci95": [0.74, 0.97],
            "false_accept_count": 0,
            "abstention_count": 0,
        },
    )
    _write_json(
        root,
        mod.EXP3314_REL_PATH,
        {
            "experiment_id": "exp3314",
            "distributional_repair_audit_ready": True,
            "repair_case_count": len(cases),
            "exact_acceptance_authority_preserved": True,
            "uncertainty_abstention_policy": {
                "uncertainty_score_block_threshold": 0.6,
                "provenance_risk_score_block_threshold": 0.5,
                "model_identity_coverage_risk_block_threshold": 0.5,
                "headline_promotion_blocked": not clean_policy,
                "exact_acceptance_remains_final_authority": True,
            },
            "distributional_energy_schema": {"row_components": {"abstention": {"authority": "advisory_policy_only"}}},
            "model_identity_confound_check": {
                "used_model_ids": [QWEN, GEMMA31, GEMMA26],
                "missing_mandated_model_ids": [],
                "model_identity_coverage_risk": 0.0,
            },
            "provenance_risk_features": {
                "runtime_contract_ready": True,
                "critical_adversarial_flag_count": 0,
                "provenance_risk_score": 0.0,
            },
            "no_new_model_execution": True,
        },
    )
    _write_json(
        root,
        mod.EXP3315_REL_PATH,
        {
            "experiment_id": "exp3315",
            "vgb_repair_policy_ready": True,
            "proposal_budget": {
                "max_attempts_per_case": 4,
                "max_backtracks_per_case": 3,
                "max_total_attempts": len(cases) * 4,
            },
            "exact_acceptance_rules": {
                "final_acceptance_authority": "exact_verifier_only",
                "llm_judge_final_acceptance_allowed": False,
            },
            "verifier_confidence_thresholds": {
                "process_accept_confidence_min": 0.8,
                "row_uncertainty_abstain_threshold": 0.6,
                "provenance_risk_abstain_threshold": 0.5,
                "model_identity_coverage_risk_abstain_threshold": 0.5,
                "critical_adversarial_flag_count_max": 0,
            },
            "exp3316_handoff": {
                "headline_promotion_blocked_until_policy_clears": not clean_policy,
                "required_artifact_fields": ["candidate_attempts", "exact_outcome_summary"],
            },
            "no_new_model_execution": True,
        },
    )
    return cases


def _checker_versions() -> dict[str, str]:
    return {
        "live_runtime_provenance_contract": mod.RUNTIME_CONTRACT_VERSION,
        "executable_checker_path": mod.RUNTIME_CONTRACT_CHECKER_PATH,
        "checker_file_sha256": "sha256:test-checker",
        "adversarial_verify": "scripts/adversarial_verify.py@sha256:test",
        "spec_coverage": "scripts/check_spec_coverage.py@sha256:test",
        "llama_cpp_python": "0.3.16",
        "selected_python_cuda_probe": "selected_python_cuda@sha256:test",
    }


def _inventory(root: Path, *, cached: bool = True) -> dict[str, Any]:
    available = []
    missing = []
    mandated = {}
    roles = {QWEN: "moe", GEMMA31: "dense", GEMMA26: "moe"}
    for index, model_id in enumerate((QWEN, GEMMA31, GEMMA26)):
        path = root / "models" / f"{index}.gguf"
        if cached:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("gguf", encoding="utf-8")
        record = {
            "model_id": model_id,
            "hf_id": model_id,
            "name": model_id.rsplit("/", 1)[-1],
            "role": roles[model_id],
            "expected_quantization": "Q4_K_M",
            "quantization": "Q4_K_M",
            "cached": cached,
            "model_path": str(path) if cached else None,
            "cache_root": str(path.parent) if cached else None,
            "snapshot_revision": f"rev-{index}",
            "size_bytes": 1024 + index,
            "gpu": index % 2,
            "source": "test_inventory",
            "legacy_small_model": False,
        }
        mandated[model_id] = record
        if cached:
            available.append(record)
        else:
            missing.append(record | {"reason": "not_cached"})
    return {
        "cached_sota_pair_attempted": True,
        "cached_sota_pair_available": cached,
        "available_models": available,
        "missing_model_specs": missing,
        "mandated_models": mandated,
    }


def _runtime_payload(cases: list[dict[str, Any]], models: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "command": [".venv/bin/python", "scripts/experiment_3316_sota_repair_rerun_v12_runtime_clean.py"],
        "argv": [".venv/bin/python", "scripts/experiment_3316_sota_repair_rerun_v12_runtime_clean.py"],
        "cwd": "/tmp/repo",
        "pid": 3316,
        "cuda_visible_devices": "0,1",
        "selected_python_cuda": {"cuda_available": True, "cuda_device_count": 2},
        "wall_clock_duration_s": 90.0,
        "model_load_started_at": "2026-05-29T00:00:00Z",
        "model_load_finished_at": "2026-05-29T00:00:18Z",
        "model_load_duration_s": 18.0,
        "generation_started_at": "2026-05-29T00:00:18Z",
        "generation_finished_at": "2026-05-29T00:01:30Z",
        "gpu_memory_samples": [
            {"phase": "before_load", "gpus": [{"index": 0, "memory_used_mib": 4}]},
            {"phase": "after_load", "gpus": [{"index": 0, "memory_used_mib": 18152}]},
            {"phase": "after_generation", "gpus": [{"index": 0, "memory_used_mib": 18220}]},
        ],
        "per_model_load": [
            {
                "model_id": model["model_id"],
                "model_path": model["model_path"],
                "load_started_at": "2026-05-29T00:00:00Z",
                "load_finished_at": "2026-05-29T00:00:06Z",
                "model_load_duration_s": 6.0,
            }
            for model in models
        ],
        "per_case_generation": [
            {
                "case_id": case["case_id"],
                "model_id": models[index % len(models)]["model_id"],
                "started_at": "2026-05-29T00:00:18Z",
                "finished_at": "2026-05-29T00:00:20Z",
                "generated_tokens": 2,
            }
            for index, case in enumerate(cases)
        ],
    }


def _passed_probe(name: str) -> dict[str, Any]:
    return {"name": name, "passed": True, "detail": "test probe"}


def test_req_verify_3316_spec_anchor_declares_runtime_clean_rerun() -> None:
    """REQ-VERIFY-3316: OpenSpec declares the v12 rerun contract."""

    spec = (mod.REPO_ROOT / mod.SPEC_REL_PATH).read_text(encoding="utf-8")

    assert "REQ-VERIFY-3316" in spec
    assert "SCENARIO-VERIFY-3316" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.EXP3312_REL_PATH.as_posix() in spec
    assert mod.EXP3309_REL_PATH.as_posix() in spec
    assert mod.EXP3314_REL_PATH.as_posix() in spec
    assert mod.EXP3315_REL_PATH.as_posix() in spec
    assert "scripts/research_conductor.py" in spec
    for field in REQUIRED_FIELDS:
        assert field in spec


def test_scenario_verify_3316_runtime_clean_vgb_rerun_allows_headline_claim(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3316: clean live evidence can be headline-audit-ready."""

    cases = _stage_sources(tmp_path)
    inventory = _inventory(tmp_path, cached=True)

    def runner(panel: list[dict[str, Any]], models: list[dict[str, Any]], policy: dict[str, Any], seed: int) -> dict[str, Any]:
        assert [row["case_id"] for row in panel] == [row["case_id"] for row in cases]
        assert {row["model_id"] for row in models} == {QWEN, GEMMA31, GEMMA26}
        assert policy["max_attempts_per_case"] == 4
        assert seed == mod.RANDOM_SEED
        attempts: list[dict[str, Any]] = []
        for index, case in enumerate(panel):
            model = models[index % len(models)]
            attempts.append(
                {
                    "case_id": case["case_id"],
                    "attempt_index": 1,
                    "proposal_id": f"{case['case_id']}-a1",
                    "candidate_answer": case["expected_answer"],
                    "verifier_output_text": "ACCEPT",
                    "process_verifier_confidence": 0.93,
                    "model_id": model["model_id"],
                    "token_counts": {"completion_tokens": 2, "total_tokens": 10},
                }
            )
        attempts[3]["verifier_output_text"] = "REJECT"
        attempts[3]["process_verifier_confidence"] = 0.91
        attempts.insert(
            4,
            {
                "case_id": panel[3]["case_id"],
                "attempt_index": 2,
                "proposal_id": f"{panel[3]['case_id']}-a2",
                "parent_attempt_id": f"{panel[3]['case_id']}-a1",
                "candidate_answer": panel[3]["expected_answer"],
                "verifier_output_text": "ACCEPT",
                "process_verifier_confidence": 0.94,
                "model_id": models[0]["model_id"],
                "token_counts": {"completion_tokens": 2, "total_tokens": 10},
            },
        )
        return {
            "candidate_attempts": attempts,
            "gpu_mem_used_mib": 18220,
            "runtime_provenance": _runtime_payload(panel, models),
            "checker_versions": _checker_versions(),
        }

    artifact = mod.build_artifact(
        tmp_path,
        nvidia_probe=lambda: _passed_probe("nvidia_smi"),
        python_cuda_probe=lambda: _passed_probe("selected_python_cuda"),
        model_inventory=inventory,
        candidate_runner=runner,
        started_s=0.0,
        now_s=90.0,
        tests_run=["SCENARIO-VERIFY-3316"],
    )

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["repair_rerun_v12_ready"] is True
    assert artifact["repair_panel_ran"] is True
    assert artifact["headline_repair_panel_ready"] is True
    assert artifact["headline_claim_allowed"] is True
    assert artifact["runtime_provenance_clean"] is True
    assert artifact["duration_contract_passed"] is True
    assert artifact["substrate_consistency_passed"] is True
    assert artifact["panel_case_count"] == 30
    assert artifact["verified_success_count"] == 30
    assert artifact["false_accept_count"] == 0
    assert artifact["abstention_count"] == 0
    assert artifact["repair_success_rate"] == pytest.approx(1.0)
    assert len(artifact["confidence_interval"]) == 2
    assert {row["model_id"] for row in artifact["model_specs_used"]} == {QWEN, GEMMA31, GEMMA26}
    assert artifact["same_or_superset_manifest_case_hashes_recorded"] is True
    assert any(row["policy_action"] == "backtracked" for row in artifact["candidate_attempts"])
    assert artifact["candidate_results"][3]["accepted_attempt_index"] == 2
    assert artifact["distributional_ebm_sidecar"]["headline_promotion_blocked"] is False
    assert artifact["tests_run"] == ["SCENARIO-VERIFY-3316"]
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_scenario_verify_3316_blocks_when_live_substrate_is_unavailable(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3316: failed live checks write an honest blocked artifact."""

    _stage_sources(tmp_path)
    called = False

    def runner(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    artifact = mod.build_artifact(
        tmp_path,
        nvidia_probe=lambda: {"name": "nvidia_smi", "passed": False, "error": "NVML mismatch"},
        python_cuda_probe=lambda: {"name": "selected_python_cuda", "passed": False, "cuda_available": False},
        model_inventory=_inventory(tmp_path, cached=False),
        candidate_runner=runner,
        started_s=4.0,
        now_s=5.0,
    )

    assert called is False
    assert artifact["repair_rerun_v12_ready"] is False
    assert artifact["repair_panel_ran"] is False
    assert artifact["headline_repair_panel_ready"] is False
    assert artifact["headline_claim_allowed"] is False
    assert artifact["runtime_provenance_clean"] is False
    assert artifact["duration_contract_passed"] is False
    assert artifact["substrate_consistency_passed"] is False
    assert artifact["panel_case_count"] == 0
    assert artifact["verified_success_count"] == 0
    assert artifact["false_accept_count"] == 0
    assert artifact["abstention_count"] == 0
    assert artifact["repair_success_rate"] == 0.0
    assert artifact["confidence_interval"] == [0.0, 0.0]
    assert artifact["model_specs_used"] == []
    assert "nvidia_smi_unavailable" in artifact["blocked_reasons"]
    assert "selected_python_cuda_unavailable" in artifact["blocked_reasons"]
    assert "mandated_sota_gguf_unavailable" in artifact["blocked_reasons"]
    assert "honestly_blocked" in artifact["honest_verdict"]
    mod.validate_artifact(artifact)


def test_req_verify_3316_counts_false_accepts_abstentions_and_rejects_overclaims(tmp_path: Path) -> None:
    """REQ-VERIFY-3316: exact authority and VGB routing determine metrics."""

    cases = _stage_sources(tmp_path)
    inventory = _inventory(tmp_path, cached=True)

    def runner(panel: list[dict[str, Any]], models: list[dict[str, Any]], policy: dict[str, Any], seed: int) -> dict[str, Any]:
        del policy, seed
        attempts: list[dict[str, Any]] = []
        for index, case in enumerate(panel):
            candidate = case["expected_answer"]
            verifier = "ACCEPT"
            confidence = 0.93
            if index == 0:
                candidate = case["failing_candidate"]
            if index == 1:
                verifier = "ABSTAIN"
                confidence = 0.50
            attempts.append(
                {
                    "case_id": case["case_id"],
                    "attempt_index": 1,
                    "candidate_answer": candidate,
                    "verifier_output_text": verifier,
                    "process_verifier_confidence": confidence,
                    "model_id": models[index % len(models)]["model_id"],
                    "token_counts": {"completion_tokens": 1, "total_tokens": 5},
                }
            )
        return {
            "candidate_attempts": attempts,
            "gpu_mem_used_mib": 18100,
            "runtime_provenance": _runtime_payload(panel, models),
            "checker_versions": _checker_versions(),
        }

    artifact = mod.build_artifact(
        tmp_path,
        nvidia_probe=lambda: _passed_probe("nvidia_smi"),
        python_cuda_probe=lambda: _passed_probe("selected_python_cuda"),
        model_inventory=inventory,
        candidate_runner=runner,
        started_s=0.0,
        now_s=90.0,
    )

    assert artifact["repair_panel_ran"] is True
    assert artifact["headline_claim_allowed"] is False
    assert artifact["verified_success_count"] == 28
    assert artifact["false_accept_count"] == 1
    assert artifact["abstention_count"] == 2
    assert artifact["candidate_results"][0]["false_accept"] is True
    assert artifact["candidate_results"][0]["final_policy_action"] == "abstained"
    assert artifact["candidate_results"][1]["final_policy_action"] == "abstained"
    assert "false_accept_count_nonzero" in artifact["blocked_reasons"]
    assert mod.rate(2, 4) == 0.5
    assert mod.rate(1, 0) == 0.0
    assert mod.duration(5.0, 4.0) == 0.0
    assert mod.mapping_list("bad") == []
    assert mod.string_list(["x", 1, None]) == ["x", "1", "None"]
    assert mod.numeric(True, 9.0) == 9.0
    assert mod.count_value(True) == 0
    assert mod.count_value("7") == 7
    assert mod.critical_flags({"adversarial_verify_flags": [{"kind": "X", "severity": "critical"}]}) == [
        {"kind": "X", "severity": "critical", "detail": ""}
    ]
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert mod.read_json_object(bad) == {}
    assert mod.sha256_file(tmp_path / "missing.bin") is None

    saved_path = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        nvidia_probe=lambda: _passed_probe("nvidia_smi"),
        python_cuda_probe=lambda: _passed_probe("selected_python_cuda"),
        model_inventory=inventory,
        candidate_runner=runner,
        started_s=0.0,
        now_s=90.0,
        tests_run=["writer"],
    )
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved["tests_run"] == ["writer"]
    assert len(saved["reproducibility_checksum"]) == 64

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="repair_rerun_v12_ready"):
        mod.validate_artifact(saved | {"repair_rerun_v12_ready": "yes"})
    with pytest.raises(ValueError, match="panel_case_count"):
        mod.validate_artifact(saved | {"panel_case_count": True})
    with pytest.raises(ValueError, match="repair_success_rate"):
        mod.validate_artifact(saved | {"repair_success_rate": 2.0})
    with pytest.raises(ValueError, match="confidence_interval"):
        mod.validate_artifact(saved | {"confidence_interval": [0.0]})
    with pytest.raises(ValueError, match="model_specs_used"):
        mod.validate_artifact(saved | {"repair_panel_ran": True, "model_specs_used": []})
    with pytest.raises(ValueError, match="false accepts"):
        mod.validate_artifact(saved | {"headline_claim_allowed": True, "false_accept_count": 1})


def test_req_verify_3316_fail_closed_helper_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3316: helper branches fail closed on malformed evidence."""

    case = _cases(1)[0]
    bad_sources = {
        "exp3312": {
            "adversarial_verify_flags": [{"kind": "CRIT", "severity": "critical"}],
        },
        "exp3309": {"runtime_contract_ready": False, "minimum_live_duration_s": 60.0},
        "exp3314": {"distributional_repair_audit_ready": False},
        "exp3315": {"vgb_repair_policy_ready": False, "exact_acceptance_rules": {"llm_judge_final_acceptance_allowed": True}},
        "exp3302": {},
    }
    blockers = mod.preflight_blockers(
        sources=bad_sources,
        manifest_check={"valid": False},
        clean_check={"ready": False},
        nvidia={"passed": False},
        cuda={"passed": False},
        selected_models=[{"model_id": GEMMA26, "legacy_small_model": True}],
        inventory={"cached_sota_pair_attempted": False},
    )
    assert {
        "exp3312_quality_cleanup_not_ready",
        "exp3309_runtime_contract_not_ready",
        "exp3314_distributional_audit_not_ready",
        "exp3315_vgb_policy_not_ready",
        "exp3301_fixed_manifest_unavailable",
        "exp3287_clean_verifier_unavailable",
        "legacy_small_model_substitution_disallowed",
        "critical_adversarial_verify_flags_present",
        "cached_sota_pair_not_attempted",
    } <= set(blockers)
    assert mod.proposal_budget_context({}, 3)["max_total_attempts"] == 12
    assert mod.normalize_raw_attempts([case], [], {"candidate_attempts": [{"case_id": "unknown"}]}) == []

    routed, final = mod.route_and_summarize_candidates(
        [case],
        [],
        [],
        {},
        {"provenance_risk_score": 0.0, "critical_adversarial_flag_count": 0},
        {"model_identity_coverage_risk": 0.0},
    )
    assert routed == []
    assert final[0]["final_policy_action"] == "abstained"
    abstained_route, abstained_final = mod.route_and_summarize_candidates(
        [case],
        [
            {
                "case_id": case["case_id"],
                "attempt_index": 1,
                "candidate_answer": case["expected_answer"],
                "exact_check_passed": True,
                "exact_checker_type": case["exact_checker_type"],
                "calibrated_clean_verifier_decision": "accept",
                "process_verifier_confidence": 0.95,
                "false_accept": False,
            }
        ],
        [{"case_id": case["case_id"], "attempt_index": 1, "uncertainty_score": 0.0}],
        {},
        {"provenance_risk_score": 1.0, "critical_adversarial_flag_count": 0},
        {"model_identity_coverage_risk": 0.0},
    )
    assert abstained_route[0]["policy_action"] == "abstained"
    assert abstained_final[0]["final_policy_action"] == "abstained"
    assert mod.vgb_thresholds({})["process_accept_confidence_min"] == pytest.approx(0.8)

    provenance = mod.sidecar_provenance_features(
        sources=bad_sources,
        source_status={"missing": {"readable": False}},
        repair_panel_ran=True,
        preflight_blockers=["blocked"],
        runtime_provenance={"wall_clock_duration_s": 1.0},
    )
    assert provenance["provenance_risk_score"] == pytest.approx(1.0)
    assert provenance["source_duration_below_live_floor"] is True
    one_family = mod.model_identity_check([{"model_id": GEMMA26}])
    assert one_family["model_identity_coverage_risk"] >= 0.5
    assert mod.file_status(tmp_path / "missing.json")["readable"] is False
    assert mod.string_list("scalar") == []
    assert mod.numeric("bad") == 0.0
    assert mod.count_value("bad") == 0

    critical_blockers = mod.final_blocked_reasons(
        {
            "repair_panel_ran": True,
            "runtime_provenance_clean": True,
            "duration_contract_passed": True,
            "substrate_consistency_passed": True,
            "false_accept_count": 0,
            "adversarial_verify_flags": [{"kind": "CRIT", "severity": "critical"}],
        },
        [],
        {"headline_promotion_blocked": False},
    )
    assert critical_blockers == ["critical_adversarial_verify_flags_present"]

    valid = mod.build_artifact(
        tmp_path,
        nvidia_probe=lambda: {"name": "nvidia_smi", "passed": False},
        python_cuda_probe=lambda: {"name": "selected_python_cuda", "passed": False},
        model_inventory=_inventory(tmp_path, cached=False),
        started_s=1.0,
        now_s=1.0,
    )
    with pytest.raises(ValueError, match="model_specs_used"):
        mod.validate_artifact(valid | {"model_specs_used": {}})
    for field, message in [
        ("headline_repair_panel_ready", "headline_repair_panel_ready"),
        ("runtime_provenance_clean", "runtime_provenance_clean"),
        ("duration_contract_passed", "duration_contract_passed"),
        ("substrate_consistency_passed", "substrate_consistency_passed"),
    ]:
        overclaim = valid | {
            "headline_claim_allowed": True,
            "headline_repair_panel_ready": True,
            "runtime_provenance_clean": True,
            "duration_contract_passed": True,
            "substrate_consistency_passed": True,
            "false_accept_count": 0,
        }
        overclaim[field] = False
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(overclaim)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(valid | {"honest_verdict": "blocked"})
