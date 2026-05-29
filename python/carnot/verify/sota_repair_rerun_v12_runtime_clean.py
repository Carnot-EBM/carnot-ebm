"""Build the Exp 3316 SOTA repair rerun v12 runtime-clean artifact.

Spec refs: REQ-VERIFY-3316, SCENARIO-VERIFY-3316.

The rerun is a live-evidence gate, not an audit-only paper exercise. It checks
the `.306` cleanup artifacts, the runtime contract, the fixed exact manifest,
the Distributional-EBM sidecar policy, the VGB backtracking policy, CUDA
visibility, and mandated local GGUF cache state before attempting generation.
If any live substrate precondition fails, the artifact is written as honestly
blocked with zero repair denominators rather than substituting a smoke model.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from carnot.reporting import live_runtime_provenance_contract_3309 as runtime_contract
from carnot.verify import distributional_ebm_repair_uncertainty_audit_v1 as dist
from carnot.verify import headline_sota_repair_panel_v11 as panel_v11
from carnot.verify import vgb_backtracking_repair_policy_v1 as vgb


JsonDict = dict[str, Any]
Probe = Callable[[], JsonDict]
CandidateRunner = Callable[[list[JsonDict], list[JsonDict], JsonDict, int], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.sota_repair_rerun_v12_runtime_clean.v1"
EXPERIMENT_ID = "exp3316"
TASK_ID = "exp3316-gated-sota-repair-rerun-v12-runtime-clean"
ARTIFACT = "experiment_3316_sota_repair_rerun_v12_runtime_clean"
MILESTONE = "2026.05.306"
RUN_DATE = "20260529"
RANDOM_SEED = 3316

SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")
OUTPUT_REL_PATH = Path("results/experiment_3316_sota_repair_rerun_v12_runtime_clean.json")
EXP3312_REL_PATH = Path("results/experiment_3312_dataflip_garak_quality_clean_rerun_v4.json")
EXP3309_REL_PATH = Path("results/experiment_3309_live_runtime_provenance_contract_v1.json")
EXP3314_REL_PATH = Path("results/experiment_3314_distributional_ebm_repair_uncertainty_audit_v1.json")
EXP3315_REL_PATH = Path("results/experiment_3315_vgb_backtracking_repair_policy_v1.json")
EXP3302_REL_PATH = Path("results/experiment_3302_headline_sota_repair_panel_v11.json")
EXP3301_REL_PATH = panel_v11.EXP3301_REL_PATH
EXP3287_REL_PATH = panel_v11.EXP3287_REL_PATH
PANEL_CASES_REL_PATH = panel_v11.PANEL_CASES_REL_PATH

RUNTIME_CONTRACT_VERSION = runtime_contract.CONTRACT_VERSION
RUNTIME_CONTRACT_CHECKER_PATH = runtime_contract.EXECUTABLE_CHECKER_PATH
MIN_PANEL_CASES = 30
MANDATED_MODEL_IDS = panel_v11.MANDATED_MODEL_IDS
SUCCESS_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
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
)

DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3316_sota_repair_rerun_v12_runtime_clean.py -q -o addopts=''",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run --source=python/carnot/verify/sota_repair_rerun_v12_runtime_clean.py -m pytest -o addopts='' tests/python/test_experiment_3316_sota_repair_rerun_v12_runtime_clean.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/sota_repair_rerun_v12_runtime_clean.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)


case_hash = panel_v11.case_hash
sha256_text = panel_v11.sha256_text


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    nvidia_probe: Probe | None = None,
    python_cuda_probe: Probe | None = None,
    model_inventory: Mapping[str, Any] | None = None,
    candidate_runner: CandidateRunner | None = None,
    random_seed: int = RANDOM_SEED,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """SCENARIO-VERIFY-3316: run the gated repair panel or write a blocked artifact."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    sources = load_sources(root_path)
    exp3301_load = panel_v11.read_json_object(root_path / EXP3301_REL_PATH)
    exp3287_load = panel_v11.read_json_object(root_path / EXP3287_REL_PATH)
    manifest_path = panel_v11.manifest_cases_path(root_path, exp3301_load.payload)
    manifest_rows = panel_v11.read_jsonl_objects(manifest_path)
    manifest_check = panel_v11.validate_manifest_contract(exp3301_load, manifest_rows)
    clean_check = panel_v11.clean_verifier_contract(exp3287_load)
    nvidia = panel_v11.normalize_precondition((nvidia_probe or panel_v11.default_nvidia_smi_probe)())
    cuda = panel_v11.normalize_precondition((python_cuda_probe or panel_v11.selected_python_cuda_probe)())
    inventory = dict(model_inventory) if model_inventory is not None else panel_v11.resolve_mandated_model_inventory(root_path)
    selected_models = mapping_list(inventory.get("available_models"))
    source_status = source_artifacts(root_path, manifest_path)
    initial_blockers = preflight_blockers(
        sources=sources,
        manifest_check=manifest_check,
        clean_check=clean_check,
        nvidia=nvidia,
        cuda=cuda,
        selected_models=selected_models,
        inventory=inventory,
    )

    runner_payload: JsonDict = {}
    evaluated_attempts: list[JsonDict] = []
    candidate_attempts: list[JsonDict] = []
    candidate_results: list[JsonDict] = []
    runner_error = ""
    if not initial_blockers:
        try:
            runner = candidate_runner or run_llama_repair_panel
            runner_payload = mapping(
                runner(manifest_rows, selected_models, proposal_budget_context(sources["exp3315"], len(manifest_rows)), int(random_seed))
            )
            evaluated_attempts = evaluate_attempts(manifest_rows, selected_models, runner_payload)
        except Exception as exc:  # pragma: no cover - defensive live-run boundary.
            runner_error = f"{type(exc).__name__}: {exc}"
            initial_blockers.append("repair_candidate_runner_failed")

    repair_panel_ran = bool(evaluated_attempts) and not runner_error
    runtime_provenance = runtime_provenance_from_payload(runner_payload, repair_panel_ran)
    checker_versions = checker_versions_from_sources(root_path, sources, runner_payload)
    model_specs_used = (
        model_specs_used_from_runtime(selected_models, runtime_provenance, evaluated_attempts)
        if repair_panel_ran
        else []
    )
    provenance_features = sidecar_provenance_features(
        sources=sources,
        source_status=source_status,
        repair_panel_ran=repair_panel_ran,
        preflight_blockers=initial_blockers,
        runtime_provenance=runtime_provenance,
    )
    model_check = model_identity_check(model_specs_used)
    attempt_scores = dist.repair_row_scores(evaluated_attempts, provenance_features, model_check)
    for score, attempt in zip(attempt_scores, evaluated_attempts, strict=False):
        score["attempt_index"] = attempt.get("attempt_index")
    if repair_panel_ran:
        candidate_attempts, candidate_results = route_and_summarize_candidates(
            manifest_rows,
            evaluated_attempts,
            attempt_scores,
            sources["exp3315"],
            provenance_features,
            model_check,
        )
    final_sidecar_scores = dist.repair_row_scores(candidate_results, provenance_features, model_check)
    sidecar_policy = dist.uncertainty_abstention_policy(
        final_sidecar_scores,
        provenance_features,
        model_check,
        source_status_readable(source_status),
    )
    metrics = final_metrics(candidate_results)
    panel_case_count = len(candidate_results) if repair_panel_ran else 0
    tokens_generated = count_generated_tokens(evaluated_attempts, runner_payload, runtime_provenance) if repair_panel_ran else 0
    gpu_mem_used_mib = count_value(runner_payload.get("gpu_mem_used_mib")) if repair_panel_ran else 0
    baseline_hashes = string_list(sources["exp3302"].get("manifest_case_hashes"))
    current_hashes = [str(row.get("case_hash") or "") for row in candidate_results]
    same_or_superset = bool(baseline_hashes) and set(baseline_hashes).issubset(set(current_hashes))
    exact_authority_preserved = bool(candidate_results) and all(
        row.get("exact_acceptance_authority") is True or row.get("final_policy_action") == "abstained"
        for row in candidate_results
    )
    duration_s = duration(started, time.perf_counter() if now_s is None else float(now_s))

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3316", "SCENARIO-VERIFY-3316"],
        "evidence_tier": "headline_live" if repair_panel_ran else "blocked_preflight",
        "repair_rerun_v12_ready": False,
        "repair_panel_ran": repair_panel_ran,
        "headline_repair_panel_ready": False,
        "headline_claim_allowed": False,
        "runtime_provenance_clean": False,
        "duration_contract_passed": False,
        "substrate_consistency_passed": False,
        "panel_case_count": panel_case_count,
        "verified_success_count": metrics["verified_success_count"],
        "false_accept_count": metrics["false_accept_count"],
        "abstention_count": metrics["abstention_count"],
        "repair_success_rate": metrics["repair_success_rate"],
        "false_accept_rate": metrics["false_accept_rate"],
        "metric_lineage": metric_lineage(metrics, panel_case_count),
        "confidence_interval": panel_v11.wilson_ci95(metrics["verified_success_count"], panel_case_count),
        "model_specs_used": model_specs_used,
        "models_used": model_specs_used,
        "model_specs": model_specs_from_inventory(inventory),
        "missing_model_specs": mapping_list(inventory.get("missing_model_specs")),
        "preconditions_checked": {
            "exp3312_quality_cleanup": prerequisite_exp3312(sources["exp3312"]),
            "exp3309_runtime_contract": prerequisite_exp3309(sources["exp3309"]),
            "exp3314_distributional_audit": prerequisite_exp3314(sources["exp3314"]),
            "exp3315_vgb_policy": prerequisite_exp3315(sources["exp3315"]),
            "exp3301_fixed_manifest": {"passed": manifest_check.get("valid") is True, **dict(manifest_check)},
            "exp3287_clean_verifier": {"passed": clean_check.get("ready") is True, **dict(clean_check)},
            "nvidia_smi": nvidia,
            "selected_python_cuda": cuda,
            "mandated_gguf_cache": model_cache_check(inventory, selected_models),
        },
        "candidate_attempts": candidate_attempts,
        "candidate_results": candidate_results,
        "exact_outcome_summary": exact_outcome_summary(candidate_results, candidate_attempts),
        "distributional_ebm_sidecar": {
            "repair_row_scores": final_sidecar_scores,
            "uncertainty_abstention_policy": sidecar_policy,
            "headline_promotion_blocked": sidecar_policy.get("headline_promotion_blocked") is True,
            "model_identity_confound_check": model_check,
            "provenance_risk_features": provenance_features,
        },
        "vgb_policy_summary": vgb_policy_summary(sources["exp3315"]),
        "runtime_provenance": runtime_provenance,
        "runtime_contract_check": {},
        "checker_versions": checker_versions,
        "tokens_generated": tokens_generated,
        "gpu_mem_used_mib": gpu_mem_used_mib,
        "same_or_superset_manifest_case_hashes_recorded": same_or_superset,
        "manifest_case_hashes": current_hashes,
        "source_manifest_case_hashes": baseline_hashes,
        "exact_checker_types": sorted({str(row.get("exact_checker_type") or "") for row in candidate_results if row.get("exact_checker_type")}),
        "exact_acceptance_authority_preserved": exact_authority_preserved,
        "no_legacy_small_model_substitution": not any(row.get("legacy_small_model") is True for row in model_specs_used),
        "source_artifacts": source_status,
        "blocked_reasons": [],
        "runner_error": runner_error,
        "duration_s": duration_s,
        "inference_substrate": "live_local_sota_gguf_repair_v12_runtime_clean" if repair_panel_ran else "blocked_preflight_no_model_calls",
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "scripts_research_conductor_modified": False,
        "no_push": True,
    }
    runtime_check = runtime_contract.check_runtime_evidence_artifact(artifact)
    artifact["runtime_contract_check"] = runtime_check
    artifact["duration_contract_passed"] = repair_panel_ran and runtime_check.get("duration_contract_passed") is True
    artifact["runtime_provenance_clean"] = (
        repair_panel_ran
        and runtime_check.get("runtime_contract_passed") is True
        and not critical_flags(artifact)
    )
    artifact["substrate_consistency_passed"] = substrate_consistency_passed(
        artifact,
        same_or_superset=same_or_superset,
        exact_authority_preserved=exact_authority_preserved,
    )
    artifact["headline_repair_panel_ready"] = (
        repair_panel_ran
        and panel_case_count >= max(MIN_PANEL_CASES, count_value(sources["exp3302"].get("panel_case_count")))
        and artifact["runtime_provenance_clean"] is True
        and artifact["duration_contract_passed"] is True
        and artifact["substrate_consistency_passed"] is True
        and tokens_generated > 0
        and gpu_mem_used_mib > 0
    )
    artifact["blocked_reasons"] = final_blocked_reasons(artifact, initial_blockers, sidecar_policy)
    artifact["headline_claim_allowed"] = (
        artifact["headline_repair_panel_ready"] is True
        and metrics["false_accept_count"] == 0
        and sidecar_policy.get("headline_promotion_blocked") is not True
        and sources["exp3315"].get("vgb_repair_policy_ready") is True
        and not critical_flags(artifact)
        and not artifact["blocked_reasons"]
    )
    artifact["repair_rerun_v12_ready"] = (
        artifact["headline_repair_panel_ready"] is True
        and metrics["false_accept_count"] == 0
        and sidecar_policy.get("headline_promotion_blocked") is not True
    )
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    nvidia_probe: Probe | None = None,
    python_cuda_probe: Probe | None = None,
    model_inventory: Mapping[str, Any] | None = None,
    candidate_runner: CandidateRunner | None = None,
    random_seed: int = RANDOM_SEED,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3316 terminal artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(
        root_path,
        nvidia_probe=nvidia_probe,
        python_cuda_probe=python_cuda_probe,
        model_inventory=model_inventory,
        candidate_runner=candidate_runner,
        random_seed=random_seed,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_sources(root: Path) -> JsonDict:
    """Read the source artifacts that gate the v12 live rerun."""

    return {
        "exp3312": read_json_object(root / EXP3312_REL_PATH),
        "exp3309": read_json_object(root / EXP3309_REL_PATH),
        "exp3314": read_json_object(root / EXP3314_REL_PATH),
        "exp3315": read_json_object(root / EXP3315_REL_PATH),
        "exp3302": read_json_object(root / EXP3302_REL_PATH),
    }


def preflight_blockers(
    *,
    sources: Mapping[str, Mapping[str, Any]],
    manifest_check: Mapping[str, Any],
    clean_check: Mapping[str, Any],
    nvidia: Mapping[str, Any],
    cuda: Mapping[str, Any],
    selected_models: Sequence[Mapping[str, Any]],
    inventory: Mapping[str, Any],
) -> list[str]:
    """Translate failed prerequisites into stable blocked-reason strings."""

    blockers: list[str] = []
    if prerequisite_exp3312(sources["exp3312"])["passed"] is not True:
        blockers.append("exp3312_quality_cleanup_not_ready")
    if prerequisite_exp3309(sources["exp3309"])["passed"] is not True:
        blockers.append("exp3309_runtime_contract_not_ready")
    if prerequisite_exp3314(sources["exp3314"])["passed"] is not True:
        blockers.append("exp3314_distributional_audit_not_ready")
    if prerequisite_exp3315(sources["exp3315"])["passed"] is not True:
        blockers.append("exp3315_vgb_policy_not_ready")
    if manifest_check.get("valid") is not True:
        blockers.append("exp3301_fixed_manifest_unavailable")
    if clean_check.get("ready") is not True:
        blockers.append("exp3287_clean_verifier_unavailable")
    if nvidia.get("passed") is not True:
        blockers.append("nvidia_smi_unavailable")
    if cuda.get("passed") is not True:
        blockers.append("selected_python_cuda_unavailable")
    if not selected_models:
        blockers.append("mandated_sota_gguf_unavailable")
    if len(selected_models) < len(MANDATED_MODEL_IDS):
        blockers.append("missing_mandated_gguf_specs")
    if any(row.get("legacy_small_model") is True for row in selected_models):
        blockers.append("legacy_small_model_substitution_disallowed")
    if critical_flags(sources["exp3312"]):
        blockers.append("critical_adversarial_verify_flags_present")
    if not inventory.get("cached_sota_pair_attempted"):
        blockers.append("cached_sota_pair_not_attempted")
    return sorted(set(blockers))


def prerequisite_exp3312(payload: Mapping[str, Any]) -> JsonDict:
    """Check DataFlip and quality cleanup gates without using them as repair evidence."""

    return {
        "passed": payload.get("dataflip_gate_passed") is True
        and payload.get("quality_flags_cleared") is True
        and payload.get("runtime_provenance_clean") is True,
        "dataflip_gate_passed": payload.get("dataflip_gate_passed") is True,
        "quality_flags_cleared": payload.get("quality_flags_cleared") is True,
        "runtime_provenance_clean": payload.get("runtime_provenance_clean") is True,
    }


def prerequisite_exp3309(payload: Mapping[str, Any]) -> JsonDict:
    """Check the runtime contract artifact before making live claims."""

    return {
        "passed": payload.get("runtime_contract_ready") is True
        and payload.get("contract_version") == RUNTIME_CONTRACT_VERSION
        and numeric(payload.get("minimum_live_duration_s")) >= runtime_contract.MINIMUM_LIVE_DURATION_S,
        "runtime_contract_ready": payload.get("runtime_contract_ready") is True,
        "contract_version": str(payload.get("contract_version") or ""),
        "minimum_live_duration_s": numeric(payload.get("minimum_live_duration_s")),
    }


def prerequisite_exp3314(payload: Mapping[str, Any]) -> JsonDict:
    """Check that the Distributional-EBM audit is available for v12 handoff."""

    policy = mapping(payload.get("uncertainty_abstention_policy"))
    return {
        "passed": payload.get("distributional_repair_audit_ready") is True
        and policy.get("exact_acceptance_remains_final_authority") is True,
        "distributional_repair_audit_ready": payload.get("distributional_repair_audit_ready") is True,
        "exact_acceptance_remains_final_authority": policy.get("exact_acceptance_remains_final_authority") is True,
    }


def prerequisite_exp3315(payload: Mapping[str, Any]) -> JsonDict:
    """Check that the VGB policy is ready and preserves exact final authority."""

    exact = mapping(payload.get("exact_acceptance_rules"))
    return {
        "passed": payload.get("vgb_repair_policy_ready") is True
        and exact.get("final_acceptance_authority") == "exact_verifier_only"
        and exact.get("llm_judge_final_acceptance_allowed") is False,
        "vgb_repair_policy_ready": payload.get("vgb_repair_policy_ready") is True,
        "final_acceptance_authority": str(exact.get("final_acceptance_authority") or ""),
        "llm_judge_final_acceptance_allowed": exact.get("llm_judge_final_acceptance_allowed"),
    }


def model_cache_check(inventory: Mapping[str, Any], selected_models: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose mandated GGUF cache state and explicitly avoid legacy fallbacks."""

    return {
        "passed": bool(selected_models),
        "cached_sota_pair_attempted": inventory.get("cached_sota_pair_attempted") is True,
        "cached_sota_pair_available": inventory.get("cached_sota_pair_available") is True,
        "available_model_ids": [str(row.get("model_id") or row.get("hf_id") or "") for row in selected_models],
        "missing_model_ids": [str(row.get("model_id") or row.get("hf_id") or "") for row in mapping_list(inventory.get("missing_model_specs"))],
        "legacy_small_model_used": any(row.get("legacy_small_model") is True for row in selected_models),
    }


def proposal_budget_context(exp3315: Mapping[str, Any], case_count: int) -> JsonDict:
    """Return the VGB proposal budget passed to the candidate runner."""

    budget = mapping(exp3315.get("proposal_budget"))
    if not budget:
        budget = vgb.proposal_budget(case_count)
    return budget


def evaluate_attempts(
    cases: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    runner_payload: Mapping[str, Any],
) -> list[JsonDict]:
    """Attach exact-check and clean-verifier fields to raw candidate attempts."""

    raw_attempts = normalize_raw_attempts(cases, models, runner_payload)
    case_by_id = {str(case.get("case_id") or ""): dict(case) for case in cases}
    evaluated: list[JsonDict] = []
    for raw in raw_attempts:
        case = case_by_id.get(str(raw.get("case_id") or ""), {})
        candidate = panel_v11.clean_candidate_answer(raw.get("candidate_answer") or raw.get("output_text") or "")
        verifier_text = str(raw.get("verifier_output_text") or raw.get("verifier_decision") or "")
        decision = panel_v11.normalize_verifier_decision(verifier_text)
        exact_passed = panel_v11.exact_check(case, candidate)
        missing_candidate = candidate == ""
        false_accept = decision == "accept" and not exact_passed
        evaluated.append(
            {
                "case_id": str(case.get("case_id") or raw.get("case_id") or ""),
                "case_hash": str(case.get("case_hash") or ""),
                "family": str(case.get("family") or ""),
                "attempt_index": count_value(raw.get("attempt_index")) or 1,
                "proposal_id": str(raw.get("proposal_id") or ""),
                "parent_attempt_id": str(raw.get("parent_attempt_id") or ""),
                "candidate_answer": candidate,
                "expected_answer": str(case.get("expected_answer") or ""),
                "failing_candidate": str(case.get("failing_candidate") or ""),
                "exact_checker_type": str(case.get("exact_checker_type") or ""),
                "exact_check_passed": exact_passed,
                "calibrated_clean_verifier_decision": decision,
                "calibrated_clean_verifier_output": verifier_text,
                "process_verifier_confidence": numeric(
                    raw.get("process_verifier_confidence"),
                    vgb.inferred_process_confidence(decision),
                ),
                "verified_success": decision == "accept" and exact_passed and not missing_candidate,
                "false_accept": false_accept,
                "abstained": decision == "abstain" or missing_candidate,
                "failure_class": panel_v11.failure_class(
                    exact_passed=exact_passed,
                    verifier_decision=decision,
                    missing_candidate=missing_candidate,
                ),
                "model_id": str(raw.get("model_id") or ""),
                "model_path": str(raw.get("model_path") or ""),
                "token_counts": mapping(raw.get("token_counts")),
                "candidate_hash": stable_hash(
                    {
                        "case_id": raw.get("case_id"),
                        "attempt_index": raw.get("attempt_index"),
                        "candidate_answer": candidate,
                    }
                ),
            }
        )
    return evaluated


def normalize_raw_attempts(
    cases: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    runner_payload: Mapping[str, Any],
) -> list[JsonDict]:
    """Accept either v12 candidate attempts or v11-style final rows."""

    raw_rows = mapping_list(runner_payload.get("candidate_attempts")) or mapping_list(runner_payload.get("rows"))
    indexed_cases = {str(case.get("case_id") or ""): index for index, case in enumerate(cases)}
    normalized: list[JsonDict] = []
    per_case_counts: defaultdict[str, int] = defaultdict(int)
    for raw in raw_rows:
        case_id = str(raw.get("case_id") or "")
        if case_id not in indexed_cases:
            continue
        per_case_counts[case_id] += 1
        model = models[indexed_cases[case_id] % len(models)] if models else {}
        attempt = dict(raw)
        attempt.setdefault("attempt_index", per_case_counts[case_id])
        attempt.setdefault("model_id", model.get("model_id") or model.get("hf_id") or "")
        attempt.setdefault("model_path", model.get("model_path") or "")
        normalized.append(attempt)
    return sorted(normalized, key=lambda row: (indexed_cases.get(str(row.get("case_id") or ""), 0), count_value(row.get("attempt_index"))))


def route_and_summarize_candidates(
    cases: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    attempt_scores: Sequence[Mapping[str, Any]],
    exp3315: Mapping[str, Any],
    provenance: Mapping[str, Any],
    model_check: Mapping[str, Any],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Apply VGB routing to candidate attempts and return attempts plus final rows."""

    thresholds = vgb_thresholds(exp3315)
    max_attempts = count_value(mapping(exp3315.get("proposal_budget")).get("max_attempts_per_case")) or vgb.MAX_ATTEMPTS_PER_CASE
    score_by_key = {
        (str(score.get("case_id") or ""), count_value(score.get("attempt_index")) or index + 1): dict(score)
        for index, score in enumerate(attempt_scores)
    }
    attempts_by_case: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        attempts_by_case[str(attempt.get("case_id") or "")].append(attempt)
    routed_attempts: list[JsonDict] = []
    final_rows: list[JsonDict] = []
    for case in cases:
        case_id = str(case.get("case_id") or "")
        case_attempts = sorted(attempts_by_case.get(case_id, []), key=lambda row: count_value(row.get("attempt_index")))
        accepted: JsonDict | None = None
        last_attempt: JsonDict | None = None
        case_false_accept = False
        for attempt in case_attempts:
            key = (case_id, count_value(attempt.get("attempt_index")) or 1)
            score = mapping(score_by_key.get(key))
            route = vgb.route_candidate_attempt(
                {
                    **dict(attempt),
                    "uncertainty_score": numeric(score.get("uncertainty_score")),
                    "provenance_risk_score": numeric(provenance.get("provenance_risk_score")),
                    "model_identity_coverage_risk": numeric(model_check.get("model_identity_coverage_risk")),
                    "critical_adversarial_flag_count": count_value(provenance.get("critical_adversarial_flag_count")),
                },
                thresholds=thresholds,
                max_attempts_per_case=max_attempts,
            )
            logged = {**dict(attempt), **route, "action_reason_codes": route["reason_codes"]}
            routed_attempts.append(logged)
            last_attempt = logged
            case_false_accept = case_false_accept or attempt.get("false_accept") is True
            if route["policy_action"] == "accepted":
                accepted = logged
                break
            if route["policy_action"] == "abstained":
                break
        final = accepted or last_attempt
        if final is None:
            final_rows.append(empty_final_row(case))
            continue
        final_rows.append(final_candidate_row(case, final, accepted is not None, case_false_accept))
    return routed_attempts, final_rows


def final_candidate_row(case: Mapping[str, Any], attempt: Mapping[str, Any], accepted: bool, case_false_accept: bool) -> JsonDict:
    """Collapse one case's routed attempts into the panel outcome row."""

    return {
        "case_id": str(case.get("case_id") or ""),
        "case_hash": str(case.get("case_hash") or ""),
        "family": str(case.get("family") or ""),
        "model_id": str(attempt.get("model_id") or ""),
        "model_path": str(attempt.get("model_path") or ""),
        "candidate_answer": str(attempt.get("candidate_answer") or ""),
        "expected_answer": str(case.get("expected_answer") or ""),
        "failing_candidate": str(case.get("failing_candidate") or ""),
        "exact_checker_type": str(case.get("exact_checker_type") or ""),
        "exact_check_passed": attempt.get("exact_check_passed") is True,
        "calibrated_clean_verifier_decision": str(attempt.get("calibrated_clean_verifier_decision") or ""),
        "calibrated_clean_verifier_output": str(attempt.get("calibrated_clean_verifier_output") or ""),
        "process_verifier_confidence": numeric(attempt.get("process_verifier_confidence")),
        "verified_success": accepted,
        "false_accept": case_false_accept,
        "abstained": not accepted,
        "final_policy_action": "accepted" if accepted else "abstained",
        "accepted_attempt_index": count_value(attempt.get("attempt_index")) if accepted else None,
        "exact_acceptance_authority": accepted and attempt.get("exact_check_passed") is True,
        "token_counts": mapping(attempt.get("token_counts")),
        "candidate_hash": str(attempt.get("candidate_hash") or ""),
    }


def empty_final_row(case: Mapping[str, Any]) -> JsonDict:
    """Return an abstained row when the runner omitted a manifest case."""

    return {
        "case_id": str(case.get("case_id") or ""),
        "case_hash": str(case.get("case_hash") or ""),
        "family": str(case.get("family") or ""),
        "candidate_answer": "",
        "expected_answer": str(case.get("expected_answer") or ""),
        "failing_candidate": str(case.get("failing_candidate") or ""),
        "exact_checker_type": str(case.get("exact_checker_type") or ""),
        "exact_check_passed": False,
        "calibrated_clean_verifier_decision": "abstain",
        "verified_success": False,
        "false_accept": False,
        "abstained": True,
        "final_policy_action": "abstained",
        "accepted_attempt_index": None,
        "exact_acceptance_authority": False,
    }


def vgb_thresholds(exp3315: Mapping[str, Any]) -> JsonDict:
    """Use Exp 3315 thresholds, falling back to the executable policy defaults."""

    thresholds = mapping(exp3315.get("verifier_confidence_thresholds"))
    if thresholds:
        return thresholds
    return vgb.default_verifier_confidence_thresholds({})


def final_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute panel-level success, false-accept, and abstention metrics."""

    n_rows = len(rows)
    successes = sum(row.get("verified_success") is True for row in rows)
    false_accepts = sum(row.get("false_accept") is True for row in rows)
    abstentions = sum(row.get("abstained") is True for row in rows)
    return {
        "verified_success_count": successes,
        "false_accept_count": false_accepts,
        "abstention_count": abstentions,
        "repair_success_rate": rate(successes, n_rows),
        "false_accept_rate": rate(false_accepts, n_rows),
    }


def metric_lineage(metrics: Mapping[str, Any], panel_case_count: int) -> JsonDict:
    """Record independent formulas for runtime-contract tautology checks."""

    return {
        "repair_success_rate": {
            "numerator": count_value(metrics.get("verified_success_count")),
            "denominator": panel_case_count,
            "source_filter": "final_policy_action=accepted and exact_check_passed=true",
            "source_row_count": panel_case_count,
            "calculation_function": "verified_success_count/panel_case_count",
            "source_artifact_sha256": "current_exp3316_candidate_results",
        },
        "false_accept_rate": {
            "numerator": count_value(metrics.get("false_accept_count")),
            "denominator": panel_case_count,
            "source_filter": "any_attempt clean_verifier_accepts and exact_check_passed=false",
            "source_row_count": panel_case_count,
            "calculation_function": "false_accept_count/panel_case_count",
            "source_artifact_sha256": "current_exp3316_candidate_attempts",
        },
    }


def sidecar_provenance_features(
    *,
    sources: Mapping[str, Mapping[str, Any]],
    source_status: Mapping[str, Any],
    repair_panel_ran: bool,
    preflight_blockers: Sequence[str],
    runtime_provenance: Mapping[str, Any],
) -> JsonDict:
    """Build bounded provenance-risk features for the v12 sidecar."""

    flags = critical_flags(sources["exp3312"])
    source_readable = source_status_readable(source_status)
    duration_floor = numeric(sources["exp3309"].get("minimum_live_duration_s"), runtime_contract.MINIMUM_LIVE_DURATION_S)
    wall_duration = numeric(runtime_provenance.get("wall_clock_duration_s"))
    duration_below_floor = repair_panel_ran and wall_duration < duration_floor
    score = 0.0
    if not source_readable:
        score += 0.25
    if preflight_blockers:
        score += 0.50
    if flags:
        score += 0.35
    if duration_below_floor:
        score += 0.25
    return {
        "source_artifacts_readable": source_readable,
        "source_provenance_clean": not flags,
        "substrate_consistency_passed": not preflight_blockers,
        "critical_adversarial_flag_count": len(flags),
        "critical_adversarial_flags": flags,
        "runtime_contract_ready": sources["exp3309"].get("runtime_contract_ready") is True,
        "live_duration_floor_s": duration_floor,
        "source_duration_s": wall_duration,
        "source_duration_below_live_floor": duration_below_floor,
        "substrate_blocker_modes": list(preflight_blockers),
        "provenance_risk_score": round(min(1.0, score), 6),
    }


def model_identity_check(model_specs_used: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report mandated model coverage for headline-claim confound checks."""

    used_ids = unique_strings(row.get("model_id") or row.get("hf_id") for row in model_specs_used)
    missing_ids = [model_id for model_id in MANDATED_MODEL_IDS if model_id not in used_ids]
    used_families = sorted({dist.model_family(model_id) for model_id in used_ids})
    missing_families = sorted({dist.model_family(model_id) for model_id in missing_ids})
    coverage_risk = 1.0 if not used_ids else rate(len(missing_ids), len(MANDATED_MODEL_IDS))
    if used_ids and len(used_families) <= 1 and missing_ids:
        coverage_risk = max(coverage_risk, 0.5)
    return {
        "confound_detected": coverage_risk >= dist.MODEL_IDENTITY_COVERAGE_RISK_BLOCK_THRESHOLD,
        "used_model_ids": used_ids,
        "missing_mandated_model_ids": missing_ids,
        "mandated_model_ids": list(MANDATED_MODEL_IDS),
        "used_model_families": used_families,
        "missing_model_families": missing_families,
        "model_identity_coverage_risk": round(min(1.0, coverage_risk), 6),
    }


def model_specs_used_from_runtime(
    selected_models: Sequence[Mapping[str, Any]],
    runtime_provenance: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Combine cache provenance, load timing, and generated-token counts per model."""

    load_by_id = {str(row.get("model_id") or ""): row for row in mapping_list(runtime_provenance.get("per_model_load"))}
    generated_by_model: defaultdict[str, int] = defaultdict(int)
    for attempt in attempts:
        generated_by_model[str(attempt.get("model_id") or "")] += count_value(mapping(attempt.get("token_counts")).get("completion_tokens"))
    specs: list[JsonDict] = []
    for model in selected_models:
        model_id = str(model.get("model_id") or model.get("hf_id") or "")
        load = mapping(load_by_id.get(model_id))
        specs.append(
            {
                "model_id": model_id,
                "hf_id": str(model.get("hf_id") or model_id),
                "name": str(model.get("name") or model_id),
                "role": str(model.get("role") or ""),
                "model_path": str(model.get("model_path") or ""),
                "cache_root": str(model.get("cache_root") or Path(str(model.get("model_path") or "")).parent),
                "snapshot_revision": str(model.get("snapshot_revision") or ""),
                "size_bytes": count_value(model.get("size_bytes")),
                "quantization": str(model.get("quantization") or model.get("expected_quantization") or ""),
                "gpu": count_value(model.get("gpu")),
                "source": str(model.get("source") or ""),
                "legacy_small_model": model.get("legacy_small_model") is True,
                "load_started_at": str(load.get("load_started_at") or runtime_provenance.get("model_load_started_at") or ""),
                "load_finished_at": str(load.get("load_finished_at") or runtime_provenance.get("model_load_finished_at") or ""),
                "model_load_duration_s": numeric(load.get("model_load_duration_s"), numeric(runtime_provenance.get("model_load_duration_s"))),
                "generated_tokens": generated_by_model[model_id],
            }
        )
    return specs


def model_specs_from_inventory(inventory: Mapping[str, Any]) -> JsonDict:
    """Record all mandated model cache facts, including missing GGUFs."""

    return {
        "runtime": "llama_cpp_local_gguf_only",
        "mandated_model_ids": list(MANDATED_MODEL_IDS),
        "cached_sota_pair_attempted": inventory.get("cached_sota_pair_attempted") is True,
        "cached_sota_pair_available": inventory.get("cached_sota_pair_available") is True,
        "available_model_count": len(mapping_list(inventory.get("available_models"))),
        "missing_model_count": len(mapping_list(inventory.get("missing_model_specs"))),
        "mandated_models": mapping(inventory.get("mandated_models")),
        "legacy_small_model_policy": "disallowed_for_headline_evidence",
    }


def runtime_provenance_from_payload(runner_payload: Mapping[str, Any], repair_panel_ran: bool) -> JsonDict:
    """Return runtime provenance, adding command context even for blocked runs."""

    runtime = mapping(runner_payload.get("runtime_provenance"))
    runtime.setdefault("command", [".venv/bin/python", "scripts/experiment_3316_sota_repair_rerun_v12_runtime_clean.py"])
    runtime.setdefault("argv", [".venv/bin/python", "scripts/experiment_3316_sota_repair_rerun_v12_runtime_clean.py"])
    runtime.setdefault("cwd", str(REPO_ROOT))
    runtime.setdefault("pid", os.getpid())
    runtime.setdefault("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if not repair_panel_ran:
        runtime.setdefault("wall_clock_duration_s", 0.0)
    return runtime


def checker_versions_from_sources(root: Path, sources: Mapping[str, Mapping[str, Any]], runner_payload: Mapping[str, Any]) -> JsonDict:
    """Collect checker versions required by the runtime contract."""

    versions = mapping(runner_payload.get("checker_versions")) or mapping(sources["exp3312"].get("checker_versions"))
    defaults = {
        "live_runtime_provenance_contract": RUNTIME_CONTRACT_VERSION,
        "executable_checker_path": RUNTIME_CONTRACT_CHECKER_PATH,
        "checker_file_sha256": sha256_file(root / RUNTIME_CONTRACT_CHECKER_PATH) or "unavailable",
        "adversarial_verify": "scripts/adversarial_verify.py@unavailable",
        "spec_coverage": "scripts/check_spec_coverage.py@unavailable",
        "llama_cpp_python": "unavailable",
        "selected_python_cuda_probe": "selected_python_cuda@unavailable",
    }
    return {key: str(versions.get(key) or value) for key, value in defaults.items()}


def count_generated_tokens(
    attempts: Sequence[Mapping[str, Any]],
    runner_payload: Mapping[str, Any],
    runtime_provenance: Mapping[str, Any],
) -> int:
    """Count generated tokens from attempts, runtime rows, or a runner total."""

    attempt_total = sum(count_value(mapping(row.get("token_counts")).get("completion_tokens")) for row in attempts)
    runtime_total = sum(count_value(row.get("generated_tokens")) for row in mapping_list(runtime_provenance.get("per_case_generation")))
    return attempt_total or runtime_total or count_value(runner_payload.get("tokens_generated"))


def substrate_consistency_passed(
    artifact: Mapping[str, Any],
    *,
    same_or_superset: bool,
    exact_authority_preserved: bool,
) -> bool:
    """Check repair substrate facts that must hold before headline promotion."""

    return (
        artifact.get("repair_panel_ran") is True
        and artifact.get("runtime_provenance_clean") is True
        and same_or_superset
        and exact_authority_preserved
        and artifact.get("no_legacy_small_model_substitution") is True
        and count_value(artifact.get("tokens_generated")) > 0
        and count_value(artifact.get("gpu_mem_used_mib")) > 0
        and bool(mapping_list(artifact.get("model_specs_used")))
    )


def final_blocked_reasons(
    artifact: Mapping[str, Any],
    initial_blockers: Sequence[str],
    sidecar_policy: Mapping[str, Any],
) -> list[str]:
    """Merge preflight and post-run blockers into a stable list."""

    blockers = set(initial_blockers)
    if artifact.get("repair_panel_ran") is not True:
        blockers.add("repair_panel_not_run")
    if artifact.get("runtime_provenance_clean") is not True:
        blockers.add("runtime_provenance_not_clean")
    if artifact.get("duration_contract_passed") is not True:
        blockers.add("duration_contract_not_passed")
    if artifact.get("substrate_consistency_passed") is not True:
        blockers.add("substrate_consistency_not_passed")
    if count_value(artifact.get("false_accept_count")) > 0:
        blockers.add("false_accept_count_nonzero")
    if sidecar_policy.get("headline_promotion_blocked") is True:
        blockers.add("distributional_or_vgb_policy_blocks_headline")
    if critical_flags(artifact):
        blockers.add("critical_adversarial_verify_flags_present")
    return sorted(blockers)


def exact_outcome_summary(rows: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize exact outcomes for auditors without recomputing from prose."""

    return {
        "candidate_result_count": len(rows),
        "candidate_attempt_count": len(attempts),
        "exact_check_passed_count": sum(row.get("exact_check_passed") is True for row in rows),
        "verified_success_count": sum(row.get("verified_success") is True for row in rows),
        "false_accept_count": sum(row.get("false_accept") is True for row in rows),
        "abstention_count": sum(row.get("abstained") is True for row in rows),
        "llm_judge_dependency_count": 0,
        "final_acceptance_authority": "exact_verifier_only",
    }


def vgb_policy_summary(exp3315: Mapping[str, Any]) -> JsonDict:
    """Expose the VGB handoff that governed candidate routing."""

    return {
        "vgb_repair_policy_ready": exp3315.get("vgb_repair_policy_ready") is True,
        "proposal_budget": mapping(exp3315.get("proposal_budget")),
        "exact_acceptance_rules": mapping(exp3315.get("exact_acceptance_rules")),
        "verifier_confidence_thresholds": vgb_thresholds(exp3315),
        "exp3316_handoff": mapping(exp3315.get("exp3316_handoff")),
    }


def source_artifacts(root: Path, manifest_path: Path) -> JsonDict:
    """Return source file status rows and checksums for reproducibility."""

    return {
        "exp3312": file_status(root / EXP3312_REL_PATH),
        "exp3309": file_status(root / EXP3309_REL_PATH),
        "exp3314": file_status(root / EXP3314_REL_PATH),
        "exp3315": file_status(root / EXP3315_REL_PATH),
        "exp3302": file_status(root / EXP3302_REL_PATH),
        "exp3301": file_status(root / EXP3301_REL_PATH),
        "exp3287": file_status(root / EXP3287_REL_PATH),
        "fixed_panel_cases_jsonl": file_status(manifest_path),
    }


def source_status_readable(source_status: Mapping[str, Any]) -> bool:
    """Return true only when every source status row is readable."""

    return all(mapping(row).get("readable") is True for row in source_status.values())


def file_status(path: Path) -> JsonDict:
    """Inspect a source artifact without treating presence as correctness."""

    if not path.is_file():
        return {"path": str(path), "present": path.exists(), "readable": False, "sha256": None}
    return {"path": str(path), "present": True, "readable": True, "sha256": sha256_file(path)}


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object, returning empty evidence on missing or malformed input."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def critical_flags(*payloads: Mapping[str, Any]) -> list[JsonDict]:
    """Collect critical adversarial/corrigendum flags from JSON-like payloads."""

    flags: list[JsonDict] = []
    seen: set[tuple[str, str]] = set()
    for payload in payloads:
        for source in (payload.get("adversarial_verify_flags"), payload.get("corrigendum_pending")):
            for row in mapping_list(source):
                flag = {
                    "kind": str(row.get("kind") or "UNKNOWN"),
                    "severity": str(row.get("severity") or "warn"),
                    "detail": str(row.get("detail") or ""),
                }
                key = (flag["kind"], flag["severity"])
                if flag["severity"] == "critical" and key not in seen:
                    seen.add(key)
                    flags.append(flag)
    return flags


def mapping(value: Any) -> JsonDict:
    """Normalize maybe-dict data to a mutable JSON dictionary."""

    return dict(value) if isinstance(value, Mapping) else {}


def mapping_list(value: Any) -> list[JsonDict]:
    """Normalize maybe-list data to a list of JSON dictionaries."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def string_list(value: Any) -> list[str]:
    """Normalize a JSON sequence to strings while dropping scalar inputs."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [str(item) for item in value]


def unique_strings(values: Any) -> list[str]:
    """Return unique non-empty strings while preserving source order."""

    out: list[str] = []
    for value in values:
        rendered = str(value or "")
        if rendered and rendered not in out:
            out.append(rendered)
    return out


def numeric(value: Any, default: float = 0.0) -> float:
    """Convert JSON scalar values to finite floats while treating bools as invalid."""

    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def count_value(value: Any) -> int:
    """Convert JSON scalar counts to integers while treating bools as invalid."""

    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def rate(numerator: int, denominator: int) -> float:
    """Return a rounded rate without letting divide-by-zero imply success."""

    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def duration(started_s: float, finished_s: float) -> float:
    """Compute non-negative elapsed seconds for deterministic tests."""

    return round(max(0.0, finished_s - started_s), 6)


def stable_hash(payload: Any) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible content."""

    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Hash an input artifact, returning None when the file is unavailable."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash artifact content while excluding volatile runtime fields."""

    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "honest_verdict", "reproducibility_checksum", "tests_run"}
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict that distinguishes readiness from blocked evidence."""

    if artifact.get("headline_claim_allowed") is True:
        return (
            "complete: headline_audit_ready; "
            f"panel_case_count={artifact['panel_case_count']}; "
            f"verified_success_count={artifact['verified_success_count']}; "
            f"false_accept_count={artifact['false_accept_count']}; "
            f"abstention_count={artifact['abstention_count']}"
        )
    if artifact.get("repair_panel_ran") is True:
        return (
            "complete: honestly_blocked_after_rerun; "
            f"panel_case_count={artifact['panel_case_count']}; "
            f"verified_success_count={artifact['verified_success_count']}; "
            f"false_accept_count={artifact['false_accept_count']}; "
            f"abstention_count={artifact['abstention_count']}; "
            f"blocked_reasons={','.join(string_list(artifact.get('blocked_reasons')))}"
        )
    return (
        "complete: honestly_blocked_no_live_panel; "
        f"blocked_reasons={','.join(string_list(artifact.get('blocked_reasons')))}"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal v12 artifact and fail closed on overclaims."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    for field in (
        "repair_rerun_v12_ready",
        "repair_panel_ran",
        "headline_repair_panel_ready",
        "headline_claim_allowed",
        "runtime_provenance_clean",
        "duration_contract_passed",
        "substrate_consistency_passed",
    ):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bool")
    for field in (
        "panel_case_count",
        "verified_success_count",
        "false_accept_count",
        "abstention_count",
    ):
        value = artifact.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    success_rate = artifact.get("repair_success_rate")
    if (
        not isinstance(success_rate, int | float)
        or isinstance(success_rate, bool)
        or not 0.0 <= float(success_rate) <= 1.0
    ):
        raise ValueError("repair_success_rate must be in [0, 1]")
    interval = artifact.get("confidence_interval")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or any(not isinstance(item, int | float) or isinstance(item, bool) for item in interval)
    ):
        raise ValueError("confidence_interval must be a two-number list")
    if not isinstance(artifact.get("model_specs_used"), list):
        raise ValueError("model_specs_used must be a list")
    if artifact.get("repair_panel_ran") is True and not artifact.get("model_specs_used"):
        raise ValueError("model_specs_used required when repair_panel_ran=true")
    if artifact.get("headline_claim_allowed") is True:
        if artifact.get("headline_repair_panel_ready") is not True:
            raise ValueError("headline_claim_allowed requires headline_repair_panel_ready")
        if artifact.get("runtime_provenance_clean") is not True:
            raise ValueError("headline_claim_allowed requires runtime_provenance_clean")
        if artifact.get("duration_contract_passed") is not True:
            raise ValueError("headline_claim_allowed requires duration_contract_passed")
        if artifact.get("substrate_consistency_passed") is not True:
            raise ValueError("headline_claim_allowed requires substrate_consistency_passed")
        if count_value(artifact.get("false_accept_count")) != 0:
            raise ValueError("headline_claim_allowed requires zero false accepts")
    if not str(artifact.get("honest_verdict") or "").startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")


def run_llama_repair_panel(
    cases: list[JsonDict],
    models: list[JsonDict],
    policy: JsonDict,
    random_seed: int,
) -> JsonDict:  # pragma: no cover - exercised only on live GGUF hosts.
    """Generate v12 repair attempts with local llama.cpp GGUF models."""

    from llama_cpp import Llama, LlamaGrammar  # noqa: PLC0415

    del policy
    started_at = utc_now()
    runtime: JsonDict = {
        "command": [".venv/bin/python", "scripts/experiment_3316_sota_repair_rerun_v12_runtime_clean.py"],
        "argv": [".venv/bin/python", "scripts/experiment_3316_sota_repair_rerun_v12_runtime_clean.py"],
        "cwd": str(REPO_ROOT),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "model_load_started_at": started_at,
        "gpu_memory_samples": [{"phase": "before_load", "gpus": []}],
        "per_model_load": [],
        "per_case_generation": [],
    }
    grammar = LlamaGrammar.from_string(panel_v11.DECISION_GRAMMAR)
    attempts: list[JsonDict] = []
    loaded: list[tuple[JsonDict, Any]] = []
    load_start = time.perf_counter()
    for model in models:
        model_load_start = utc_now()
        llm = Llama(model_path=str(model["model_path"]), n_ctx=2048, n_gpu_layers=-1, seed=int(random_seed), verbose=False)
        model_load_finished = utc_now()
        runtime["per_model_load"].append(
            {
                "model_id": model["model_id"],
                "model_path": model["model_path"],
                "load_started_at": model_load_start,
                "load_finished_at": model_load_finished,
                "model_load_duration_s": 0.0,
            }
        )
        loaded.append((model, llm))
    runtime["model_load_finished_at"] = utc_now()
    runtime["model_load_duration_s"] = duration(load_start, time.perf_counter())
    runtime["generation_started_at"] = utc_now()
    for index, case in enumerate(cases):
        model, llm = loaded[index % len(loaded)]
        case_start = utc_now()
        candidate_text, candidate_tokens = panel_v11.llama_chat(
            llm,
            system="Repair one exact fixture answer. Reply only with the repaired answer.",
            user=panel_v11.repair_prompt(case),
            max_tokens=32,
        )
        candidate_answer = panel_v11.clean_candidate_answer(candidate_text)
        verifier_text, verifier_tokens = panel_v11.llama_chat(
            llm,
            system="Reply with exactly one word: ACCEPT, REJECT, or ABSTAIN.",
            user=panel_v11.verifier_prompt(case, candidate_answer),
            max_tokens=4,
            grammar=grammar,
        )
        token_counts = panel_v11.merge_token_counts(candidate_tokens, verifier_tokens)
        case_finished = utc_now()
        runtime["per_case_generation"].append(
            {
                "case_id": case["case_id"],
                "model_id": model["model_id"],
                "started_at": case_start,
                "finished_at": case_finished,
                "generated_tokens": count_value(token_counts.get("completion_tokens")),
            }
        )
        attempts.append(
            {
                "case_id": case["case_id"],
                "attempt_index": 1,
                "candidate_answer": candidate_answer,
                "raw_candidate_output": candidate_text,
                "verifier_output_text": verifier_text,
                "model_id": model["model_id"],
                "model_path": model["model_path"],
                "token_counts": token_counts,
            }
        )
    runtime["generation_finished_at"] = utc_now()
    runtime["wall_clock_duration_s"] = sum(count_value(row.get("generated_tokens")) for row in runtime["per_case_generation"])
    runtime["gpu_memory_samples"].append({"phase": "after_generation", "gpus": []})
    return {
        "candidate_attempts": attempts,
        "runtime_provenance": runtime,
        "gpu_mem_used_mib": panel_v11.gpu_memory_used_mib(),
    }


def utc_now() -> str:  # pragma: no cover - volatile live metadata.
    """Return the current UTC timestamp in the artifact's ISO-8601 format."""

    return _dt.datetime.now(tz=_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
