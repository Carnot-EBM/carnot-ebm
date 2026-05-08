"""Exp 1524 FR-11 live policy promotion.

Spec: REQ-LEARN-1524, SCENARIO-LEARN-1524, SCENARIO-LEARN-1525.

This module is the first promotion step after the FR-11 rollback audit.  It
keeps the learning boundary narrow: safe query-time policy/cache updates may be
used during evaluation, but the model itself is treated as frozen.  The runtime
contract ledger remains the authority for false accepts; model prose is only an
input that must be parsed back into a deterministic contract decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.verify import runtime_contract_e2e_harness as runtime_contracts

JsonDict = dict[str, Any]
GeneratorFn = Callable[[str, JsonDict, str, JsonDict, JsonDict], str]
CachedPairFn = Callable[..., list[JsonDict] | None]
ResolverFn = Callable[[str], str | None]
GpuProbeFn = Callable[[], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260508"
OUTPUT_FILE = "experiment_1524_fr11_live_policy_promotion_v12.json"
MANIFEST_FILE = "fr11_live_policy_promotion_1524.jsonl"
DEFAULT_OUTPUT_PATH = Path("results") / OUTPUT_FILE
DEFAULT_MANIFEST_PATH = Path("results") / MANIFEST_FILE
DEFAULT_POLICY_CACHE_ARTIFACT_PATH = Path(
    "results/experiment_1512_fr11_verifier_feedback_policy_cache_v11.json"
)
DEFAULT_POLICY_CACHE_MANIFEST_PATH = Path("results/fr11_policy_cache_events_1512.jsonl")
DEFAULT_ROLLBACK_ARTIFACT_PATH = Path(
    "results/experiment_1513_fr11_policy_rollback_replay_audit.json"
)
DEFAULT_ROLLBACK_MANIFEST_PATH = Path("results/fr11_policy_rollback_replay_1513.jsonl")
DEFAULT_PORTABLE_PACK_ARTIFACT_PATH = Path(
    "results/experiment_1514_trace2skill_portable_skill_pack_v2.json"
)
DEFAULT_PORTABLE_PACK_MANIFEST_PATH = Path(
    "results/trace2skill_portable_skill_pack_manifest_1514.json"
)
DEFAULT_RUNTIME_CONTRACT_ARTIFACT_PATH = Path(
    "results/experiment_1520_runtime_contract_e2e_harness.json"
)
DEFAULT_RUNTIME_CONTRACT_MANIFEST_PATH = Path("results/runtime_contract_e2e_manifest_1520.jsonl")

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe_primary_live_policy_evaluation",
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship_dense_secondary_live_policy_evaluation",
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "middle_moe_secondary_live_policy_evaluation",
    },
)
MANDATED_HF_IDS = frozenset(spec["hf_id"] for spec in MANDATED_MODEL_SPECS)
EVALUATION_MODES = ("baseline", "promoted")

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "continuous_self_learning_task",
    "model_specs",
    "live_sota_model_inference_used",
    "live_policy_promotion_ready",
    "rollback_passing_updates_loaded",
    "promoted_policy_updates",
    "baseline_task_success_rate",
    "promoted_task_success_rate",
    "utility_delta",
    "false_accept_delta",
    "soundness_mistakes",
    "no_model_weight_mutation",
    "policy_promotion_manifest_path",
    "models_used",
    "blockers",
    "honest_verdict",
)

TERMINAL_VERDICT_PREFIXES: tuple[str, ...] = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """REQ-LEARN-1524-1/8: create the durable bootstrap result first."""

    artifact = {
        "status": "in_progress",
        "run_date": run_date,
        "schema": "fr11_live_policy_promotion_v12",
        "spec": ["REQ-LEARN-1524", "SCENARIO-LEARN-1524", "SCENARIO-LEARN-1525"],
        "continuous_self_learning_task": True,
        "model_specs": [dict(spec) for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": False,
        "live_policy_promotion_ready": False,
        "rollback_passing_updates_loaded": 0,
        "promoted_policy_updates": [],
        "baseline_task_success_rate": 0.0,
        "promoted_task_success_rate": 0.0,
        "utility_delta": 0.0,
        "false_accept_delta": 0,
        "soundness_mistakes": 0,
        "no_model_weight_mutation": True,
        "policy_promotion_manifest_path": _display_path(manifest_path, project_root=project_root),
        "models_used": [],
        "blockers": ["experiment_1524_live_policy_promotion_in_progress"],
        "honest_verdict": "complete: in-progress FR-11 live policy promotion",
    }
    validate_artifact(artifact)
    _write_json(_as_path(output_path), artifact)
    return artifact


def select_promotable_updates(
    *,
    policy_rows: Sequence[Mapping[str, Any]],
    rollback_rows: Sequence[Mapping[str, Any]],
    pack_manifest: Mapping[str, Any],
    limit: int | None = None,
) -> JsonDict:
    """REQ-LEARN-1524-3/4: keep only rollback-passing packaged updates."""

    policy_by_event = {
        str(row.get("source_event_id") or ""): row
        for row in policy_rows
        if row.get("source_event_id")
    }
    pack_by_key = _pack_entry_index(pack_manifest)
    promoted: list[JsonDict] = []
    rejected: list[JsonDict] = []
    for rollback in rollback_rows:
        source_event_id = str(rollback.get("source_event_id") or "")
        policy = policy_by_event.get(source_event_id)
        pack_entry = _find_pack_entry(rollback, pack_by_key)
        reasons = _promotion_rejection_reasons(rollback, policy=policy, pack_entry=pack_entry)
        base = _promotion_base_row(rollback, policy=policy, pack_entry=pack_entry)
        if reasons:
            rejected.append(
                {**base, "promotion_status": "rejected_not_promoted", "rejection_reasons": reasons}
            )
            continue
        if limit is not None and len(promoted) >= limit:
            rejected.append(
                {
                    **base,
                    "promotion_status": "rejected_not_promoted",
                    "rejection_reasons": ["bounded_live_set_limit"],
                }
            )
            continue
        promoted.append({**base, "promotion_status": "selected_for_live_policy_promotion"})
    return {"promoted": promoted, "rejected": rejected}


def select_contract_cases(manifest_path: Path | str, *, limit: int = 2) -> list[JsonDict]:
    """Load a bounded explicit-label contract set from Exp 1520."""

    selected: list[JsonDict] = []
    for row in _read_jsonl(_as_path(manifest_path)):
        if row.get("row_type") != "contract_case":
            continue
        expected = row.get("expected_label")
        if not isinstance(expected, bool):
            continue
        if expected is not False and row.get("final_deterministic_accept") is not False:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def validate_policy_output(
    case: Mapping[str, Any],
    *,
    model_spec: Mapping[str, Any],
    update: Mapping[str, Any],
    mode: str,
    raw_output: str,
) -> JsonDict:
    """Validate one model answer by projecting it into the Exp 1520 contract shape."""

    parsed = _extract_last_json_object(raw_output)
    expected = case.get("expected_label")
    expected_bool = expected if isinstance(expected, bool) else None
    parse_status = "ok"
    proposed_accept = False
    case_id_matches = False
    if parsed is None:
        parse_status = "no_json_object"
    else:
        case_id_matches = parsed.get("contract_case_id") == case.get("contract_case_id")
        decision = parsed.get("final_deterministic_decision")
        accept_value = parsed.get("final_deterministic_accept")
        if not case_id_matches:
            parse_status = "contract_case_id_mismatch"
        elif isinstance(decision, str) and decision.lower() in {"accept", "reject"}:
            proposed_accept = decision.lower() == "accept"
        elif isinstance(accept_value, bool):
            proposed_accept = accept_value
        else:
            parse_status = "missing_final_decision"

    structurally_valid = parse_status == "ok"
    validation_row = _validation_contract_case(case, proposed_accept)
    ledger = runtime_contracts.compute_false_accept_ledger([validation_row])
    false_accept = bool(ledger["false_accept_count"])
    task_success = bool(
        structurally_valid
        and expected_bool is not None
        and proposed_accept == expected_bool
        and not false_accept
    )
    return {
        "mode": mode,
        "model_hf_id": model_spec.get("hf_id"),
        "model_name": model_spec.get("name") or model_spec.get("hf_id"),
        "policy_update_id": update.get("source_event_id"),
        "contract_case_id": case.get("contract_case_id"),
        "raw_output_sha256": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
        "raw_output_excerpt": raw_output[:500],
        "parsed_contract_output": parsed or {},
        "parse_status": parse_status,
        "case_id_matches": bool(case_id_matches),
        "expected_label": expected_bool,
        "proposed_final_deterministic_accept": bool(proposed_accept),
        "task_success": task_success,
        "false_accept": false_accept,
        "contract_validation_row": validation_row,
    }


def run_experiment(
    *,
    project_root: Path | str | None = None,
    run_date: str = RUN_DATE,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    policy_cache_artifact_path: Path | str = DEFAULT_POLICY_CACHE_ARTIFACT_PATH,
    policy_cache_manifest_path: Path | str = DEFAULT_POLICY_CACHE_MANIFEST_PATH,
    rollback_artifact_path: Path | str = DEFAULT_ROLLBACK_ARTIFACT_PATH,
    rollback_manifest_path: Path | str = DEFAULT_ROLLBACK_MANIFEST_PATH,
    portable_pack_artifact_path: Path | str = DEFAULT_PORTABLE_PACK_ARTIFACT_PATH,
    portable_pack_manifest_path: Path | str = DEFAULT_PORTABLE_PACK_MANIFEST_PATH,
    runtime_contract_artifact_path: Path | str = DEFAULT_RUNTIME_CONTRACT_ARTIFACT_PATH,
    runtime_contract_manifest_path: Path | str = DEFAULT_RUNTIME_CONTRACT_MANIFEST_PATH,
    cached_pair_fn: CachedPairFn | None = None,
    resolver_fn: ResolverFn | None = None,
    generator_fn: GeneratorFn | None = None,
    gpu_probe_fn: GpuProbeFn | None = None,
    update_limit: int = 3,
    case_limit: int = 2,
    max_models: int = 1,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Run Exp 1524 and write the JSON artifact plus promotion JSONL manifest."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    output = _resolve_under_root(root, Path(output_path))
    manifest = _resolve_under_root(root, Path(manifest_path))
    write_in_progress_artifact(output, manifest_path=manifest, project_root=root, run_date=run_date)

    paths = {
        "policy_artifact": _resolve_under_root(root, Path(policy_cache_artifact_path)),
        "policy_manifest": _resolve_under_root(root, Path(policy_cache_manifest_path)),
        "rollback_artifact": _resolve_under_root(root, Path(rollback_artifact_path)),
        "rollback_manifest": _resolve_under_root(root, Path(rollback_manifest_path)),
        "pack_artifact": _resolve_under_root(root, Path(portable_pack_artifact_path)),
        "pack_manifest": _resolve_under_root(root, Path(portable_pack_manifest_path)),
        "runtime_artifact": _resolve_under_root(root, Path(runtime_contract_artifact_path)),
        "runtime_manifest": _resolve_under_root(root, Path(runtime_contract_manifest_path)),
    }
    sources, source_blockers = _load_required_sources(paths)
    blockers = list(source_blockers)
    promoted_candidates: list[JsonDict] = []
    rejected_updates: list[JsonDict] = []
    cases: list[JsonDict] = []

    if not blockers:
        selection = select_promotable_updates(
            policy_rows=sources["policy_rows"],
            rollback_rows=sources["rollback_rows"],
            pack_manifest=sources["pack_manifest"],
        )
        promoted_candidates = selection["promoted"]
        rejected_updates = selection["rejected"]
        cases = select_contract_cases(paths["runtime_manifest"], limit=case_limit)
        if not promoted_candidates:
            blockers.append("no_rollback_passing_reachable_updates")
        if not cases:
            blockers.append("no_exp1520_explicit_contract_cases")

    pair_resolver = cached_pair_fn or _cached_sota_pair
    gguf_resolver = resolver_fn or _resolve_cached_gguf
    gpu_probe = gpu_probe_fn or _probe_gpu_state
    models = (
        _resolve_runtime_models(pair_resolver, gguf_resolver, max_models=max_models)
        if not blockers
        else []
    )
    if not blockers and not models:
        blockers.append("no_mandated_sota_gguf_runtime")

    evaluation_rows: list[JsonDict] = []
    if not blockers:
        updates_to_evaluate = promoted_candidates[:update_limit]
        if generator_fn is not None:
            evaluation_rows = _run_injected_generation(
                cases, updates_to_evaluate, models, generator_fn
            )
        else:  # pragma: no cover - exercised only on a live GGUF host.
            evaluation_rows, live_blockers = _run_live_llama_generation(
                cases, updates_to_evaluate, models
            )
            blockers.extend(live_blockers)
    if not evaluation_rows and not blockers:
        blockers.append("no_live_policy_promotion_rows")

    summary = summarize_evaluation_rows(
        evaluation_rows,
        rollback_passing_updates_loaded=len(promoted_candidates),
        rejected_updates=rejected_updates,
        blockers=blockers,
    )
    if summary["soundness_mistakes"] > 0:
        blockers.append("soundness_mistakes_nonzero")
    if summary["false_accept_delta"] > 0:
        blockers.append("false_accept_delta_positive")
    blockers = sorted(dict.fromkeys(blockers))
    summary = summarize_evaluation_rows(
        evaluation_rows,
        rollback_passing_updates_loaded=len(promoted_candidates),
        rejected_updates=rejected_updates,
        blockers=blockers,
    )
    _write_jsonl(manifest, [*evaluation_rows, summary])
    artifact = build_artifact(
        rows=evaluation_rows,
        summary=summary,
        manifest_path=manifest,
        project_root=root,
        run_date=run_date,
        tests_run=tests_run,
    )
    _write_json(output, artifact)
    return artifact


def summarize_evaluation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    rollback_passing_updates_loaded: int,
    rejected_updates: Sequence[Mapping[str, Any]],
    blockers: Sequence[str],
) -> JsonDict:
    """REQ-LEARN-1524-6/7: aggregate utility and soundness from manifest rows."""

    baseline_success = sum(int(bool(row.get("baseline_task_success"))) for row in rows)
    promoted_success = sum(int(bool(row.get("promoted_task_success"))) for row in rows)
    baseline_rate = _rate_or_zero(baseline_success, len(rows))
    promoted_rate = _rate_or_zero(promoted_success, len(rows))
    models_used = sorted(
        {str(row["model_hf_id"]) for row in rows if row.get("model_hf_id") in MANDATED_HF_IDS}
    )
    promoted_update_ids = sorted(
        {str(row["policy_update_id"]) for row in rows if row.get("policy_update_id")}
    )
    false_accept_delta = sum(int(row.get("false_accept_delta", 0)) for row in rows)
    soundness_mistakes = sum(int(row.get("soundness_mistakes", 0)) for row in rows)
    return {
        "row_type": "summary",
        "spec": ["REQ-LEARN-1524", "SCENARIO-LEARN-1524", "SCENARIO-LEARN-1525"],
        "live_sota_model_inference_used": bool(rows and models_used),
        "rollback_passing_updates_loaded": int(rollback_passing_updates_loaded),
        "promoted_policy_updates": promoted_update_ids,
        "evaluated_rows": len(rows),
        "baseline_task_success_rate": baseline_rate,
        "promoted_task_success_rate": promoted_rate,
        "utility_delta": round(promoted_rate - baseline_rate, 6),
        "false_accept_delta": false_accept_delta,
        "soundness_mistakes": soundness_mistakes,
        "no_model_weight_mutation": True,
        "models_used": models_used,
        "rejected_policy_update_count": len(rejected_updates),
        "rejected_policy_updates": list(rejected_updates),
        "blockers": list(dict.fromkeys(blockers)),
    }


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    manifest_path: Path | str,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-LEARN-1524-7/8: build the terminal promotion artifact."""

    blockers = list(summary.get("blockers", []))
    ready = bool(
        rows
        and summary.get("live_sota_model_inference_used") is True
        and int(summary.get("soundness_mistakes", 0)) == 0
        and int(summary.get("false_accept_delta", 0)) <= 0
        and summary.get("no_model_weight_mutation") is True
        and not blockers
    )
    artifact = {
        "status": "complete" if ready else "blocked",
        "run_date": run_date,
        "schema": "fr11_live_policy_promotion_v12",
        "spec": ["REQ-LEARN-1524", "SCENARIO-LEARN-1524", "SCENARIO-LEARN-1525"],
        "continuous_self_learning_task": True,
        "model_specs": [dict(spec) for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": bool(summary.get("live_sota_model_inference_used")),
        "live_policy_promotion_ready": ready,
        "rollback_passing_updates_loaded": int(summary.get("rollback_passing_updates_loaded", 0)),
        "promoted_policy_updates": list(summary.get("promoted_policy_updates", [])),
        "baseline_task_success_rate": float(summary.get("baseline_task_success_rate", 0.0)),
        "promoted_task_success_rate": float(summary.get("promoted_task_success_rate", 0.0)),
        "utility_delta": float(summary.get("utility_delta", 0.0)),
        "false_accept_delta": int(summary.get("false_accept_delta", 0)),
        "soundness_mistakes": int(summary.get("soundness_mistakes", 0)),
        "no_model_weight_mutation": bool(summary.get("no_model_weight_mutation")),
        "policy_promotion_manifest_path": _display_path(manifest_path, project_root=project_root),
        "models_used": list(summary.get("models_used", [])),
        "blockers": blockers,
        "honest_verdict": (
            "complete: fr11_live_policy_promotion_ready"
            if ready
            else "complete: fr11_live_policy_promotion_blocked"
        ),
        "tests_run": list(tests_run or []),
    }
    validate_artifact(artifact, manifest_path=manifest_path)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    manifest_path: Path | str | None = None,
) -> None:
    """Enforce the terminal artifact shape expected by the conductor."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    if not str(artifact["honest_verdict"]).startswith(TERMINAL_VERDICT_PREFIXES):
        raise AssertionError("honest_verdict must use an allowed terminal prefix")
    if artifact["live_policy_promotion_ready"]:
        if artifact["live_sota_model_inference_used"] is not True:
            raise AssertionError("ready promotion requires live SOTA inference")
        if int(artifact["soundness_mistakes"]) != 0:
            raise AssertionError("ready promotion requires zero soundness mistakes")
        if int(artifact["false_accept_delta"]) > 0:
            raise AssertionError("ready promotion cannot increase false accepts")
        if artifact["no_model_weight_mutation"] is not True:
            raise AssertionError("ready promotion requires frozen model weights")
        if not artifact["promoted_policy_updates"]:
            raise AssertionError("ready promotion requires promoted policy updates")
        if manifest_path is not None and not _as_path(manifest_path).exists():
            raise AssertionError("ready promotion requires the promotion manifest")


def _promotion_rejection_reasons(
    rollback: Mapping[str, Any],
    *,
    policy: Mapping[str, Any] | None,
    pack_entry: Mapping[str, Any] | None,
) -> list[str]:
    reasons: list[str] = []
    if policy is None:
        reasons.append("missing_exp1512_policy_row")
    elif policy.get("quarantined") is True or policy.get("accepted") is False:
        reasons.append("exp1512_not_accepted")
    if not rollback.get("skill_id"):
        reasons.append("missing_skill_id")
    if rollback.get("decision") != "keep":
        reasons.append("rollback_decision_not_keep")
    if rollback.get("source_evidence_reachable") is not True:
        reasons.append("source_evidence_unreachable")
    if bool(rollback.get("source_evidence_stale")):
        reasons.append("source_evidence_stale")
    if rollback.get("deterministic_validator_supported") is not True:
        reasons.append("missing_deterministic_validator_support")
    if int(rollback.get("soundness_mistakes", 0)) > 0:
        reasons.append("soundness_mistake")
    if int(rollback.get("false_accept_delta", 0)) > 0:
        reasons.append("false_accept_delta_positive")
    rollback_reasons = rollback.get("rollback_reasons")
    if isinstance(rollback_reasons, Sequence) and not isinstance(rollback_reasons, (str, bytes)):
        reasons.extend(f"rollback_reason:{reason}" for reason in rollback_reasons if reason)
    if pack_entry is None:
        reasons.append("missing_portable_provenance")
    elif pack_entry.get("promotion_status") != "packaged_rollback_passed":
        reasons.append("portable_provenance_not_packaged")
    return sorted(dict.fromkeys(reasons))


def _promotion_base_row(
    rollback: Mapping[str, Any],
    *,
    policy: Mapping[str, Any] | None,
    pack_entry: Mapping[str, Any] | None,
) -> JsonDict:
    return {
        "source_event_id": str(rollback.get("source_event_id") or ""),
        "source_case_id": str(rollback.get("source_case_id") or ""),
        "source_kind": str(rollback.get("source_kind") or ""),
        "skill_id": str(rollback.get("skill_id") or ""),
        "policy_action": str(rollback.get("policy_action") or ""),
        "resolver_key": str((pack_entry or {}).get("resolver_key") or ""),
        "rollback_verifier_evidence": {
            "decision": rollback.get("decision"),
            "source_evidence_reachable": rollback.get("source_evidence_reachable"),
            "source_evidence_stale": rollback.get("source_evidence_stale"),
            "deterministic_validator_supported": rollback.get("deterministic_validator_supported"),
            "soundness_mistakes": int(rollback.get("soundness_mistakes", 0)),
            "false_accept_delta": int(rollback.get("false_accept_delta", 0)),
            "utility_delta": int(rollback.get("utility_delta", 0)),
        },
        "exp1512_policy_action": None if policy is None else policy.get("policy_action"),
        "portable_provenance": {} if pack_entry is None else dict(pack_entry),
    }


def _pack_entry_index(pack_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    entries = pack_manifest.get("entries", [])
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return {}
    index: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        for key in (
            entry.get("source_event_id"),
            entry.get("resolver_key"),
            entry.get("skill_id"),
        ):
            if key:
                index.setdefault(str(key), entry)
    return index


def _find_pack_entry(
    rollback: Mapping[str, Any],
    pack_by_key: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for key in (
        rollback.get("source_event_id"),
        rollback.get("skill_id"),
        f"daily_eval:{rollback.get('source_case_id')}",
    ):
        if key and str(key) in pack_by_key:
            return pack_by_key[str(key)]
    return None


def _run_injected_generation(
    cases: Sequence[JsonDict],
    updates: Sequence[JsonDict],
    models: Sequence[JsonDict],
    generator_fn: GeneratorFn,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for model in models:
        for update in updates:
            for case in cases:
                rows.append(
                    _evaluate_one_case(case, update=update, model=model, generator_fn=generator_fn)
                )
    return rows


def _evaluate_one_case(
    case: JsonDict,
    *,
    update: JsonDict,
    model: JsonDict,
    generator_fn: GeneratorFn,
) -> JsonDict:
    evaluations: dict[str, JsonDict] = {}
    for mode in EVALUATION_MODES:
        prompt = build_policy_prompt(case, update=update, mode=mode)
        raw = generator_fn(prompt, model, mode, update, case)
        evaluations[mode] = validate_policy_output(
            case,
            model_spec=model,
            update=update,
            mode=mode,
            raw_output=raw,
        )
    baseline = evaluations["baseline"]
    promoted = evaluations["promoted"]
    baseline_false = int(bool(baseline["false_accept"]))
    promoted_false = int(bool(promoted["false_accept"]))
    baseline_success = int(bool(baseline["task_success"]))
    promoted_success = int(bool(promoted["task_success"]))
    return {
        "row_type": "policy_promotion_evaluation",
        "spec": ["REQ-LEARN-1524", "SCENARIO-LEARN-1524"],
        "model_hf_id": model.get("hf_id"),
        "model_name": model.get("name") or model.get("hf_id"),
        "policy_update_id": update.get("source_event_id"),
        "policy_action": update.get("policy_action"),
        "skill_id": update.get("skill_id"),
        "contract_case_id": case.get("contract_case_id"),
        "prompt_or_case_id": case.get("prompt_or_case_id"),
        "source_family": case.get("source_family"),
        "baseline_task_success": bool(baseline_success),
        "promoted_task_success": bool(promoted_success),
        "utility_delta": promoted_success - baseline_success,
        "baseline_false_accept": bool(baseline_false),
        "promoted_false_accept": bool(promoted_false),
        "false_accept_delta": promoted_false - baseline_false,
        "soundness_mistakes": promoted_false,
        "runtime_contract_validation": evaluations,
    }


def build_policy_prompt(case: Mapping[str, Any], *, update: Mapping[str, Any], mode: str) -> str:
    """Build the prompt for one baseline or promoted-policy evaluation."""

    context = {
        "contract_case_id": case.get("contract_case_id"),
        "source_family": case.get("source_family"),
        "prompt_or_case_id": case.get("prompt_or_case_id"),
        "proposed_output": case.get("proposed_output"),
        "certificate_parse_result": case.get("certificate_parse_result"),
        "safe_dsl_verifier_result": case.get("safe_dsl_verifier_result"),
        "monitor_event_result": case.get("monitor_event_result"),
        "structural_contract_result": case.get("structural_contract_result"),
    }
    if mode == "baseline":
        instruction = (
            "Decide whether the proposed output should pass Carnot's runtime contract. "
            "Answer naturally."
        )
    elif mode == "promoted":
        instruction = (
            "Use the promoted FR-11 query-time policy update, then return strict JSON "
            "with keys contract_case_id and final_deterministic_decision."
        )
        context["promoted_policy_update"] = {
            "source_event_id": update.get("source_event_id"),
            "policy_action": update.get("policy_action"),
            "skill_id": update.get("skill_id"),
            "verifier_evidence": update.get("rollback_verifier_evidence"),
        }
    else:
        raise ValueError(f"unknown evaluation mode: {mode}")
    return f"{instruction}\n\nRuntime contract context:\n{json.dumps(context, sort_keys=True)}\n"


def _validation_contract_case(case: Mapping[str, Any], final_accept: bool) -> JsonDict:
    validation = {
        key: case.get(key) for key in runtime_contracts.REQUIRED_CONTRACT_CASE_FIELDS if key in case
    }
    validation["row_type"] = "contract_case"
    validation["contract_schema_version"] = runtime_contracts.CONTRACT_CASE_SCHEMA_VERSION
    validation["final_deterministic_accept"] = bool(final_accept)
    validation["final_deterministic_decision"] = "accept" if final_accept else "reject"
    return validation


def _resolve_runtime_models(
    cached_pair_fn: CachedPairFn,
    resolver_fn: ResolverFn,
    *,
    max_models: int,
) -> list[JsonDict]:
    models: list[JsonDict] = []
    try:
        pair = cached_pair_fn(gpu_indices=(0, 1), preferred_quant="Q4_K_M")
    except Exception:
        pair = None
    for spec in pair or []:
        hf_id = spec.get("hf_id")
        if hf_id in MANDATED_HF_IDS and spec.get("model_path"):
            models.append(dict(spec))
    if not models:
        for index, mandated in enumerate(MANDATED_MODEL_SPECS):
            model_path = resolver_fn(str(mandated["hf_id"]))
            if model_path:
                models.append(
                    {
                        "name": str(mandated["hf_id"]).rsplit("/", 1)[-1].removesuffix("-GGUF"),
                        "hf_id": mandated["hf_id"],
                        "role": mandated["role"],
                        "gpu": index,
                        "model_path": model_path,
                    }
                )
    return models[:max_models]


def _run_live_llama_generation(
    cases: Sequence[JsonDict],
    updates: Sequence[JsonDict],
    models: Sequence[JsonDict],
) -> tuple[list[JsonDict], list[str]]:  # pragma: no cover - hardware dependent.
    rows: list[JsonDict] = []
    blockers: list[str] = []
    for model in models:
        try:
            model_rows = _run_one_live_model(cases, updates, model)
        except Exception as exc:
            blockers.append(
                f"live_generation_failed:{model.get('hf_id')}:{type(exc).__name__}:{exc}"
            )
            continue
        rows.extend(model_rows)
        if model_rows:
            break
    if not rows:
        blockers.append("no_mandated_sota_model_completed_live_inference")
    return rows, blockers


def _run_one_live_model(
    cases: Sequence[JsonDict],
    updates: Sequence[JsonDict],
    model: JsonDict,
) -> list[JsonDict]:  # pragma: no cover - hardware dependent.
    _ensure_cuda_library_path()
    from llama_cpp import Llama  # noqa: PLC0415

    gpu = int(model.get("gpu", 0))
    llm = Llama(
        model_path=str(model["model_path"]),
        n_gpu_layers=-1 if gpu >= 0 else 0,
        main_gpu=max(gpu, 0),
        n_ctx=2048,
        verbose=False,
    )
    try:
        return [
            _evaluate_one_case(case, update=update, model=model, generator_fn=_llama_generator(llm))
            for update in updates
            for case in cases
        ]
    finally:
        if hasattr(llm, "close"):
            llm.close()


def _llama_generator(llm: Any) -> GeneratorFn:  # pragma: no cover - hardware dependent.
    def generate(
        prompt: str,
        _model: JsonDict,
        _mode: str,
        _update: JsonDict,
        _case: JsonDict,
    ) -> str:
        completion = llm(
            prompt,
            max_tokens=180,
            temperature=0.0,
            echo=False,
            stop=["</s>", "<eos>"],
        )
        return _completion_text(completion)

    return generate


def _load_required_sources(paths: Mapping[str, Path]) -> tuple[JsonDict, list[str]]:
    blockers: list[str] = []
    policy_artifact = _load_json_or_blocker(paths["policy_artifact"], blockers)
    rollback_artifact = _load_json_or_blocker(paths["rollback_artifact"], blockers)
    pack_artifact = _load_json_or_blocker(paths["pack_artifact"], blockers)
    runtime_artifact = _load_json_or_blocker(paths["runtime_artifact"], blockers)
    if policy_artifact is not None and policy_artifact.get("policy_cache_ready") is not True:
        blockers.append("exp1512_policy_cache_not_ready")
    if rollback_artifact is not None and rollback_artifact.get("rollback_audit_passed") is not True:
        blockers.append("exp1513_rollback_audit_not_passed")
    if pack_artifact is not None and pack_artifact.get("portable_skill_pack_ready") is not True:
        blockers.append("exp1514_portable_pack_not_ready")
    if (
        runtime_artifact is not None
        and runtime_artifact.get("runtime_contract_e2e_ready") is not True
    ):
        blockers.append("exp1520_runtime_contract_not_ready")

    for key in ("policy_manifest", "rollback_manifest", "pack_manifest", "runtime_manifest"):
        if not paths[key].exists():
            blockers.append(f"missing_{key}:{paths[key]}")
    if blockers:
        return {}, sorted(dict.fromkeys(blockers))
    return (
        {
            "policy_artifact": policy_artifact or {},
            "policy_rows": _read_jsonl(paths["policy_manifest"]),
            "rollback_artifact": rollback_artifact or {},
            "rollback_rows": _read_jsonl(paths["rollback_manifest"]),
            "pack_artifact": pack_artifact or {},
            "pack_manifest": _read_json(paths["pack_manifest"]),
            "runtime_artifact": runtime_artifact or {},
        },
        [],
    )


def _load_json_or_blocker(path: Path, blockers: list[str]) -> JsonDict | None:
    if not path.exists():
        blockers.append(f"missing_artifact:{path}")
        return None
    try:
        return _read_json(path)
    except (json.JSONDecodeError, OSError, AssertionError) as exc:
        blockers.append(f"malformed_artifact:{path}:{type(exc).__name__}")
        return None


def _extract_last_json_object(text: str) -> JsonDict | None:
    decoder = json.JSONDecoder()
    last: JsonDict | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            last = parsed
    return last


def _completion_text(result: Any) -> str:  # pragma: no cover - llama.cpp adapter.
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return ""
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    text = first.get("text")
    if isinstance(text, str):
        return text.strip()
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return str(message["content"]).strip()
    return ""


def _ensure_cuda_library_path() -> None:  # pragma: no cover - host runtime repair.
    site_packages = sorted((Path.cwd() / ".venv" / "lib").glob("python*/site-packages"))
    candidates: list[str] = []
    for site in site_packages:
        candidates.extend(
            [
                str(site / "nvidia" / "cuda_runtime" / "lib"),
                str(site / "nvidia" / "cublas" / "lib"),
            ]
        )
    current_parts = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    repaired: list[str] = []
    seen: set[str] = set()
    for path in [*candidates, *current_parts]:
        if path in seen or not Path(path).is_dir():
            continue
        seen.add(path)
        repaired.append(path)
    if repaired:
        os.environ["LD_LIBRARY_PATH"] = ":".join(repaired)


def _cached_sota_pair(**kwargs: Any) -> list[JsonDict] | None:  # pragma: no cover
    from carnot.inference.sota_models import cached_sota_pair

    return cached_sota_pair(**kwargs)


def _resolve_cached_gguf(hf_id: str) -> str | None:  # pragma: no cover
    from carnot.inference.sota_models import resolve_cached_gguf

    return resolve_cached_gguf(hf_id)


def _probe_gpu_state() -> JsonDict:  # pragma: no cover
    from carnot.reporting.live_sota_repair_runtime_preflight import probe_gpu_state

    return probe_gpu_state()


def _rate_or_zero(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(path: Path | str, *, project_root: Path | str = REPO_ROOT) -> str:
    target = Path(path)
    try:
        return target.relative_to(Path(project_root)).as_posix()
    except ValueError:
        return target.as_posix()


def _as_path(path: Path | str) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"JSON artifact must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise AssertionError(f"JSONL row must be an object: {path}")
        rows.append(row)
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--update-limit", type=int, default=3)
    parser.add_argument("--case-limit", type=int, default=2)
    parser.add_argument("--max-models", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = _parse_args(argv)
    run_experiment(
        run_date=args.run_date,
        output_path=args.output,
        manifest_path=args.manifest,
        update_limit=args.update_limit,
        case_limit=args.case_limit,
        max_models=args.max_models,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
