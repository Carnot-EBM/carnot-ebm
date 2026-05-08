"""Exp 1525 MARCH claim-isolation verifier ablation.

Spec: REQ-VERIFY-1525, SCENARIO-VERIFY-1525.

MARCH-style checking is useful only if the information-asymmetry trick does not
become a new trust boundary.  This module therefore lets an LLM checker provide
auxiliary full-context and claim-isolated feedback, then projects that feedback
back into the Exp 1520 runtime-contract ledger.  The deterministic ledger is
the authority for false accepts.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.verify import runtime_contract_e2e_harness as runtime_contracts

JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[JsonDict] | None]
CheckerFn = Callable[[str, JsonDict, str, JsonDict], str]
ResolverFn = Callable[[str], str | None]

RUN_DATE = "20260508"
OUTPUT_FILE = "experiment_1525_march_claim_isolation_verifier_ablation.json"
MANIFEST_FILE = "march_claim_isolation_1525.jsonl"
DEFAULT_OUTPUT_PATH = Path("results") / OUTPUT_FILE
DEFAULT_MANIFEST_PATH = Path("results") / MANIFEST_FILE
DEFAULT_PROMOTION_ARTIFACT_PATH = Path(
    "results/experiment_1524_fr11_live_policy_promotion_v12.json"
)
DEFAULT_PROMOTION_MANIFEST_PATH = Path("results/fr11_live_policy_promotion_1524.jsonl")
DEFAULT_RUNTIME_CONTRACT_ARTIFACT_PATH = Path(
    "results/experiment_1520_runtime_contract_e2e_harness.json"
)
DEFAULT_RUNTIME_CONTRACT_MANIFEST_PATH = Path("results/runtime_contract_e2e_manifest_1520.jsonl")

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe_primary_claim_proposer_checker",
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship_dense_secondary_claim_proposer_checker",
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "middle_moe_secondary_claim_proposer_checker",
    },
)
MANDATED_HF_IDS = frozenset(str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS)
MAX_CLAIMS_PER_CASE = 4

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "live_sota_model_inference_used",
    "claim_isolation_ablation_ready",
    "cases_loaded",
    "claims_extracted",
    "full_context_accept_rate",
    "claim_isolated_accept_rate",
    "claim_isolation_delta",
    "verifier_calls_full_context",
    "verifier_calls_claim_isolated",
    "budget_delta",
    "false_accept_count",
    "false_accept_rate",
    "claim_isolation_manifest_path",
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
    project_root: Path | str | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """REQ-VERIFY-1525: persist a restartable bootstrap artifact first."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    artifact = _artifact_from_summary(
        status="in_progress",
        ready=False,
        cases_loaded=0,
        claims_extracted=0,
        full_context_accept_rate=None,
        claim_isolated_accept_rate=None,
        claim_isolation_delta=None,
        verifier_calls_full_context=0,
        verifier_calls_claim_isolated=0,
        budget_delta=None,
        false_accept_count=0,
        false_accept_rate=None,
        manifest_path=Path(manifest_path),
        project_root=root,
        models_used=[],
        blockers=["experiment_1525_claim_isolation_in_progress"],
        run_date=run_date,
    )
    validate_artifact(artifact)
    _write_json(Path(output_path), artifact)
    return artifact


def extract_atomic_claims(
    promotion_row: Mapping[str, Any],
    *,
    runtime_case: Mapping[str, Any],
) -> list[JsonDict]:
    """REQ-VERIFY-1525: extract bounded, stable claim rows from one answer."""

    contract_case_id = str(
        promotion_row.get("contract_case_id") or runtime_case.get("contract_case_id") or ""
    )
    source_mode = "promoted"
    claim_text = _claim_source_text(promotion_row, runtime_case=runtime_case)
    pieces = _split_atomic_claims(claim_text)
    if not pieces:
        pieces = [_fallback_claim_text(promotion_row, runtime_case=runtime_case)]

    claims: list[JsonDict] = []
    for index, text in enumerate(pieces[:MAX_CLAIMS_PER_CASE], start=1):
        deterministic_accept = bool(runtime_case.get("final_deterministic_accept"))
        claims.append(
            {
                "claim_id": f"{contract_case_id}:{source_mode}:{index:03d}",
                "contract_case_id": contract_case_id,
                "source_mode": source_mode,
                "source_family": str(
                    runtime_case.get("source_family") or promotion_row.get("source_family") or ""
                ),
                "claim_text": text,
                "deterministic_expected_label": _explicit_bool(runtime_case.get("expected_label")),
                "deterministic_final_accept": deterministic_accept,
                "deterministic_final_decision": ("accept" if deterministic_accept else "reject"),
            }
        )
    return claims


def route_checker_verdict(
    claim: Mapping[str, Any],
    *,
    runtime_case: Mapping[str, Any],
    checker_accept: bool,
    mode: str,
    model_spec: Mapping[str, Any],
    raw_output: str = "",
    parse_status: str = "ok",
) -> JsonDict:
    """REQ-VERIFY-1525: score checker feedback with the deterministic ledger."""

    validation_row = _validation_contract_case(runtime_case, final_accept=checker_accept)
    ledger = runtime_contracts.compute_false_accept_ledger([validation_row])
    expected = _explicit_bool(validation_row.get("expected_label"))
    return {
        "mode": mode,
        "model_hf_id": model_spec.get("hf_id"),
        "claim_id": claim.get("claim_id"),
        "checker_accept": bool(checker_accept),
        "deterministic_expected_label": expected,
        "deterministic_final_accept": bool(checker_accept),
        "deterministic_final_decision": "accept" if checker_accept else "reject",
        "false_accept": bool(ledger["false_accept_count"]),
        "false_accept_rate": ledger["false_accept_rate"],
        "auxiliary_disagreement": bool(
            isinstance(expected, bool) and bool(checker_accept) != expected
        ),
        "original_answer_visible_to_checker": mode == "full_context",
        "parse_status": parse_status,
        "raw_output_sha256": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
    }


def run_experiment(
    *,
    project_root: Path | str | None = None,
    run_date: str = RUN_DATE,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    promotion_artifact_path: Path | str = DEFAULT_PROMOTION_ARTIFACT_PATH,
    promotion_manifest_path: Path | str = DEFAULT_PROMOTION_MANIFEST_PATH,
    runtime_contract_artifact_path: Path | str = DEFAULT_RUNTIME_CONTRACT_ARTIFACT_PATH,
    runtime_contract_manifest_path: Path | str = DEFAULT_RUNTIME_CONTRACT_MANIFEST_PATH,
    cached_pair_fn: CachedPairFn | None = None,
    resolver_fn: ResolverFn | None = None,
    checker_fn: CheckerFn | None = None,
    max_models: int = 1,
    case_limit: int = 4,
) -> JsonDict:
    """SCENARIO-VERIFY-1525: run full-context and isolated checker modes."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    output = _resolve_under_root(root, Path(output_path))
    manifest = _resolve_under_root(root, Path(manifest_path))
    write_in_progress_artifact(output, manifest_path=manifest, project_root=root, run_date=run_date)

    paths = {
        "promotion_artifact": _resolve_under_root(root, Path(promotion_artifact_path)),
        "promotion_manifest": _resolve_under_root(root, Path(promotion_manifest_path)),
        "runtime_artifact": _resolve_under_root(root, Path(runtime_contract_artifact_path)),
        "runtime_manifest": _resolve_under_root(root, Path(runtime_contract_manifest_path)),
    }
    promotion_rows, runtime_rows, blockers = _load_required_sources(paths)
    cases = _select_joined_cases(promotion_rows, runtime_rows, limit=case_limit)
    if not cases:
        blockers.append("no_promoted_runtime_contract_cases")
    claim_cases = _claim_cases(cases)
    claims_extracted = sum(len(case["claims"]) for case in claim_cases)
    if not claims_extracted:
        blockers.append("no_atomic_claims_extracted")

    models = (
        _resolve_runtime_models(
            cached_pair_fn or _cached_sota_pair,
            resolver_fn or _resolve_cached_gguf,
            max_models=max_models,
        )
        if not blockers
        else []
    )
    if not blockers and not models:
        blockers.append("no_mandated_sota_gguf_runtime")

    rows: list[JsonDict] = []
    if not blockers:
        if checker_fn is None:  # pragma: no cover
            rows, live_blockers = _run_live_checker_modes(claim_cases, models[0])
            blockers.extend(live_blockers)
        else:
            rows = _run_checker_modes(claim_cases, model=models[0], checker_fn=checker_fn)

    summary = summarize_rows(
        rows,
        cases_loaded=len(cases),
        claims_extracted=claims_extracted,
        blockers=sorted(dict.fromkeys(blockers)),
    )
    _write_jsonl(manifest, [*rows, summary])
    artifact = build_artifact(
        summary=summary,
        manifest_path=manifest,
        project_root=root,
        run_date=run_date,
    )
    _write_json(output, artifact)
    return artifact


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    cases_loaded: int,
    claims_extracted: int,
    blockers: Sequence[str],
) -> JsonDict:
    """REQ-VERIFY-1525: aggregate accept rates, budgets, and false accepts."""

    full_by_call: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        full_by_call.setdefault(str(row.get("full_context_call_id")), row["full_context"])
    isolated = [row["claim_isolated"] for row in rows]
    full_accept_rate = _rate_or_none(
        sum(int(bool(verdict["checker_accept"])) for verdict in full_by_call.values()),
        len(full_by_call),
    )
    isolated_accept_rate = _rate_or_none(
        sum(int(bool(verdict["checker_accept"])) for verdict in isolated),
        len(isolated),
    )
    false_accept_count = sum(
        int(bool(verdict["false_accept"])) for verdict in full_by_call.values()
    )
    false_accept_count += sum(int(bool(verdict["false_accept"])) for verdict in isolated)
    reject_opportunities = sum(
        int(verdict.get("deterministic_expected_label") is False)
        for verdict in full_by_call.values()
    )
    reject_opportunities += sum(
        int(verdict.get("deterministic_expected_label") is False) for verdict in isolated
    )
    return {
        "row_type": "summary",
        "spec": ["REQ-VERIFY-1525", "SCENARIO-VERIFY-1525"],
        "cases_loaded": int(cases_loaded),
        "claims_extracted": int(claims_extracted),
        "full_context_accept_rate": full_accept_rate,
        "claim_isolated_accept_rate": isolated_accept_rate,
        "claim_isolation_delta": (
            None
            if full_accept_rate is None or isolated_accept_rate is None
            else round(isolated_accept_rate - full_accept_rate, 6)
        ),
        "verifier_calls_full_context": len(full_by_call),
        "verifier_calls_claim_isolated": len(isolated),
        "budget_delta": len(isolated) - len(full_by_call) if rows else None,
        "false_accept_count": false_accept_count,
        "false_accept_rate": _rate_or_none(false_accept_count, reject_opportunities),
        "models_used": sorted(
            {str(row["model_hf_id"]) for row in rows if row.get("model_hf_id") in MANDATED_HF_IDS}
        ),
        "blockers": list(dict.fromkeys(blockers)),
    }


def build_artifact(
    *,
    summary: Mapping[str, Any],
    manifest_path: Path | str,
    project_root: Path | str | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """REQ-VERIFY-1525: build the terminal experiment artifact."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    blockers = list(summary.get("blockers", []))
    ready = bool(
        summary.get("verifier_calls_full_context")
        and summary.get("verifier_calls_claim_isolated")
        and summary.get("false_accept_rate") is not None
        and summary.get("models_used")
        and not blockers
    )
    artifact = _artifact_from_summary(
        status="complete" if ready else "blocked",
        ready=ready,
        cases_loaded=int(summary.get("cases_loaded", 0)),
        claims_extracted=int(summary.get("claims_extracted", 0)),
        full_context_accept_rate=summary.get("full_context_accept_rate"),
        claim_isolated_accept_rate=summary.get("claim_isolated_accept_rate"),
        claim_isolation_delta=summary.get("claim_isolation_delta"),
        verifier_calls_full_context=int(summary.get("verifier_calls_full_context", 0)),
        verifier_calls_claim_isolated=int(summary.get("verifier_calls_claim_isolated", 0)),
        budget_delta=summary.get("budget_delta"),
        false_accept_count=int(summary.get("false_accept_count", 0)),
        false_accept_rate=summary.get("false_accept_rate"),
        manifest_path=Path(manifest_path),
        project_root=root,
        models_used=list(summary.get("models_used", [])),
        blockers=blockers,
        run_date=run_date,
    )
    validate_artifact(artifact, manifest_path=manifest_path)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    manifest_path: Path | str | None = None,
) -> None:
    """Enforce the Exp 1525 terminal schema used by the conductor."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:  # pragma: no cover
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:  # pragma: no cover
        raise AssertionError(f"unsupported status: {artifact['status']}")
    if not str(artifact["honest_verdict"]).startswith(
        TERMINAL_VERDICT_PREFIXES
    ):  # pragma: no cover
        raise AssertionError("honest_verdict must use an allowed terminal prefix")
    if artifact["claim_isolation_ablation_ready"]:
        if artifact["live_sota_model_inference_used"] is not True:  # pragma: no cover
            raise AssertionError("ready ablation requires live SOTA inference")
        if not artifact["models_used"]:  # pragma: no cover
            raise AssertionError("ready ablation requires at least one mandated SOTA model")
        if artifact["false_accept_rate"] is None:  # pragma: no cover
            raise AssertionError("ready ablation requires reported false_accept_rate")
        if manifest_path is not None and not Path(manifest_path).exists():  # pragma: no cover
            raise AssertionError("ready ablation requires the claim-isolation manifest")


def _run_checker_modes(
    claim_cases: Sequence[JsonDict],
    *,
    model: JsonDict,
    checker_fn: CheckerFn,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for case in claim_cases:
        claims = list(case["claims"])
        full_context_call_id = f"{case['contract_case_id']}:full_context"
        aggregate_claim = dict(claims[0])
        aggregate_claim["claim_id"] = full_context_call_id
        aggregate_claim["claim_text"] = " ".join(str(claim["claim_text"]) for claim in claims)
        full_prompt = build_checker_prompt(
            aggregate_claim,
            runtime_case=case["runtime_case"],
            original_answer=case["original_answer"],
            mode="full_context",
        )
        full_raw = checker_fn(full_prompt, model, "full_context", aggregate_claim)
        full_accept, full_parse_status = parse_checker_output(full_raw)
        for claim in claims:
            isolated_prompt = build_checker_prompt(
                claim,
                runtime_case=case["runtime_case"],
                original_answer=case["original_answer"],
                mode="claim_isolated",
            )
            isolated_raw = checker_fn(isolated_prompt, model, "claim_isolated", claim)
            isolated_accept, isolated_parse_status = parse_checker_output(isolated_raw)
            full_routed = route_checker_verdict(
                claim,
                runtime_case=case["runtime_case"],
                checker_accept=full_accept,
                mode="full_context",
                model_spec=model,
                raw_output=full_raw,
                parse_status=full_parse_status,
            )
            isolated_routed = route_checker_verdict(
                claim,
                runtime_case=case["runtime_case"],
                checker_accept=isolated_accept,
                mode="claim_isolated",
                model_spec=model,
                raw_output=isolated_raw,
                parse_status=isolated_parse_status,
            )
            rows.append(
                {
                    "row_type": "claim_isolation_evaluation",
                    "spec": ["REQ-VERIFY-1525", "SCENARIO-VERIFY-1525"],
                    "model_hf_id": model.get("hf_id"),
                    "model_name": model.get("name") or model.get("hf_id"),
                    "contract_case_id": claim["contract_case_id"],
                    "prompt_or_case_id": case["runtime_case"].get("prompt_or_case_id"),
                    "source_family": claim["source_family"],
                    "policy_update_id": case["promotion_row"].get("policy_update_id"),
                    "claim": dict(claim),
                    "full_context_call_id": full_context_call_id,
                    "full_context": full_routed,
                    "claim_isolated": isolated_routed,
                    "checker_accept_delta": int(isolated_accept) - int(full_accept),
                    "false_accept": bool(
                        full_routed["false_accept"] or isolated_routed["false_accept"]
                    ),
                }
            )
    return rows


def parse_checker_output(raw_output: str) -> tuple[bool, str]:
    """Parse a strict checker verdict, defaulting malformed feedback to reject."""

    parsed = _extract_last_json_object(raw_output)
    if parsed is None:
        return False, "no_json_object"
    if isinstance(parsed.get("checker_accept"), bool):
        return bool(parsed["checker_accept"]), "ok"
    decision = parsed.get("checker_decision")
    if isinstance(decision, str) and decision.lower() in {"accept", "reject"}:
        return decision.lower() == "accept", "ok"
    return False, "missing_checker_decision"


def build_checker_prompt(
    claim: Mapping[str, Any],
    *,
    runtime_case: Mapping[str, Any],
    original_answer: str,
    mode: str,
) -> str:
    """Build the full-context or claim-isolated checker prompt."""

    envelope = {
        "contract_case_id": claim.get("contract_case_id"),
        "claim_id": claim.get("claim_id"),
        "claim_text": claim.get("claim_text"),
        "source_family": claim.get("source_family"),
        "expected_label_available": isinstance(runtime_case.get("expected_label"), bool),
    }
    if mode == "full_context":
        return (
            "Check the claim with the original answer visible. "
            "Return JSON with claim_id and checker_decision.\n\n"
            f"Original answer:\n{original_answer}\n\n"
            f"Claim packet:\n{json.dumps(envelope, sort_keys=True)}\n"
        )
    return (
        "Check the isolated claim without using the original answer. "
        "Return JSON with claim_id and checker_decision.\n\n"
        f"Claim packet:\n{json.dumps(envelope, sort_keys=True)}\n"
    )


def _claim_cases(cases: Sequence[JsonDict]) -> list[JsonDict]:
    claim_cases: list[JsonDict] = []
    for case in cases:
        claims = extract_atomic_claims(case["promotion_row"], runtime_case=case["runtime_case"])
        claim_cases.append({**case, "claims": claims, "original_answer": _original_answer(case)})
    return claim_cases


def _select_joined_cases(
    promotion_rows: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> list[JsonDict]:
    runtime_by_id = {
        str(row.get("contract_case_id")): row
        for row in runtime_rows
        if row.get("row_type") == "contract_case" and row.get("contract_case_id")
    }
    cases: list[JsonDict] = []
    for row in promotion_rows:
        if row.get("row_type") != "policy_promotion_evaluation":
            continue
        runtime_case = runtime_by_id.get(str(row.get("contract_case_id")))
        if runtime_case is None:
            continue
        cases.append(
            {
                "contract_case_id": str(row.get("contract_case_id")),
                "promotion_row": dict(row),
                "runtime_case": dict(runtime_case),
            }
        )
        if len(cases) >= limit:
            break
    return cases


def _load_required_sources(
    paths: Mapping[str, Path],
) -> tuple[list[JsonDict], list[JsonDict], list[str]]:
    blockers: list[str] = []
    promotion_artifact = _load_json_or_blocker(paths["promotion_artifact"], blockers)
    runtime_artifact = _load_json_or_blocker(paths["runtime_artifact"], blockers)
    if (
        promotion_artifact is not None
        and promotion_artifact.get("live_policy_promotion_ready") is not True
    ):  # pragma: no cover
        blockers.append("exp1524_live_policy_promotion_not_ready")
    if (
        runtime_artifact is not None
        and runtime_artifact.get("runtime_contract_e2e_ready") is not True
    ):  # pragma: no cover
        blockers.append("exp1520_runtime_contract_not_ready")
    for key in ("promotion_manifest", "runtime_manifest"):
        if not paths[key].exists():  # pragma: no cover
            blockers.append(f"missing_{key}:{paths[key]}")
    if blockers:
        return [], [], sorted(dict.fromkeys(blockers))
    return _read_jsonl(paths["promotion_manifest"]), _read_jsonl(paths["runtime_manifest"]), []


def _resolve_runtime_models(
    cached_pair_fn: CachedPairFn,
    resolver_fn: ResolverFn,
    *,
    max_models: int,
) -> list[JsonDict]:
    models: list[JsonDict] = []
    try:
        pair = cached_pair_fn(gpu_indices=(0, 1), preferred_quant="Q4_K_M")
    except Exception:  # pragma: no cover
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


def _claim_source_text(
    promotion_row: Mapping[str, Any],
    *,
    runtime_case: Mapping[str, Any],
) -> str:
    promoted = _promoted_validation(promotion_row)
    parsed = promoted.get("parsed_contract_output")
    if isinstance(parsed, Mapping):
        for key in ("rationale", "reasoning", "explanation", "answer", "final_answer"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value
        if parsed:
            return json.dumps(dict(parsed), sort_keys=True)
    raw = promoted.get("raw_output_excerpt")
    if isinstance(raw, str) and raw.strip():
        return raw[:1200]
    proposed = runtime_case.get("proposed_output")
    return str(proposed or "")


def _fallback_claim_text(
    promotion_row: Mapping[str, Any],
    *,
    runtime_case: Mapping[str, Any],
) -> str:
    promoted = _promoted_validation(promotion_row)
    parsed = promoted.get("parsed_contract_output")
    if isinstance(parsed, Mapping) and parsed.get("final_deterministic_decision"):
        return f"final deterministic decision is {parsed['final_deterministic_decision']}"
    return f"runtime contract {runtime_case.get('contract_case_id')} is checked"


def _split_atomic_claims(text: str) -> list[str]:
    candidates: list[str] = []
    for chunk in re.split(r"(?<=[.!?])\s+|\n+", text):
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", chunk).strip()
        cleaned = cleaned.strip(" \t\r\n.;:!?")
        if len(cleaned) >= 3:
            candidates.append(cleaned)
    return candidates


def _promoted_validation(promotion_row: Mapping[str, Any]) -> JsonDict:
    validation = promotion_row.get("runtime_contract_validation")
    if not isinstance(validation, Mapping):
        return {}
    promoted = validation.get("promoted")
    return dict(promoted) if isinstance(promoted, Mapping) else {}


def _original_answer(case: Mapping[str, Any]) -> str:
    promoted = _promoted_validation(case["promotion_row"])
    raw = promoted.get("raw_output_excerpt")
    if isinstance(raw, str) and raw.strip():
        return raw
    parsed = promoted.get("parsed_contract_output")
    if isinstance(parsed, Mapping) and parsed:
        return json.dumps(dict(parsed), sort_keys=True)
    return str(case["runtime_case"].get("proposed_output") or "")


def _validation_contract_case(case: Mapping[str, Any], *, final_accept: bool) -> JsonDict:
    validation = {
        key: case.get(key) for key in runtime_contracts.REQUIRED_CONTRACT_CASE_FIELDS if key in case
    }
    validation["row_type"] = "contract_case"
    validation["contract_schema_version"] = runtime_contracts.CONTRACT_CASE_SCHEMA_VERSION
    validation["final_deterministic_accept"] = bool(final_accept)
    validation["final_deterministic_decision"] = "accept" if final_accept else "reject"
    return validation


def _artifact_from_summary(
    *,
    status: str,
    ready: bool,
    cases_loaded: int,
    claims_extracted: int,
    full_context_accept_rate: float | None,
    claim_isolated_accept_rate: float | None,
    claim_isolation_delta: float | None,
    verifier_calls_full_context: int,
    verifier_calls_claim_isolated: int,
    budget_delta: int | None,
    false_accept_count: int,
    false_accept_rate: float | None,
    manifest_path: Path,
    project_root: Path,
    models_used: list[str],
    blockers: list[str],
    run_date: str,
) -> JsonDict:
    return {
        "status": status,
        "run_date": run_date,
        "schema": "march_claim_isolation_verifier_ablation_v1",
        "spec": ["REQ-VERIFY-1525", "SCENARIO-VERIFY-1525"],
        "model_specs": [dict(spec) for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": bool(models_used),
        "claim_isolation_ablation_ready": bool(ready),
        "cases_loaded": int(cases_loaded),
        "claims_extracted": int(claims_extracted),
        "full_context_accept_rate": full_context_accept_rate,
        "claim_isolated_accept_rate": claim_isolated_accept_rate,
        "claim_isolation_delta": claim_isolation_delta,
        "verifier_calls_full_context": int(verifier_calls_full_context),
        "verifier_calls_claim_isolated": int(verifier_calls_claim_isolated),
        "budget_delta": budget_delta,
        "false_accept_count": int(false_accept_count),
        "false_accept_rate": false_accept_rate,
        "claim_isolation_manifest_path": _display_path(manifest_path, project_root=project_root),
        "models_used": list(models_used),
        "blockers": list(blockers),
        "honest_verdict": (
            "complete: march_claim_isolation_ablation_ready"
            if ready
            else "complete: march_claim_isolation_ablation_blocked"
        ),
    }


def _load_json_or_blocker(path: Path, blockers: list[str]) -> JsonDict | None:
    if not path.exists():  # pragma: no cover
        blockers.append(f"missing_artifact:{path}")
        return None
    try:
        return _read_json(path)
    except (json.JSONDecodeError, OSError, AssertionError) as exc:  # pragma: no cover
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


def _rate_or_none(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _explicit_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(path: Path | str, *, project_root: Path | str | None = None) -> str:
    target = Path(path)
    root = Path(project_root) if project_root is not None else Path.cwd()
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return target.as_posix()


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):  # pragma: no cover
        raise AssertionError(f"JSON artifact must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):  # pragma: no cover
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


def _cached_sota_pair(**kwargs: Any) -> list[JsonDict] | None:  # pragma: no cover
    from carnot.inference.sota_models import cached_sota_pair

    return cached_sota_pair(**kwargs)


def _resolve_cached_gguf(hf_id: str) -> str | None:  # pragma: no cover
    from carnot.inference.sota_models import resolve_cached_gguf

    return resolve_cached_gguf(hf_id)


def _run_live_checker_modes(
    claim_cases: Sequence[JsonDict],
    model: JsonDict,
) -> tuple[list[JsonDict], list[str]]:  # pragma: no cover
    try:
        from llama_cpp import Llama

        llm = Llama(
            model_path=str(model["model_path"]),
            n_gpu_layers=-1,
            main_gpu=max(int(model.get("gpu", 0)), 0),
            n_ctx=2048,
            verbose=False,
        )
    except Exception as exc:
        return [], [f"live_checker_load_failed:{model.get('hf_id')}:{type(exc).__name__}:{exc}"]

    def checker(prompt: str, _model: JsonDict, _mode: str, _claim: JsonDict) -> str:
        completion = llm(
            prompt,
            max_tokens=96,
            temperature=0.0,
            echo=False,
            stop=["</s>", "<eos>"],
        )
        return _completion_text(completion)

    try:
        return _run_checker_modes(claim_cases, model=model, checker_fn=checker), []
    except Exception as exc:
        return [], [
            f"live_checker_generation_failed:{model.get('hf_id')}:{type(exc).__name__}:{exc}"
        ]
    finally:
        if hasattr(llm, "close"):
            llm.close()


def _completion_text(result: Any) -> str:  # pragma: no cover
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
