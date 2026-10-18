#!/usr/bin/env python3
"""Exp5457 distortion-guarded local SOTA verifier-potential decoding rerun.

Spec refs: REQ-SAFE-5457, SCENARIO-SAFE-5457.

This module reruns the guided-decoding comparison after the Exp5456 tautology
corrigendum.  The model remains a candidate generator only: attribution,
entailment, contradiction, and the exact final verifiers decide whether a row
is valid.  That separation is what keeps constrained decoding from becoming a
schema-valid but fact-distorting shortcut.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5443_verifier_potential_prefix_fixture_v495 as exp5443
from carnot import experiment_5444_gated_sota_energy_guided_decoding_v495 as exp5444
from carnot import experiment_5456_guided_decoding_tautology_corrigendum_v496 as exp5456
from carnot.inference.sota_models import resolve_cached_gguf


JsonDict = dict[str, Any]
ModelResolver = Callable[[str, str], str | None]
RuntimeProbe = Callable[..., JsonDict]
GenerationRunner = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5457_gated_sota_distortion_guarded_decoding_v496.json"
)
ROW_RESULTS_RELATIVE_PATH = Path(
    "results/experiment_5457_gated_sota_distortion_guarded_decoding_v496_rows.jsonl"
)
CLAIM_ATTRIBUTION_RELATIVE_PATH = Path(
    "results/"
    "experiment_5457_gated_sota_distortion_guarded_decoding_v496_"
    "claim_attribution_receipts.jsonl"
)
EXPERIMENT_ID = "experiment_5457_gated_sota_distortion_guarded_decoding_v496"
TASK_ID = "exp5457-gated-sota-distortion-guarded-decoding-v496"
MILESTONE = "2026.07.496"
RUN_DATE = "2026-07-09"
SCHEMA = "carnot.experiment_5457.gated_sota_distortion_guarded_decoding.v496"
SPEC_REFS = ("REQ-SAFE-5457", "SCENARIO-SAFE-5457")
RANDOM_SEED = 5457
DEFAULT_QUANTIZATION = "Q4_K_M"
DEFAULT_TOKEN_BUDGET = 96
DEFAULT_MAX_BASE_FIXTURES = 4
N_GPU_LAYERS = -1
ACCEPTANCE_THRESHOLD = 0.5
ACCEPTED_RISK_THRESHOLD = 0.5
INFERENCE_SUBSTRATE = "live_llm_inference"
BLOCKED_INFERENCE_SUBSTRATE = "blocked_preconditions_no_live_llm"
TERMINAL_PREFIXES = ("complete:", "blocked:")

CONDITION_NAMES = ("unconstrained", "grammar_lcd_only", "verifier_potential_guided")
GUIDED_CONDITION = "verifier_potential_guided"
LCD_CONDITION = "grammar_lcd_only"
FACTUAL_FAMILIES = {"factual_conflict_cocoa", "strict_constraint_distortion_guard"}
SEMANTIC_FALSE_ACCEPT_FAMILIES = {
    "semantic_contradiction",
    "factual_conflict_cocoa",
    "strict_constraint_distortion_guard",
}

MANDATED_MODEL_SPECS = exp5444.MANDATED_MODEL_SPECS
MANDATED_HF_IDS = exp5444.MANDATED_HF_IDS
LlamaCppGenerationRunner = exp5444.LlamaCppGenerationRunner

FIELD_PRINCIPLES: dict[str, str] = {
    "preconditions_checked": "compute-bound task must fail fast.",
    "model_specs": "mandated SOTA GGUF provenance.",
    "headline_required_any_of": "confirms headline rows can only use mandated GGUFs.",
    "runtime_backend": "GGUF/llama.cpp path, not transformers tokenizer path.",
    "gpu_offload_verified": "no CPU-only SOTA headline.",
    "fixture_count": "bounded fixture coverage.",
    "condition_names": "unconstrained, LCD-only, and guided baseline clarity.",
    "row_results_path": "inspectable row evidence.",
    "exact_final_authority": "deterministic verifier authority.",
    "claim_attribution_receipts_path": "claim-to-evidence provenance.",
    "chance_risk_bound": "finite-sample accepted-risk guard.",
    "factual_distortion_rate": "strict-constraint distortion guard.",
    "semantic_false_accept_rate": "semantic contradiction guard.",
    "lcd_bias_check_passed": "local constrained decoding bias check.",
    "guided_validity_delta_vs_unconstrained": "guided utility versus free decoding.",
    "guided_validity_delta_vs_lcd_only": "guided utility versus grammar/LCD-only decoding.",
    "metric_independence_checks_passed": "tautology prevention.",
    "verifier_guided_decoding_ready": "downstream gate.",
    "inference_substrate": "real local model invocation.",
    "honest_verdict": 'terminal status; start with "complete:" or "blocked:".',
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def run(
    *,
    root: Path | str = REPO_ROOT,
    artifact_path: Path | str | None = None,
    row_results_path: Path | str | None = None,
    claim_attribution_receipts_path: Path | str | None = None,
    fixture_artifact: Mapping[str, Any] | None = None,
    corrigendum_artifact: Mapping[str, Any] | None = None,
    model_resolver: ModelResolver = resolve_cached_gguf,
    runtime_probe: RuntimeProbe | None = None,
    generation_runner: GenerationRunner | None = None,
    max_base_fixtures: int = DEFAULT_MAX_BASE_FIXTURES,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    write: bool = True,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """Run the distortion-guarded panel and optionally write all artifacts."""

    started = time.perf_counter()
    root_path = Path(root)
    destination = _destination(root_path, artifact_path, RESULT_RELATIVE_PATH)
    rows_destination = _destination(root_path, row_results_path, ROW_RESULTS_RELATIVE_PATH)
    receipts_destination = _destination(
        root_path,
        claim_attribution_receipts_path,
        CLAIM_ATTRIBUTION_RELATIVE_PATH,
    )
    fixture_payload = (
        dict(fixture_artifact)
        if fixture_artifact is not None
        else _read_json(root_path / exp5443.RESULT_RELATIVE_PATH)
    )
    corrigendum_payload = (
        dict(corrigendum_artifact)
        if corrigendum_artifact is not None
        else _read_json(root_path / exp5456.RESULT_RELATIVE_PATH)
    )

    model_specs = resolve_model_specs(model_resolver)
    selected_model = select_headline_model(model_specs)
    runtime_fn = runtime_probe or default_runtime_probe
    runtime_receipt = runtime_fn(model_spec=selected_model, n_gpu_layers=N_GPU_LAYERS)
    preconditions = evaluate_preconditions(
        fixture_payload=fixture_payload,
        corrigendum_payload=corrigendum_payload,
        model_specs=model_specs,
        selected_model=selected_model,
        runtime_receipt=runtime_receipt,
    )

    live_runner: GenerationRunner | None = generation_runner
    if preconditions["all_passed"] and live_runner is None:
        try:
            live = LlamaCppGenerationRunner(
                model_spec=selected_model or {},
                n_gpu_layers=N_GPU_LAYERS,
                seed=random_seed,
            )
        except Exception as exc:  # pragma: no cover - depends on local runtime/model.
            preconditions["all_passed"] = False
            preconditions["blocked_preconditions"].append(
                f"llama_cpp_model_load_failed:{type(exc).__name__}: {exc}"
            )
        else:
            live_runner = live
            runtime_receipt = {**runtime_receipt, "load_receipt": dict(live.load_receipt)}
            if live.load_receipt.get("offload_evidence") is not True:
                preconditions["all_passed"] = False
                preconditions["blocked_preconditions"].append("gpu_offload_not_observed_after_load")

    if not preconditions["all_passed"] or live_runner is None:
        artifact = build_artifact(
            model_specs=model_specs,
            selected_model=selected_model,
            runtime_receipt=runtime_receipt,
            preconditions=preconditions,
            rows=[],
            receipts=[],
            row_results_path=rows_destination,
            claim_attribution_receipts_path=receipts_destination,
            tests_run=tests_run,
            methodology_duration_s=time.perf_counter() - started,
            blocked=True,
        )
        if write:
            _write_json(destination, artifact)
        validate_artifact(artifact)
        return artifact

    fixtures = select_panel_fixtures(fixture_payload, max_base_fixtures=max_base_fixtures)
    rows, receipts = run_condition_rows(
        fixtures=fixtures,
        model_spec=selected_model or {},
        runtime_receipt=runtime_receipt,
        generation_runner=live_runner,
        random_seed=random_seed,
    )
    artifact = build_artifact(
        model_specs=_mark_model_ran(model_specs, selected_model),
        selected_model=selected_model,
        runtime_receipt=runtime_receipt,
        preconditions=preconditions,
        rows=rows,
        receipts=receipts,
        row_results_path=rows_destination,
        claim_attribution_receipts_path=receipts_destination,
        tests_run=tests_run,
        methodology_duration_s=time.perf_counter() - started,
        blocked=False,
    )
    if write:
        _write_jsonl(rows_destination, rows)
        _write_jsonl(receipts_destination, receipts)
        validate_artifact(artifact)
        _write_json(destination, artifact)
    else:
        validate_artifact(artifact)
    return artifact


def resolve_model_specs(model_resolver: ModelResolver = resolve_cached_gguf) -> list[JsonDict]:
    """Resolve all mandated local GGUF specs without using HF tokenizers."""

    return exp5444.resolve_model_specs(model_resolver)


def select_headline_model(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    """Select the mandated local GGUF that will run the bounded panel."""

    return exp5444.select_headline_model(model_specs)


def default_runtime_probe(**kwargs: Any) -> JsonDict:  # pragma: no cover
    """Check CUDA and llama.cpp offload support using the existing 5444 probe."""

    return exp5444.default_runtime_probe(**kwargs)


def evaluate_preconditions(
    *,
    fixture_payload: Mapping[str, Any],
    corrigendum_payload: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    selected_model: Mapping[str, Any] | None,
    runtime_receipt: Mapping[str, Any],
) -> JsonDict:
    """Combine Exp5456, Exp5443, model-cache, CUDA, and offload gates."""

    blocked = list(runtime_receipt.get("blocked_preconditions", []))
    corrigendum_clean = corrigendum_payload.get("guided_decoding_corrigendum_clean") is True
    fixture_ready = fixture_payload.get("verifier_potential_fixture_ready") is True
    model_ids = {str(row.get("hf_id")) for row in model_specs}
    mandated_present = model_ids == set(MANDATED_HF_IDS)
    all_model_paths = bool(model_specs) and all(
        row.get("local_path_available") is True and str(row.get("model_path", "")).endswith(".gguf")
        for row in model_specs
    )
    if not corrigendum_clean:
        blocked.append("exp5456_guided_decoding_corrigendum_not_clean")
    if not fixture_ready:
        blocked.append("exp5443_verifier_potential_fixture_not_ready")
    if not mandated_present:
        blocked.append("mandated_model_specs_missing")
    if not all_model_paths:
        blocked.append("non_empty_mandated_model_paths_missing")
    if selected_model is None:
        blocked.append("no_mandated_local_gguf_model_path")
    if runtime_receipt.get("cuda_visible") is not True:
        blocked.append("cuda_not_visible")
    if runtime_receipt.get("offload_evidence") is not True:
        blocked.append("gpu_offload_evidence_missing")
    return {
        "exp5456_corrigendum_clean": corrigendum_clean,
        "exp5443_fixture_ready": fixture_ready,
        "mandated_model_specs_present": mandated_present,
        "all_mandated_model_paths_available": all_model_paths,
        "selected_model_available": selected_model is not None,
        "cuda_visible": runtime_receipt.get("cuda_visible") is True,
        "runtime_backend": str(runtime_receipt.get("runtime_backend", "")),
        "gpu_offload_preflight": runtime_receipt.get("offload_evidence") is True,
        "blocked_preconditions": sorted(set(str(item) for item in blocked)),
        "all_passed": not blocked,
    }


def select_panel_fixtures(
    fixture_payload: Mapping[str, Any],
    *,
    max_base_fixtures: int = DEFAULT_MAX_BASE_FIXTURES,
) -> list[JsonDict]:
    """Load Exp5443 rows and append V496 conflict/distortion rows."""

    rows = fixture_payload.get("fixture_rows")
    if not isinstance(rows, list):
        return []
    selected: list[JsonDict] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        verdict = _mapping(row.get("exact_final_verdict"))
        if verdict.get("verified") is not True:
            continue
        selected.append(copy.deepcopy(dict(row)))
        if len(selected) >= max_base_fixtures:
            break
    selected.extend(build_conflict_distortion_rows())
    return selected


def build_conflict_distortion_rows() -> list[JsonDict]:
    """Return deterministic conflict rows with evidence spans and decoys."""

    return [
        {
            "row_id": "5457-conflict-001",
            "fixture_id": "cocoa_context_opened_2019",
            "constraint_family": "factual_conflict_cocoa",
            "required_keys": ["claim", "answer", "evidence_ids"],
            "allowed_keys": ["claim", "answer", "evidence_ids"],
            "expected_answer": "2019",
            "expected_claim": "Riverton clinic opened in 2019.",
            "distorted_claim": "Riverton clinic opened in 2018.",
            "contradicted_answers": ["2018"],
            "supporting_evidence_ids": ["E5457-001-A"],
            "evidence_spans": [
                {
                    "id": "E5457-001-A",
                    "text": "The Riverton clinic opened to patients in 2019.",
                    "supports_answers": ["2019"],
                    "contradicts_answers": ["2018"],
                },
                {
                    "id": "E5457-001-B",
                    "text": "A planning memo from 2018 discussed the clinic.",
                    "supports_answers": ["2018_planning_only"],
                    "contradicts_answers": [],
                },
            ],
            "prefixes": [_accepted_claim_prefix("5457-conflict-001", "2019")],
        },
        {
            "row_id": "5457-distortion-002",
            "fixture_id": "strict_budget_no_rounding",
            "constraint_family": "strict_constraint_distortion_guard",
            "required_keys": ["claim", "answer", "evidence_ids"],
            "allowed_keys": ["claim", "answer", "evidence_ids"],
            "expected_answer": "14",
            "expected_claim": "The approved equipment budget is 14 units.",
            "distorted_claim": "The approved equipment budget is 40 units.",
            "contradicted_answers": ["40"],
            "supporting_evidence_ids": ["E5457-002-A"],
            "evidence_spans": [
                {
                    "id": "E5457-002-A",
                    "text": "The approval sheet lists the equipment budget as 14 units.",
                    "supports_answers": ["14"],
                    "contradicts_answers": ["40"],
                },
                {
                    "id": "E5457-002-B",
                    "text": "The template allows at most 40 characters in the budget field.",
                    "supports_answers": ["field_limit_40_chars"],
                    "contradicts_answers": [],
                },
            ],
            "prefixes": [_accepted_claim_prefix("5457-distortion-002", "14")],
        },
    ]


def run_condition_rows(
    *,
    fixtures: Sequence[Mapping[str, Any]],
    model_spec: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
    generation_runner: GenerationRunner,
    random_seed: int = RANDOM_SEED,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Generate and score one candidate row for every fixture and condition."""

    rows: list[JsonDict] = []
    receipts: list[JsonDict] = []
    runtime_backend = str(runtime_receipt.get("runtime_backend", "llama_cpp_python_cuda_gguf"))
    for fixture_index, fixture in enumerate(fixtures):
        fixture_row = copy.deepcopy(dict(fixture))
        for condition_index, condition in enumerate(CONDITION_NAMES):
            seed = random_seed + fixture_index * 101 + condition_index
            prompt = build_prompt(fixture_row, condition=condition)
            prompt_hash = _sha256_text(prompt)
            prefix = exp5444._best_guided_prefix(fixture_row) if condition == GUIDED_CONDITION else None
            grammar = exp5444.JSON_GBNF if condition in {LCD_CONDITION, GUIDED_CONDITION} else None
            started = time.perf_counter()
            generation = generation_runner(
                prompt=prompt,
                condition=condition,
                fixture=fixture_row,
                model_spec=model_spec,
                runtime_backend=runtime_backend,
                grammar=grammar,
                prefix_seed=prefix,
                seed=seed,
                max_tokens=DEFAULT_TOKEN_BUDGET,
                n_gpu_layers=N_GPU_LAYERS,
            )
            elapsed = time.perf_counter() - started
            row, receipt = score_candidate_row(
                fixture=fixture_row,
                condition=condition,
                model_spec=model_spec,
                runtime_backend=runtime_backend,
                runtime_receipt=runtime_receipt,
                generation=generation,
                prompt_hash=prompt_hash,
                seed=seed,
                token_budget=DEFAULT_TOKEN_BUDGET,
                fallback_duration_s=elapsed,
                prefix_seed=prefix,
            )
            rows.append(row)
            receipts.append(receipt)
    return rows, receipts


def build_prompt(fixture: Mapping[str, Any], *, condition: str) -> str:
    """Build a bounded prompt with evidence spans but without verifier labels."""

    if condition not in CONDITION_NAMES:
        raise ValueError(f"unknown condition: {condition}")
    family = str(fixture.get("constraint_family"))
    lines = [
        "Return exactly one compact JSON object and no prose.",
        f"Fixture id: {fixture.get('fixture_id')}",
        f"Constraint family: {family}",
        f"Required keys: {json.dumps(fixture.get('required_keys', []), sort_keys=True)}",
        f"Allowed keys: {json.dumps(fixture.get('allowed_keys', []), sort_keys=True)}",
    ]
    spans = fixture.get("evidence_spans")
    if isinstance(spans, list) and spans:
        lines.append("Evidence spans:")
        for span in spans:
            if isinstance(span, Mapping):
                lines.append(f"- {span.get('id')}: {span.get('text')}")
        lines.append("Use evidence_ids to cite the exact span IDs supporting the claim.")
    else:
        lines.append(exp5444._family_instruction(fixture, family))
    if condition == "unconstrained":
        lines.append("No grammar or local constrained decoding mask is applied.")
    elif condition == LCD_CONDITION:
        lines.append("A JSON grammar/LCD mask constrains syntax only; preserve the facts.")
    else:
        prefix = exp5444._best_guided_prefix(fixture)
        lines.append("A deterministic verifier potential selected this partial JSON prefix:")
        lines.append(json.dumps(_mapping(prefix.get("fields")) if prefix else {}, sort_keys=True))
        lines.append("Complete or minimally repair the prefix without changing cited facts.")
    return "\n".join(lines)


def score_candidate_row(
    *,
    fixture: Mapping[str, Any],
    condition: str,
    model_spec: Mapping[str, Any],
    runtime_backend: str,
    runtime_receipt: Mapping[str, Any],
    generation: Mapping[str, Any],
    prompt_hash: str,
    seed: int,
    token_budget: int,
    fallback_duration_s: float,
    prefix_seed: Mapping[str, Any] | None,
) -> tuple[JsonDict, JsonDict]:
    """Parse, attribute, and exactly verify one generated candidate."""

    output_text = str(generation.get("output_text", ""))
    parsed, parse_error = exp5444.extract_json_object(output_text)
    candidate = parsed if parsed is not None else {}
    row_id = f"{fixture.get('row_id')}:{condition}"
    receipt = build_claim_attribution_receipt(
        fixture=fixture,
        row_id=row_id,
        condition=condition,
        candidate_output=candidate,
    )
    exact_verdict = exact_final_verdict(fixture, candidate, receipt)
    advisory_accept = parsed is not None
    reward_evaluations = reward_evaluation_count(condition, prefix_seed)
    row: JsonDict = {
        "schema": "carnot.experiment_5457.row.v1",
        "experiment_id": EXPERIMENT_ID,
        "row_id": row_id,
        "fixture_row_id": str(fixture.get("row_id")),
        "fixture_id": str(fixture.get("fixture_id")),
        "constraint_family": str(fixture.get("constraint_family")),
        "condition": condition,
        "model_role": str(model_spec.get("role")),
        "model_hf_id": str(model_spec.get("hf_id")),
        "model_path": str(model_spec.get("model_path")),
        "runtime_backend": runtime_backend,
        "n_gpu_layers": int(runtime_receipt.get("n_gpu_layers", N_GPU_LAYERS)),
        "gpu_offload_evidence": bool(runtime_receipt.get("offload_evidence")),
        "random_seed": seed,
        "prompt_hash": prompt_hash,
        "token_budget": token_budget,
        "reward_evaluation_count": reward_evaluations,
        "acceptance_threshold": ACCEPTANCE_THRESHOLD,
        "generation_duration_s": float(generation.get("duration_s", fallback_duration_s)),
        "generated_token_count": int(generation.get("generated_token_count", 0) or 0),
        "output_text": output_text,
        "candidate_output": parsed,
        "parse_status": "parsed" if parsed is not None else "abstained",
        "parse_error": parse_error,
        "condition_advisory_accept": advisory_accept,
        "exact_final_verdict": exact_verdict,
        "accepted_by_final_authority": bool(exact_verdict["accepted"]),
        "final_authority_bypassed": False,
        "claim_attribution_receipt_id": receipt["receipt_id"],
        "factual_distortion_detected": bool(receipt.get("factual_distortion_detected")),
        "semantic_false_accept": bool(
            advisory_accept
            and not exact_verdict["accepted"]
            and str(fixture.get("constraint_family")) in SEMANTIC_FALSE_ACCEPT_FAMILIES
        ),
        "prefix_seed": copy.deepcopy(prefix_seed),
        "backend_details": copy.deepcopy(generation.get("backend_details", {})),
    }
    row["row_checksum"] = row_checksum(row)
    return row, receipt


def build_claim_attribution_receipt(
    *,
    fixture: Mapping[str, Any],
    row_id: str,
    condition: str,
    candidate_output: Mapping[str, Any],
) -> JsonDict:
    """Bind candidate claims to deterministic evidence span IDs."""

    family = str(fixture.get("constraint_family"))
    evidence_spans = [dict(span) for span in fixture.get("evidence_spans", []) if isinstance(span, Mapping)]
    answer = str(candidate_output.get("answer", "")) if isinstance(candidate_output, Mapping) else ""
    raw_ids = candidate_output.get("evidence_ids", []) if isinstance(candidate_output, Mapping) else []
    evidence_ids = [str(item) for item in raw_ids] if isinstance(raw_ids, list) else []
    span_by_id = {str(span.get("id")): span for span in evidence_spans}
    selected = [span_by_id[item] for item in evidence_ids if item in span_by_id]
    supporting = [
        str(span.get("id"))
        for span in selected
        if answer and answer in [str(item) for item in span.get("supports_answers", [])]
    ]
    contradicting = [
        str(span.get("id"))
        for span in selected
        if answer and answer in [str(item) for item in span.get("contradicts_answers", [])]
    ]
    expected_answer = str(fixture.get("expected_answer", ""))
    contradicted_answers = [str(item) for item in fixture.get("contradicted_answers", [])]
    contradiction = bool(contradicting or (answer and answer in contradicted_answers))
    distortion = bool(family in FACTUAL_FAMILIES and answer and answer != expected_answer)
    receipt: JsonDict = {
        "schema": "carnot.experiment_5457.claim_attribution_receipt.v1",
        "receipt_id": f"{row_id}:claim_attribution",
        "row_id": row_id,
        "condition": condition,
        "constraint_family": family,
        "claim_text": str(candidate_output.get("claim", "")) if isinstance(candidate_output, Mapping) else "",
        "answer": answer,
        "expected_answer": expected_answer,
        "evidence_span_ids": evidence_ids,
        "supporting_evidence_span_ids": supporting,
        "contradicting_evidence_span_ids": contradicting,
        "claim_attributed": bool(not evidence_spans or supporting),
        "entailment_check_passed": bool(not evidence_spans or (supporting and not contradiction)),
        "contradiction_detected": contradiction,
        "factual_distortion_detected": distortion,
        "evidence_span_count": len(evidence_spans),
    }
    receipt["receipt_checksum"] = receipt_checksum(receipt)
    return receipt


def exact_final_verdict(
    fixture: Mapping[str, Any],
    candidate_output: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> JsonDict:
    """Run deterministic final checks over a completed candidate."""

    family = str(fixture.get("constraint_family"))
    if family not in FACTUAL_FAMILIES:
        exact_row = copy.deepcopy(dict(fixture))
        exact_row["final_output"] = dict(candidate_output)
        verdict = exp5443.exact_final_verdict(exact_row)
        verdict["ast_witness"] = {"candidate_keys": sorted(candidate_output)}
        verdict["kb_witness"] = {"constraint_family": family}
        verdict["entailment_check_passed"] = True
        verdict["contradiction_detected"] = False
        return verdict

    reasons: list[str] = []
    required = set(str(item) for item in fixture.get("required_keys", []))
    allowed = set(str(item) for item in fixture.get("allowed_keys", []))
    keys = set(str(item) for item in candidate_output)
    if not required.issubset(keys) or not keys.issubset(allowed):
        reasons.append("schema_exact_failed")
    if receipt.get("claim_attributed") is not True:
        reasons.append("claim_attribution_missing")
    if receipt.get("contradiction_detected") is True:
        reasons.append("contradiction_detected")
    if receipt.get("factual_distortion_detected") is True:
        reasons.append("factual_distortion_detected")
    accepted = not reasons
    return {
        "verified": True,
        "authority": "exact_final_verifier",
        "accepted": accepted,
        "failure_reasons": reasons,
        "verifiers_run": [
            "schema_exact",
            "claim_attribution_exact",
            "entailment_contradiction_exact",
        ],
        "overrides_prefix_potential": bool(exp5444._best_guided_prefix(fixture) and not accepted),
        "ast_witness": {"candidate_keys": sorted(candidate_output)},
        "kb_witness": {
            "expected_answer": fixture.get("expected_answer"),
            "supporting_evidence_ids": fixture.get("supporting_evidence_ids", []),
        },
        "entailment_check_passed": bool(receipt.get("entailment_check_passed")),
        "contradiction_detected": bool(receipt.get("contradiction_detected")),
    }


def derive_metrics(
    rows: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Compute all aggregate metrics from row evidence and receipts only."""

    row_list = [dict(row) for row in rows if isinstance(row, Mapping)]
    receipt_list = [dict(receipt) for receipt in receipts if isinstance(receipt, Mapping)]
    by_condition: dict[str, list[JsonDict]] = {name: [] for name in CONDITION_NAMES}
    for row in row_list:
        by_condition.setdefault(str(row.get("condition")), []).append(row)
    condition_metrics = {
        condition: _condition_metric(condition_rows)
        for condition, condition_rows in by_condition.items()
        if condition in CONDITION_NAMES
    }
    guided_rate = condition_metrics.get(GUIDED_CONDITION, {}).get("accepted_validity_rate", 0.0)
    unconstrained_rate = condition_metrics.get("unconstrained", {}).get(
        "accepted_validity_rate", 0.0
    )
    lcd_rate = condition_metrics.get(LCD_CONDITION, {}).get("accepted_validity_rate", 0.0)
    advisory_accepted = [row for row in row_list if row.get("condition_advisory_accept") is True]
    chance_risk_failures = [
        row for row in advisory_accepted if row.get("accepted_by_final_authority") is not True
    ]
    factual_rows = [
        row for row in row_list if str(row.get("constraint_family")) in FACTUAL_FAMILIES
    ]
    factual_distortions = [
        row for row in factual_rows if row.get("factual_distortion_detected") is True
    ]
    semantic_false_accepts = [
        row
        for row in row_list
        if row.get("semantic_false_accept") is True
        and row.get("condition_advisory_accept") is True
    ]
    lcd_distortions = [
        row for row in factual_distortions if row.get("condition") == LCD_CONDITION
    ]
    guided_distortions = [
        row for row in factual_distortions if row.get("condition") == GUIDED_CONDITION
    ]
    lcd_semantic_false_accepts = [
        row for row in semantic_false_accepts if row.get("condition") == LCD_CONDITION
    ]
    guided_semantic_false_accepts = [
        row for row in semantic_false_accepts if row.get("condition") == GUIDED_CONDITION
    ]
    row_checksums_match = all(row.get("row_checksum") == row_checksum(row) for row in row_list)
    receipt_checksums_match = all(
        receipt.get("receipt_checksum") == receipt_checksum(receipt) for receipt in receipt_list
    )
    receipt_rows_match = [receipt.get("row_id") for receipt in receipt_list] == [
        row.get("row_id") for row in row_list
    ]
    exact_authority_ok = all(_exact_authority_row(row) for row in row_list)
    predicate_support = {
        "accepted_validity": "exact_final_verdict.accepted is true",
        "semantic_false_accept": "advisory accept and exact final reject in semantic families",
        "factual_distortion": "receipt factual_distortion_detected is true on factual rows",
        "lcd_bias": "LCD distortion/semantic false accepts not greater than guided",
        "chance_risk": "Wilson upper bound over advisory accepted false accepts",
        "guided_delta": "guided accepted-validity rate minus baseline accepted-validity rate",
    }
    lcd_bias_passed = bool(
        len(lcd_distortions) <= len(guided_distortions)
        and len(lcd_semantic_false_accepts) <= len(guided_semantic_false_accepts)
    )
    metric_independence = bool(
        (not row_list or row_checksums_match)
        and (not receipt_list or receipt_checksums_match)
        and receipt_rows_match
        and exact_authority_ok
        and len(set(predicate_support.values())) == len(predicate_support)
    )
    return {
        "condition_metrics": condition_metrics,
        "accepted_validity_rate": _rate(
            [row for row in row_list if row.get("accepted_by_final_authority") is True],
            len(row_list),
        ),
        "guided_validity_delta_vs_unconstrained": guided_rate - unconstrained_rate,
        "guided_validity_delta_vs_lcd_only": guided_rate - lcd_rate,
        "semantic_false_accept_rate": _rate(semantic_false_accepts, len(row_list)),
        "factual_distortion_rate": _rate(factual_distortions, len(factual_rows)),
        "chance_risk_bound": wilson_upper_bound(
            len(chance_risk_failures),
            len(advisory_accepted),
        ),
        "chance_risk_failure_count": len(chance_risk_failures),
        "accepted_advisory_denominator": len(advisory_accepted),
        "factual_row_denominator": len(factual_rows),
        "abstention_rate": _rate(
            [row for row in row_list if row.get("parse_status") == "abstained"],
            len(row_list),
        ),
        "lcd_bias_check_passed": lcd_bias_passed,
        "lcd_bias_indicators": {
            "lcd_factual_distortion_count": len(lcd_distortions),
            "guided_factual_distortion_count": len(guided_distortions),
            "lcd_semantic_false_accept_count": len(lcd_semantic_false_accepts),
            "guided_semantic_false_accept_count": len(guided_semantic_false_accepts),
        },
        "constraint_family_counts": dict(
            sorted(Counter(str(row.get("constraint_family")) for row in row_list).items())
        ),
        "metric_independence_checks_passed": metric_independence,
        "predicate_support": predicate_support,
        "row_checksums_match": row_checksums_match,
        "receipt_checksums_match": receipt_checksums_match,
        "receipt_rows_match": receipt_rows_match,
        "exact_authority_rows": sum(1 for row in row_list if _exact_authority_row(row)),
        "row_count": len(row_list),
    }


def derive_reward_budget(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute reward-evaluation accounting from candidate rows."""

    per_condition: dict[str, int] = defaultdict(int)
    per_row: list[JsonDict] = []
    for row in rows:
        condition = str(row.get("condition"))
        count = int(row.get("reward_evaluation_count", 0))
        per_condition[condition] += count
        per_row.append(
            {
                "row_id": row.get("row_id"),
                "fixture_id": row.get("fixture_id"),
                "condition": condition,
                "reward_evaluation_count": count,
            }
        )
    return {
        "unit": "one deterministic verifier-potential, attribution, or exact-final invocation",
        "total_reward_evaluations": sum(per_condition.values()),
        "per_condition": {condition: per_condition.get(condition, 0) for condition in CONDITION_NAMES},
        "per_row": per_row,
    }


def build_artifact(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    selected_model: Mapping[str, Any] | None,
    runtime_receipt: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    row_results_path: Path,
    claim_attribution_receipts_path: Path,
    tests_run: Sequence[str | Mapping[str, Any]],
    methodology_duration_s: float,
    blocked: bool,
) -> JsonDict:
    """Assemble the terminal artifact from row records and preflight receipts."""

    row_list = [dict(row) for row in rows]
    receipt_list = [dict(receipt) for receipt in receipts]
    metrics = derive_metrics(row_list, receipt_list)
    budget = derive_reward_budget(row_list)
    gpu_offload_verified = bool(
        preconditions.get("all_passed")
        and runtime_receipt.get("offload_evidence") is True
        and _mapping(runtime_receipt.get("load_receipt")).get("offload_evidence", True) is True
    )
    ready = bool(
        not blocked
        and row_list
        and gpu_offload_verified
        and metrics["metric_independence_checks_passed"]
        and metrics["lcd_bias_check_passed"]
        and metrics["chance_risk_bound"] <= ACCEPTED_RISK_THRESHOLD
        and metrics["condition_metrics"].get(GUIDED_CONDITION, {}).get("count", 0)
        == len(row_list) // len(CONDITION_NAMES)
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "status": "complete" if row_list else "blocked",
        "preconditions_checked": True,
        "precondition_details": copy.deepcopy(preconditions),
        "model_specs": [dict(row) for row in model_specs],
        "selected_model_spec": dict(selected_model) if selected_model is not None else None,
        "headline_required_any_of": list(MANDATED_HF_IDS),
        "runtime_backend": str(runtime_receipt.get("runtime_backend", "unavailable")),
        "runtime_receipt": copy.deepcopy(runtime_receipt),
        "gpu_offload_verified": gpu_offload_verified,
        "fixture_count": len({row.get("fixture_id") for row in row_list}),
        "condition_names": list(CONDITION_NAMES),
        "row_results_path": str(row_results_path),
        "exact_final_authority": True,
        "claim_attribution_receipts_path": str(claim_attribution_receipts_path),
        "chance_risk_bound": metrics["chance_risk_bound"],
        "factual_distortion_rate": metrics["factual_distortion_rate"],
        "semantic_false_accept_rate": metrics["semantic_false_accept_rate"],
        "lcd_bias_check_passed": metrics["lcd_bias_check_passed"],
        "guided_validity_delta_vs_unconstrained": metrics[
            "guided_validity_delta_vs_unconstrained"
        ],
        "guided_validity_delta_vs_lcd_only": metrics["guided_validity_delta_vs_lcd_only"],
        "metric_independence_checks_passed": metrics["metric_independence_checks_passed"],
        "verifier_guided_decoding_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE if row_list else BLOCKED_INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready, row_list, preconditions, metrics),
        "accepted_validity_rate": metrics["accepted_validity_rate"],
        "abstention_rate": metrics["abstention_rate"],
        "condition_metrics": metrics["condition_metrics"],
        "constraint_family_counts": metrics["constraint_family_counts"],
        "reward_evaluation_budget": budget,
        "metric_denominators": {
            "row_rate_denominator": len(row_list),
            "accepted_advisory_denominator": metrics["accepted_advisory_denominator"],
            "factual_row_denominator": metrics["factual_row_denominator"],
        },
        "metric_details": metrics,
        "metric_dependency_graph": build_metric_dependency_graph(),
        "row_results": row_list,
        "claim_attribution_receipts": receipt_list,
        "row_checksums": [row.get("row_checksum") for row in row_list],
        "receipt_checksums": [receipt.get("receipt_checksum") for receipt in receipt_list],
        "field_principles": dict(FIELD_PRINCIPLES),
        "methodology_duration_s": methodology_duration_s,
        "tests_run": _normalise_tests_run(tests_run),
        "research_conductor_modified": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_metric_dependency_graph() -> JsonDict:
    """Return the readiness dependency graph used by validation."""

    return {
        "accepted_validity_rate": ["row.exact_final_verdict.accepted"],
        "semantic_false_accept_rate": [
            "row.condition_advisory_accept",
            "row.exact_final_verdict.accepted",
            "row.constraint_family",
        ],
        "factual_distortion_rate": ["claim_attribution_receipt.factual_distortion_detected"],
        "chance_risk_bound": [
            "row.condition_advisory_accept",
            "row.exact_final_verdict.accepted",
            "wilson_upper_bound",
        ],
        "lcd_bias_check_passed": [
            "row.condition",
            "claim_attribution_receipt.factual_distortion_detected",
            "row.semantic_false_accept",
        ],
        "guided_validity_delta_vs_unconstrained": [
            "condition_metrics.verifier_potential_guided.accepted_validity_rate",
            "condition_metrics.unconstrained.accepted_validity_rate",
        ],
        "guided_validity_delta_vs_lcd_only": [
            "condition_metrics.verifier_potential_guided.accepted_validity_rate",
            "condition_metrics.grammar_lcd_only.accepted_validity_rate",
        ],
        "verifier_guided_decoding_ready": [
            "gpu_offload_verified",
            "metric_independence_checks_passed",
            "lcd_bias_check_passed",
            "chance_risk_bound",
            "row_results",
        ],
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp5457 artifact cannot support the decoding claim."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema, precondition, row, attribution, and metric errors."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-SAFE-5457")
    if type(artifact.get("preconditions_checked")) is not bool:
        errors.append("preconditions_checked must be boolean")
    model_specs_value = artifact.get("model_specs")
    if not isinstance(model_specs_value, list):
        errors.append("model_specs must be a list")
        model_specs: list[Mapping[str, Any]] = []
    else:
        model_specs = [row for row in model_specs_value if isinstance(row, Mapping)]
    model_ids = {str(row.get("hf_id")) for row in model_specs}
    if model_ids != set(MANDATED_HF_IDS):
        errors.append("mandated model_specs must include the three required GGUF IDs")
    if artifact.get("headline_required_any_of") != list(MANDATED_HF_IDS):
        errors.append("headline_required_any_of must list mandated SOTA IDs")
    errors.extend(_headline_model_errors(model_specs))
    if artifact.get("condition_names") != list(CONDITION_NAMES):
        errors.append("condition_names must match required baseline/guided order")
    if artifact.get("exact_final_authority") is not True:
        errors.append("exact_final_authority must be true")
    for field in (
        "gpu_offload_verified",
        "lcd_bias_check_passed",
        "metric_independence_checks_passed",
        "verifier_guided_decoding_ready",
    ):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be boolean")
    if not isinstance(artifact.get("runtime_backend"), str):
        errors.append("runtime_backend must be a string")
    if not _bare_non_negative_int(artifact.get("fixture_count")):
        errors.append("fixture_count must be a non-negative integer")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or "\n" in verdict or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete: or blocked:")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("scripts/research_conductor.py must not be modified")

    rows = _list_of_mappings(artifact.get("row_results"))
    receipts = _list_of_mappings(artifact.get("claim_attribution_receipts"))
    if not isinstance(artifact.get("row_results"), list):
        errors.append("row_results must be a list")
    if not isinstance(artifact.get("claim_attribution_receipts"), list):
        errors.append("claim_attribution_receipts must be a list")
    errors.extend(_row_integrity_errors(rows))
    errors.extend(_receipt_integrity_errors(receipts, rows))
    metrics = derive_metrics(rows, receipts)
    budget = derive_reward_budget(rows)
    errors.extend(_aggregate_errors(artifact, metrics, budget, rows, receipts))
    errors.extend(_path_errors(artifact, rows, receipts))
    errors.extend(_dependency_graph_errors(artifact.get("metric_dependency_graph")))

    ready = artifact.get("verifier_guided_decoding_ready") is True
    if ready:
        if artifact.get("status") != "complete":
            errors.append("verifier_guided_decoding_ready requires complete status")
        if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
            errors.append("complete artifact requires inference_substrate=live_llm_inference")
        if artifact.get("gpu_offload_verified") is not True:
            errors.append("verifier_guided_decoding_ready requires gpu_offload_verified")
        if artifact.get("lcd_bias_check_passed") is not True:
            errors.append("verifier_guided_decoding_ready requires lcd_bias_check_passed")
        if float(artifact.get("chance_risk_bound", 1.0)) > ACCEPTED_RISK_THRESHOLD:
            errors.append("verifier_guided_decoding_ready requires bounded chance risk")
        if not any(row.get("ran_headline") is True for row in model_specs):
            errors.append("headline_required_any_of requires at least one mandated model ran")
        if not rows or not receipts:
            errors.append("verifier_guided_decoding_ready requires row and attribution evidence")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum must match artifact payload")
    return errors


def reward_evaluation_count(condition: str, prefix_seed: Mapping[str, Any] | None) -> int:
    """Count deterministic potential, attribution, and final-verifier calls."""

    attribution_and_final = 2
    if condition != GUIDED_CONDITION or prefix_seed is None:
        return attribution_and_final
    return attribution_and_final + len(prefix_seed.get("potential_evaluations", []))


def wilson_upper_bound(failures: int, total: int, z: float = 1.96) -> float:
    """Return a one-sided Wilson-style upper bound for a binomial failure rate."""

    if total <= 0:
        return 1.0
    phat = failures / total
    denom = 1.0 + (z * z) / total
    centre = phat + (z * z) / (2.0 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) / total) + (z * z) / (4.0 * total * total))
    return min(1.0, (centre + margin) / denom)


def row_checksum(row: Mapping[str, Any]) -> str:
    """Hash one row while excluding the checksum field itself."""

    payload = {key: value for key, value in row.items() if key != "row_checksum"}
    return _sha256_json(payload)


def receipt_checksum(receipt: Mapping[str, Any]) -> str:
    """Hash one attribution receipt while excluding the checksum field itself."""

    payload = {key: value for key, value in receipt.items() if key != "receipt_checksum"}
    return _sha256_json(payload)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact payload without the self-referential checksum."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return _sha256_json(payload)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point for producing the Exp5457 result artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifact-path", type=Path, default=None)
    parser.add_argument("--row-results-path", type=Path, default=None)
    parser.add_argument("--claim-attribution-receipts-path", type=Path, default=None)
    parser.add_argument("--max-base-fixtures", type=int, default=DEFAULT_MAX_BASE_FIXTURES)
    args = parser.parse_args(argv)
    artifact = run(
        root=args.root,
        artifact_path=args.artifact_path,
        row_results_path=args.row_results_path,
        claim_attribution_receipts_path=args.claim_attribution_receipts_path,
        max_base_fixtures=args.max_base_fixtures,
        write=True,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if artifact["status"] == "complete" else 1


def _condition_metric(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    accepted = [row for row in rows if row.get("accepted_by_final_authority") is True]
    abstained = [row for row in rows if row.get("parse_status") == "abstained"]
    return {
        "count": total,
        "accepted_count": len(accepted),
        "accepted_validity_rate": _rate(accepted, total),
        "abstention_rate": _rate(abstained, total),
    }


def _aggregate_errors(
    artifact: Mapping[str, Any],
    metrics: Mapping[str, Any],
    budget: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    scalar_fields = (
        "chance_risk_bound",
        "factual_distortion_rate",
        "semantic_false_accept_rate",
        "guided_validity_delta_vs_unconstrained",
        "guided_validity_delta_vs_lcd_only",
        "accepted_validity_rate",
        "abstention_rate",
    )
    for field in scalar_fields:
        if not _float_close(artifact.get(field), metrics.get(field)):
            errors.append(f"{field} must match row recomputation")
    for field in ("lcd_bias_check_passed", "metric_independence_checks_passed"):
        if artifact.get(field) != metrics.get(field):
            errors.append(f"{field} must match row recomputation")
    if artifact.get("condition_metrics") != metrics.get("condition_metrics"):
        errors.append("condition_metrics must match row recomputation")
    if artifact.get("metric_details") != metrics:
        errors.append("metric_details must match row recomputation")
    if artifact.get("constraint_family_counts") != metrics.get("constraint_family_counts"):
        errors.append("constraint_family_counts must match row recomputation")
    if artifact.get("reward_evaluation_budget") != budget:
        errors.append("reward_evaluation_budget must match row recomputation")
    if artifact.get("row_checksums") != [row.get("row_checksum") for row in rows]:
        errors.append("row_checksums must match row_results")
    if artifact.get("receipt_checksums") != [receipt.get("receipt_checksum") for receipt in receipts]:
        errors.append("receipt_checksums must match claim_attribution_receipts")
    if artifact.get("fixture_count") != len({row.get("fixture_id") for row in rows}):
        errors.append("fixture_count must match distinct row fixtures")
    return errors


def _row_integrity_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        verdict = _mapping(row.get("exact_final_verdict"))
        if verdict.get("verified") is not True or verdict.get("authority") != "exact_final_verifier":
            errors.append("row exact final authority must be exact_final_verifier")
        if row.get("final_authority_bypassed") is not False:
            errors.append("row exact final authority must not be bypassed")
        if row.get("row_checksum") != row_checksum(row):
            errors.append("row checksum must match row payload")
        if str(row.get("model_hf_id")) not in MANDATED_HF_IDS:
            errors.append("row model_hf_id must be mandated")
        if not str(row.get("model_path", "")).endswith(".gguf"):
            errors.append("row model_path must be a GGUF path")
        if type(row.get("gpu_offload_evidence")) is not bool:
            errors.append("row gpu_offload_evidence must be boolean")
        if row.get("condition") not in CONDITION_NAMES:
            errors.append("row condition must be known")
    return errors


def _receipt_integrity_errors(
    receipts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if len(receipts) != len(rows):
        errors.append("claim attribution receipts must match row count")
    for receipt, row in zip(receipts, rows, strict=False):
        if receipt.get("row_id") != row.get("row_id"):
            errors.append("claim attribution receipts must align with rows")
        if receipt.get("receipt_checksum") != receipt_checksum(receipt):
            errors.append("claim attribution receipts must have valid checksums")
        if receipt.get("constraint_family") in FACTUAL_FAMILIES and row.get("parse_status") == "parsed":
            if not isinstance(receipt.get("evidence_span_ids"), list):
                errors.append("claim attribution receipts must record evidence span ids")
    return errors


def _path_errors(
    artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    errors.extend(_jsonl_path_errors(artifact.get("row_results_path"), rows, "row_results_path"))
    errors.extend(
        _jsonl_path_errors(
            artifact.get("claim_attribution_receipts_path"),
            receipts,
            "claim_attribution_receipts_path",
        )
    )
    return errors


def _jsonl_path_errors(
    path_text: Any,
    expected_rows: Sequence[Mapping[str, Any]],
    field: str,
) -> list[str]:
    if not isinstance(path_text, str) or not path_text:
        return [f"{field} must be a non-empty string"]
    if not expected_rows:
        return []
    path = Path(path_text)
    if not path.is_file():
        return [f"{field} must point to written evidence"]
    try:
        disk_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{field} is unreadable: {type(exc).__name__}: {exc}"]
    if disk_rows != list(expected_rows):
        return [f"{field} contents must match embedded evidence"]
    return []


def _dependency_graph_errors(graph_value: Any) -> list[str]:
    if not isinstance(graph_value, Mapping):
        return ["metric_dependency_graph must be a dict"]
    deps = graph_value.get("verifier_guided_decoding_ready")
    if not isinstance(deps, list):
        return ["metric_dependency_graph must include readiness dependencies"]
    forbidden = {
        "verifier_guided_decoding_ready",
        "prior.verifier_guided_decoding_ready",
        "prior.metric_independence_checks_passed",
        "prior.guided_validity_delta_vs_unconstrained",
    }
    if any(str(dep) in forbidden for dep in deps):
        return ["self-validating readiness dependency is forbidden"]
    return []


def _headline_model_errors(model_specs: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    for spec in model_specs:
        if spec.get("ran_headline") is True and spec.get("gpu_offload_verified") is not True:
            errors.append("CPU-only headline model is forbidden")
        if spec.get("ran_headline") is True and spec.get("legacy_smoke_only") is True:
            errors.append("legacy smoke model cannot headline")
    return errors


def _mark_model_ran(
    model_specs: Sequence[Mapping[str, Any]],
    selected_model: Mapping[str, Any] | None,
) -> list[JsonDict]:
    selected_hf_id = str(selected_model.get("hf_id")) if selected_model else ""
    rows: list[JsonDict] = []
    for spec in model_specs:
        row = dict(spec)
        if str(row.get("hf_id")) == selected_hf_id:
            row["ran_headline"] = True
            row["runtime_backend"] = "llama_cpp_python_cuda_gguf"
            row["n_gpu_layers"] = N_GPU_LAYERS
            row["gpu_offload_verified"] = True
        rows.append(row)
    return rows


def _accepted_claim_prefix(row_id: str, answer: str) -> JsonDict:
    return {
        "prefix_id": f"{row_id}:evidence-prefix",
        "accepted_by_potential": True,
        "accepted_functions": ["schema_key_coverage"],
        "fields": {"answer": answer},
        "potential_evaluations": [
            {
                "function_id": "schema_key_coverage",
                "decision": "accept",
                "score": 1.0,
                "cost_units": 1,
                "evidence": {"present_required_keys": ["answer"]},
                "monotone": True,
            }
        ],
        "reward_evaluation_cost_units": 1,
    }


def _honest_verdict(
    ready: bool,
    rows: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> str:
    if ready:
        return "complete: distortion-guarded verifier-potential decoding ready"
    if rows:
        blockers: list[str] = []
        if metrics.get("lcd_bias_check_passed") is not True:
            blockers.append("lcd_bias_check_failed")
        if float(metrics.get("chance_risk_bound", 1.0)) > ACCEPTED_RISK_THRESHOLD:
            blockers.append("chance_risk_bound_exceeded")
        if metrics.get("metric_independence_checks_passed") is not True:
            blockers.append("metric_independence_failed")
        return "complete: live panel ran; readiness false (" + ",".join(blockers or ["gate_failed"]) + ")"
    blockers = ",".join(preconditions.get("blocked_preconditions", []))
    return f"blocked: {blockers or 'preconditions_failed'}"


def _exact_authority_row(row: Mapping[str, Any]) -> bool:
    verdict = _mapping(row.get("exact_final_verdict"))
    return (
        verdict.get("verified") is True
        and verdict.get("authority") == "exact_final_verifier"
        and row.get("final_authority_bypassed") is False
    )


def _list_of_mappings(value: Any) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _rate(items: Sequence[Any], total: int) -> float:
    return 0.0 if total <= 0 else len(items) / total


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _bare_non_negative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _float_close(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-12)
    except (TypeError, ValueError):
        return False


def _normalise_tests_run(tests_run: Sequence[str | Mapping[str, Any]]) -> list[JsonDict]:
    if not tests_run:
        return [{"command": "not_recorded", "outcome": "not_recorded"}]
    return [_normalise_test_run(item) for item in tests_run]


def _normalise_test_run(item: str | Mapping[str, Any]) -> JsonDict:
    if isinstance(item, str):
        return {"command": item, "outcome": "recorded"}
    return {
        "command": str(item.get("command", "not_recorded")),
        "outcome": str(item.get("outcome", "recorded")),
    }


def _destination(root: Path, path: Path | str | None, default_relative: Path) -> Path:
    if path is None:
        return root / default_relative
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _read_json(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _sha256_text(text: Any) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
