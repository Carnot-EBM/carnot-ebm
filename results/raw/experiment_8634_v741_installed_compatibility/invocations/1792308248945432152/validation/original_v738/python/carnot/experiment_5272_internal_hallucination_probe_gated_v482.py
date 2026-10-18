"""Exp 5272: gated internal/logit hallucination probe.

Spec refs: REQ-VERIFY-5272, SCENARIO-VERIFY-5272.

This runner is deliberately narrow. It uses the Exp 5271 telemetry receipts as
the capability gate, then measures only the logit or logprob fields that the
local GGUF runtime actually exposed. If those receipts are absent, the honest
result is blocked or null, not replaced with a text judge.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
from typing import Any


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5272
EXPERIMENT_NAME = "experiment_5272_internal_hallucination_probe_gated_v482"
RESULT_RELATIVE_PATH = Path("results/experiment_5272_internal_hallucination_probe_gated_v482.json")
EXP5271_RELATIVE_PATH = Path("results/experiment_5271_sota_telemetry_receipt_harness_v482.json")
SCHEMA = "carnot.experiment_5272.internal_hallucination_probe_gated.v482"
SPEC_REFS = ("REQ-VERIFY-5272", "SCENARIO-VERIFY-5272")
INFERENCE_SUBSTRATE = "live_llm_internal_telemetry_local_gguf_sota"
RANDOM_SEED = 5272
TERMINAL_PREFIXES = ("complete:", "blocked_")
TELEMETRY_KEYS = ("logits", "token_logprobs", "hidden_states", "attention_summaries")
USABLE_AVAILABILITY = {"available"}
HEADLINE_ROLES = ("flagship_moe", "flagship_dense")
OPTIONAL_ROLES = ("middle_moe",)

GGUF_PROBE_CONFIG: JsonDict = {
    "n_gpu_layers": -1,
    "n_ctx": 512,
    "max_tokens": 4,
    "temperature": 0.0,
    "seed": RANDOM_SEED,
    "logprobs": 5,
    "logits_all": True,
}

MANDATED_MODEL_IDS: dict[str, str] = {
    "flagship_moe": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "flagship_dense": "unsloth/gemma-4-31B-it-GGUF",
    "middle_moe": "unsloth/gemma-4-26B-A4B-it-GGUF",
}

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal Exp 5272 verdict; starts with complete: or blocked_ and states "
        "whether the exposed internal/logit hallucination signal was positive, "
        "null, harmful, or unmeasured."
    ),
    "inference_substrate": (
        "Declares live local SOTA GGUF internal telemetry, not cached text scoring, "
        "external judging, or a tiny-model smoke path."
    ),
    "preconditions_checked": (
        "Records the Exp 5271 gate, exposed telemetry fields, fixture-label adequacy, "
        "retired-scorer exclusion, and local model readiness before any quality number "
        "is interpreted."
    ),
    "MODEL_SPECS": (
        "Records mandated SOTA GGUF model IDs, roles, quantization/file receipts, "
        "and which roles contributed headline telemetry."
    ),
    "internal_signal_available": (
        "Boolean gate showing whether logits, token logprobs, hidden states, or attention "
        "receipts were actually available from Exp 5271 and live fixture rows."
    ),
    "delta_over_lexical_baseline": (
        "Internal AUROC minus lexical baseline AUROC; negative or zero values honestly "
        "show that the internal signal did not beat the cheap fixture-only control."
    ),
    "auroc": (
        "Reports AUROC only with explicit sample counts and controls; null is required "
        "for blocked or under-supported measurements."
    ),
    "false_accepts": (
        "Counts unsupported or contradiction fixtures accepted as safe under the fixed "
        "supported-threshold policy."
    ),
    "telemetry_receipts": (
        "Records field availability, per-model durations, fixture checksums, and control "
        "metrics so a downstream auditor can separate real telemetry from text-only fallback."
    ),
    "retired_external_scorer_reopened": (
        "Must remain false so the retired Phase D generated-text scorer, LLM judge, "
        "LoRA-EBM, uPRM, and EBRM reranker paths stay closed."
    ),
    "tests_run": (
        "Commands run to validate the 5272 module, artifact schema, new-code coverage, "
        "and repository test status."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "MODEL_SPECS",
    "internal_signal_available",
    "delta_over_lexical_baseline",
    "auroc",
    "false_accepts",
    "telemetry_receipts",
    "retired_external_scorer_reopened",
    "tests_run",
)
WRAPPED_FIELDS = tuple(field for field in REQUIRED_ARTIFACT_FIELDS if field != "tests_run")


@dataclass(frozen=True)
class FactualFixture:
    """One local evidence-relative factual claim with an external label."""

    fixture_id: str
    evidence: str
    claim: str
    case_type: str
    unsupported_label: bool
    label_source: str = "curated_local_evidence_label"


GenerationRunner = Callable[[FactualFixture, JsonDict, JsonDict, int], JsonDict]


def default_fixtures() -> list[FactualFixture]:
    """Return a bounded supported/unsupported/contradiction factual panel."""

    return [
        FactualFixture(
            fixture_id="ihp-001-supported-runtime",
            evidence="Larkspur audit memo: the Aster-9 battery ran 47 minutes under the amber-load test.",
            claim="The Aster-9 battery ran 47 minutes under the amber-load test.",
            case_type="supported",
            unsupported_label=False,
        ),
        FactualFixture(
            fixture_id="ihp-002-supported-enrollment",
            evidence="Mira Vale clinic snippet: Trial Nacre enrolled 18 participants and used a saline control.",
            claim="Trial Nacre enrolled 18 participants.",
            case_type="supported",
            unsupported_label=False,
        ),
        FactualFixture(
            fixture_id="ihp-003-supported-route",
            evidence="Noma transit note: Route 6 skipped Pear Gate during the lantern parade.",
            claim="Route 6 skipped Pear Gate during the lantern parade.",
            case_type="supported",
            unsupported_label=False,
        ),
        FactualFixture(
            fixture_id="ihp-004-unsupported-manufacturer",
            evidence="Solenne lab note: reagent Pavo turned violet at pH 5.2.",
            claim="Reagent Pavo was manufactured by HelioChem.",
            case_type="unsupported",
            unsupported_label=True,
        ),
        FactualFixture(
            fixture_id="ihp-005-unsupported-sensor",
            evidence="Vela cold-room note: the freezer used a brass sensor and alarmed at minus 18 C.",
            claim="The Vela freezer used a platinum sensor.",
            case_type="unsupported",
            unsupported_label=True,
        ),
        FactualFixture(
            fixture_id="ihp-006-unsupported-bench-count",
            evidence="Tavian school board note: the Cedar lab opened in 2024 with six benches.",
            claim="The Cedar lab opened with eleven benches.",
            case_type="unsupported",
            unsupported_label=True,
        ),
        FactualFixture(
            fixture_id="ihp-007-contradiction-runtime",
            evidence="Larkspur audit memo: the Aster-9 battery ran 47 minutes under the amber-load test.",
            claim="The Aster-9 battery ran 74 minutes under the amber-load test.",
            case_type="contradiction",
            unsupported_label=True,
        ),
        FactualFixture(
            fixture_id="ihp-008-contradiction-enrollment",
            evidence="Mira Vale clinic snippet: Trial Nacre enrolled 18 participants and used a saline control.",
            claim="Trial Nacre enrolled 81 participants.",
            case_type="contradiction",
            unsupported_label=True,
        ),
        FactualFixture(
            fixture_id="ihp-009-contradiction-year",
            evidence="Tavian school board note: the Cedar lab opened in 2024 with six benches.",
            claim="The Cedar lab opened in 2023.",
            case_type="contradiction",
            unsupported_label=True,
        ),
    ]


def render_prompt(fixture: FactualFixture) -> str:
    """Render the local evidence prompt without including the ground-truth label."""

    return (
        "Use only the evidence below. Reply with exactly SUPPORTED or UNSUPPORTED.\n\n"
        f"Evidence:\n{fixture.evidence}\n\n"
        f"Claim:\n{fixture.claim}\n\n"
        "Verdict:"
    )


def fixture_checksums(fixtures: Sequence[FactualFixture]) -> JsonDict:
    """Return stable checksums for prompts, labels, and case composition."""

    prompt_checksums = {fixture.fixture_id: sha16(render_prompt(fixture)) for fixture in fixtures}
    labels = {
        fixture.fixture_id: {
            "unsupported_label": bool(fixture.unsupported_label),
            "case_type": fixture.case_type,
        }
        for fixture in fixtures
    }
    payload = [
        {
            "fixture_id": fixture.fixture_id,
            "evidence_sha16": sha16(fixture.evidence),
            "claim_sha16": sha16(fixture.claim),
            "case_type": fixture.case_type,
            "unsupported_label": fixture.unsupported_label,
            "prompt_sha16": prompt_checksums[fixture.fixture_id],
        }
        for fixture in fixtures
    ]
    case_counts = Counter(fixture.case_type for fixture in fixtures)
    return {
        "fixture_count": len(fixtures),
        "prompt_checksums": prompt_checksums,
        "label_checksum": sha16(_stable_json(labels)),
        "fixture_set_checksum": sha16(_stable_json(payload)),
        "label_source": "curated_local_evidence_label",
        "supported_count": sum(1 for fixture in fixtures if not fixture.unsupported_label),
        "unsupported_count": sum(1 for fixture in fixtures if fixture.unsupported_label),
        "case_type_counts": {key: case_counts.get(key, 0) for key in sorted(case_counts)},
    }


def compute_internal_features(
    generation: Mapping[str, Any],
    *,
    exposed_fields: Mapping[str, Any],
) -> JsonDict:
    """Compute logit/logprob features from only the fields exposed by Exp 5271."""

    token_logprobs = (
        _numeric_values(generation.get("token_logprobs"))
        if _field_available(exposed_fields, "token_logprobs")
        else []
    )
    top_rows = (
        _top_logprob_rows(generation.get("top_logprobs"))
        if _field_available(exposed_fields, "token_logprobs")
        else []
    )
    final_logits = _final_logits(generation) if _field_available(exposed_fields, "logits") else []
    full_logit_summary = _full_logit_summary(final_logits) if final_logits else {}
    if not full_logit_summary and _field_available(exposed_fields, "logits"):
        full_logit_summary = _summary_from_generation(generation)

    top_distribution = _softmax_log_values(list(top_rows[-1].values())) if top_rows else []
    entropy_logprob = _entropy(top_distribution) if top_distribution else None
    selected_final_prob = _selected_final_probability(token_logprobs)
    final_top1_probability = max(top_distribution) if top_distribution else selected_final_prob

    sequence_spilled = _mean([1.0 - math.exp(min(0.0, value)) for value in token_logprobs])
    sequence_marginal = -_mean(token_logprobs) if token_logprobs else None
    final_spilled = 1.0 - final_top1_probability if final_top1_probability is not None else None
    full_logit_top1 = _optional_float(full_logit_summary.get("top1_probability"))
    full_logit_spilled = 1.0 - full_logit_top1 if full_logit_top1 is not None else None

    primary = sequence_marginal
    primary_name = "sequence_marginal_energy"
    if primary is None:
        primary = full_logit_spilled
        primary_name = "full_logit_spilled_energy"
    if primary is None:
        primary = final_spilled
        primary_name = "final_token_spilled_energy"
    signal_available = any(
        value is not None
        for value in (sequence_marginal, full_logit_spilled, final_spilled, entropy_logprob)
    )
    return {
        "signal_available": signal_available,
        "field_usage": {key: _availability(exposed_fields, key) for key in TELEMETRY_KEYS},
        "token_count": _token_count(generation, token_logprobs),
        "token_logprob_count": len(token_logprobs),
        "top_logprobs_count": len(top_rows),
        "final_logits_count": len(final_logits),
        "sequence_spilled_energy": sequence_spilled,
        "sequence_marginal_energy": sequence_marginal,
        "entropy_logprob_baseline": entropy_logprob,
        "final_token_top1_probability": final_top1_probability,
        "final_token_spilled_energy": final_spilled,
        "full_logit_top1_probability": full_logit_top1,
        "full_logit_spilled_energy": full_logit_spilled,
        "full_logit_entropy_topk": full_logit_summary.get("entropy_topk"),
        "primary_internal_score": primary,
        "primary_internal_score_name": primary_name,
    }


def lexical_risk_score(fixture: FactualFixture) -> float:
    """Return a cheap evidence-overlap baseline where higher means less supported."""

    claim_terms = _content_terms(fixture.claim)
    if not claim_terms:
        return 0.0
    evidence_terms = set(_content_terms(fixture.evidence))
    missing = sum(1 for term in claim_terms if term not in evidence_terms)
    return missing / len(claim_terms)


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize internal scores against lexical and shuffled-label controls."""

    scored = [
        row
        for row in rows
        if _optional_float(_nested_value(row.get("scores", {}), "internal")) is not None
    ]
    labels = [1 if bool(row.get("unsupported_label")) else 0 for row in scored]
    internal_scores = [float(_nested_value(row.get("scores", {}), "internal")) for row in scored]
    lexical_scores = [float(_nested_value(row.get("scores", {}), "lexical")) for row in scored]
    entropy_scores = [
        _optional_float(_nested_value(row.get("scores", {}), "entropy_logprob")) for row in scored
    ]
    entropy_pairs = [
        (label, score)
        for label, score in zip(labels, entropy_scores, strict=False)
        if score is not None
    ]

    internal = _score_summary(labels, internal_scores)
    lexical = _score_summary(labels, lexical_scores)
    entropy = _score_summary(
        [label for label, _score in entropy_pairs],
        [float(score) for _label, score in entropy_pairs],
    )
    internal_auroc = internal["auroc"]
    lexical_auroc = lexical["auroc"]
    delta = (
        float(internal_auroc) - float(lexical_auroc)
        if internal_auroc is not None and lexical_auroc is not None
        else 0.0
    )
    threshold = _supported_threshold(labels, internal_scores)
    false_accepts = (
        sum(
            1
            for label, score in zip(labels, internal_scores, strict=False)
            if label == 1 and threshold is not None and score <= threshold
        )
        if threshold is not None
        else 0
    )
    shuffled_labels = _rotated_labels(labels)
    shuffled = _score_summary(shuffled_labels, internal_scores)
    case_counts = Counter(str(row.get("case_type")) for row in scored)
    return {
        "sample_count": len(scored),
        "label_counts": {
            "supported": sum(1 for label in labels if label == 0),
            "unsupported_or_contradiction": sum(1 for label in labels if label == 1),
        },
        "case_type_counts": {key: case_counts.get(key, 0) for key in sorted(case_counts)},
        "higher_score_means": "greater_unsupported_risk",
        "internal": internal,
        "lexical": lexical,
        "entropy_logprob": entropy,
        "delta_over_lexical_baseline": delta,
        "supported_threshold": threshold,
        "false_accepts": int(false_accepts),
        "shuffled_label_control": {
            "sample_count": len(shuffled_labels),
            "auroc": shuffled["auroc"],
            "label_rotation": "left_by_one",
        },
    }


def run_probe(
    *,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    telemetry_artifact: Mapping[str, Any] | None = None,
    generation_runner: GenerationRunner | None = None,
    fixtures: Sequence[FactualFixture] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
    root: Path = REPO_ROOT,
    write: bool = True,
) -> JsonDict:
    """Run the 5271-gated probe and write a terminal artifact."""

    started = time.perf_counter()
    root = Path(root)
    active_telemetry = dict(telemetry_artifact or load_telemetry_artifact(root))
    active_fixtures = list(fixtures or default_fixtures())
    model_specs = _model_specs_from_telemetry(active_telemetry)
    exposed_fields = _exposed_fields_from_telemetry(active_telemetry)
    selected_roles, blocker = _selected_roles(model_specs, exposed_fields)

    if not active_telemetry.get("telemetry_harness_ready"):
        artifact = _blocked_artifact(
            root=root,
            telemetry_artifact=active_telemetry,
            model_specs=model_specs,
            exposed_fields=exposed_fields,
            fixtures=active_fixtures,
            tests_run=tests_run,
            blocker="blocked_telemetry_harness_not_ready",
            duration_s=time.perf_counter() - started,
        )
        validate_artifact(artifact)
        _write_json_if_requested(result_path, artifact, write)
        return artifact

    if blocker:
        artifact = _blocked_artifact(
            root=root,
            telemetry_artifact=active_telemetry,
            model_specs=model_specs,
            exposed_fields=exposed_fields,
            fixtures=active_fixtures,
            tests_run=tests_run,
            blocker=blocker,
            duration_s=time.perf_counter() - started,
        )
        validate_artifact(artifact)
        _write_json_if_requested(result_path, artifact, write)
        return artifact

    if not _fixture_labels_adequate(active_fixtures):
        artifact = _blocked_artifact(
            root=root,
            telemetry_artifact=active_telemetry,
            model_specs=model_specs,
            exposed_fields=exposed_fields,
            fixtures=active_fixtures,
            tests_run=tests_run,
            blocker="blocked_fixture_labels_inadequate",
            duration_s=time.perf_counter() - started,
        )
        validate_artifact(artifact)
        _write_json_if_requested(result_path, artifact, write)
        return artifact

    runner = generation_runner or live_llama_cpp_generation_runner()
    pilot_rows = [
        _evaluate_fixture(
            fixture=fixture,
            model_spec=model_specs[role],
            exposed_fields=exposed_fields[role],
            generation_runner=runner,
            seed=RANDOM_SEED + (role_index * 1000) + fixture_index,
        )
        for role_index, role in enumerate(selected_roles)
        for fixture_index, fixture in enumerate(active_fixtures)
    ]
    roles_with_signal = sorted(
        {
            str(row.get("model_role"))
            for row in pilot_rows
            if row.get("internal_features", {}).get("signal_available")
        }
    )
    if not all(role in roles_with_signal for role in HEADLINE_ROLES):
        artifact = _blocked_artifact(
            root=root,
            telemetry_artifact=active_telemetry,
            model_specs=_mark_headline_roles(model_specs, roles_with_signal),
            exposed_fields=exposed_fields,
            fixtures=active_fixtures,
            tests_run=tests_run,
            blocker="blocked_live_internal_signal_unmeasured",
            duration_s=time.perf_counter() - started,
            pilot_rows=pilot_rows,
        )
        validate_artifact(artifact)
        _write_json_if_requested(result_path, artifact, write)
        return artifact

    summary = summarize_rows(pilot_rows)
    artifact = _complete_artifact(
        root=root,
        telemetry_artifact=active_telemetry,
        model_specs=_mark_headline_roles(model_specs, roles_with_signal),
        exposed_fields=exposed_fields,
        fixtures=active_fixtures,
        pilot_rows=pilot_rows,
        summary=summary,
        tests_run=tests_run,
        duration_s=time.perf_counter() - started,
    )
    validate_artifact(artifact)
    _write_json_if_requested(result_path, artifact, write)
    return artifact


def live_llama_cpp_generation_runner() -> GenerationRunner:  # pragma: no cover
    """Build a lazy llama.cpp runner that returns only exposed telemetry receipts."""

    from llama_cpp import Llama  # noqa: PLC0415

    loaded: dict[str, Any] = {}

    def run(
        fixture: FactualFixture,
        model_spec: JsonDict,
        exposed_fields: JsonDict,
        seed: int,
    ) -> JsonDict:
        role = str(model_spec["role"])
        model_path = str(
            model_spec.get("model_path") or model_spec.get("file_receipts", {}).get("path")
        )
        if not model_path:
            raise RuntimeError(f"no model path for {role}")
        if role not in loaded:
            loaded[role] = Llama(
                model_path=model_path,
                n_gpu_layers=int(GGUF_PROBE_CONFIG["n_gpu_layers"]),
                n_ctx=int(GGUF_PROBE_CONFIG["n_ctx"]),
                seed=int(GGUF_PROBE_CONFIG["seed"]),
                logits_all=_field_available(exposed_fields, "logits"),
                verbose=False,
            )
        llm = loaded[role]
        response = llm(
            render_prompt(fixture),
            max_tokens=int(GGUF_PROBE_CONFIG["max_tokens"]),
            temperature=float(GGUF_PROBE_CONFIG["temperature"]),
            logprobs=int(GGUF_PROBE_CONFIG["logprobs"])
            if _field_available(exposed_fields, "token_logprobs")
            else None,
            echo=False,
            seed=int(seed),
        )
        choice = _first_choice(response)
        logprobs = choice.get("logprobs") if isinstance(choice, Mapping) else {}
        token_logprobs = _numeric_values(_nested_value(logprobs, "token_logprobs"))
        top_logprobs = _top_logprob_rows(_nested_value(logprobs, "top_logprobs"))
        logits_summary: JsonDict = {}
        if _field_available(exposed_fields, "logits"):
            logits = getattr(llm, "eval_logits", None)
            if logits:
                logits_summary = _full_logit_summary(logits[-1])
                logits_summary["steps"] = len(logits)
        return {
            "raw_response": str(choice.get("text", ""))
            if isinstance(choice, Mapping)
            else str(response),
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
            "tokens": list(logprobs.get("tokens", [])) if isinstance(logprobs, Mapping) else [],
            "token_count": len(token_logprobs),
            "logit_receipt": logits_summary,
            "final_logits_summary": logits_summary,
            "seed": int(seed),
        }

    return run


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 5272 artifact violates the required schema."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise AssertionError("; ".join(errors))


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    for field in WRAPPED_FIELDS:
        value = artifact.get(field)
        if not isinstance(value, Mapping) or "value" not in value or "principle" not in value:
            errors.append(f"{field} must be principle-wrapped")
            continue
        if value.get("principle") != FIELD_PRINCIPLES[field]:
            errors.append(f"{field} principle mismatch")

    verdict = _wrapped_value(artifact, "honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict.value must start with complete: or blocked_")
    elif not any(word in verdict for word in ("positive", "null", "harmful", "unmeasured")):
        errors.append("honest_verdict.value must state positive, null, harmful, or unmeasured")

    if _wrapped_value(artifact, "inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate.value must be {INFERENCE_SUBSTRATE}")
    if not isinstance(_wrapped_value(artifact, "internal_signal_available"), bool):
        errors.append("internal_signal_available.value must be bool")
    delta = _wrapped_value(artifact, "delta_over_lexical_baseline")
    if not isinstance(delta, int | float) or isinstance(delta, bool):
        errors.append("delta_over_lexical_baseline.value must be numeric")
    auroc = _wrapped_value(artifact, "auroc")
    if auroc is not None and (not isinstance(auroc, int | float) or isinstance(auroc, bool)):
        errors.append("auroc.value must be numeric or null")
    false_accepts = _wrapped_value(artifact, "false_accepts")
    if not isinstance(false_accepts, int) or isinstance(false_accepts, bool):
        errors.append("false_accepts.value must be integer")
    if _wrapped_value(artifact, "retired_external_scorer_reopened") is not False:
        errors.append("retired_external_scorer_reopened.value must be false")
    if not isinstance(artifact.get("tests_run"), list):
        errors.append("tests_run must be a list")

    model_specs = _wrapped_value(artifact, "MODEL_SPECS")
    if not isinstance(model_specs, Mapping):
        errors.append("MODEL_SPECS.value must be an object")
    else:
        for role, hf_id in MANDATED_MODEL_IDS.items():
            row = model_specs.get(role)
            if not isinstance(row, Mapping):
                errors.append(f"MODEL_SPECS.value missing role {role}")
            elif row.get("hf_id") != hf_id:
                errors.append(f"MODEL_SPECS.value.{role}.hf_id mismatch")

    receipts = _wrapped_value(artifact, "telemetry_receipts")
    if not isinstance(receipts, Mapping):
        errors.append("telemetry_receipts.value must be an object")
    else:
        if "field_availability" not in receipts:
            errors.append("telemetry_receipts.value.field_availability missing")
        if "duration" not in receipts:
            errors.append("telemetry_receipts.value.duration missing")
    return errors


def load_telemetry_artifact(root: Path = REPO_ROOT) -> JsonDict:  # pragma: no cover
    """Load the Exp 5271 gate artifact."""

    path = root / EXP5271_RELATIVE_PATH
    if not path.exists():
        return {"telemetry_harness_ready": False, "blocker": f"missing {path}"}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return dict(loaded) if isinstance(loaded, Mapping) else {"telemetry_harness_ready": False}


def _complete_artifact(
    *,
    root: Path,
    telemetry_artifact: Mapping[str, Any],
    model_specs: Mapping[str, JsonDict],
    exposed_fields: Mapping[str, JsonDict],
    fixtures: Sequence[FactualFixture],
    pilot_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    delta = float(summary.get("delta_over_lexical_baseline") or 0.0)
    auroc = summary.get("internal", {}).get("auroc")
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(duration_s, 6),
        "random_seed": RANDOM_SEED,
        "honest_verdict": _wrap(
            "honest_verdict",
            _complete_verdict(delta=delta, auroc=auroc, sample_count=int(summary["sample_count"])),
        ),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "preconditions_checked": _preconditions(
            root=root,
            telemetry_artifact=telemetry_artifact,
            model_specs=model_specs,
            exposed_fields=exposed_fields,
            fixtures=fixtures,
            blocker="",
        ),
        "MODEL_SPECS": _wrap("MODEL_SPECS", dict(model_specs)),
        "internal_signal_available": _wrap("internal_signal_available", True),
        "delta_over_lexical_baseline": _wrap("delta_over_lexical_baseline", delta),
        "auroc": _wrap("auroc", auroc),
        "false_accepts": _wrap("false_accepts", int(summary.get("false_accepts") or 0)),
        "telemetry_receipts": _wrap(
            "telemetry_receipts",
            _telemetry_receipts(
                telemetry_artifact=telemetry_artifact,
                exposed_fields=exposed_fields,
                fixtures=fixtures,
                pilot_rows=pilot_rows,
                summary=summary,
                duration_s=duration_s,
            ),
        ),
        "retired_external_scorer_reopened": _wrap("retired_external_scorer_reopened", False),
        "tests_run": [dict(row) for row in tests_run],
        "pilot_rows": [dict(row) for row in pilot_rows],
        "control_summary": dict(summary),
        "reproducibility_checksum": sha16(
            _stable_json(
                {
                    "schema": SCHEMA,
                    "fixtures": fixture_checksums(fixtures),
                    "summary": summary,
                    "model_specs": model_specs,
                    "field_availability": exposed_fields,
                }
            )
        ),
    }


def _blocked_artifact(
    *,
    root: Path,
    telemetry_artifact: Mapping[str, Any],
    model_specs: Mapping[str, JsonDict],
    exposed_fields: Mapping[str, JsonDict],
    fixtures: Sequence[FactualFixture],
    tests_run: Sequence[Mapping[str, Any]],
    blocker: str,
    duration_s: float,
    pilot_rows: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    summary = {
        "sample_count": 0,
        "label_counts": {"supported": 0, "unsupported_or_contradiction": 0},
        "internal": {"auroc": None},
        "lexical": {"auroc": None},
        "entropy_logprob": {"auroc": None},
        "delta_over_lexical_baseline": 0.0,
        "false_accepts": 0,
        "shuffled_label_control": {"sample_count": 0, "auroc": None},
        "blocked_reason": blocker,
    }
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(duration_s, 6),
        "random_seed": RANDOM_SEED,
        "honest_verdict": _wrap("honest_verdict", f"{blocker}: unmeasured internal/logit signal"),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "preconditions_checked": _preconditions(
            root=root,
            telemetry_artifact=telemetry_artifact,
            model_specs=model_specs,
            exposed_fields=exposed_fields,
            fixtures=fixtures,
            blocker=blocker,
        ),
        "MODEL_SPECS": _wrap("MODEL_SPECS", dict(model_specs)),
        "internal_signal_available": _wrap("internal_signal_available", False),
        "delta_over_lexical_baseline": _wrap("delta_over_lexical_baseline", 0.0),
        "auroc": _wrap("auroc", None),
        "false_accepts": _wrap("false_accepts", 0),
        "telemetry_receipts": _wrap(
            "telemetry_receipts",
            _telemetry_receipts(
                telemetry_artifact=telemetry_artifact,
                exposed_fields=exposed_fields,
                fixtures=fixtures,
                pilot_rows=pilot_rows,
                summary=summary,
                duration_s=duration_s,
            ),
        ),
        "retired_external_scorer_reopened": _wrap("retired_external_scorer_reopened", False),
        "tests_run": [dict(row) for row in tests_run],
        "pilot_rows": [dict(row) for row in pilot_rows],
        "control_summary": summary,
        "blocker": blocker,
    }


def _evaluate_fixture(
    *,
    fixture: FactualFixture,
    model_spec: JsonDict,
    exposed_fields: JsonDict,
    generation_runner: GenerationRunner,
    seed: int,
) -> JsonDict:
    started = time.perf_counter()
    generation: JsonDict
    try:
        generation = generation_runner(fixture, model_spec, exposed_fields, seed)
        runtime_error = None
    except Exception as exc:  # pragma: no cover - defensive for live runtime failures.
        generation = {"raw_response": "", "runtime_error": f"{type(exc).__name__}: {exc}"}
        runtime_error = traceback.format_exc()
    elapsed = time.perf_counter() - started
    features = compute_internal_features(generation, exposed_fields=exposed_fields)
    raw_response = str(generation.get("raw_response") or generation.get("response_text") or "")
    internal_score = _optional_float(features.get("primary_internal_score"))
    entropy_score = _optional_float(features.get("entropy_logprob_baseline"))
    return {
        "fixture_id": fixture.fixture_id,
        "model_role": model_spec["role"],
        "model_hf_id": model_spec["hf_id"],
        "case_type": fixture.case_type,
        "unsupported_label": bool(fixture.unsupported_label),
        "label_source": fixture.label_source,
        "claim_checksum": sha16(fixture.claim),
        "evidence_checksum": sha16(fixture.evidence),
        "prompt_checksum": sha16(render_prompt(fixture)),
        "raw_response_checksum": sha16(raw_response),
        "raw_response_excerpt": raw_response[:160],
        "internal_features": features,
        "scores": {
            "internal": internal_score,
            "lexical": lexical_risk_score(fixture),
            "entropy_logprob": entropy_score,
        },
        "signal_receipts": {
            "token_logprob_count": features["token_logprob_count"],
            "top_logprobs_count": features["top_logprobs_count"],
            "final_logits_count": features["final_logits_count"],
            "logit_receipt": dict(
                generation.get("logit_receipt") or generation.get("final_logits_summary") or {}
            ),
            "field_usage": features["field_usage"],
            "runtime_error": runtime_error,
        },
        "seed": int(seed),
        "duration_s": round(elapsed, 6),
    }


def _preconditions(
    *,
    root: Path,
    telemetry_artifact: Mapping[str, Any],
    model_specs: Mapping[str, JsonDict],
    exposed_fields: Mapping[str, JsonDict],
    fixtures: Sequence[FactualFixture],
    blocker: str,
) -> JsonDict:
    selected_roles, role_blocker = _selected_roles(model_specs, exposed_fields)
    checksums = fixture_checksums(fixtures)
    return _wrap(
        "preconditions_checked",
        {
            "exp5271_artifact_path": str(root / EXP5271_RELATIVE_PATH),
            "exp5271_telemetry_harness_ready": bool(
                telemetry_artifact.get("telemetry_harness_ready")
            ),
            "exp5271_telemetry_harness_ready_principle": telemetry_artifact.get(
                "telemetry_harness_ready_principle"
            ),
            "exposed_telemetry_fields": _field_availability(exposed_fields),
            "headline_roles_required": list(HEADLINE_ROLES),
            "headline_roles_ready": [
                role for role in HEADLINE_ROLES if _role_ready(model_specs, exposed_fields, role)
            ],
            "selected_model_roles": selected_roles,
            "role_blocker": role_blocker,
            "fixture_label_counts": {
                "supported": checksums["supported_count"],
                "unsupported": checksums["unsupported_count"],
                "case_type_counts": checksums["case_type_counts"],
            },
            "fixture_labels_adequate": _fixture_labels_adequate(fixtures),
            "retired_external_scorer_reopened": False,
            "disallowed_external_scorers": [
                "external_generated_text_scorer",
                "llm_judge",
                "lora_ebm",
                "uprm",
                "ebrm_text_reranker",
                "nli_text_scorer",
            ],
            "legacy_tiny_models_headline": False,
            "blocker": blocker,
        },
    )


def _telemetry_receipts(
    *,
    telemetry_artifact: Mapping[str, Any],
    exposed_fields: Mapping[str, JsonDict],
    fixtures: Sequence[FactualFixture],
    pilot_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    per_model_duration: JsonDict = {}
    for row in pilot_rows:
        role = str(row.get("model_role"))
        per_model_duration[role] = per_model_duration.get(role, 0.0) + float(
            row.get("duration_s") or 0.0
        )
    return {
        "exp5271_telemetry_harness_ready": bool(telemetry_artifact.get("telemetry_harness_ready")),
        "field_availability": _field_availability(exposed_fields),
        "upstream_duration_receipts": _nested_value(
            telemetry_artifact.get("duration_receipts"), "value"
        )
        or _nested_value(telemetry_artifact, "duration_receipts"),
        "duration": {
            "total_wall_clock_s": round(duration_s, 6),
            "per_model_fixture_duration_s": {
                role: round(value, 6) for role, value in sorted(per_model_duration.items())
            },
        },
        "fixture_checksums": fixture_checksums(fixtures),
        "model_roles_scored": _unique_ordered(str(row.get("model_role")) for row in pilot_rows),
        "sample_count": int(summary.get("sample_count") or 0),
        "label_counts": summary.get("label_counts"),
        "controls": {
            "lexical": summary.get("lexical"),
            "entropy_logprob": summary.get("entropy_logprob"),
            "shuffled_label_control": summary.get("shuffled_label_control"),
            "higher_score_means": summary.get("higher_score_means"),
        },
    }


def _model_specs_from_telemetry(telemetry_artifact: Mapping[str, Any]) -> JsonDict:
    upstream = _nested_value(telemetry_artifact.get("MODEL_SPECS"), "value")
    if not isinstance(upstream, Mapping):
        upstream = {}
    specs: JsonDict = {}
    for role, hf_id in MANDATED_MODEL_IDS.items():
        raw = upstream.get(role, {})
        row = dict(raw) if isinstance(raw, Mapping) else {}
        file_receipts = row.get("file_receipts")
        path = row.get("model_path") or (
            file_receipts.get("path") if isinstance(file_receipts, Mapping) else None
        )
        specs[role] = {
            "role": role,
            "hf_id": str(row.get("hf_id") or hf_id),
            "quantization": str(row.get("quantization") or row.get("preferred_quant") or "Q4_K_M"),
            "runtime_status": row.get("runtime_status") or row.get("status") or "missing_receipt",
            "status": row.get("status") or "missing_receipt",
            "model_path": path,
            "file_receipts": file_receipts,
            "headline_required": role in HEADLINE_ROLES,
            "optional": role in OPTIONAL_ROLES,
            "headline_metric_role": False,
            "legacy_tiny_model": False,
        }
    return specs


def _exposed_fields_from_telemetry(telemetry_artifact: Mapping[str, Any]) -> JsonDict:
    upstream = _nested_value(telemetry_artifact.get("exposed_telemetry_fields"), "value")
    if not isinstance(upstream, Mapping):
        upstream = {}
    fields: JsonDict = {}
    for role in MANDATED_MODEL_IDS:
        raw = upstream.get(role, {})
        role_fields = dict(raw) if isinstance(raw, Mapping) else {}
        fields[role] = {
            key: dict(role_fields.get(key, {"availability": "missing_receipt"}))
            if isinstance(role_fields.get(key), Mapping)
            else {"availability": "missing_receipt"}
            for key in TELEMETRY_KEYS
        }
    return fields


def _selected_roles(
    model_specs: Mapping[str, JsonDict],
    exposed_fields: Mapping[str, JsonDict],
) -> tuple[list[str], str]:
    missing_required = [
        role for role in HEADLINE_ROLES if not _role_ready(model_specs, exposed_fields, role)
    ]
    if missing_required:
        return [], f"blocked_headline_models_unavailable: {','.join(missing_required)}"
    selected = list(HEADLINE_ROLES)
    for role in OPTIONAL_ROLES:
        if _role_ready(model_specs, exposed_fields, role):
            selected.append(role)
    return selected, ""


def _role_ready(
    model_specs: Mapping[str, JsonDict],
    exposed_fields: Mapping[str, JsonDict],
    role: str,
) -> bool:
    spec = model_specs.get(role, {})
    if not spec.get("model_path"):
        return False
    if spec.get("legacy_tiny_model"):
        return False
    return any(_field_available(exposed_fields.get(role, {}), key) for key in TELEMETRY_KEYS)


def _mark_headline_roles(
    model_specs: Mapping[str, JsonDict], roles_with_signal: Sequence[str]
) -> JsonDict:
    role_set = set(roles_with_signal)
    out: JsonDict = {}
    for role, spec in model_specs.items():
        row = dict(spec)
        row["headline_metric_role"] = role in role_set
        out[role] = row
    return out


def _fixture_labels_adequate(fixtures: Sequence[FactualFixture]) -> bool:
    case_types = {fixture.case_type for fixture in fixtures}
    labels = {fixture.unsupported_label for fixture in fixtures}
    return labels == {False, True} and {"supported", "unsupported", "contradiction"}.issubset(
        case_types
    )


def _complete_verdict(*, delta: float, auroc: Any, sample_count: int) -> str:
    if delta > 0.05:
        quality = "positive"
    elif delta < -0.05:
        quality = "harmful"
    else:
        quality = "null"
    return (
        f"complete: {quality} internal/logit signal "
        f"delta_over_lexical={delta:.6f} auroc={_format_optional_float(auroc)} "
        f"sample_count={sample_count}"
    )


def _score_summary(labels: Sequence[int], scores: Sequence[float]) -> JsonDict:
    positives = [score for label, score in zip(labels, scores, strict=False) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=False) if label == 0]
    return {
        "sample_count": len(scores),
        "auroc": _auroc(labels, scores),
        "supported_mean": _mean(negatives),
        "unsupported_mean": _mean(positives),
        "unsupported_minus_supported": (
            _mean(positives) - _mean(negatives) if positives and negatives else None
        ),
    }


def _unique_ordered(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _supported_threshold(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    supported = [score for label, score in zip(labels, scores, strict=False) if label == 0]
    return max(supported) if supported else None


def _rotated_labels(labels: Sequence[int]) -> list[int]:
    if len(labels) < 2:
        return list(labels)
    return list(labels[1:]) + [labels[0]]


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores, strict=False) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=False) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for positive in positives:
        for negative in negatives:
            total += 1
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / total if total else None


def _field_availability(exposed_fields: Mapping[str, JsonDict]) -> JsonDict:
    return {
        role: {key: _availability(field_map, key) for key in TELEMETRY_KEYS}
        for role, field_map in exposed_fields.items()
    }


def _field_available(exposed_fields: Mapping[str, Any], key: str) -> bool:
    return _availability(exposed_fields, key) in USABLE_AVAILABILITY


def _availability(exposed_fields: Mapping[str, Any], key: str) -> str:
    value = exposed_fields.get(key)
    if isinstance(value, Mapping):
        return str(value.get("availability") or "missing_receipt")
    return "missing_receipt"


def _content_terms(text: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "under",
        "used",
        "was",
        "with",
    }
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return [term for term in cleaned.split() if term and term not in stopwords]


def _first_choice(response: Any) -> JsonDict:
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            return dict(choices[0])
    return {"text": str(response), "logprobs": {}}


def _full_logit_summary(values: Any, *, top_k: int = 8) -> JsonDict:
    numeric = _numeric_values(values)
    if not numeric:
        return {}
    maximum = max(numeric)
    weights = [math.exp(value - maximum) for value in numeric]
    total = sum(weights)
    if total <= 0.0:  # pragma: no cover - finite exponentials have positive sum.
        return {}
    top = sorted(enumerate(numeric), key=lambda item: item[1], reverse=True)[:top_k]
    top_probs = [math.exp(value - maximum) / total for _index, value in top]
    top_mass = sum(top_probs)
    normalized_top = [prob / top_mass for prob in top_probs] if top_mass > 0 else []
    return {
        "vocab_size": len(numeric),
        "top1_probability": max(weights) / total,
        "top_mass_probability": top_mass,
        "entropy_topk": _entropy(normalized_top),
        "top_logits": [
            {"token_index": int(index), "logit": float(value), "probability": float(prob)}
            for (index, value), prob in zip(top, top_probs, strict=False)
        ],
    }


def _summary_from_generation(generation: Mapping[str, Any]) -> JsonDict:
    summary = generation.get("final_logits_summary") or generation.get("logit_receipt")
    return dict(summary) if isinstance(summary, Mapping) else {}


def _final_logits(generation: Mapping[str, Any]) -> list[float]:
    for key in ("final_logits", "logits"):
        values = generation.get(key)
        if isinstance(values, list):
            if values and isinstance(values[-1], list):
                return _numeric_values(values[-1])
            return _numeric_values(values)
    return []


def _top_logprob_rows(value: Any) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not isinstance(value, list):
        return rows
    for item in value:
        if isinstance(item, Mapping):
            converted = {
                str(key): numeric
                for key, raw in item.items()
                if (numeric := _optional_float(raw)) is not None
            }
            if converted:
                rows.append(converted)
    return rows


def _numeric_values(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    values: list[float] = []
    for item in value:
        number = _optional_float(item)
        if number is not None:
            values.append(number)
    return values


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _softmax_log_values(values: Sequence[float]) -> list[float]:
    numeric: list[float] = []
    for value in values:
        number = _optional_float(value)
        if number is not None:
            numeric.append(number)
    if not numeric:
        return []
    maximum = max(numeric)
    weights = [math.exp(value - maximum) for value in numeric]
    total = sum(weights)
    return [weight / total for weight in weights] if total else []


def _entropy(probabilities: Sequence[float]) -> float | None:
    if not probabilities:
        return None
    return -sum(prob * math.log(max(prob, 1e-12)) for prob in probabilities)


def _selected_final_probability(token_logprobs: Sequence[float]) -> float | None:
    if not token_logprobs:
        return None
    return max(0.0, min(1.0, math.exp(min(0.0, float(token_logprobs[-1])))))


def _token_count(generation: Mapping[str, Any], token_logprobs: Sequence[float]) -> int:
    explicit = generation.get("token_count") or generation.get("tokens_generated")
    if isinstance(explicit, int) and explicit >= 0:
        return explicit
    tokens = generation.get("tokens")
    if isinstance(tokens, list):
        return len(tokens)
    return len(token_logprobs)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _nested_value(payload: Any, field: str) -> Any:
    if isinstance(payload, Mapping):
        value = payload.get(field)
        if isinstance(value, Mapping) and "value" in value:
            return value["value"]
        return value
    return None


def _wrapped_value(artifact: Mapping[str, Any], field: str) -> Any:
    value = artifact.get(field)
    return value.get("value") if isinstance(value, Mapping) else None


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def sha16(value: str | bytes) -> str:
    """Return a short stable checksum for prompts, labels, and receipts."""

    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _format_optional_float(value: Any) -> str:
    number = _optional_float(value)
    return "null" if number is None else f"{number:.6f}"


def _write_json_if_requested(path: Path, artifact: Mapping[str, Any], write: bool) -> None:
    if not write:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_tests_run_argument(value: str | None) -> list[JsonDict]:  # pragma: no cover
    if not value:
        return [
            {
                "command": (
                    ".venv/bin/python -m carnot.experiment_5272_internal_hallucination_probe_gated_v482 "
                    "--output results/experiment_5272_internal_hallucination_probe_gated_v482.json"
                ),
                "outcome": "completed module invocation",
            }
        ]
    path = Path(value)
    text = path.read_text(encoding="utf-8") if path.exists() else value
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("--tests-run-json must decode to a list")
    return [dict(row) for row in parsed]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--telemetry-artifact", default=str(REPO_ROOT / EXP5271_RELATIVE_PATH))
    parser.add_argument("--tests-run-json", default=None)
    args = parser.parse_args(argv)
    telemetry = json.loads(Path(args.telemetry_artifact).read_text(encoding="utf-8"))
    artifact = run_probe(
        result_path=Path(args.output),
        telemetry_artifact=telemetry,
        tests_run=_load_tests_run_argument(args.tests_run_json),
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
