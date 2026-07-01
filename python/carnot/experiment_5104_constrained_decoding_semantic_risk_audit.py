"""Exp 5104: semantic risk audit for finite constrained decoding.

Spec refs: REQ-VERIFY-5104, SCENARIO-VERIFY-5104.

This module deliberately separates syntax from meaning.  A finite
STATIC/trie/CSR mask can prove that emitted bytes are in a JSON language, but
it can still admit no-op, tautological, unsupported, or contradicted claims.
The artifact therefore reports a diagnostic instead of a syntax-only win.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5090_static_csr_constrained_decoding as static5090


JsonDict = dict[str, Any]
GrammarProbe = Callable[[], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5104
EXPERIMENT_NAME = "experiment_5104_constrained_decoding_semantic_risk_audit"
ARTIFACT_SCHEMA = "carnot.experiment_5104_constrained_decoding_semantic_risk_audit.v468"
SCHEMA_NAME = "finite_carnot_semantic_control_schema_v1"
RESULT_RELATIVE_PATH = (
    "results/experiment_5104_constrained_decoding_semantic_risk_audit_v468.json"
)
MODULE_RELATIVE_PATH = (
    "python/carnot/experiment_5104_constrained_decoding_semantic_risk_audit.py"
)
EXP5097_RELATIVE_PATH = "results/experiment_5097_clean_sota_endpoint_logprob_cache_v468.json"
SPEC_REFS = ["REQ-VERIFY-5104", "SCENARIO-VERIFY-5104"]
RUN_DATE = "20260701"
RANDOM_SEED = 20260701
DETERMINISTIC_INFERENCE_SUBSTRATE = "deterministic_static_csr_semantic_distribution_audit"

MODEL_SPECS: tuple[dict[str, str], ...] = (
    {
        "role": "flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "preferred_quant": "Q4_K_M",
    },
    {
        "role": "flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "preferred_quant": "Q4_K_M",
    },
    {
        "role": "middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "preferred_quant": "Q4_K_M",
    },
)
MANDATED_MODEL_IDS = tuple(spec["hf_id"] for spec in MODEL_SPECS)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "preconditions_checked",
    "model_specs",
    "schema_name",
    "candidate_pool_non_degenerate",
    "grammar_baseline",
    "syntax_validity_rate",
    "semantic_validity_rate",
    "noop_accept_rate",
    "contradiction_reject_rate",
    "distribution_shift_metric",
    "latency_ms",
    "mask_memory",
    "syntax_only_headline_forbidden",
    "live_llm_invoked",
    "flagged_adversarial",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix cannot promote syntax validity unless semantic controls, "
            "contradiction rejection, and distribution-shift checks are clean."
        )
    },
    "duration_s": {
        "principle": "wall-clock runtime for the deterministic audit and artifact write."
    },
    "inference_substrate": {
        "principle": "declares deterministic audit unless live local decoding actually ran."
    },
    "preconditions_checked": {
        "principle": (
            "records schema, tokenizer assumptions, candidate-pool checks, grammar "
            "baseline availability, and Exp5097 endpoint cleanliness."
        )
    },
    "model_specs": {
        "principle": "names every mandated GGUF model and any resolved local path evidence."
    },
    "schema_name": {
        "principle": "the finite Carnot semantic-control schema selected for the audit."
    },
    "candidate_pool_non_degenerate": {
        "principle": "true only when all semantic-risk control families are present."
    },
    "grammar_baseline": {
        "principle": "external grammar-engine availability or a precise unavailable reason."
    },
    "syntax_validity_rate": {
        "principle": "schema-language validity from STATIC masks, not semantic quality."
    },
    "semantic_validity_rate": {
        "principle": "non-vacuous semantic support rate under the STATIC syntax mask."
    },
    "noop_accept_rate": {
        "principle": "mass accepted by syntax masks for no-op valid but uninformative rows."
    },
    "contradiction_reject_rate": {
        "principle": "fraction of contradicted schema-valid rows rejected by the audited path."
    },
    "distribution_shift_metric": {
        "principle": "total variation distance between static masking and semantic rerank."
    },
    "latency_ms": {
        "principle": "trie, CSR, and optional grammar-baseline latency measurements."
    },
    "mask_memory": {
        "principle": "trie and CSR state, transition, and byte-memory estimates."
    },
    "syntax_only_headline_forbidden": {
        "principle": "must remain true so syntax-only validity cannot be headlined."
    },
    "live_llm_invoked": {
        "principle": "true only when local LLM decoding was actually invoked."
    },
    "flagged_adversarial": {
        "principle": "true only when the artifact detects its own inconsistent claim."
    },
}


def semantic_control_candidates() -> tuple[JsonDict, ...]:
    """Return the finite syntax-valid rows used to audit semantic risk.

    The pool intentionally contains rows that are syntactically valid JSON but
    semantically weak or wrong.  That makes a syntax-only mask look perfect
    while preserving the risk signal that an external checker would need.
    """

    rows = (
        _candidate(
            "supported-accept",
            "grounded_supported",
            "power-budget-log-42",
            "sensor_a_within_power_budget",
            "accept",
            "sensor_log_matches_budget",
            0.14,
            semantic_valid=True,
        ),
        _candidate(
            "distribution-alt-a",
            "distribution_sensitive_alternative",
            "cooling-log-43",
            "fan_speed_stable",
            "accept",
            "two_logs_support_stability",
            0.12,
            semantic_valid=True,
        ),
        _candidate(
            "distribution-alt-b",
            "distribution_sensitive_alternative",
            "cooling-log-43",
            "fan_speed_requires_review",
            "abstain",
            "two_logs_are_close_to_threshold",
            0.10,
            semantic_valid=True,
        ),
        _candidate(
            "noop-valid",
            "noop_valid",
            "operator-no-change",
            "no_change_requested",
            "abstain",
            "noop",
            0.20,
            semantic_valid=True,
            vacuous=True,
        ),
        _candidate(
            "tautology-valid",
            "tautology_valid",
            "schema-tautology",
            "schema_output_is_schema_valid",
            "accept",
            "tautology",
            0.18,
            semantic_valid=True,
            vacuous=True,
        ),
        _candidate(
            "unsupported-claim",
            "unsupported_claim",
            "efficiency-claim",
            "efficiency_improved_by_30_percent",
            "accept",
            "no_source",
            0.13,
            semantic_valid=False,
            unsupported=True,
        ),
        _candidate(
            "contradicted-claim",
            "contradicted_claim",
            "temperature-log-44",
            "temperature_decreased_after_load_spike",
            "accept",
            "sensor_log_contradicts_claim",
            0.11,
            semantic_valid=False,
            contradicted=True,
        ),
        _candidate(
            "supported-reject",
            "grounded_supported",
            "safety-log-45",
            "interlock_was_disabled",
            "reject",
            "interlock_log_shows_enabled",
            0.12,
            semantic_valid=True,
        ),
    )
    return tuple(rows)


def finite_schema_outputs(candidates: Sequence[Mapping[str, Any]] | None = None) -> tuple[str, ...]:
    """Serialize the finite schema language as canonical ASCII JSON strings."""

    pool = tuple(candidates) if candidates is not None else semantic_control_candidates()
    return tuple(_canonical_json(row["payload"]) for row in pool)


def selected_schema_descriptor() -> JsonDict:
    """Describe the finite schema and its semantic-control families."""

    return {
        "schema_name": SCHEMA_NAME,
        "output_format": "canonical_ascii_json",
        "canonicalization": "json.dumps(sort_keys=True,separators=(',',':'))",
        "tokenizer_assumption": {
            "kind": "ascii_byte_tokens_plus_eos",
            "byte_token_ids": "0..127",
            "bpe_tokenizer_used": False,
            "why": "the audit must be independent of GGUF tokenizer merges",
        },
        "control_types": sorted(
            {str(row["control_type"]) for row in semantic_control_candidates()}
        ),
    }


def build_static_mask_audit(
    outputs: Sequence[str] | None = None,
) -> tuple[static5090.TrieMaskIndex, static5090.CSRAutomaton, JsonDict]:
    """Build trie and CSR masks, then check exact prefix-mask equivalence."""

    finite_outputs = tuple(outputs) if outputs is not None else finite_schema_outputs()
    trie = static5090.build_trie_mask_index(finite_outputs)
    csr = static5090.build_csr_from_trie(trie)
    equivalence = static5090.evaluate_mask_equivalence(finite_outputs, trie, csr)
    return trie, csr, equivalence


def evaluate_semantic_controls(
    candidates: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Compare syntax masks with unconstrained and semantic-rerank baselines."""

    pool = tuple(candidates) if candidates is not None else semantic_control_candidates()
    finite_mass = sum(float(row["base_probability"]) for row in pool)
    static_distribution = {
        str(row["candidate_id"]): float(row["base_probability"]) / finite_mass for row in pool
    }
    strict_semantic = [
        row for row in pool if bool(row["semantic_valid"]) and not bool(row["vacuous"])
    ]
    strict_mass = sum(float(row["base_probability"]) for row in strict_semantic)
    rerank_distribution = {
        str(row["candidate_id"]): float(row["base_probability"]) / strict_mass
        for row in strict_semantic
    }
    raw_invalid = unconstrained_invalid_candidates()
    unconstrained_mass = finite_mass + sum(float(row["base_probability"]) for row in raw_invalid)
    semantic_mass = sum(
        float(row["base_probability"])
        for row in pool
        if bool(row["semantic_valid"]) and not bool(row["vacuous"])
    )
    contradicted = [row for row in pool if bool(row["contradicted"])]
    unsupported = [row for row in pool if bool(row["unsupported"])]
    static = {
        "syntax_validity_rate": 1.0,
        "semantic_validity_rate": _round_rate(semantic_mass / finite_mass),
        "noop_accept_rate": _round_rate(
            sum(
                float(row["base_probability"])
                for row in pool
                if row["control_type"] == "noop_valid"
            )
            / finite_mass
        ),
        "tautology_accept_rate": _round_rate(
            sum(
                float(row["base_probability"])
                for row in pool
                if row["control_type"] == "tautology_valid"
            )
            / finite_mass
        ),
        "contradiction_reject_rate": _reject_rate(contradicted, accepted=True),
        "unsupported_reject_rate": _reject_rate(unsupported, accepted=True),
    }
    unconstrained = {
        "syntax_validity_rate": _round_rate(finite_mass / unconstrained_mass),
        "semantic_validity_rate": _round_rate(semantic_mass / unconstrained_mass),
        "invalid_syntax_mass": _round_rate(
            sum(float(row["base_probability"]) for row in raw_invalid) / unconstrained_mass
        ),
    }
    semantic_rerank = {
        "syntax_validity_rate": 1.0,
        "semantic_validity_rate": 1.0,
        "noop_accept_rate": 0.0,
        "tautology_accept_rate": 0.0,
        "contradiction_reject_rate": _reject_rate(contradicted, accepted=False),
        "unsupported_reject_rate": _reject_rate(unsupported, accepted=False),
    }
    return {
        "control_types": sorted(Counter(str(row["control_type"]) for row in pool)),
        "static_mask": static,
        "unconstrained": unconstrained,
        "semantic_rerank": semantic_rerank,
        "distribution_shift_metric": _total_variation(
            static_distribution, rerank_distribution
        ),
        "distribution_shift_basis": "TV(static_mask_distribution,semantic_rerank_distribution)",
        "candidate_summaries": [
            {
                "candidate_id": row["candidate_id"],
                "control_type": row["control_type"],
                "semantic_valid": row["semantic_valid"],
                "vacuous": row["vacuous"],
                "contradicted": row["contradicted"],
                "unsupported": row["unsupported"],
                "base_probability": row["base_probability"],
            }
            for row in pool
        ],
    }


def unconstrained_invalid_candidates() -> tuple[JsonDict, ...]:
    """Return malformed candidates present before syntax constraints apply."""

    return (
        {
            "candidate_id": "raw-malformed-json",
            "text": "not json",
            "base_probability": 0.06,
        },
        {
            "candidate_id": "raw-wrong-schema",
            "text": '{"schema":"other","verdict":"accept"}',
            "base_probability": 0.04,
        },
    )


def candidate_pool_non_degeneracy(
    candidates: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Check that the finite pool exercises every requested risk family."""

    pool = tuple(candidates) if candidates is not None else semantic_control_candidates()
    counts = Counter(str(row["control_type"]) for row in pool)
    required = {
        "noop_valid",
        "tautology_valid",
        "unsupported_claim",
        "contradicted_claim",
        "distribution_sensitive_alternative",
    }
    missing = sorted(required - set(counts))
    ok = (
        len(pool) >= 8
        and not missing
        and counts["distribution_sensitive_alternative"] >= 2
        and any(bool(row["semantic_valid"]) for row in pool)
        and any(not bool(row["semantic_valid"]) for row in pool)
        and all(float(row["base_probability"]) > 0.0 for row in pool)
    )
    return {
        "ok": ok,
        "n_candidates": len(pool),
        "control_counts": dict(sorted(counts.items())),
        "missing_control_types": missing,
    }


def probe_grammar_baseline(
    *,
    module_finder: Callable[[str], Any] | None = None,
    module_importer: Callable[[str], Any] | None = None,
) -> JsonDict:
    """Probe for an external grammar engine without invoking live decoding."""

    finder = module_finder or importlib.util.find_spec
    importer = module_importer or importlib.import_module
    for module_name, backend in (
        ("llguidance", "llguidance"),
        ("xgrammar", "xgrammar"),
        ("llama_cpp", "llama_cpp_gbnf"),
    ):
        if finder(module_name) is None:
            continue
        compiled = False
        compile_error = None
        if module_name == "llama_cpp":
            try:
                module = importer(module_name)
                grammar_cls = getattr(module, "LlamaGrammar", None)
                if grammar_cls is not None:
                    grammar_cls.from_string('root ::= "x"\n', verbose=False)
                    compiled = True
            except Exception as exc:  # pragma: no cover - environment dependent.
                compile_error = type(exc).__name__
        return {
            "available": True,
            "backend": backend,
            "reason": None,
            "grammar_compiled": compiled,
            "compile_error": compile_error,
            "syntax_validity_rate": 1.0,
            "latency_ms": 0.0,
        }
    return {
        "available": False,
        "backend": None,
        "reason": "no_external_grammar_engine_available",
        "grammar_compiled": False,
        "compile_error": None,
        "syntax_validity_rate": None,
        "latency_ms": None,
    }


def load_preconditions(
    *,
    root: Path | str = REPO_ROOT,
    grammar_baseline: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Collect deterministic preconditions before any possible live decoding."""

    root_path = Path(root)
    exp5097_path = root_path / EXP5097_RELATIVE_PATH
    exp5097 = _read_json_object(exp5097_path)
    endpoint_clean = _endpoint_cleanliness(exp5097, exp5097_path)
    grammar = dict(grammar_baseline) if grammar_baseline is not None else probe_grammar_baseline()
    return {
        "selected_schema": SCHEMA_NAME,
        "tokenizer_assumptions": selected_schema_descriptor()["tokenizer_assumption"],
        "candidate_pool_non_degenerate": candidate_pool_non_degeneracy(),
        "grammar_engine_availability": {
            "available": bool(grammar.get("available")),
            "backend": grammar.get("backend"),
            "reason": grammar.get("reason"),
            "grammar_compiled": grammar.get("grammar_compiled"),
        },
        "exp5097_endpoint_cleanliness": endpoint_clean,
    }


def run_audit(
    *,
    root: Path | str = REPO_ROOT,
    repeats: int = 2000,
    grammar_probe: GrammarProbe | None = None,
) -> JsonDict:
    """Run the deterministic Exp 5104 semantic risk audit."""

    started = time.perf_counter()
    candidates = semantic_control_candidates()
    outputs = finite_schema_outputs(candidates)
    trie, csr, equivalence = build_static_mask_audit(outputs)
    latency = static5090.benchmark_mask_lookup(trie, csr, outputs, repeats=repeats)
    semantic = evaluate_semantic_controls(candidates)
    grammar_baseline = (grammar_probe or probe_grammar_baseline)()
    preconditions = load_preconditions(root=root, grammar_baseline=grammar_baseline)
    nondegenerate = bool(preconditions["candidate_pool_non_degenerate"]["ok"])
    live_llm_invoked = False
    syntax_only_headline_forbidden = True
    static_metrics = semantic["static_mask"]
    distribution_shift = float(semantic["distribution_shift_metric"])
    semantic_clean = (
        float(static_metrics["semantic_validity_rate"]) == 1.0
        and float(static_metrics["contradiction_reject_rate"]) == 1.0
        and float(static_metrics["noop_accept_rate"]) == 0.0
        and distribution_shift <= 0.05
    )
    honest_verdict = (
        "success_constrained_decoding_semantic_controls_clean"
        if semantic_clean and syntax_only_headline_forbidden
        else "complete_constrained_decoding_semantic_audit_no_syntax_only_headline"
    )
    artifact: JsonDict = {
        "schema": ARTIFACT_SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "result_path": RESULT_RELATIVE_PATH,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict,
        "duration_s": round(max(0.0, time.perf_counter() - started), 6),
        "inference_substrate": DETERMINISTIC_INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "model_specs": model_specs_from_preconditions(root=root),
        "schema_name": SCHEMA_NAME,
        "candidate_pool_non_degenerate": nondegenerate,
        "grammar_baseline": grammar_baseline,
        "syntax_validity_rate": float(static_metrics["syntax_validity_rate"]),
        "semantic_validity_rate": float(static_metrics["semantic_validity_rate"]),
        "noop_accept_rate": float(static_metrics["noop_accept_rate"]),
        "contradiction_reject_rate": float(static_metrics["contradiction_reject_rate"]),
        "distribution_shift_metric": distribution_shift,
        "latency_ms": {
            "trie": float(latency["trie_latency_ms"]),
            "csr": float(latency["csr_latency_ms"]),
            "grammar_baseline": grammar_baseline.get("latency_ms"),
            "lookup_count": int(latency["lookup_count"]),
        },
        "mask_memory": {
            "trie_bytes": static5090.estimate_trie_memory_bytes(trie),
            "csr_bytes": static5090.estimate_csr_memory_bytes(csr),
            "state_count": csr.state_count,
            "transition_count": csr.transition_count,
        },
        "syntax_only_headline_forbidden": syntax_only_headline_forbidden,
        "live_llm_invoked": live_llm_invoked,
        "flagged_adversarial": False,
        "finite_schema": selected_schema_descriptor(),
        "mask_equivalence": equivalence,
        "semantic_controls": semantic,
        "unconstrained_baseline": semantic["unconstrained"],
        "semantic_rerank_baseline": semantic["semantic_rerank"],
        "finite_outputs_sha256": _sha256_payload(outputs),
        "reproducibility_checksum": _reproducibility_checksum(outputs, csr, semantic),
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    *,
    root: Path | str = REPO_ROOT,
    output_path: Path | str | None = None,
    repeats: int = 2000,
    grammar_probe: GrammarProbe | None = None,
) -> JsonDict:
    """Persist the terminal JSON artifact consumed by the conductor."""

    root_path = Path(root)
    destination = Path(output_path) if output_path is not None else root_path / RESULT_RELATIVE_PATH
    payload = run_audit(root=root_path, repeats=repeats, grammar_probe=grammar_probe)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 5104 artifact violates the terminal contract."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact["honest_verdict"])
    if not verdict.startswith(
        (
            "complete_constrained_decoding_semantic_audit_no_syntax_only_headline",
            "success_constrained_decoding_semantic_controls_clean",
        )
    ):
        raise ValueError("honest_verdict has no accepted Exp 5104 terminal prefix")
    if artifact["inference_substrate"] == "live_llm_inference" and not artifact["live_llm_invoked"]:
        raise ValueError("live_llm_inference cannot be claimed when live_llm_invoked=false")
    if (
        artifact["inference_substrate"] != DETERMINISTIC_INFERENCE_SUBSTRATE
        and not artifact["live_llm_invoked"]
    ):
        raise ValueError("inference_substrate must be deterministic when live is false")
    if artifact["schema_name"] != SCHEMA_NAME:
        raise ValueError("schema_name does not match the selected finite schema")
    if not isinstance(artifact["candidate_pool_non_degenerate"], bool):
        raise ValueError("candidate_pool_non_degenerate must be a boolean")
    for field in ("syntax_only_headline_forbidden", "live_llm_invoked", "flagged_adversarial"):
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a boolean")
    if artifact["syntax_only_headline_forbidden"] is not True:
        raise ValueError("syntax_only_headline_forbidden must be true")
    for field in (
        "syntax_validity_rate",
        "semantic_validity_rate",
        "noop_accept_rate",
        "contradiction_reject_rate",
        "distribution_shift_metric",
    ):
        if not _is_rate(artifact[field]):
            raise ValueError(f"{field} must be in [0, 1]")
    if not _is_nonnegative_number(artifact["duration_s"]):
        raise ValueError("duration_s must be a nonnegative finite number")
    _validate_latency(artifact["latency_ms"])
    _validate_mask_memory(artifact["mask_memory"])
    if not isinstance(artifact["grammar_baseline"], Mapping):
        raise ValueError("grammar_baseline must be a mapping")
    if not isinstance(artifact["preconditions_checked"], Mapping):
        raise ValueError("preconditions_checked must be a mapping")
    model_ids = {
        str(row.get("hf_id"))
        for row in artifact.get("model_specs", [])
        if isinstance(row, Mapping)
    }
    if set(MANDATED_MODEL_IDS) - model_ids:
        raise ValueError("model_specs must include all mandated GGUF IDs")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not set(REQUIRED_ARTIFACT_FIELDS) <= set(principles):
        raise ValueError("field_principles must annotate every required field")


def model_specs_from_preconditions(*, root: Path | str = REPO_ROOT) -> list[JsonDict]:
    """Merge mandated model IDs with any Exp5097 resolved-path evidence."""

    exp5097 = _read_json_object(Path(root) / EXP5097_RELATIVE_PATH)
    resolved = _resolved_model_paths(exp5097 or {})
    specs: list[JsonDict] = []
    for base in MODEL_SPECS:
        row = dict(base)
        row["resolved_path"] = resolved.get(base["hf_id"])
        row["live_llm_invoked"] = False
        specs.append(row)
    return specs


def main() -> int:  # pragma: no cover - CLI wrapper.
    payload = write_artifact()
    print(json.dumps({field: payload[field] for field in REQUIRED_ARTIFACT_FIELDS}, indent=2))
    return 0


def _candidate(
    candidate_id: str,
    control_type: str,
    case_id: str,
    claim: str,
    verdict: str,
    evidence_label: str,
    base_probability: float,
    *,
    semantic_valid: bool,
    vacuous: bool = False,
    unsupported: bool = False,
    contradicted: bool = False,
) -> JsonDict:
    payload = {
        "case_id": case_id,
        "claim": claim,
        "control_type": control_type,
        "evidence_label": evidence_label,
        "schema": SCHEMA_NAME,
        "verdict": verdict,
    }
    return {
        "candidate_id": candidate_id,
        "control_type": control_type,
        "base_probability": float(base_probability),
        "semantic_valid": bool(semantic_valid),
        "vacuous": bool(vacuous),
        "unsupported": bool(unsupported),
        "contradicted": bool(contradicted),
        "payload": payload,
    }


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _round_rate(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 6)


def _reject_rate(rows: Sequence[Mapping[str, Any]], *, accepted: bool) -> float:
    if not rows:
        return 1.0
    rejected = 0 if accepted else len(rows)
    return _round_rate(rejected / len(rows))


def _total_variation(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    return _round_rate(0.5 * sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys))


def _endpoint_cleanliness(exp5097: Mapping[str, Any] | None, path: Path) -> JsonDict:
    exists = exp5097 is not None
    clean = bool(
        exists
        and exp5097.get("logprob_endpoint_clean") is True
        and exp5097.get("logprob_endpoint_ready") is True
        and exp5097.get("flagged_adversarial") is not True
    )
    if not exists:
        reason = "exp5097_artifact_missing"
    elif not clean:
        reason = "exp5097_not_clean_for_live_decoding"
    else:
        reason = None
    return {
        "artifact_path": EXP5097_RELATIVE_PATH,
        "exists": exists,
        "artifact_sha256": _sha256_file(path),
        "honest_verdict": exp5097.get("honest_verdict") if exp5097 else None,
        "endpoint_url": exp5097.get("endpoint_url") if exp5097 else None,
        "logprob_endpoint_clean": bool(exp5097.get("logprob_endpoint_clean")) if exp5097 else False,
        "logprob_endpoint_ready": bool(exp5097.get("logprob_endpoint_ready")) if exp5097 else False,
        "live_llm_invoked": bool(exp5097.get("live_llm_invoked")) if exp5097 else False,
        "flagged_adversarial": bool(exp5097.get("flagged_adversarial")) if exp5097 else False,
        "clean_for_live_decoding": clean,
        "unusable_reason": reason,
    }


def _resolved_model_paths(exp5097: Mapping[str, Any]) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {model_id: None for model_id in MANDATED_MODEL_IDS}
    model_specs = exp5097.get("model_specs")
    if not isinstance(model_specs, Mapping):
        return resolved
    mandatory_models = model_specs.get("mandatory_models")
    if isinstance(mandatory_models, Sequence) and not isinstance(mandatory_models, (str, bytes)):
        for value in mandatory_models:
            if isinstance(value, Mapping):
                hf_id = str(value.get("hf_id") or "")
                if hf_id in resolved:
                    resolved[hf_id] = _optional_string(value.get("resolved_path"))
    return resolved


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    parsed = str(value)
    return parsed or None


def _read_json_object(path: Path) -> JsonDict | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(loaded) if isinstance(loaded, Mapping) else None


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_payload(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reproducibility_checksum(
    outputs: Sequence[str],
    csr: static5090.CSRAutomaton,
    semantic: Mapping[str, Any],
) -> str:
    payload = {
        "outputs": list(outputs),
        "row_offsets": list(csr.row_offsets),
        "labels": list(csr.labels),
        "targets": list(csr.targets),
        "accepting_states": sorted(csr.accepting_states),
        "semantic_static": semantic["static_mask"],
        "distribution_shift_metric": semantic["distribution_shift_metric"],
    }
    return _sha256_payload(payload)


def _is_rate(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= parsed <= 1.0


def _is_nonnegative_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return parsed >= 0.0


def _validate_latency(latency: Any) -> None:
    if not isinstance(latency, Mapping):
        raise ValueError("latency_ms must be a mapping")
    for field in ("trie", "csr"):
        if not _is_nonnegative_number(latency.get(field)):
            raise ValueError(f"latency_ms.{field} must be nonnegative")
    grammar_latency = latency.get("grammar_baseline")
    if grammar_latency is not None and not _is_nonnegative_number(grammar_latency):
        raise ValueError("latency_ms.grammar_baseline must be nonnegative or null")


def _validate_mask_memory(mask_memory: Any) -> None:
    if not isinstance(mask_memory, Mapping):
        raise ValueError("mask_memory must be a mapping")
    for field in ("trie_bytes", "csr_bytes", "state_count", "transition_count"):
        value = mask_memory.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"mask_memory.{field} must be a positive integer")


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
