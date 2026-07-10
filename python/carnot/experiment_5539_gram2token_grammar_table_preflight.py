"""Exp5539 Gram2Token-style grammar-table preflight for hard/soft schema rows.

Spec refs: REQ-VERIFY-5539, SCENARIO-VERIFY-5539.

This module is a local preflight only. It proves which grammar compile surface
and deterministic schema-table evidence are reachable for the Exp5512
hard/soft candidate schema. It does not load a model, invoke an LLM, measure
decoding, or claim model quality.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5512_structured_output_positive_control as positive


JsonDict = dict[str, Any]
ModuleAvailable = Callable[[str], bool]
LlamaGrammarCompiler = Callable[[str], object]
LlGuidanceGrammarCompiler = Callable[[Mapping[str, Any]], str | None]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5539_gram2token_grammar_table_preflight.json")

SCHEMA = "carnot.experiment_5539.gram2token_grammar_table_preflight.v502"
EXPERIMENT = 5539
EXPERIMENT_ID = "exp5539-gram2token-grammar-table-preflight"
MILESTONE = "2026.07.502"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5539
INFERENCE_SUBSTRATE = "deterministic_grammar_table_preflight_no_llm"
SPEC_REFS = ("REQ-VERIFY-5539", "SCENARIO-VERIFY-5539")

REQUIRED_ARTIFACT_FIELDS = (
    "grammar_backend_candidates",
    "selected_backend",
    "backend_available",
    "schema_hash",
    "table_hashes",
    "valid_fixture_acceptance_rate",
    "invalid_fixture_rejection_rate",
    "unsupported_schema_features",
    "llm_invoked",
    "no_model_specs_required",
    "decoding_speedup_claim",
    "grammar_table_preflight_ready",
    "tests_added_or_reused",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
)

TESTS_ADDED_OR_REUSED = (
    "tests/python/test_experiment_5539_gram2token_grammar_table_preflight.py",
    "tests/python/test_experiment_5512_structured_output_positive_control.py",
    "tests/python/test_experiment_5525_sota_schema_failure_taxonomy.py",
    "tests/python/test_experiment_5526_sota_structured_repair_loop.py",
)

FIELD_PRINCIPLES: JsonDict = {
    "grammar_backend_candidates": "Lists each local grammar or schema helper checked before selecting a backend.",
    "selected_backend": "Names only the constrained grammar backend whose compile path was actually reached.",
    "backend_available": "Separates reachable constrained grammar infrastructure from parser-only fallback evidence.",
    "schema_hash": "Pins the preflight to the reused Exp5512 hard/soft candidate schema.",
    "table_hashes": "Content-addresses available grammar, schema-transition, and fixture acceptance tables without inventing hidden token tables.",
    "valid_fixture_acceptance_rate": "Confirms schema-valid hard/soft fixture rows still reach the exact validators.",
    "invalid_fixture_rejection_rate": "Confirms malformed or semantically invalid fixture rows are rejected locally.",
    "unsupported_schema_features": "Keeps cross-field and exact-validator semantics out of the grammar-table claim.",
    "llm_invoked": "Must stay false because this preflight checks infrastructure, not generation quality.",
    "no_model_specs_required": "Prevents model metadata from appearing when no model was loaded.",
    "decoding_speedup_claim": "Must stay false because no decoding runtime or timing comparison ran.",
    "grammar_table_preflight_ready": "Opens only when a constrained backend, table hashes, and fixture gates are all clean.",
    "tests_added_or_reused": "Links the artifact to tests that exercise backend selection and fixture gates.",
    "field_principles": "Explains why each headline and gate field must remain in future artifacts.",
    "inference_substrate": "Declares deterministic no-LLM grammar-table preflight as the evidence substrate.",
    "honest_verdict": "Provides a terminal status that cannot promote reachability into speed or quality.",
}

BASE_UNSUPPORTED_SCHEMA_FEATURES = (
    "instance_specific_assignment_keys_require_exp5499_exact_validator",
    "assignment_domain_values_require_exp5499_exact_validator",
    "validator_target_cross_field_equality_requires_exp5499_exact_validator",
    "hard_constraint_feasibility_not_json_schema_expressible",
    "soft_preference_optimality_not_json_schema_expressible",
    "abstention_correctness_not_json_schema_expressible",
    "premise_and_rule_text_truth_not_grammar_checkable",
    "free_form_string_lengths_not_token_table_optimized_here",
)


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically so hashes are stable across runs."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    """Return a SHA-256 hex digest for UTF-8 text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a SHA-256 hex digest for a JSON-compatible value."""

    return sha256_text(canonical_json(value))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking the self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def grammar_backend_candidates(
    *,
    module_available: ModuleAvailable | None = None,
    llama_grammar_compiler: LlamaGrammarCompiler | None = None,
    llguidance_grammar_compiler: LlGuidanceGrammarCompiler | None = None,
) -> list[JsonDict]:
    """Inspect local grammar/runtime helpers without loading any model."""

    available = module_available or _module_available
    schema = positive.candidate_schema()
    gbnf = positive.build_llama_cpp_json_grammar()
    candidates = [
        _llama_cpp_candidate(
            available=available,
            grammar=gbnf,
            compiler=llama_grammar_compiler,
        ),
        _llguidance_candidate(
            available=available,
            schema=schema,
            compiler=llguidance_grammar_compiler,
        ),
        _xgrammar_candidate(available=available),
        {
            "name": "repository_json_schema_validator",
            "available": True,
            "import_available": True,
            "grammar_compiled": True,
            "constrained_generation": False,
            "table_exposed": True,
            "schema_support": "post-decode validation plus deterministic schema-transition table",
            "failure_reason": None,
            "notes": "Useful as fixture/table evidence but not a constrained decoding backend.",
        },
        {
            "name": "experiment_5512_parser_exact_validator",
            "available": True,
            "import_available": True,
            "grammar_compiled": True,
            "constrained_generation": False,
            "table_exposed": True,
            "schema_support": "parser handoff plus Exp5499 exact hard/soft validators",
            "failure_reason": None,
            "notes": "Validates semantics after JSON is available; not a token mask runtime.",
        },
    ]
    return candidates


def select_backend(candidates: Sequence[Mapping[str, Any]]) -> tuple[str, bool]:
    """Select the first constrained grammar backend with a reached compile path."""

    priority = ("llama_cpp_gbnf", "llguidance_json_schema", "xgrammar_json_schema")
    by_name = {str(row.get("name")): row for row in candidates}
    for name in priority:
        row = by_name.get(name, {})
        if (
            row.get("available") is True
            and row.get("grammar_compiled") is True
            and row.get("constrained_generation") is True
        ):
            return name, True
    return "none", False


def build_schema_transition_table(schema: Mapping[str, Any]) -> list[JsonDict]:
    """Build a deterministic schema-shape table from the bounded JSON schema."""

    rows: list[JsonDict] = []
    _walk_schema("$", schema, rows)
    return sorted(rows, key=lambda row: str(row["path"]))


def build_invalid_fixture_payloads(valid_payloads: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Create deterministic malformed and semantically invalid fixture rows."""

    if len(valid_payloads) < 2:
        return []

    missing_required = _json_clone(valid_payloads[0])
    missing_required.pop("candidate_schema_version", None)

    target_mismatch = _json_clone(valid_payloads[1])
    target = dict(target_mismatch["validator_target"])
    target["instance_id"] = "wrong_instance_for_target"
    target_mismatch["validator_target"] = target

    assignment_domain = _json_clone(
        next(
            row
            for row in valid_payloads
            if row.get("conclusion", {}).get("status") == "candidate"
        )
    )
    assignment = dict(assignment_domain["conclusion"]["assignment"])
    first_key = sorted(assignment)[0]
    assignment[first_key] = "not_in_fixture_domain"
    conclusion = dict(assignment_domain["conclusion"])
    conclusion["assignment"] = assignment
    assignment_domain["conclusion"] = conclusion

    return [missing_required, target_mismatch, assignment_domain]


def evaluate_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    fixture: Mapping[str, Any] | None = None,
) -> list[JsonDict]:
    """Classify fixture payloads through the reused Exp5512 validator handoff."""

    fixture_payload = dict(fixture or positive.load_fixture_artifact()["fixture"])
    rows = []
    for payload in payloads:
        classified = positive.classify_candidate_payload(payload, fixture=fixture_payload)
        accepted = _accepted(classified)
        rows.append(
            {
                "instance_id": str(payload.get("instance_id", classified.get("instance_id", ""))),
                "candidate_id": str(payload.get("candidate_id", classified.get("candidate_id", ""))),
                "accepted": accepted,
                "acceptance_status": str(classified.get("parse_status", "")),
                "schema_valid": bool(classified.get("schema_valid") is True),
                "parseable": bool(classified.get("parseable") is True),
                "exact_validator_verdict": str(classified.get("exact_validator_verdict", "")),
                "exact_validator_correct": bool(classified.get("exact_validator_correct") is True),
                "schema_errors": [str(error) for error in classified.get("schema_errors", [])],
            }
        )
    return rows


def acceptance_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return the fraction of rows accepted by schema plus exact-validator handoff."""

    return _rate(sum(int(row.get("accepted") is True) for row in rows), len(rows))


def rejection_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return the fraction of rows rejected by schema or exact-validator handoff."""

    return _rate(sum(int(row.get("accepted") is not True) for row in rows), len(rows))


def build_artifact(
    *,
    module_available: ModuleAvailable | None = None,
    llama_grammar_compiler: LlamaGrammarCompiler | None = None,
    llguidance_grammar_compiler: LlGuidanceGrammarCompiler | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5539 no-LLM grammar-table preflight artifact."""

    schema = positive.candidate_schema()
    fixture = positive.load_fixture_artifact()["fixture"]
    valid_payloads = positive.build_fixture_candidate_payloads(fixture)
    invalid_payloads = build_invalid_fixture_payloads(valid_payloads)
    valid_rows = evaluate_payloads(valid_payloads, fixture=fixture)
    invalid_rows = evaluate_payloads(invalid_payloads, fixture=fixture)
    candidates = grammar_backend_candidates(
        module_available=module_available,
        llama_grammar_compiler=llama_grammar_compiler,
        llguidance_grammar_compiler=llguidance_grammar_compiler,
    )
    selected_backend, backend_available = select_backend(candidates)
    schema_table = build_schema_transition_table(schema)
    table_hashes = _table_hashes(
        candidates=candidates,
        selected_backend=selected_backend,
        schema_table=schema_table,
        valid_rows=valid_rows,
        invalid_rows=invalid_rows,
    )
    valid_rate = acceptance_rate(valid_rows)
    invalid_rate = rejection_rate(invalid_rows)
    ready = bool(backend_available and table_hashes and valid_rate == 1.0 and invalid_rate == 1.0)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "grammar_backend_candidates": candidates,
        "selected_backend": selected_backend,
        "backend_available": backend_available,
        "schema_hash": positive.sha256_json(schema),
        "table_hashes": table_hashes,
        "valid_fixture_acceptance_rate": valid_rate,
        "invalid_fixture_rejection_rate": invalid_rate,
        "unsupported_schema_features": unsupported_schema_features(candidates, selected_backend),
        "llm_invoked": False,
        "no_model_specs_required": True,
        "decoding_speedup_claim": False,
        "grammar_table_preflight_ready": ready,
        "tests_added_or_reused": list(TESTS_ADDED_OR_REUSED),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(ready, selected_backend),
        "candidate_schema_version": positive.CANDIDATE_SCHEMA_VERSION,
        "candidate_schema_path": positive.SCHEMA_PATH,
        "schema_transition_table_row_count": len(schema_table),
        "schema_transition_table": schema_table,
        "valid_fixture_rows": valid_rows,
        "invalid_fixture_rows": invalid_rows,
        "grammar_table_evidence_scope": (
            "schema/grammar compile and deterministic fixture table only; "
            "no tokenizer transition table, decoding path, speed, or model quality measured"
        ),
        "research_conductor_modified": False,
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    module_available: ModuleAvailable | None = None,
    llama_grammar_compiler: LlamaGrammarCompiler | None = None,
    llguidance_grammar_compiler: LlGuidanceGrammarCompiler | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the Exp5539 result JSON."""

    artifact = build_artifact(
        module_available=module_available,
        llama_grammar_compiler=llama_grammar_compiler,
        llguidance_grammar_compiler=llguidance_grammar_compiler,
        tests_run=tests_run,
    )
    if write:
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate Exp5539 fields and fail closed on speed, model, or LLM claims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(str(artifact.get("honest_verdict", "")).startswith(("complete:", "blocked:")), "honest_verdict")
    _require(artifact.get("llm_invoked") is False, "llm_invoked")
    _require(artifact.get("no_model_specs_required") is True, "no_model_specs_required")
    _require(artifact.get("decoding_speedup_claim") is False, "decoding_speedup_claim")
    _require("model_specs" not in artifact, "model_specs")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    _require(isinstance(artifact.get("grammar_backend_candidates"), list), "grammar_backend_candidates")
    _require(isinstance(artifact.get("selected_backend"), str), "selected_backend")
    _require(isinstance(artifact.get("backend_available"), bool), "backend_available")
    _require(isinstance(artifact.get("grammar_table_preflight_ready"), bool), "grammar_table_preflight_ready")
    _require(artifact.get("schema_hash") == positive.sha256_json(positive.candidate_schema()), "schema_hash")
    _require(isinstance(artifact.get("table_hashes"), list), "table_hashes")
    _require(isinstance(artifact.get("unsupported_schema_features"), list), "unsupported_schema_features")
    _require(
        set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact.get("field_principles", {})),
        "field_principles",
    )
    for field in ("valid_fixture_acceptance_rate", "invalid_fixture_rejection_rate"):
        value = float(artifact.get(field, -1.0))
        _require(0.0 <= value <= 1.0, field)
    for row in artifact.get("table_hashes", []):
        _require(isinstance(row, Mapping), "table_hashes")
        digest = str(row.get("hash", ""))
        _require(len(digest) == 64 and all(char in "0123456789abcdef" for char in digest), "table_hashes")
    if artifact.get("grammar_table_preflight_ready") is True:
        _require(artifact.get("backend_available") is True, "backend_available")
        _require(artifact.get("selected_backend") != "none", "selected_backend")
        _require(artifact.get("valid_fixture_acceptance_rate") == 1.0, "valid_fixture_acceptance_rate")
        _require(artifact.get("invalid_fixture_rejection_rate") == 1.0, "invalid_fixture_rejection_rate")
        _require(str(artifact.get("honest_verdict", "")).startswith("complete:"), "honest_verdict")
    if artifact.get("backend_available") is False:
        _require(artifact.get("grammar_table_preflight_ready") is False, "grammar_table_preflight_ready")
    _require(artifact.get("tests_added_or_reused") == list(TESTS_ADDED_OR_REUSED), "tests_added_or_reused")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def unsupported_schema_features(
    candidates: Sequence[Mapping[str, Any]],
    selected_backend: str,
) -> list[str]:
    """List grammar-table limitations that still require parser or validator checks."""

    features = list(BASE_UNSUPPORTED_SCHEMA_FEATURES)
    by_name = {str(row.get("name")): row for row in candidates}
    if selected_backend == "llama_cpp_gbnf" and by_name.get("llama_cpp_gbnf", {}).get("table_exposed") is False:
        features.append("llama_cpp_token_transition_table_not_exposed")
    if selected_backend == "none":
        features.append("no_constrained_grammar_backend_reachable")
    for name in ("llguidance_json_schema", "xgrammar_json_schema"):
        failure = by_name.get(name, {}).get("failure_reason")
        if failure:
            features.append(str(failure))
    return sorted(set(features))


def honest_verdict(ready: bool, selected_backend: str) -> str:
    """Return a terminal verdict that cannot imply generation speed or quality."""

    if ready:
        return (
            "complete: gram2token_preflight_ready_"
            f"{selected_backend}_schema_reachable_no_llm_no_speedup_or_quality_claim"
        )
    return "blocked: gram2token_preflight_not_ready_no_constrained_grammar_backend"


def _llama_cpp_candidate(
    *,
    available: ModuleAvailable,
    grammar: str,
    compiler: LlamaGrammarCompiler | None,
) -> JsonDict:
    import_available = available("llama_cpp")
    if not import_available:
        return {
            "name": "llama_cpp_gbnf",
            "available": False,
            "import_available": False,
            "grammar_compiled": False,
            "constrained_generation": True,
            "table_exposed": False,
            "schema_support": "GBNF compile path for JSON grammar",
            "failure_reason": "llama_cpp_not_installed",
        }
    try:
        compile_fn = compiler or positive._compile_llama_cpp_grammar
        compile_fn(grammar)
    except Exception as exc:
        return {
            "name": "llama_cpp_gbnf",
            "available": False,
            "import_available": True,
            "grammar_compiled": False,
            "constrained_generation": True,
            "table_exposed": False,
            "schema_support": "GBNF compile path for JSON grammar",
            "failure_reason": f"llama_cpp_grammar_compile_failed:{type(exc).__name__}",
        }
    return {
        "name": "llama_cpp_gbnf",
        "available": True,
        "import_available": True,
        "grammar_compiled": True,
        "constrained_generation": True,
        "table_exposed": False,
        "schema_support": "GBNF compile path for JSON grammar; no internal token table exposed here",
        "failure_reason": None,
        "grammar_hash": sha256_text(grammar),
    }


def _llguidance_candidate(
    *,
    available: ModuleAvailable,
    schema: Mapping[str, Any],
    compiler: LlGuidanceGrammarCompiler | None,
) -> JsonDict:
    import_available = available("llguidance")
    if not import_available:
        return {
            "name": "llguidance_json_schema",
            "available": False,
            "import_available": False,
            "grammar_compiled": False,
            "constrained_generation": True,
            "table_exposed": False,
            "schema_support": "llguidance JSON-schema grammar compiler",
            "failure_reason": "llguidance_not_installed",
        }
    try:
        compile_fn = compiler or _compile_llguidance_schema_grammar
        grammar = compile_fn(schema)
        if not grammar:
            raise ValueError("empty grammar")
    except Exception as exc:
        return {
            "name": "llguidance_json_schema",
            "available": False,
            "import_available": True,
            "grammar_compiled": False,
            "constrained_generation": True,
            "table_exposed": False,
            "schema_support": "llguidance JSON-schema grammar compiler",
            "failure_reason": f"llguidance_grammar_compile_failed:{type(exc).__name__}",
        }
    return {
        "name": "llguidance_json_schema",
        "available": True,
        "import_available": True,
        "grammar_compiled": True,
        "constrained_generation": True,
        "table_exposed": False,
        "schema_support": "llguidance JSON-schema grammar compiler",
        "failure_reason": None,
        "grammar_hash": sha256_text(str(grammar)),
    }


def _xgrammar_candidate(*, available: ModuleAvailable) -> JsonDict:
    import_available = available("xgrammar")
    if not import_available:
        return {
            "name": "xgrammar_json_schema",
            "available": False,
            "import_available": False,
            "grammar_compiled": False,
            "constrained_generation": True,
            "table_exposed": False,
            "schema_support": "XGrammar JSON-schema compiler when locally wired",
            "failure_reason": "xgrammar_not_installed",
        }
    return {
        "name": "xgrammar_json_schema",
        "available": False,
        "import_available": True,
        "grammar_compiled": False,
        "constrained_generation": True,
        "table_exposed": False,
        "schema_support": "XGrammar import was present, but this preflight has no wired compile API",
        "failure_reason": "xgrammar_compile_api_not_wired_in_this_preflight",
    }


def _table_hashes(
    *,
    candidates: Sequence[Mapping[str, Any]],
    selected_backend: str,
    schema_table: Sequence[Mapping[str, Any]],
    valid_rows: Sequence[Mapping[str, Any]],
    invalid_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    hashes: list[JsonDict] = [
        {
            "name": "hard_soft_schema_transition_table",
            "backend": "repository_json_schema_validator",
            "kind": "schema_transition_table",
            "hash": sha256_json(list(schema_table)),
            "row_count": len(schema_table),
            "source": "Exp5512 candidate_schema traversed by Exp5539",
        },
        {
            "name": "valid_fixture_acceptance_table",
            "backend": "experiment_5512_parser_exact_validator",
            "kind": "fixture_acceptance_table",
            "hash": sha256_json(list(valid_rows)),
            "row_count": len(valid_rows),
            "source": "Exp5512 fixture candidate rows",
        },
        {
            "name": "invalid_fixture_rejection_table",
            "backend": "experiment_5512_parser_exact_validator",
            "kind": "fixture_rejection_table",
            "hash": sha256_json(list(invalid_rows)),
            "row_count": len(invalid_rows),
            "source": "Exp5539 deterministic invalid fixture mutations",
        },
    ]
    if selected_backend == "llama_cpp_gbnf":
        hashes.insert(
            0,
            {
                "name": "llama_cpp_json_gbnf",
                "backend": "llama_cpp_gbnf",
                "kind": "compiled_grammar_source",
                "hash": sha256_text(positive.build_llama_cpp_json_grammar()),
                "row_count": 0,
                "source": "Exp5512 build_llama_cpp_json_grammar",
            },
        )
    for candidate in candidates:
        if candidate.get("name") == "llguidance_json_schema" and candidate.get("grammar_hash"):
            hashes.append(
                {
                    "name": "llguidance_schema_grammar",
                    "backend": "llguidance_json_schema",
                    "kind": "compiled_grammar_source",
                    "hash": str(candidate["grammar_hash"]),
                    "row_count": 0,
                    "source": "llguidance LLMatcher.grammar_from_json_schema",
                }
            )
    return hashes


def _walk_schema(path: str, spec: Mapping[str, Any], rows: list[JsonDict]) -> None:
    node_type = spec.get("type", "any")
    row: JsonDict = {
        "path": path,
        "type": node_type,
    }
    if "enum" in spec:
        row["enum"] = list(spec["enum"])
    if node_type == "object":
        properties = spec.get("properties", {})
        property_names = sorted(str(name) for name in properties) if isinstance(properties, Mapping) else []
        required = sorted(str(name) for name in spec.get("required", []))
        row.update(
            {
                "properties": property_names,
                "required": required,
                "additional_properties": spec.get("additionalProperties", None),
                "dynamic_object": not property_names and spec.get("additionalProperties", None) is not False,
            }
        )
        rows.append(row)
        if isinstance(properties, Mapping):
            for name in property_names:
                subschema = properties.get(name)
                if isinstance(subschema, Mapping):
                    _walk_schema(f"{path}.{name}", subschema, rows)
        return
    if node_type == "array":
        row.update(
            {
                "min_items": spec.get("minItems", None),
                "max_items": spec.get("maxItems", None),
                "has_item_schema": isinstance(spec.get("items"), Mapping),
            }
        )
        rows.append(row)
        item_schema = spec.get("items")
        if isinstance(item_schema, Mapping):
            _walk_schema(f"{path}[]", item_schema, rows)
        return
    for key in ("minimum", "maximum", "minLength", "maxLength", "pattern"):
        if key in spec:
            row[key] = spec[key]
    rows.append(row)


def _compile_llguidance_schema_grammar(schema: Mapping[str, Any]) -> str | None:
    module = importlib.import_module("llguidance")
    matcher = getattr(module, "LLMatcher", None)
    grammar_from_schema = getattr(matcher, "grammar_from_json_schema", None)
    if not callable(grammar_from_schema):
        raise AttributeError("LLMatcher.grammar_from_json_schema")
    return str(grammar_from_schema(_json_clone(schema), overrides={"whitespace_flexible": False}))


def _accepted(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("schema_valid") is True
        and row.get("parseable") is True
        and row.get("exact_validator_correct") is True
        and row.get("exact_validator_verdict") != "not_handed_off"
    )


def _json_clone(value: Any) -> Any:
    try:
        return json.loads(canonical_json(value))
    except TypeError:
        return deepcopy(value)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _rate(numerator: int | float, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:  # pragma: no cover
    artifact = run()
    print(
        json.dumps(
            {
                "result": RESULT_RELATIVE_PATH.as_posix(),
                "selected_backend": artifact["selected_backend"],
                "grammar_table_preflight_ready": artifact["grammar_table_preflight_ready"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
