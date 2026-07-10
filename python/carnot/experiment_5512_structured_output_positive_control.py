"""Exp5512 structured-output positive control for hard/soft claim candidates.

Spec refs: REQ-VERIFY-5512, SCENARIO-VERIFY-5512.

This module proves the parser and exact-validator handoff before another SOTA
hard/soft panel spends GGUF runtime. The deterministic positive control builds
candidate proof/claim rows from Exp5499's tiny typed fixture, validates a
bounded JSON schema, and then sends only schema-valid rows to the same exact
hard/soft validators that own the final verdict.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import gc
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5499_preference_maxsat_minimal_fixture_v499 as fixture_mod
from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf
from carnot.verifiers.dccd_adapter import extract_json_object, validate_json_schema


JsonDict = dict[str, Any]
CacheResolver = Callable[[str, str], str | None]
SmokeRunner = Callable[[Mapping[str, Any], str, str | None], str]
ModuleAvailable = Callable[[str], bool]
LlamaCudaProbe = Callable[[], bool]
LlamaGrammarCompiler = Callable[[str], object]
PairResolver = Callable[[], Sequence[Mapping[str, Any]] | None]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5512_structured_output_positive_control.json")
FIXTURE_ARTIFACT_RELATIVE_PATH = fixture_mod.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5512.structured_output_positive_control.v500"
CANDIDATE_SCHEMA_VERSION = "carnot.hard_soft_claim_candidate.v1"
EXPERIMENT = 5512
EXPERIMENT_ID = "exp5512-structured-output-positive-control"
MILESTONE = "2026.07.500"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5512
N_GPU_LAYERS = -1
PREFERRED_QUANT = "Q4_K_M"
INFERENCE_SUBSTRATE = "structured_output_fixture_or_live_llm_smoke"
SCHEMA_PATH = "python/carnot/experiment_5512_structured_output_positive_control.py::candidate_schema"
PARSER_PATH = (
    "python/carnot/experiment_5512_structured_output_positive_control.py::classify_candidate_text"
)
SPEC_REFS = ("REQ-VERIFY-5512", "SCENARIO-VERIFY-5512")

MANDATED_HEADLINE_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
_REGISTRY_BY_ID = {row["hf_id"]: row for row in SOTA_GGUF_MODELS}
MODEL_SPECS: list[JsonDict] = [
    {
        "name": _REGISTRY_BY_ID[hf_id]["name"],
        "hf_id": hf_id,
        "role": _REGISTRY_BY_ID[hf_id]["role"],
        "preferred_quant": PREFERRED_QUANT,
        "headline_eligible": True,
    }
    for hf_id in MANDATED_HEADLINE_MODEL_IDS
]

TESTS_ADDED_OR_REUSED = (
    "tests/python/test_experiment_5512_structured_output_positive_control.py",
    "tests/python/test_experiment_5499_preference_maxsat_minimal_fixture_v499.py",
    "tests/python/test_experiment_5500_sota_concept_claim_panel_v499.py",
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema_path",
    "parser_path",
    "tests_added_or_reused",
    "model_specs",
    "smoke_models_used",
    "grammar_runtime_available",
    "parser_only_fallback_used",
    "schema_validity_rate",
    "parseable_candidate_rows",
    "missing_candidate_rows",
    "exact_validator_handoff_ready",
    "structured_output_positive_control_ready",
    "llama_cpp_cuda_available",
    "inference_substrate",
    "honest_verdict",
)


def candidate_schema() -> JsonDict:
    """Return the bounded row schema used for each hard/soft claim candidate."""

    string_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "candidate_schema_version",
            "instance_id",
            "candidate_id",
            "premises",
            "rules_or_constraints",
            "conclusion",
            "abstention_reason",
            "validator_target",
        ],
        "properties": {
            "candidate_schema_version": {
                "type": "string",
                "enum": [CANDIDATE_SCHEMA_VERSION],
            },
            "instance_id": {"type": "string"},
            "candidate_id": {"type": "string"},
            "premises": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["premise_id", "text", "source"],
                    "properties": {
                        "premise_id": {"type": "string"},
                        "text": {"type": "string"},
                        "source": {"type": "string"},
                    },
                },
            },
            "rules_or_constraints": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["rule_id", "kind", "text", "validator_constraint_id"],
                    "properties": {
                        "rule_id": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": [
                                "hard_constraint",
                                "soft_preference",
                                "domain_rule",
                                "abstention_rule",
                            ],
                        },
                        "text": {"type": "string"},
                        "validator_constraint_id": {"type": "string"},
                    },
                },
            },
            "conclusion": {
                "type": "object",
                "additionalProperties": False,
                "required": ["status", "assignment", "confidence"],
                "properties": {
                    "status": {"type": "string", "enum": ["candidate", "abstain"]},
                    "assignment": {"type": "object"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
            "abstention_reason": {"type": "string"},
            "validator_target": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "instance_id",
                    "expected_status",
                    "reference_solver_path",
                    "hard_constraint_ids",
                    "soft_preference_ids",
                    "typed_claim_names",
                ],
                "properties": {
                    "instance_id": {"type": "string"},
                    "expected_status": {"type": "string", "enum": ["optimal", "infeasible"]},
                    "reference_solver_path": {"type": "string"},
                    "hard_constraint_ids": string_array,
                    "soft_preference_ids": string_array,
                    "typed_claim_names": string_array,
                },
            },
        },
    }


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically so checksum and smoke prompts are stable."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value after stable serialization."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def schema_errors(payload: Any) -> list[str]:
    """Return row-schema errors using Carnot's bounded JSON Schema validator."""

    return validate_json_schema(candidate_schema(), payload)


def load_fixture_artifact(
    path: Path = REPO_ROOT / FIXTURE_ARTIFACT_RELATIVE_PATH,
) -> JsonDict:
    """Load Exp5499's artifact, or rebuild the fixture when only source exists."""

    if path.exists():
        artifact = json.loads(path.read_text(encoding="utf-8"))
        fixture_mod.validate_fixture(artifact["fixture"])
        return artifact
    fixture = fixture_mod.build_fixture()
    fixture_mod.validate_fixture(fixture)
    return {"fixture": fixture}


def build_fixture_candidate_payloads(
    fixture: Mapping[str, Any] | None = None,
) -> list[JsonDict]:
    """Build deterministic positive-control rows from Exp5499 exact references."""

    fixture_payload = dict(fixture or load_fixture_artifact()["fixture"])
    fixture_mod.validate_fixture(fixture_payload)
    return [build_candidate_payload(instance) for instance in fixture_payload["instances"]]


def build_candidate_payload(instance: Mapping[str, Any]) -> JsonDict:
    """Create one schema-valid candidate row for an Exp5499 fixture instance."""

    reference = fixture_mod.solve_reference(instance)
    instance_id = str(instance["instance_id"])
    if reference["status"] == "optimal":
        conclusion = {
            "status": "candidate",
            "assignment": dict(reference["assignment"]),
            "confidence": 1.0,
        }
        abstention_reason = ""
        candidate_suffix = "exact_assignment"
    else:
        conclusion = {"status": "abstain", "assignment": {}, "confidence": 1.0}
        hard_ids = ", ".join(str(row["id"]) for row in instance["hard_constraints"])
        abstention_reason = f"hard_constraints_infeasible: {hard_ids}"
        candidate_suffix = "explicit_abstention"

    return {
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "instance_id": instance_id,
        "candidate_id": f"{instance_id}_{candidate_suffix}_5512",
        "premises": _premises(instance),
        "rules_or_constraints": _rules_or_constraints(instance, reference["status"]),
        "conclusion": conclusion,
        "abstention_reason": abstention_reason,
        "validator_target": _validator_target(instance),
    }


def classify_candidate_text(
    text: str,
    fixture: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Extract one JSON object from model text and classify its parser handoff."""

    parsed = extract_json_object(text)
    if parsed is None:
        return _failure_row(
            parse_status="no_json_object",
            schema_errors=["$ is not a JSON object"],
            parsed_payload={},
        )
    return classify_candidate_payload(parsed, fixture=fixture)


def classify_candidate_payload(
    payload: Mapping[str, Any],
    fixture: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Validate a candidate row and, when safe, hand it to Exp5499 validators."""

    errors = schema_errors(payload)
    if errors:
        return _failure_row(
            parse_status="schema_invalid",
            schema_errors=errors,
            parsed_payload=dict(payload),
            instance_id=str(payload.get("instance_id", "")) or None,
            candidate_id=str(payload.get("candidate_id", "")) or None,
        )

    fixture_payload = dict(fixture or load_fixture_artifact()["fixture"])
    instances = {str(row["instance_id"]): row for row in fixture_payload["instances"]}
    instance_id = str(payload["instance_id"])
    instance = instances.get(instance_id)
    if instance is None:
        return _semantic_failure(payload, "unknown_instance")

    if not _validator_target_matches(payload, instance):
        return _semantic_failure(payload, "validator_target_mismatch")

    conclusion = payload["conclusion"]
    conclusion_status = str(conclusion["status"])
    if conclusion_status == "abstain":
        if not str(payload.get("abstention_reason", "")).strip():
            return _semantic_failure(payload, "abstention_reason_missing")
        return _score_abstention(payload, instance)

    assignment = {str(key): str(value) for key, value in conclusion["assignment"].items()}
    domains = fixture_mod.domains_from_instance(instance)
    if set(assignment) != set(domains):
        return _semantic_failure(payload, "invalid_assignment_keys")
    if any(assignment[name] not in domains[name] for name in domains):
        return _semantic_failure(payload, "invalid_assignment_domain")
    return _score_assignment(payload, instance, assignment)


def evaluate_candidate_payloads(
    payloads: Sequence[Mapping[str, Any]],
    fixture: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Evaluate parser coverage and exact-validator handoff over candidate rows."""

    fixture_payload = dict(fixture or load_fixture_artifact()["fixture"])
    expected_ids = {str(row["instance_id"]) for row in fixture_payload["instances"]}
    rows = [
        classify_candidate_payload(payload, fixture=fixture_payload)
        for payload in payloads
    ]
    parseable_rows = [row for row in rows if row["parseable"] is True]
    schema_valid_rows = [row for row in rows if row["schema_valid"] is True]
    parseable_ids = {str(row["instance_id"]) for row in parseable_rows if row.get("instance_id")}
    exact_handoff_ready = bool(parseable_rows) and all(
        row["exact_validator_verdict"] != "not_handed_off" for row in parseable_rows
    )
    exact_rows_correct = bool(parseable_rows) and all(
        row["exact_validator_correct"] is True for row in parseable_rows
    )
    missing_candidate_rows = len(expected_ids - parseable_ids)
    schema_validity_rate = _rate(len(schema_valid_rows), len(expected_ids))
    positive_ready = (
        schema_validity_rate == 1.0
        and len(parseable_rows) == len(expected_ids)
        and missing_candidate_rows == 0
        and exact_handoff_ready
        and exact_rows_correct
    )
    return {
        "schema_validity_rate": schema_validity_rate,
        "parseable_candidate_rows": len(parseable_rows),
        "missing_candidate_rows": missing_candidate_rows,
        "exact_validator_handoff_ready": exact_handoff_ready,
        "structured_output_positive_control_ready": positive_ready,
        "candidate_rows": rows,
        "parse_failure_counts": _parse_failure_counts(rows),
    }


def probe_structured_runtime(
    *,
    module_available: ModuleAvailable | None = None,
    llama_cpp_cuda_probe: LlamaCudaProbe | None = None,
    llama_grammar_compiler: LlamaGrammarCompiler | None = None,
) -> JsonDict:
    """Report whether this host exposes a constrained structured-output runtime."""

    available = module_available or _module_available
    blockers: list[str] = []

    llguidance_available = available("llguidance")
    llguidance_schema_available = _llguidance_schema_available() if llguidance_available else False
    if not llguidance_available:
        blockers.append("llguidance_not_installed")
    elif not llguidance_schema_available:
        blockers.append("llguidance_schema_compiler_unavailable")

    xgrammar_available = available("xgrammar")
    if not xgrammar_available:
        blockers.append("xgrammar_not_installed")

    llama_cpp_available = available("llama_cpp")
    if not llama_cpp_available:
        blockers.append("llama_cpp_not_installed")
    llama_cpp_grammar_available = False
    if llama_cpp_available:
        try:
            compiler = llama_grammar_compiler or _compile_llama_cpp_grammar
            compiler(build_llama_cpp_json_grammar())
            llama_cpp_grammar_available = True
        except Exception as exc:  # noqa: BLE001
            blockers.append(f"llama_cpp_grammar_unavailable:{type(exc).__name__}")

    llama_cpp_cuda_available = False
    if llama_cpp_available:
        try:
            probe = llama_cpp_cuda_probe or _llama_cpp_cuda_available
            llama_cpp_cuda_available = bool(probe())
        except Exception as exc:  # noqa: BLE001
            blockers.append(f"llama_cpp_cuda_probe_failed:{type(exc).__name__}")
    if llama_cpp_available and not llama_cpp_cuda_available:
        blockers.append("llama_cpp_cuda_unavailable")

    grammar_runtime_available = (
        llguidance_schema_available or xgrammar_available or llama_cpp_grammar_available
    )
    return {
        "grammar_runtime_available": grammar_runtime_available,
        "parser_only_fallback_used": not grammar_runtime_available,
        "llama_cpp_cuda_available": llama_cpp_cuda_available,
        "llama_cpp_grammar_available": llama_cpp_grammar_available,
        "llguidance_available": llguidance_schema_available,
        "xgrammar_available": xgrammar_available,
        "runtime_blockers": sorted(set(blockers)),
    }


def build_llama_cpp_json_grammar() -> str:
    """Return a small GBNF grammar that constrains llama.cpp output to JSON."""

    return r'''
root ::= object
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (string ws ":" ws value ("," ws string ws ":" ws value)*)? "}" ws
array ::= "[" ws (value ("," ws value)*)? "]" ws
string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
ws ::= [ \t\n\r]*
'''.strip()


def resolve_model_specs(cache_resolver: CacheResolver = resolve_cached_gguf) -> list[JsonDict]:
    """Resolve mandated GGUF model IDs without touching HuggingFace tokenizers."""

    resolved: list[JsonDict] = []
    for base in MODEL_SPECS:
        model_path = cache_resolver(str(base["hf_id"]), str(base["preferred_quant"]))
        path = Path(model_path) if model_path else None
        local_model_present = bool(path and path.is_file())
        resolved.append(
            {
                **base,
                "model_path": str(path) if path else None,
                "local_model_present": local_model_present,
                "model_filename": path.name if path else None,
                "model_size_bytes": path.stat().st_size if path and path.exists() else None,
            }
        )
    return resolved


def build_artifact(
    *,
    runtime_status: Mapping[str, Any] | None = None,
    cache_resolver: CacheResolver = resolve_cached_gguf,
    pair_resolver: PairResolver = cached_sota_pair,
    smoke_runner: SmokeRunner | None = None,
    max_smoke_models: int = 1,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5512 terminal artifact, including an optional live smoke."""

    fixture = load_fixture_artifact()["fixture"]
    fixture_payloads = build_fixture_candidate_payloads(fixture)
    report = evaluate_candidate_payloads(fixture_payloads, fixture=fixture)
    runtime = dict(runtime_status or probe_structured_runtime())
    model_specs = resolve_model_specs(cache_resolver)
    live_smoke_rows = _run_live_smoke(
        model_specs=model_specs,
        runtime_status=runtime,
        pair_resolver=pair_resolver,
        smoke_runner=smoke_runner or default_smoke_runner,
        max_smoke_models=max_smoke_models,
    )
    smoke_models_used = [
        str(row["model_hf_id"])
        for row in live_smoke_rows
        if row.get("model_hf_id") and row.get("runtime_error") is None
    ]
    live_smoke_parseable = any(row.get("parseable") is True for row in live_smoke_rows)
    positive_ready = bool(report["structured_output_positive_control_ready"])
    sota_gate_open = (
        positive_ready
        and bool(runtime.get("grammar_runtime_available"))
        and live_smoke_parseable
        and bool(smoke_models_used)
        and not bool(runtime.get("parser_only_fallback_used"))
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "schema_path": SCHEMA_PATH,
        "parser_path": PARSER_PATH,
        "tests_added_or_reused": list(TESTS_ADDED_OR_REUSED),
        "model_specs": model_specs,
        "smoke_models_used": smoke_models_used,
        "grammar_runtime_available": bool(runtime.get("grammar_runtime_available")),
        "parser_only_fallback_used": bool(runtime.get("parser_only_fallback_used")),
        "schema_validity_rate": report["schema_validity_rate"],
        "parseable_candidate_rows": report["parseable_candidate_rows"],
        "missing_candidate_rows": report["missing_candidate_rows"],
        "exact_validator_handoff_ready": report["exact_validator_handoff_ready"],
        "structured_output_positive_control_ready": positive_ready,
        "llama_cpp_cuda_available": bool(runtime.get("llama_cpp_cuda_available")),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(
            positive_ready=positive_ready,
            parser_only_fallback_used=bool(runtime.get("parser_only_fallback_used")),
            smoke_models_used=smoke_models_used,
            live_smoke_parseable=live_smoke_parseable,
        ),
        "sota_panel_gate_open": sota_gate_open,
        "fixture_artifact": FIXTURE_ARTIFACT_RELATIVE_PATH.as_posix(),
        "candidate_schema": candidate_schema(),
        "schema_sha256": sha256_json(candidate_schema()),
        "fixture_sha256": fixture_mod.sha256_json(fixture),
        "fixture_candidate_payloads": fixture_payloads,
        "candidate_rows": report["candidate_rows"],
        "parse_failure_counts": report["parse_failure_counts"],
        "runtime_status": runtime,
        "runtime_blockers": list(runtime.get("runtime_blockers", [])),
        "live_smoke_rows": live_smoke_rows,
        "live_smoke_parseable_rows": sum(int(row.get("parseable") is True) for row in live_smoke_rows),
        "no_autotokenizer_on_gguf": True,
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
    runtime_status: Mapping[str, Any] | None = None,
    cache_resolver: CacheResolver = resolve_cached_gguf,
    pair_resolver: PairResolver = cached_sota_pair,
    smoke_runner: SmokeRunner | None = None,
    max_smoke_models: int = 1,
    tests_run: Sequence[Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the Exp5512 result JSON."""

    artifact = build_artifact(
        runtime_status=runtime_status,
        cache_resolver=cache_resolver,
        pair_resolver=pair_resolver,
        smoke_runner=smoke_runner,
        max_smoke_models=max_smoke_models,
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
    """Validate the Exp5512 artifact contract and fail closed on false gates."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(artifact.get("schema_path") == SCHEMA_PATH, "schema_path")
    _require(artifact.get("parser_path") == PARSER_PATH, "parser_path")
    _require(artifact.get("tests_added_or_reused") == list(TESTS_ADDED_OR_REUSED), "tests")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("no_autotokenizer_on_gguf") is True, "no_autotokenizer_on_gguf")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    _require(str(artifact.get("honest_verdict", "")).startswith(("complete:", "blocked:")), "verdict")
    _require(
        [row.get("hf_id") for row in artifact.get("model_specs", [])]
        == list(MANDATED_HEADLINE_MODEL_IDS),
        "model_specs",
    )
    _require(
        set(artifact.get("smoke_models_used", [])).issubset(MANDATED_HEADLINE_MODEL_IDS),
        "smoke_models_used",
    )
    _require(0.0 <= float(artifact.get("schema_validity_rate", -1.0)) <= 1.0, "schema_rate")
    _require(int(artifact.get("parseable_candidate_rows", -1)) >= 0, "parseable_candidate_rows")
    _require(int(artifact.get("missing_candidate_rows", -1)) >= 0, "missing_candidate_rows")
    _require(isinstance(artifact.get("exact_validator_handoff_ready"), bool), "handoff_ready")
    _require(isinstance(artifact.get("structured_output_positive_control_ready"), bool), "ready")
    _require(isinstance(artifact.get("llama_cpp_cuda_available"), bool), "llama_cpp_cuda_available")
    _require(isinstance(artifact.get("grammar_runtime_available"), bool), "grammar_runtime_available")
    _require(isinstance(artifact.get("parser_only_fallback_used"), bool), "fallback")
    if artifact.get("parser_only_fallback_used") is True:
        _require(artifact.get("sota_panel_gate_open") is False, "sota_panel_gate_open")
    if artifact.get("sota_panel_gate_open") is True:
        _require(int(artifact.get("live_smoke_parseable_rows", 0)) > 0, "sota_panel_gate_open")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def honest_verdict(
    *,
    positive_ready: bool,
    parser_only_fallback_used: bool,
    smoke_models_used: Sequence[str],
    live_smoke_parseable: bool = False,
) -> str:
    """Return a terminal verdict that separates parser proof from SOTA claims."""

    if not positive_ready:
        return "blocked: structured_output_positive_control_not_ready"
    if parser_only_fallback_used:
        return "complete: structured_output_positive_control_ready_parser_only_fallback_sota_gate_closed"
    if smoke_models_used and live_smoke_parseable:
        return "complete: structured_output_positive_control_ready_live_llm_smoke_sota_gate_open"
    if smoke_models_used:
        return "complete: structured_output_positive_control_ready_live_smoke_unparseable_sota_gate_closed"
    return "complete: structured_output_positive_control_ready_no_live_smoke_sota_gate_closed"


def default_smoke_runner(spec: Mapping[str, Any], prompt: str, grammar: str | None) -> str:  # pragma: no cover
    """Run one tiny local GGUF smoke sample through llama.cpp."""

    from llama_cpp import Llama, LlamaGrammar  # noqa: PLC0415

    compiled_grammar = LlamaGrammar.from_string(grammar) if grammar else None
    llm = Llama(
        model_path=str(spec["model_path"]),
        n_ctx=2048,
        n_batch=64,
        n_gpu_layers=N_GPU_LAYERS,
        seed=RANDOM_SEED,
        verbose=False,
    )
    try:
        result = llm.create_completion(
            prompt=prompt,
            max_tokens=768,
            temperature=0.0,
            top_p=1.0,
            seed=RANDOM_SEED,
            echo=False,
            grammar=compiled_grammar,
            stop=["</s>", "<end_of_turn>"],
        )
        choices = result.get("choices", []) if isinstance(result, Mapping) else []
        return str(choices[0].get("text", "")) if choices else ""
    finally:
        llm = None
        gc.collect()


def build_smoke_prompt(target_payload: Mapping[str, Any]) -> str:
    """Build a tiny transcribe-and-validate prompt for the local structured path."""

    return (
        "Return exactly one JSON object and no prose. "
        "Transcribe this structured hard/soft claim candidate exactly so the "
        "local parser can validate the schema and exact validator target.\n"
        f"JSON schema:\n{canonical_json(candidate_schema())}\n"
        f"Candidate:\n{canonical_json(target_payload)}\n"
    )


def _premises(instance: Mapping[str, Any]) -> list[JsonDict]:
    premises = []
    for claim in instance["typed_claims"]:
        name = str(claim["name"])
        domain = ", ".join(str(value) for value in claim["domain"])
        premises.append(
            {
                "premise_id": f"DOMAIN_{name.upper()}",
                "text": f"{name} is a typed claim with finite domain [{domain}]",
                "source": "exp5499_fixture_typed_claim",
            }
        )
    return premises


def _rules_or_constraints(instance: Mapping[str, Any], reference_status: str) -> list[JsonDict]:
    rows = []
    for constraint in instance["hard_constraints"]:
        rows.append(
            {
                "rule_id": str(constraint["id"]),
                "kind": "hard_constraint",
                "text": canonical_json({"literals": constraint["literals"]}),
                "validator_constraint_id": str(constraint["id"]),
            }
        )
    for preference in instance["soft_preferences"]:
        rows.append(
            {
                "rule_id": str(preference["id"]),
                "kind": "soft_preference",
                "text": canonical_json(
                    {
                        "variable": preference["variable"],
                        "value": preference["value"],
                        "weight": preference["weight"],
                    }
                ),
                "validator_constraint_id": str(preference["id"]),
            }
        )
    if reference_status == "infeasible":
        rows.append(
            {
                "rule_id": "ABSTAIN_WHEN_HARD_CONSTRAINTS_CONFLICT",
                "kind": "abstention_rule",
                "text": "Return abstain when the exact validator finds no hard-feasible state.",
                "validator_constraint_id": "exact_validator_infeasible",
            }
        )
    return rows


def _validator_target(instance: Mapping[str, Any]) -> JsonDict:
    return {
        "instance_id": str(instance["instance_id"]),
        "expected_status": str(instance["expected_status"]),
        "reference_solver_path": fixture_mod.REFERENCE_SOLVER_PATH,
        "hard_constraint_ids": [str(row["id"]) for row in instance["hard_constraints"]],
        "soft_preference_ids": [str(row["id"]) for row in instance["soft_preferences"]],
        "typed_claim_names": [str(row["name"]) for row in instance["typed_claims"]],
    }


def _validator_target_matches(payload: Mapping[str, Any], instance: Mapping[str, Any]) -> bool:
    return payload.get("validator_target") == _validator_target(instance)


def _score_assignment(
    payload: Mapping[str, Any],
    instance: Mapping[str, Any],
    assignment: Mapping[str, str],
) -> JsonDict:
    reference = fixture_mod.solve_reference(instance)
    hard_ok = fixture_mod.hard_constraints_pass(instance, assignment)
    soft_score = fixture_mod.soft_score(instance, assignment)
    soft_optimal = bool(
        reference["status"] == "optimal"
        and hard_ok
        and soft_score == reference["objective_score"]
    )
    reference_agreement = bool(
        soft_optimal
        and assignment == reference["assignment"]
        and fixture_mod.assignment_hash(assignment) == reference["assignment_hash"]
    )
    if not hard_ok:
        verdict = "hard_constraint_violation"
    elif reference_agreement:
        verdict = "exact_match"
    else:
        verdict = "soft_suboptimal"
    return {
        **_success_base(payload, "schema_valid_assignment"),
        "assignment": dict(assignment),
        "hard_constraints_pass": hard_ok,
        "soft_score": soft_score,
        "soft_optimal": soft_optimal,
        "reference_agreement": reference_agreement,
        "exact_validator_correct": reference_agreement,
        "exact_validator_verdict": verdict,
    }


def _score_abstention(payload: Mapping[str, Any], instance: Mapping[str, Any]) -> JsonDict:
    reference = fixture_mod.solve_reference(instance)
    correct = reference["status"] == "infeasible"
    return {
        **_success_base(payload, "schema_valid_abstention"),
        "assignment": None,
        "hard_constraints_pass": False,
        "soft_score": None,
        "soft_optimal": False,
        "reference_agreement": correct,
        "exact_validator_correct": correct,
        "exact_validator_verdict": "correct_abstention" if correct else "abstention_on_feasible",
    }


def _success_base(payload: Mapping[str, Any], parse_status: str) -> JsonDict:
    return {
        "instance_id": str(payload["instance_id"]),
        "candidate_id": str(payload["candidate_id"]),
        "schema_valid": True,
        "schema_errors": [],
        "parse_status": parse_status,
        "parseable": True,
        "validator_target_ready": True,
        "abstention_reason": str(payload.get("abstention_reason", "")),
    }


def _semantic_failure(payload: Mapping[str, Any], parse_status: str) -> JsonDict:
    return _failure_row(
        parse_status=parse_status,
        schema_errors=[],
        parsed_payload=dict(payload),
        schema_valid=True,
        instance_id=str(payload.get("instance_id", "")) or None,
        candidate_id=str(payload.get("candidate_id", "")) or None,
        abstention_reason=str(payload.get("abstention_reason", "")),
    )


def _failure_row(
    *,
    parse_status: str,
    schema_errors: Sequence[str],
    parsed_payload: Mapping[str, Any],
    schema_valid: bool = False,
    instance_id: str | None = None,
    candidate_id: str | None = None,
    abstention_reason: str = "",
) -> JsonDict:
    return {
        "instance_id": instance_id,
        "candidate_id": candidate_id,
        "schema_valid": schema_valid,
        "schema_errors": list(schema_errors),
        "parse_status": parse_status,
        "parseable": False,
        "validator_target_ready": False,
        "abstention_reason": abstention_reason,
        "assignment": None,
        "hard_constraints_pass": False,
        "soft_score": None,
        "soft_optimal": False,
        "reference_agreement": False,
        "exact_validator_correct": False,
        "exact_validator_verdict": "not_handed_off",
        "parsed_payload": dict(parsed_payload),
    }


def _parse_failure_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("parseable") is True:
            continue
        status = str(row.get("parse_status"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _run_live_smoke(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    runtime_status: Mapping[str, Any],
    pair_resolver: PairResolver,
    smoke_runner: SmokeRunner,
    max_smoke_models: int,
) -> list[JsonDict]:
    if (
        max_smoke_models <= 0
        or runtime_status.get("grammar_runtime_available") is not True
        or runtime_status.get("llama_cpp_cuda_available") is not True
    ):
        return []
    selected = _select_smoke_specs(
        model_specs,
        pair_resolver=pair_resolver,
        max_smoke_models=max_smoke_models,
    )
    if not selected:
        return []
    target_payload = build_fixture_candidate_payloads()[0]
    prompt = build_smoke_prompt(target_payload)
    grammar = build_llama_cpp_json_grammar() if runtime_status.get("llama_cpp_grammar_available") else None
    rows = []
    for spec in selected:
        try:
            output = smoke_runner(spec, prompt, grammar)
            row = classify_candidate_text(output)
            row["model_hf_id"] = spec["hf_id"]
            row["raw_output_preview"] = output[:500]
            row["runtime_error"] = None
        except Exception as exc:  # noqa: BLE001
            row = _failure_row(
                parse_status="runtime_error",
                schema_errors=[],
                parsed_payload={},
            )
            row["model_hf_id"] = spec["hf_id"]
            row["runtime_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def _select_smoke_specs(
    model_specs: Sequence[Mapping[str, Any]],
    *,
    pair_resolver: PairResolver,
    max_smoke_models: int,
) -> list[JsonDict]:
    selected: list[JsonDict] = []
    by_id = {str(row["hf_id"]): row for row in model_specs}
    pair = pair_resolver() or []
    for pair_row in pair:
        spec = by_id.get(str(pair_row.get("hf_id")))
        if spec and spec.get("local_model_present") is True:
            selected.append(dict(spec))
    for spec in model_specs:
        if spec.get("local_model_present") is True and str(spec["hf_id"]) not in {
            row["hf_id"] for row in selected
        }:
            selected.append(dict(spec))
    return selected[: max(0, max_smoke_models)]


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _llguidance_schema_available() -> bool:  # pragma: no cover
    try:
        module = importlib.import_module("llguidance")
    except ImportError:
        return False
    matcher = getattr(module, "LLMatcher", None)
    return callable(getattr(matcher, "grammar_from_json_schema", None))


def _compile_llama_cpp_grammar(grammar: str) -> object:  # pragma: no cover
    from llama_cpp import LlamaGrammar  # noqa: PLC0415

    return LlamaGrammar.from_string(grammar)


def _llama_cpp_cuda_available() -> bool:  # pragma: no cover
    from llama_cpp import llama_cpp  # noqa: PLC0415

    return bool(llama_cpp.llama_supports_gpu_offload())


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
                "honest_verdict": artifact["honest_verdict"],
                "smoke_models_used": artifact["smoke_models_used"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
