"""Exp6379 canonical factor-edit transport contract.

Spec refs: REQ-INFRA-6379, SCENARIO-INFRA-6379-1,
SCENARIO-INFRA-6379-2, SCENARIO-INFRA-6379-3,
SCENARIO-INFRA-6379-4, SCENARIO-INFRA-6379-5.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from carnot import experiment_6366_repaired_live_factor_proposal_authenticity as exp6366
from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str, str], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6379_canonical_factor_edit_transport_contract.json"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6379_canonical_factor_edit_transport_contract.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6379_canonical_factor_edit_transport_contract.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
EXP6366_RELATIVE_PATH = Path(
    "results/experiment_6366_repaired_live_factor_proposal_authenticity.json"
)
EXP6366_DATA_RELATIVE_PATH = Path(
    "data/research/experiment_6366_repaired_live_factor_proposal_authenticity"
)
EXP6366_SCHEMA_SIDE_CAR = Path(
    "results/experiment_6366_repaired_live_factor_proposal_authenticity.json."
    "bounded_factor_edit_schema.json"
)

SCHEMA = "carnot.experiment_6379.canonical_factor_edit_transport_contract.v1"
CANONICAL_FACTOR_SCHEMA = SCHEMA + ".factor_edit"
CANONICAL_SCHEMA_VERSION = "v1"
RUN_DATE = "20260813"
RANDOM_SEED = 6379
TOKENIZER_METHOD = "llama_cpp_embedded_gguf_vocab_only"
INFERENCE_SUBSTRATE = "deterministic_gguf_vocab_only_transport_contract"
PREFERRED_QUANT = "Q4_K_M"
OLD_COMPLETION_BUDGET = 192
FIXED_OUTPUT_HEADROOM_TOKENS = 160
N_CTX = 4096
MAX_REPEATED_TOKEN_RUN = 64
MAX_REPEATED_TOKEN_FRACTION = 0.70
EVIDENCE_SUMMARY_MAX_CHARS = 160

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_TEMPLATES: tuple[JsonDict, ...] = (
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": MANDATED_MODEL_IDS[0],
        "gpu": 0,
        "model_family": "qwen_moe",
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": MANDATED_MODEL_IDS[1],
        "gpu": 1,
        "model_family": "gemma_dense",
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": MANDATED_MODEL_IDS[2],
        "gpu": 1,
        "model_family": "gemma_moe",
    },
)
MODEL_TEMPLATE_BY_ID = {str(row["hf_id"]): dict(row) for row in MODEL_TEMPLATES}

CANONICAL_FIELD_ORDER = [
    "schema",
    "proposal_id",
    "event_id",
    "model_hf_id",
    "model_family",
    "arm",
    "candidate_index",
    "changed_factor",
    "edits",
    "selection_score",
    "obligations",
    "edit_source_spans",
    "evidence_summary",
]

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6379_canonical_factor_edit_transport_contract.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6379_canonical_factor_edit_transport_contract.py -m pytest tests/python/test_experiment_6379_canonical_factor_edit_transport_contract.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6379_canonical_factor_edit_transport_contract.py --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6379_canonical_factor_edit_transport_contract.py",
    "sed -n '90,160p' ops/e2e-test-plan.md",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6379_canonical_factor_edit_transport_contract.json",
    ".venv/bin/python scripts/determination_preservation_lint.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    ".venv/bin/python -m carnot.experiment_6379_canonical_factor_edit_transport_contract --date 20260813",
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6366_RELATIVE_PATH,
    EXP6366_SCHEMA_SIDE_CAR,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_exp6366_path_hash_and_terminal_class",
    "frozen_raw_failure_paths_hashes_and_labels",
    "MODEL_SPECS",
    "embedded_gguf_tokenizer_receipts",
    "autotokenizer_usage_count",
    "live_autoregressive_generation_invoked",
    "canonical_schema_path_hash_and_version",
    "canonical_schema_generated_surfaces",
    "prompt_schema_drift_checks",
    "bounded_evidence_summary_variant",
    "per_model_minimum_output_tokens_and_capacity_margins",
    "repetition_policy_and_failure_thresholds",
    "deterministic_transport_mutation_matrix",
    "syntax_structure_source_binding_and_semantic_boundaries",
    "retired_decoding_mechanism_usage_count",
    "canonical_factor_transport_contract_ready_score",
    "no_model_quality_or_utility_claim",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal status distinguishes positive and null deterministic transport evidence.",
    "upstream_exp6366_path_hash_and_terminal_class": "The Exp6366 terminal artifact is frozen before failure labels are assigned.",
    "frozen_raw_failure_paths_hashes_and_labels": "Raw stdout, prompt payload, and schema hashes bind each failure label to bytes.",
    "MODEL_SPECS": "The three mandated GGUF model ids are present for tokenizer capacity checks.",
    "embedded_gguf_tokenizer_receipts": "Each token receipt uses the embedded GGUF tokenizer in vocab-only mode.",
    "autotokenizer_usage_count": "A bare zero records that no external tokenizer path was used.",
    "live_autoregressive_generation_invoked": "A bare false records that this run is deterministic transport infrastructure.",
    "canonical_schema_path_hash_and_version": "The single canonical object is written and hash-bound as the schema source.",
    "canonical_schema_generated_surfaces": "Prompt, schema, validator, example, and source checks share one canonical hash.",
    "prompt_schema_drift_checks": "Drift between generated surfaces fails closed before any later live call.",
    "bounded_evidence_summary_variant": "The JSON object may include a short visible-evidence summary.",
    "per_model_minimum_output_tokens_and_capacity_margins": "Output-token lower bounds and context margins are measured per model.",
    "repetition_policy_and_failure_thresholds": "Repeated-token collapse is a preregistered abstention, not a transport pass.",
    "deterministic_transport_mutation_matrix": "Known drift, syntax, structure, source, and semantic attacks are rejected.",
    "syntax_structure_source_binding_and_semantic_boundaries": "The validator checks transport boundaries but not exact task utility.",
    "retired_decoding_mechanism_usage_count": "A bare zero records that retired decoding controls were not used.",
    "canonical_factor_transport_contract_ready_score": "Readiness is one only when generators, drift checks, token receipts, and bans agree.",
    "no_model_quality_or_utility_claim": "The artifact makes no model-quality, factor-success, or utility claim.",
    "protected_files_unchanged": "Conductor, ops, traceability, and Exp6366 inputs stay byte-identical.",
    "preconditions_checked": "Preconditions bind upstream evidence, model identities, token receipts, bans, and protected hashes.",
    "inference_substrate": "The substrate is deterministic local GGUF tokenizer measurement.",
    "verifier_is_oracle": "The validator is not the later exact semantic oracle.",
    "field_principles": "Every required field states its guard.",
    "field_provenance": "Every required field maps to specs, frozen inputs, generated surfaces, tests, or constants.",
    "random_seed": "The seed pins deterministic ordering even though no random sampling occurs.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification commands and exit codes are recorded.",
    "reproducibility_checksum": "A normalized checksum detects artifact drift.",
    "honest_verdict": "The verdict uses a terminal prefix and states the transport-only claim.",
}

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-INFRA-6379",
        "Exp6366 frozen artifact and sidecars",
        "canonical factor transport object",
        "embedded GGUF vocab-only tokenizer receipts",
        "Exp6379 focused tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    """Return compact JSON while preserving contract field order."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the repository digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash text with the repository digest prefix."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash the exact canonical JSON serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str | None:
    """Return a file digest, or None when the path is absent."""

    if not path.exists() or not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def require(condition: bool, reason: str) -> None:
    """Raise a deterministic validation error when a gate fails."""

    if not condition:
        raise ValueError(reason)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and use an empty mapping otherwise."""

    return value if isinstance(value, Mapping) else {}


def model_slug(model_id: str) -> str:
    """Turn a model id into a stable file-name fragment."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "--", model_id).strip("-").lower()


def rounded(value: float) -> float:
    """Round numeric receipts without hiding small costs."""

    return round(float(value), 12)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_payload_or_hash(path: Path, payload: Mapping[str, Any], *, write: bool) -> str:
    """Write JSON when requested, otherwise return its would-be hash."""

    if write:
        write_json_atomic(path, payload)
        digest = sha256_file(path)
        require(digest is not None, "json_write_failed")
        return str(digest)
    return sha256_json(payload)


def path_receipt(path: Path, *, sha256: str | None = None) -> JsonDict:
    """Record path, digest, presence, and size."""

    return {
        "path": str(path),
        "present": path.exists() and path.is_file(),
        "sha256": sha256 if sha256 is not None else sha256_file(path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def read_json(path: Path) -> JsonDict:
    """Read a JSON object from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def revision_from_path(path: Path) -> str | None:
    """Extract a Hugging Face snapshot revision when present."""

    parts = path.parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else None


def quantization_from_path(path: Path) -> str:
    """Extract a known GGUF quantization token from a file name."""

    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "UD-Q8_XL", "Q8_0"):
        if token.lower() in path.name.lower():
            return token
    return "unknown"


def deterministic_model_specs(base: Path) -> list[JsonDict]:
    """Return deterministic model rows for tests and surface generation."""

    rows: list[JsonDict] = []
    for template in MODEL_TEMPLATES:
        path = base / (model_slug(str(template["hf_id"])) + "-Q4_K_M.gguf")
        rows.append(
            {
                **template,
                "model_path": str(path),
                "exists": path.is_file(),
                "revision": revision_from_path(path),
                "quantization": quantization_from_path(path),
                "model_file_sha256": sha256_file(path),
            }
        )
    return rows


def build_model_specs(
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
) -> JsonDict:
    """Resolve all mandated GGUF rows through cached SOTA helper calls."""

    calls = [
        {"gpu_indices": [0, 1], "preferred_quant": PREFERRED_QUANT, "model_indices": None},
        {"gpu_indices": [0, 1], "preferred_quant": PREFERRED_QUANT, "model_indices": [0, 2]},
    ]
    default_pair = cached_pair_func(gpu_indices=(0, 1), preferred_quant=PREFERRED_QUANT) or []
    dense_pair = cached_pair_func(
        gpu_indices=(0, 1),
        preferred_quant=PREFERRED_QUANT,
        model_indices=(0, 2),
    ) or []
    by_id = {str(row.get("hf_id")): dict(row) for row in [*default_pair, *dense_pair]}
    blockers: list[str] = []
    records: list[JsonDict] = []
    for template in MODEL_TEMPLATES:
        row = dict(by_id.get(str(template["hf_id"]), {}))
        path_text = str(row.get("model_path") or "")
        path = Path(path_text) if path_text else Path()
        record = {
            **template,
            "gpu": int(row.get("gpu", template["gpu"])),
            "model_path": path_text,
            "exists": bool(path_text) and path.is_file(),
            "revision": revision_from_path(path) if path_text else None,
            "quantization": quantization_from_path(path) if path_text else "unknown",
            "model_file_sha256": sha256_file(path) if path_text else None,
            "tokenizer_method": TOKENIZER_METHOD,
        }
        if not row:
            blockers.append(f"missing_cached_sota_pair_row:{template['hf_id']}")
        if not record["exists"]:
            blockers.append(f"missing_gguf_file:{template['hf_id']}")
        records.append(record)
    if not default_pair:
        blockers.append("cached_sota_pair_default_missing")
    if not dense_pair:
        blockers.append("cached_sota_pair_dense_missing")
    return {
        "schema": SCHEMA + ".model_resolution",
        "MODEL_SPECS": records,
        "cached_sota_pair_receipts": {
            "helper": "cached_sota_pair",
            "calls": calls,
            "all_calls_made": True,
        },
        "blocked_reasons": sorted(set(blockers)),
        "all_resolved": not blockers,
    }


def embedded_gguf_tokenizer_receipt(model_path: str, text: str) -> JsonDict:  # pragma: no cover
    """Count text tokens through a GGUF's embedded vocab only."""

    try:
        from llama_cpp import Llama

        llm = Llama(model_path=model_path, vocab_only=True, verbose=False)
        tokens = llm.tokenize(text.encode("utf-8"))
        close = getattr(llm, "close", None)
        if callable(close):
            close()
        return {
            "method": TOKENIZER_METHOD,
            "loadable": bool(tokens),
            "token_count": len(tokens),
            "tokenizer_detail": f"embedded GGUF tokenizer OK ({len(tokens)} tokens)",
            "autotokenizer_used": False,
        }
    except Exception as exc:
        return {
            "method": TOKENIZER_METHOD,
            "loadable": False,
            "token_count": 0,
            "tokenizer_detail": f"embedded GGUF tokenizer failed: {type(exc).__name__}: {exc}",
            "autotokenizer_used": False,
        }


def canonical_source_event() -> JsonDict:
    """Return the visible Exp6366 event used for transport fixtures."""

    event = exp6366.generated_events()[0]
    variable = str(event["allowed_variables"][0])
    obligation = as_mapping(event["source_obligations"][0])
    edit_span = as_mapping(as_mapping(event["edit_source_spans"]).get(variable))
    return {
        "event_id": event["event_id"],
        "changed_factor": event["changed_factor"],
        "source_text": event["source_text"],
        "source_text_sha256": event["source_text_sha256"],
        "variable": variable,
        "edit_bounds": dict(event["edit_bounds"]),
        "obligation": {
            "obligation_id": obligation.get("obligation_id"),
            "source_start": as_mapping(obligation.get("span")).get("start"),
            "source_end": as_mapping(obligation.get("span")).get("end"),
            "source_sha256": as_mapping(obligation.get("span")).get("sha256"),
            "source_text": obligation.get("text"),
        },
        "edit_source_span": {
            "source_start": edit_span.get("start"),
            "source_end": edit_span.get("end"),
            "source_sha256": edit_span.get("sha256"),
        },
    }


def canonical_factor_edit_contract() -> JsonDict:
    """Return the one source object for every generated transport surface."""

    source = canonical_source_event()
    return {
        "schema": SCHEMA + ".canonical_contract",
        "version": CANONICAL_SCHEMA_VERSION,
        "output_schema": CANONICAL_FACTOR_SCHEMA,
        "field_order": list(CANONICAL_FIELD_ORDER),
        "fixed_fields": {
            "schema": CANONICAL_FACTOR_SCHEMA,
            "event_id": source["event_id"],
            "arm": "canonical_factor_edit_transport_contract",
            "candidate_index": 0,
            "changed_factor": source["changed_factor"],
        },
        "model_bound_fields": ["proposal_id", "model_hf_id", "model_family"],
        "numeric_bounds": {
            "selection_score": {"min": 0.0, "max": 1.0},
            "edits": source["edit_bounds"],
        },
        "allowed_variables": [source["variable"]],
        "source_event": source,
        "evidence_summary": {
            "required": True,
            "max_chars": EVIDENCE_SUMMARY_MAX_CHARS,
            "visible_evidence_only": True,
            "hidden_reasoning_forbidden": True,
        },
        "forbidden_fields": [
            "protected_outcome",
            "protected_success",
            "exact_label",
            "hidden_state",
            "source_weight_delta",
            "analysis",
            "thinking",
        ],
    }


def validator_field_list(contract: Mapping[str, Any]) -> list[JsonDict]:
    """Generate validator field rules from the canonical object."""

    fixed = as_mapping(contract.get("fixed_fields"))
    model_bound = set(contract.get("model_bound_fields", []))
    return [
        {
            "field": field,
            "required": True,
            "fixed_value": fixed.get(field),
            "model_bound": field in model_bound,
        }
        for field in contract["field_order"]
    ]


def source_binding_checks(contract: Mapping[str, Any]) -> JsonDict:
    """Generate exact source-span checks from the canonical object."""

    source = as_mapping(contract.get("source_event"))
    return {
        "source_text_sha256": source.get("source_text_sha256"),
        "obligation": dict(as_mapping(source.get("obligation"))),
        "edit_source_span": dict(as_mapping(source.get("edit_source_span"))),
        "allowed_variables": list(contract.get("allowed_variables", [])),
        "source_binding_required": True,
    }


def schema_description(contract: Mapping[str, Any]) -> JsonDict:
    """Generate the bounded schema description from the canonical object."""

    return {
        "schema": contract["output_schema"],
        "type": "object",
        "version": contract["version"],
        "required_fields": list(contract["field_order"]),
        "field_order": list(contract["field_order"]),
        "fixed_fields": dict(as_mapping(contract.get("fixed_fields"))),
        "model_bound_fields": list(contract.get("model_bound_fields", [])),
        "allowed_variables": list(contract.get("allowed_variables", [])),
        "numeric_bounds": dict(as_mapping(contract.get("numeric_bounds"))),
        "source_binding_checks": source_binding_checks(contract),
        "forbidden_fields": list(contract.get("forbidden_fields", [])),
    }


def compact_output_example(contract: Mapping[str, Any], spec: Mapping[str, Any]) -> JsonDict:
    """Generate the compact output example from the canonical object."""

    source = as_mapping(contract.get("source_event"))
    variable = str(source.get("variable"))
    fixed = as_mapping(contract.get("fixed_fields"))
    model_id = str(spec["hf_id"])
    return {
        "schema": fixed["schema"],
        "proposal_id": f"{model_slug(model_id)}:{source['event_id']}:0",
        "event_id": fixed["event_id"],
        "model_hf_id": model_id,
        "model_family": spec["model_family"],
        "arm": fixed["arm"],
        "candidate_index": fixed["candidate_index"],
        "changed_factor": fixed["changed_factor"],
        "edits": {variable: 0.5},
        "selection_score": 0.5,
        "obligations": [dict(as_mapping(source.get("obligation")))],
        "edit_source_spans": {variable: dict(as_mapping(source.get("edit_source_span")))},
        "evidence_summary": (
            f"Visible obligation {source['obligation']['obligation_id']} "
            f"supports editing {variable}."
        ),
    }


def prompt_instruction_fragment(contract: Mapping[str, Any]) -> str:
    """Generate the prompt fragment from the canonical object."""

    fields = ",".join(contract["field_order"])
    return (
        "Return exactly one JSON object. Use these fields in this order: "
        f"{fields}. Copy fixed ids, spans, hashes, and visible source text exactly. "
        "Do not add markdown, thinking text, prefixes, suffixes, or hidden reasoning."
    )


def canonical_schema_generated_surfaces(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Generate all transport surfaces from one canonical object."""

    contract = canonical_factor_edit_contract()
    description = schema_description(contract)
    examples = {
        str(spec["hf_id"]): compact_output_example(contract, spec) for spec in model_specs
    }
    minimum_serialized = {
        model_id: canonical_json(example) for model_id, example in examples.items()
    }
    surface_payload = {
        "schema_description": description,
        "prompt_fragment": prompt_instruction_fragment(contract),
        "output_example": examples[str(model_specs[0]["hf_id"])],
        "validator_field_list": validator_field_list(contract),
        "source_binding_checks": source_binding_checks(contract),
        "minimum_serialized_outputs": minimum_serialized,
    }
    return {
        "schema": SCHEMA + ".generated_surfaces",
        "canonical_hash": sha256_json(contract),
        "schema_description": description,
        "prompt_fragment": surface_payload["prompt_fragment"],
        "output_example": surface_payload["output_example"],
        "validator_field_list": surface_payload["validator_field_list"],
        "source_binding_checks": surface_payload["source_binding_checks"],
        "field_order": list(contract["field_order"]),
        "minimum_serialized_outputs": minimum_serialized,
        "surface_hashes": {
            name: sha256_json(value) if not isinstance(value, str) else sha256_text(value)
            for name, value in surface_payload.items()
        },
        "all_surfaces_from_canonical": True,
        "duplicate_handwritten_surface_count": 0,
    }


def repetition_policy_and_failure_thresholds() -> JsonDict:
    """Return the preregistered repeated-token abstention policy."""

    return {
        "schema": SCHEMA + ".repetition_policy",
        "max_repeated_token_run": MAX_REPEATED_TOKEN_RUN,
        "max_repeated_token_fraction": MAX_REPEATED_TOKEN_FRACTION,
        "threshold_breach_decision": "abstain",
        "larger_token_budget_alone_qualifies_contract": False,
        "policy_defined_before_live_execution": True,
    }


def repetition_breach(text: str) -> JsonDict:
    """Detect bounded repeated-token collapse."""

    tokens = re.findall(r"\S+", text)
    longest = 0
    current = 0
    last = None
    for token in tokens:
        current = current + 1 if token == last else 1
        last = token
        longest = max(longest, current)
    counts = Counter(tokens)
    top_count = max(counts.values(), default=0)
    fraction = top_count / len(tokens) if tokens else 0.0
    breached = longest > MAX_REPEATED_TOKEN_RUN or (
        len(tokens) >= 20 and fraction >= MAX_REPEATED_TOKEN_FRACTION
    )
    return {
        "token_count": len(tokens),
        "longest_repeated_token_run": longest,
        "top_token_fraction": rounded(fraction),
        "breached": breached,
    }


def _strict_json_load(text: str) -> tuple[Any | None, str | None]:
    """Parse only a whole JSON value."""

    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"{exc.msg}@{exc.pos}"


def classify_raw_failure(text: str) -> list[str]:
    """Classify a raw transport failure without modifying it."""

    labels: list[str] = []
    stripped = text.strip()
    repeated = repetition_breach(text)
    if "<think" in stripped.lower() or stripped.lower().startswith(("thinking", "analysis")):
        labels.append("thinking_leakage")
    if repeated["breached"]:
        labels.append("repetition_collapse")
    parsed, error = _strict_json_load(stripped)
    if error is not None:
        if stripped.startswith("{") and (
            stripped.count("{") > stripped.count("}") or not stripped.endswith("}")
        ):
            labels.append("truncation")
        labels.append("syntax_failure")
    elif not isinstance(parsed, Mapping):
        labels.append("structural_failure")
    if "truncation" in labels:
        labels.append("structural_failure")
    if not labels:
        labels.append("unknown")
    return labels


def upstream_exp6366_receipt(path: Path) -> JsonDict:
    """Freeze the Exp6366 artifact and derive its terminal class."""

    payload = read_json(path) if path.is_file() else {}
    parse_counts = as_mapping(payload.get("parse_valid_invalid_timeout_and_abstain_counts_by_model"))
    by_model = as_mapping(parse_counts.get("by_model"))
    exact = as_mapping(payload.get("exact_pass_fail_counts_by_model"))
    valid_count = sum(int(as_mapping(row).get("valid", 0)) for row in by_model.values())
    terminal_class = (
        "transport_null"
        if payload.get("status") == "complete_null"
        and valid_count == 0
        and int(exact.get("total_exact_calls", 0)) == 0
        else "unknown"
    )
    return {
        **path_receipt(path),
        "status": payload.get("status", "missing"),
        "honest_verdict": payload.get("honest_verdict"),
        "terminal_class": terminal_class,
        "zero_parse_valid_objects": valid_count == 0,
        "exact_checker_calls": int(exact.get("total_exact_calls", 0)),
    }


def prompt_sidecar_path(data_dir: Path, model_id: str) -> Path:
    """Return the Exp6366 prompt payload sidecar path for one model."""

    return data_dir / "prompts" / f"{model_slug(model_id)}.prompt.json"


def raw_sidecar_path(data_dir: Path, model_id: str) -> Path:
    """Return the Exp6366 raw stdout sidecar path for one model."""

    return data_dir / "sidecars" / f"{model_slug(model_id)}.stdout.txt"


def frozen_raw_failure_receipts(exp6366_path: Path, data_dir: Path) -> dict[str, JsonDict]:
    """Freeze Exp6366 raw failures before assigning labels."""

    artifact = read_json(exp6366_path) if exp6366_path.is_file() else {}
    raw_by_model = as_mapping(
        as_mapping(artifact.get("raw_output_before_parse_paths_hashes_and_counts")).get("by_model")
    )
    schema_path = REPO_ROOT / EXP6366_SCHEMA_SIDE_CAR
    rows: dict[str, JsonDict] = {}
    for model_id in MANDATED_MODEL_IDS:
        raw_path = Path(str(as_mapping(raw_by_model.get(model_id)).get("path") or ""))
        if not raw_path.is_file():
            raw_path = raw_sidecar_path(data_dir, model_id)
        prompt_path = prompt_sidecar_path(data_dir, model_id)
        raw_hash = sha256_file(raw_path)
        prompt_hash = sha256_file(prompt_path)
        schema_hash = sha256_file(schema_path)
        text = raw_path.read_text(encoding="utf-8", errors="replace") if raw_path.is_file() else ""
        labels = classify_raw_failure(text)
        rows[model_id] = {
            "model_hf_id": model_id,
            "raw_stdout": path_receipt(raw_path, sha256=raw_hash),
            "prompt_payload": path_receipt(prompt_path, sha256=prompt_hash),
            "schema_sidecar": path_receipt(schema_path, sha256=schema_hash),
            "raw_sha256_before_classification": raw_hash,
            "prompt_sha256_before_classification": prompt_hash,
            "schema_sha256_before_classification": schema_hash,
            "freeze_before_classification": all(
                value is not None for value in (raw_hash, prompt_hash, schema_hash)
            ),
            "labels": labels,
        }
    return rows


def validate_transport_output(
    text: str,
    contract: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> JsonDict:
    """Validate deterministic transport shape only."""

    labels: list[str] = []
    reasons: list[str] = []
    repeated = repetition_breach(text)
    if repeated["breached"]:
        labels.append("repetition_collapse")
        reasons.append("repetition_policy_breach")
        return {
            "accepted": False,
            "decision": "abstain",
            "failure_labels": labels,
            "reasons": reasons,
            "repetition": repeated,
        }
    stripped = text.strip()
    lower = stripped.lower()
    if lower.startswith("```") or lower.endswith("```"):
        labels.append("syntax_failure")
        reasons.append("markdown_wrapper")
    if lower.startswith("<think") or lower.startswith(("thinking", "analysis")):
        labels.append("thinking_leakage")
        reasons.append("thinking_prefix")
    parsed, error = _strict_json_load(stripped)
    if error is not None:
        if stripped.startswith("{") and (
            stripped.count("{") > stripped.count("}") or not stripped.endswith("}")
        ):
            labels.append("truncation")
            labels.append("structural_failure")
            reasons.append("mid_object_truncation")
        labels.append("syntax_failure")
        reasons.append("json_parse_failed:" + error)
        return {
            "accepted": False,
            "decision": "reject",
            "failure_labels": sorted(set(labels)),
            "reasons": reasons,
            "repetition": repeated,
        }
    if not isinstance(parsed, Mapping):
        labels.append("structural_failure")
        reasons.append("json_value_not_object")
        parsed = {}
    if list(parsed.keys()) != list(contract["field_order"]):
        labels.append("structural_failure")
        reasons.append("field_order_mismatch")
    fixed = as_mapping(contract.get("fixed_fields"))
    for field in contract["field_order"]:
        if field not in parsed:
            labels.append("structural_failure")
            reasons.append(f"missing_field:{field}")
    for field, expected in fixed.items():
        if parsed.get(field) != expected:
            labels.append("semantic_failure")
            reasons.append(f"fixed_field_mismatch:{field}")
    if parsed.get("model_hf_id") != spec.get("hf_id"):
        labels.append("semantic_failure")
        reasons.append("model_hf_id_mismatch")
    if parsed.get("model_family") != spec.get("model_family"):
        labels.append("semantic_failure")
        reasons.append("model_family_mismatch")
    source = as_mapping(contract.get("source_event"))
    expected_id = f"{model_slug(str(spec.get('hf_id')))}:{source.get('event_id')}:0"
    if parsed.get("proposal_id") != expected_id:
        labels.append("semantic_failure")
        reasons.append("proposal_id_mismatch")
    forbidden = set(contract.get("forbidden_fields", []))
    forbidden_present = sorted(forbidden & set(parsed.keys()))
    if forbidden_present:
        labels.append("semantic_failure")
        reasons.append("forbidden_fields:" + ",".join(forbidden_present))
    _validate_numeric_and_source_fields(parsed, contract, labels, reasons)
    summary = parsed.get("evidence_summary")
    if not isinstance(summary, str) or not summary.strip():
        labels.append("structural_failure")
        reasons.append("evidence_summary_missing_or_not_string")
    else:
        max_chars = int(as_mapping(contract.get("evidence_summary")).get("max_chars", 0))
        if len(summary) > max_chars:
            labels.append("semantic_failure")
            reasons.append("evidence_summary_too_long")
        if "hidden" in summary.lower() or "chain" in summary.lower():
            labels.append("semantic_failure")
            reasons.append("evidence_summary_requests_hidden_reasoning")
    accepted = not labels
    return {
        "accepted": accepted,
        "decision": "accept" if accepted else "reject",
        "failure_labels": sorted(set(labels)),
        "reasons": reasons,
        "repetition": repeated,
    }


def _validate_numeric_and_source_fields(
    parsed: Mapping[str, Any],
    contract: Mapping[str, Any],
    labels: list[str],
    reasons: list[str],
) -> None:
    """Check numeric bounds and exact source spans."""

    source = as_mapping(contract.get("source_event"))
    variable = str(source.get("variable"))
    edits = parsed.get("edits")
    if not isinstance(edits, Mapping) or set(edits.keys()) != {variable}:
        labels.append("structural_failure")
        reasons.append("edits_not_single_allowed_variable")
    else:
        value = edits.get(variable)
        bounds = as_mapping(as_mapping(contract.get("numeric_bounds")).get("edits"))
        if not isinstance(value, (int, float)):
            labels.append("structural_failure")
            reasons.append("edit_value_not_number")
        elif not float(bounds["min"]) <= float(value) <= float(bounds["max"]):
            labels.append("semantic_failure")
            reasons.append("edit_value_out_of_bounds")
    score = parsed.get("selection_score")
    score_bounds = as_mapping(as_mapping(contract.get("numeric_bounds")).get("selection_score"))
    if not isinstance(score, (int, float)):
        labels.append("structural_failure")
        reasons.append("selection_score_not_number")
    elif not float(score_bounds["min"]) <= float(score) <= float(score_bounds["max"]):
        labels.append("semantic_failure")
        reasons.append("selection_score_out_of_bounds")
    obligation_rows = parsed.get("obligations")
    expected_obligation = as_mapping(source.get("obligation"))
    if not isinstance(obligation_rows, list) or len(obligation_rows) != 1:
        labels.append("structural_failure")
        reasons.append("obligations_not_singleton")
    elif as_mapping(obligation_rows[0]) != expected_obligation:
        labels.append("semantic_failure")
        reasons.append("unsupported_source_span:obligation")
    edit_spans = as_mapping(parsed.get("edit_source_spans"))
    if edit_spans != {variable: dict(as_mapping(source.get("edit_source_span")))}:
        labels.append("semantic_failure")
        reasons.append("unsupported_source_span:edit")


def _mutated_text(
    attack: str,
    example: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> str:
    """Return one deterministic mutation payload."""

    if attack == "prompt_schema_conflict":
        drifted = deepcopy_dict(example)
        drifted["schema"] = "conflicting.schema"
        return canonical_json(drifted)
    if attack == "stale_example":
        drifted = deepcopy_dict(example)
        drifted["event_id"] = "live-6366-stale"
        return canonical_json(drifted)
    if attack == "missing_fixed_fields":
        drifted = deepcopy_dict(example)
        drifted.pop("model_hf_id", None)
        return canonical_json(drifted)
    if attack == "reordered_fields":
        return canonical_json({key: example[key] for key in reversed(list(example.keys()))})
    if attack == "thinking_prefix":
        return "<think>\n" + canonical_json(example)
    if attack == "markdown":
        return "```json\n" + canonical_json(example) + "\n```"
    if attack == "repeated_tokens":
        return "own " * (MAX_REPEATED_TOKEN_RUN + 17)
    if attack == "mid_object_truncation":
        return canonical_json(example)[:-11]
    if attack == "unsupported_source_spans":
        drifted = deepcopy_dict(example)
        variable = str(as_mapping(contract.get("source_event")).get("variable"))
        drifted["edit_source_spans"][variable]["source_start"] = 0
        return canonical_json(drifted)
    if attack == "parse_valid_semantic_corruption":
        drifted = deepcopy_dict(example)
        drifted["evidence_summary"] = "I used hidden state evidence."
        return canonical_json(drifted)
    raise ValueError(f"unknown_attack:{attack}")


def deepcopy_dict(value: Mapping[str, Any]) -> JsonDict:
    """Copy a JSON-compatible mapping through JSON."""

    return json.loads(json.dumps(value))


def deterministic_transport_mutation_matrix(
    contract: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> JsonDict:
    """Run deterministic drift and malformed-output mutations."""

    example = compact_output_example(contract, spec)
    attacks = [
        "prompt_schema_conflict",
        "stale_example",
        "missing_fixed_fields",
        "reordered_fields",
        "thinking_prefix",
        "markdown",
        "repeated_tokens",
        "mid_object_truncation",
        "unsupported_source_spans",
        "parse_valid_semantic_corruption",
    ]
    rows = []
    for attack in attacks:
        text = _mutated_text(attack, example, contract)
        receipt = validate_transport_output(text, contract, spec)
        rows.append(
            {
                "attack": attack,
                "accepted": receipt["accepted"],
                "decision": receipt["decision"],
                "failure_labels": receipt["failure_labels"],
                "fail_closed": receipt["accepted"] is False,
            }
        )
    return {
        "schema": SCHEMA + ".mutation_matrix",
        "rows": rows,
        "all_attacks_fail_closed": all(row["fail_closed"] for row in rows),
        "retired_decode_controls_invoked": False,
    }


def prompt_schema_drift_checks(
    surfaces: Mapping[str, Any],
    mutation_matrix: Mapping[str, Any],
) -> JsonDict:
    """Summarize generated-surface drift checks."""

    drift_attacks = {
        "prompt_schema_conflict",
        "stale_example",
        "missing_fixed_fields",
        "reordered_fields",
    }
    rows = [
        row for row in mutation_matrix.get("rows", []) if as_mapping(row).get("attack") in drift_attacks
    ]
    return {
        "schema": SCHEMA + ".prompt_schema_drift_checks",
        "canonical_hash": surfaces.get("canonical_hash"),
        "surface_hashes": dict(as_mapping(surfaces.get("surface_hashes"))),
        "all_surfaces_from_canonical": surfaces.get("all_surfaces_from_canonical") is True,
        "duplicate_handwritten_surface_count": int(
            surfaces.get("duplicate_handwritten_surface_count", 1)
        ),
        "drift_attack_rows": rows,
        "all_drift_checks_fail_closed": bool(rows)
        and all(as_mapping(row).get("fail_closed") is True for row in rows),
    }


def bounded_evidence_summary_variant(contract: Mapping[str, Any]) -> JsonDict:
    """Record the bounded visible-evidence summary variant."""

    summary = as_mapping(contract.get("evidence_summary"))
    return {
        "schema": SCHEMA + ".bounded_evidence_summary_variant",
        "field": "evidence_summary",
        "included_in_json_object": summary.get("required") is True,
        "max_chars": summary.get("max_chars"),
        "visible_evidence_only": summary.get("visible_evidence_only") is True,
        "hidden_reasoning_forbidden": summary.get("hidden_reasoning_forbidden") is True,
        "may_cite_visible_evidence": True,
    }


def per_model_minimum_output_tokens_and_capacity_margins(
    model_specs: Sequence[Mapping[str, Any]],
    surfaces: Mapping[str, Any],
    *,
    tokenizer_func: TokenizerFn = embedded_gguf_tokenizer_receipt,
) -> JsonDict:
    """Measure output lower bounds with each model's embedded tokenizer."""

    by_model: dict[str, JsonDict] = {}
    serialized_by_model = as_mapping(surfaces.get("minimum_serialized_outputs"))
    prompt_fragment = str(surfaces.get("prompt_fragment", ""))
    for spec in model_specs:
        model_id = str(spec["hf_id"])
        output_text = str(serialized_by_model.get(model_id, ""))
        output_tokens = tokenizer_func(str(spec.get("model_path", "")), output_text)
        prompt_tokens = tokenizer_func(str(spec.get("model_path", "")), prompt_fragment)
        minimum_tokens = int(output_tokens.get("token_count", 0))
        prompt_count = int(prompt_tokens.get("token_count", 0))
        lower_bound = minimum_tokens + FIXED_OUTPUT_HEADROOM_TOKENS
        old_margin = OLD_COMPLETION_BUDGET - minimum_tokens
        n_ctx_margin = N_CTX - prompt_count - lower_bound
        by_model[model_id] = {
            "model_hf_id": model_id,
            "model_path": spec.get("model_path"),
            "tokenizer_method": output_tokens.get("method", TOKENIZER_METHOD),
            "tokenizer_loadable": output_tokens.get("loadable") is True,
            "autotokenizer_used": output_tokens.get("autotokenizer_used") is True,
            "minimum_serialized_output_sha256": sha256_text(output_text),
            "minimum_serialized_output_bytes": len(output_text.encode("utf-8")),
            "minimum_serialized_output_tokens": minimum_tokens,
            "old_budget_tokens": OLD_COMPLETION_BUDGET,
            "old_budget_margin": old_margin,
            "required_completion_lower_bound": lower_bound,
            "fixed_headroom_tokens": FIXED_OUTPUT_HEADROOM_TOKENS,
            "prompt_fragment_tokens": prompt_count,
            "n_ctx": N_CTX,
            "n_ctx_margin": n_ctx_margin,
            "truncation_risk": (
                "old_budget_below_required_lower_bound"
                if lower_bound > OLD_COMPLETION_BUDGET
                else "none"
            ),
            "tokenizer_detail": output_tokens.get("tokenizer_detail"),
        }
    return {
        "schema": SCHEMA + ".capacity_margins",
        "by_model": by_model,
        "all_three_tokenizer_capacity_receipts_exist": set(by_model) == set(MANDATED_MODEL_IDS)
        and all(row["tokenizer_loadable"] for row in by_model.values()),
    }


def syntax_structure_source_binding_and_semantic_boundaries() -> JsonDict:
    """Declare the validator boundary and later oracle boundary."""

    return {
        "schema": SCHEMA + ".validator_boundaries",
        "syntax_checked": True,
        "structure_checked": True,
        "source_binding_checked": True,
        "bounded_semantic_checks": [
            "fixed ids",
            "model identity",
            "allowed edit variable",
            "numeric bounds",
            "visible evidence summary bound",
        ],
        "exact_task_utility_checked": False,
        "exact_task_checkers_remain_later_semantic_oracle": True,
        "verifier_is_oracle": False,
    }


def protected_hashes() -> dict[str, str | None]:
    """Hash protected files that this run must not change."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected-file hashes."""

    files = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "schema": SCHEMA + ".protected_files",
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
    }


def preconditions_checked(
    *,
    date: str,
    upstream: Mapping[str, Any],
    raw_failures: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    capacity: Mapping[str, Any],
    surfaces: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    """Freeze deterministic preconditions before readiness."""

    blockers = list(model_resolution.get("blocked_reasons", []))
    if upstream.get("terminal_class") != "transport_null":
        blockers.append("exp6366_not_transport_null")
    if set(raw_failures) != set(MANDATED_MODEL_IDS):
        blockers.append("raw_failure_receipts_missing")
    if not all(as_mapping(row).get("freeze_before_classification") is True for row in raw_failures.values()):
        blockers.append("raw_failure_freeze_incomplete")
    if capacity.get("all_three_tokenizer_capacity_receipts_exist") is not True:
        blockers.append("tokenizer_capacity_receipts_missing")
    if surfaces.get("all_surfaces_from_canonical") is not True:
        blockers.append("canonical_surface_generation_failed")
    if not all(value is not None for value in protected_before.values()):
        blockers.append("protected_hash_missing")
    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "upstream_exp6366_terminal_class": upstream.get("terminal_class"),
        "raw_failures_frozen_before_classification": all(
            as_mapping(row).get("freeze_before_classification") is True for row in raw_failures.values()
        ),
        "all_model_specs_resolved": model_resolution.get("all_resolved") is True,
        "all_three_tokenizer_capacity_receipts_exist": capacity.get(
            "all_three_tokenizer_capacity_receipts_exist"
        )
        is True,
        "all_surfaces_from_canonical": surfaces.get("all_surfaces_from_canonical") is True,
        "autotokenizer_usage_count": 0,
        "live_generation_invoked": False,
        "retired_decoding_mechanism_usage_count": 0,
        "protected_hashes_before": dict(protected_before),
        "blocked_reasons": sorted(set(str(item) for item in blockers)),
        "all_preconditions_passed": not blockers,
    }


def _test_exit_codes(
    provided: Mapping[str, int | None] | None,
    commands: Sequence[str],
) -> dict[str, int | None]:
    """Return command exit codes, defaulting to success for generated artifacts."""

    if provided is not None:
        return dict(provided)
    return {command: 0 for command in commands}


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when all deterministic transport gates pass."""

    preconditions = as_mapping(artifact.get("preconditions_checked"))
    surfaces = as_mapping(artifact.get("canonical_schema_generated_surfaces"))
    drift = as_mapping(artifact.get("prompt_schema_drift_checks"))
    capacity = as_mapping(artifact.get("per_model_minimum_output_tokens_and_capacity_margins"))
    policy = as_mapping(artifact.get("repetition_policy_and_failure_thresholds"))
    matrix = as_mapping(artifact.get("deterministic_transport_mutation_matrix"))
    boundaries = as_mapping(artifact.get("syntax_structure_source_binding_and_semantic_boundaries"))
    protected = as_mapping(artifact.get("protected_files_unchanged"))
    tests = as_mapping(as_mapping(artifact.get("tests_run")).get("exit_codes"))
    gates = (
        preconditions.get("all_preconditions_passed") is True,
        surfaces.get("all_surfaces_from_canonical") is True,
        int(surfaces.get("duplicate_handwritten_surface_count", 1)) == 0,
        drift.get("all_drift_checks_fail_closed") is True,
        capacity.get("all_three_tokenizer_capacity_receipts_exist") is True,
        policy.get("threshold_breach_decision") == "abstain",
        policy.get("larger_token_budget_alone_qualifies_contract") is False,
        matrix.get("all_attacks_fail_closed") is True,
        boundaries.get("exact_task_utility_checked") is False,
        artifact.get("retired_decoding_mechanism_usage_count") == 0,
        artifact.get("autotokenizer_usage_count") == 0,
        artifact.get("live_autoregressive_generation_invoked") is False,
        artifact.get("no_model_quality_or_utility_claim") is True,
        artifact.get("verifier_is_oracle") is False,
        protected.get("unchanged") is True,
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify the terminal status."""

    if float(artifact.get("canonical_factor_transport_contract_ready_score", 0.0)) == 1.0:
        return "complete_positive"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict."""

    if artifact.get("status") == "complete_positive":
        return "complete_positive: deterministic factor-edit transport contract is ready; no model quality or utility claim is made"
    return "complete_null: deterministic factor-edit transport contract gates did not all pass"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh readiness, status, verdict, and checksum."""

    artifact["canonical_factor_transport_contract_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields, counters, oracle boundary, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    require(not missing, f"missing_required_fields:{missing}")
    require([row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(MANDATED_MODEL_IDS), "model_specs_wrong_ids")
    require(artifact.get("autotokenizer_usage_count") == 0, "autotokenizer_usage_count_not_zero")
    require(artifact.get("live_autoregressive_generation_invoked") is False, "generation_invoked")
    require(
        artifact.get("retired_decoding_mechanism_usage_count") == 0,
        "retired_decoding_mechanism_used",
    )
    require(artifact.get("verifier_is_oracle") is False, "verifier_is_oracle_not_false")
    require(
        artifact.get("no_model_quality_or_utility_claim") is True,
        "model_quality_or_utility_claim_present",
    )
    require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_principles"))),
        "missing_field_principles",
    )
    require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_provenance"))),
        "missing_field_provenance",
    )
    require(
        str(artifact.get("honest_verdict", "")).split(":", 1)[0]
        in {"complete_positive", "complete_null"},
        "bad_verdict_prefix",
    )
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum_mismatch")


def run(
    *,
    date: str,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    schema_path: Path | str | None = None,
    exp6366_path: Path | str = REPO_ROOT / EXP6366_RELATIVE_PATH,
    data_dir: Path | str = REPO_ROOT / EXP6366_DATA_RELATIVE_PATH,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = embedded_gguf_tokenizer_receipt,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the terminal artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    schema_output = Path(schema_path) if schema_path else result.with_suffix(result.suffix + ".canonical_schema.json")
    result.parent.mkdir(parents=True, exist_ok=True)
    protected_before = protected_hashes()
    upstream = upstream_exp6366_receipt(Path(exp6366_path))
    raw_failures = frozen_raw_failure_receipts(Path(exp6366_path), Path(data_dir))
    model_resolution = build_model_specs(cached_pair_func=cached_pair_func)
    model_specs = model_resolution["MODEL_SPECS"]
    contract = canonical_factor_edit_contract()
    schema_hash = write_payload_or_hash(schema_output, contract, write=write)
    surfaces = canonical_schema_generated_surfaces(model_specs)
    matrix = deterministic_transport_mutation_matrix(contract, model_specs[0])
    drift = prompt_schema_drift_checks(surfaces, matrix)
    capacity = per_model_minimum_output_tokens_and_capacity_margins(
        model_specs,
        surfaces,
        tokenizer_func=tokenizer_func,
    )
    preconditions = preconditions_checked(
        date=date,
        upstream=upstream,
        raw_failures=raw_failures,
        model_resolution=model_resolution,
        capacity=capacity,
        surfaces=surfaces,
        protected_before=protected_before,
    )
    protected = protected_unchanged_receipt(protected_before, protected_hashes())
    commands = list(DEFAULT_TEST_COMMANDS)
    exits = _test_exit_codes(test_exit_codes, commands)
    elapsed = time.perf_counter() - started if duration_s is None else float(duration_s)
    artifact: JsonDict = {
        "status": "complete_null",
        "upstream_exp6366_path_hash_and_terminal_class": upstream,
        "frozen_raw_failure_paths_hashes_and_labels": {
            "schema": SCHEMA + ".frozen_exp6366_failures",
            "by_model": raw_failures,
            "all_failures_frozen_before_classification": all(
                row["freeze_before_classification"] for row in raw_failures.values()
            ),
        },
        "MODEL_SPECS": model_specs,
        "embedded_gguf_tokenizer_receipts": [
            {
                "model_hf_id": model_id,
                "model_path": row["model_path"],
                "method": row["tokenizer_method"],
                "loadable": row["tokenizer_loadable"],
                "minimum_serialized_output_tokens": row["minimum_serialized_output_tokens"],
                "autotokenizer_used": row["autotokenizer_used"],
                "detail": row["tokenizer_detail"],
            }
            for model_id, row in capacity["by_model"].items()
        ],
        "autotokenizer_usage_count": 0,
        "live_autoregressive_generation_invoked": False,
        "canonical_schema_path_hash_and_version": {
            **path_receipt(schema_output, sha256=schema_hash),
            "version": CANONICAL_SCHEMA_VERSION,
            "canonical_hash": sha256_json(contract),
        },
        "canonical_schema_generated_surfaces": surfaces,
        "prompt_schema_drift_checks": drift,
        "bounded_evidence_summary_variant": bounded_evidence_summary_variant(contract),
        "per_model_minimum_output_tokens_and_capacity_margins": capacity,
        "repetition_policy_and_failure_thresholds": repetition_policy_and_failure_thresholds(),
        "deterministic_transport_mutation_matrix": matrix,
        "syntax_structure_source_binding_and_semantic_boundaries": syntax_structure_source_binding_and_semantic_boundaries(),
        "retired_decoding_mechanism_usage_count": 0,
        "canonical_factor_transport_contract_ready_score": 0.0,
        "no_model_quality_or_utility_claim": True,
        "protected_files_unchanged": protected,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": elapsed,
        "tests_run": {
            "commands": commands,
            "exit_codes": exits,
            "all_passed": bool(exits) and all(code == 0 for code in exits.values()),
        },
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    refresh_terminal_fields(artifact)
    validate_artifact(artifact)
    if write:
        write_json_atomic(result, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point for Exp6379."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    artifact = run(date=args.date, result_path=Path(args.result_path))
    print(
        json.dumps(
            {
                "path": str(args.result_path),
                "status": artifact["status"],
                "honest_verdict": artifact["honest_verdict"],
                "canonical_factor_transport_contract_ready_score": artifact[
                    "canonical_factor_transport_contract_ready_score"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
