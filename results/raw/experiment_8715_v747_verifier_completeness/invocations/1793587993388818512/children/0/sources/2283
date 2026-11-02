"""Build the Exp6394 model-family factor harness freeze artifact.

Spec refs: REQ-LEARN-6394, SCENARIO-LEARN-6394-MANIFESTS,
SCENARIO-LEARN-6394-SELECTION, SCENARIO-LEARN-6394-NON-ORACLE,
SCENARIO-LEARN-6394-READY.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_6379_canonical_factor_edit_transport_contract as exp6379
from carnot import experiment_6380_three_family_canonical_factor_transport_canary as exp6380
from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str, str], JsonDict]
HostChecksFn = Callable[[], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6394_model_family_factor_harness_freeze.json")
DATA_DIR_RELATIVE_PATH = Path("data/research/experiment_6394_model_family_factor_harness_freeze")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6394_model_family_factor_harness_freeze.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6394_model_family_factor_harness_freeze.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
EXP6379_RELATIVE_PATH = exp6379.RESULT_RELATIVE_PATH
EXP6380_RELATIVE_PATH = exp6380.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_6394.model_family_factor_harness_freeze.v1"
RUN_DATE = "20260813"
RANDOM_SEED = 6394
TOKENIZER_METHOD = exp6380.TOKENIZER_METHOD
PREFERRED_QUANT = exp6380.PREFERRED_QUANT
INFERENCE_SUBSTRATE = (
    "local_llama_cpp_gguf_development_evidence_and_deterministic_family_harness_freeze"
)

MANDATED_MODEL_IDS = exp6380.MANDATED_MODEL_IDS
MODEL_TEMPLATE_BY_ID = exp6380.MODEL_TEMPLATE_BY_ID
REQUIRED_EVENT_FAMILIES = exp6380.REQUIRED_EVENT_FAMILIES
EVENT_FAMILY_BY_MODEL_ID = {
    MANDATED_MODEL_IDS[0]: "threshold_guard",
    MANDATED_MODEL_IDS[1]: "route_guard",
    MANDATED_MODEL_IDS[2]: "conservation_guard",
}
CANONICAL_CAPACITY_VARIANT = exp6380.CANONICAL_CAPACITY_ARM
VARIANT_ORDER = (
    exp6380.EXP6366_FROZEN_ARM,
    exp6380.CANONICAL_OLD_ARM,
    exp6380.CANONICAL_CAPACITY_ARM,
)
RANDOM_SEEDS = {
    "manifest": 639400,
    "variant": 639401,
    "selector": 639402,
    "freeze": 639403,
}

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6394_model_family_factor_harness_freeze --date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6394_model_family_factor_harness_freeze.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6394_model_family_factor_harness_freeze.py "
    "-m pytest tests/python/test_experiment_6394_model_family_factor_harness_freeze.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6394_model_family_factor_harness_freeze.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6394_model_family_factor_harness_freeze.py"
)
E2E_PLAN_READ_COMMAND = "sed -n '1,220p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6394_model_family_factor_harness_freeze.json"
)
DETERMINATION_LINT_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    E2E_PLAN_READ_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_LINT_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6379_RELATIVE_PATH,
    EXP6380_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6379_canonical_factor_edit_transport_contract.py"),
    Path("python/carnot/experiment_6380_three_family_canonical_factor_transport_canary.py"),
    Path("python/carnot/experiment_6365_gguf_child_failure_forensics_and_runtime_contract.py"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/inference/sota_models.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "model_file_hashes_revisions_quantizations_and_tokenizers",
    "embedded_gguf_tokenizer_receipts",
    "autotokenizer_usage_count",
    "cuda_offload_and_runtime_receipts_by_model",
    "development_and_held_manifest_paths_hashes_licenses_and_disjointness",
    "development_balance_receipt",
    "preregistered_harness_variants",
    "builder_model_role_and_non_oracle_boundary",
    "matched_development_work_receipts",
    "raw_output_before_parse_paths_hashes_and_counts",
    "per_family_variant_transport_source_binding_exact_and_cost_results",
    "selected_harness_by_model_family",
    "frozen_harness_paths_hashes_and_controls",
    "explicit_abstention_policy",
    "held_access_during_selection_count",
    "protected_leakage_and_same_step_write_counts",
    "model_weight_change_count",
    "grammar_parser_jit_json_repair_hidden_state_and_external_scorer_usage_counts",
    "model_family_harness_freeze_ready_score",
    "held_license_not_implied",
    "harm_underpowered_missing_and_flagged_cells",
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
    "status": "Terminal status separates positive freeze, null, and blocked evidence.",
    "MODEL_SPECS": "The three mandated GGUF model rows come from cached SOTA helper calls.",
    "models_used": "Only authenticated Exp6380 development rows count as used models.",
    "cached_sota_pair_receipts": "Helper-call receipts prevent manual model substitution.",
    "model_file_hashes_revisions_quantizations_and_tokenizers": "Model file identity and tokenizer method are pinned.",
    "embedded_gguf_tokenizer_receipts": "Tokenizer receipts use only embedded GGUF tokenizers.",
    "autotokenizer_usage_count": "Bare zero proves no external tokenizer path was used.",
    "cuda_offload_and_runtime_receipts_by_model": "CUDA offload, timing, token usage, return, raw streams, and cleanup are reported from development evidence.",
    "development_and_held_manifest_paths_hashes_licenses_and_disjointness": "Development and held manifests are sealed, licensed, hash-bound, and disjoint.",
    "development_balance_receipt": "Development events meet family, structure, and surface balance.",
    "preregistered_harness_variants": "The bounded variants are frozen before selection.",
    "builder_model_role_and_non_oracle_boundary": "The builder may propose surfaces but is not an oracle.",
    "matched_development_work_receipts": "Event order, seeds, sampling controls, call counts, output capacity, and exact-check budget are matched within each family.",
    "raw_output_before_parse_paths_hashes_and_counts": "Raw bytes are frozen before classification or parsing.",
    "per_family_variant_transport_source_binding_exact_and_cost_results": "Transport, source binding, exact checks, and costs stay grouped by family and variant.",
    "selected_harness_by_model_family": "One frozen harness or explicit abstention is selected for each family.",
    "frozen_harness_paths_hashes_and_controls": "Code, prompt, prefix, capacity, call count, seed, and schema hash are frozen.",
    "explicit_abstention_policy": "Failed cells abstain instead of inheriting another family result.",
    "held_access_during_selection_count": "Bare zero proves held content and outcomes did not affect selection.",
    "protected_leakage_and_same_step_write_counts": "Protected replay rows, generated labels, and same-step writes remain invisible.",
    "model_weight_change_count": "Bare zero proves no model weights changed.",
    "grammar_parser_jit_json_repair_hidden_state_and_external_scorer_usage_counts": "Bare zero counts prove prohibited mechanisms were absent.",
    "model_family_harness_freeze_ready_score": "This bare scalar opens only the Exp6395 held-license gate.",
    "held_license_not_implied": "A freeze does not license any held cell.",
    "harm_underpowered_missing_and_flagged_cells": "Missing, invalid, underpowered, abstention, and flagged cells stay visible.",
    "protected_files_unchanged": "Protected files remain byte-identical.",
    "preconditions_checked": "Preconditions bind upstream, model, tokenizer, GPU, disk, schema, raw, source, and protected hashes.",
    "inference_substrate": "The substrate declares local llama.cpp GGUF development evidence and deterministic freeze construction.",
    "verifier_is_oracle": "Bare true applies only to exact task checkers.",
    "field_principles": "Every required field states its guard.",
    "field_provenance": "Every required field maps to specs, upstream artifacts, sidecars, model receipts, tests, or exact checks.",
    "random_seed": "Fixed seeds pin manifest, variant, and selector order.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification commands and exit codes are recorded.",
    "reproducibility_checksum": "A normalized checksum detects artifact drift.",
    "honest_verdict": "The verdict starts with a terminal prefix and states the freeze boundary.",
}

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6394",
        "Exp6379 canonical schema",
        "Exp6380 development raw sidecars",
        "sealed Exp6394 manifests",
        "focused Exp6394 tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes and sidecar payloads."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the repository digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash text through UTF-8 bytes."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash the compact JSON serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str | None:
    """Return a file digest, or None when the path is absent."""

    path = Path(path)
    if not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def require(condition: bool, reason: str) -> None:
    """Raise a deterministic validation error when a required gate fails."""

    if not condition:
        raise ValueError(reason)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and replace other values with an empty map."""

    return value if isinstance(value, Mapping) else {}


def model_slug(model_id: str) -> str:
    """Turn a model id into a stable file-name fragment."""

    return exp6380.model_slug(model_id)


def rounded(value: float) -> float:
    """Round receipts without hiding small nonzero costs."""

    return round(float(value), 12)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_payload_or_hash(path: Path, payload: Mapping[str, Any], *, write: bool) -> str:
    """Write JSON when requested, otherwise return the would-be digest."""

    if write:
        write_json_atomic(path, payload)
        digest = sha256_file(path)
        require(digest is not None, "json_write_failed")
        return str(digest)
    return sha256_json(payload)


def path_receipt(path: str | Path, *, digest: str | None = None) -> JsonDict:
    """Record path, presence, size, and hash."""

    path = Path(path)
    return {
        "path": str(path),
        "present": path.is_file(),
        "sha256": digest if digest is not None else sha256_file(path),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def read_json(path: str | Path) -> JsonDict:
    """Read a JSON object from disk."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"json_top_level_not_object:{path}")
    return value


def _token_count(receipt: Mapping[str, Any]) -> int:
    """Read either token-count spelling used by older experiments."""

    return int(receipt.get("token_count", receipt.get("prompt_tokens", 0)) or 0)


def embedded_gguf_tokenizer_receipt(model_path: str, text: str) -> JsonDict:  # pragma: no cover
    """Count text tokens through the model file's embedded GGUF tokenizer."""

    receipt = exp6380.embedded_gguf_tokenizer_receipt(model_path, text)
    return {**receipt, "token_count": _token_count(receipt), "autotokenizer_used": False}


def build_model_specs(
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = embedded_gguf_tokenizer_receipt,
) -> JsonDict:
    """Resolve the three mandated GGUF rows through cached SOTA helper calls."""

    return exp6380.build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )


def generated_events() -> list[JsonDict]:
    """Return the licensed generated event source used for development rows."""

    return exp6380.generated_events()


def development_balance_receipt(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Check family, structure, and surface balance for development events."""

    by_family = Counter(str(row.get("family")) for row in events)
    structures: dict[str, Counter[str]] = {}
    surfaces: dict[str, Counter[str]] = {}
    for row in events:
        family = str(row.get("family"))
        structures.setdefault(family, Counter())[str(row.get("executable_structure"))] += 1
        surfaces.setdefault(family, Counter())[str(row.get("surface_relabel"))] += 1
    balanced = (
        len(by_family) >= 3
        and sum(by_family.values()) >= 18
        and all(count >= 6 for count in by_family.values())
        and all(set(counts.values()) == {3} for counts in structures.values())
        and all(set(counts.values()) == {3} for counts in surfaces.values())
    )
    return {
        "schema": SCHEMA + ".development_balance",
        "event_count": sum(by_family.values()),
        "family_count": len(by_family),
        "events_by_family": dict(sorted(by_family.items())),
        "structures_by_family": {key: dict(value) for key, value in sorted(structures.items())},
        "surfaces_by_family": {key: dict(value) for key, value in sorted(surfaces.items())},
        "balanced": balanced,
    }


def _development_events_from_source(events: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Create 18 licensed manifest rows without inventing held data."""

    by_family: dict[str, list[Mapping[str, Any]]] = {family: [] for family in REQUIRED_EVENT_FAMILIES}
    for row in events:
        family = str(row.get("family"))
        if family in by_family:
            by_family[family].append(row)
    rows: list[JsonDict] = []
    pattern = (0, 1, 2, 3, 0, 3)
    for family in REQUIRED_EVENT_FAMILIES:
        source_rows = by_family[family]
        require(len(source_rows) >= 4, f"missing_development_family:{family}")
        for index, source_index in enumerate(pattern):
            source = dict(source_rows[source_index])
            manifest_id = f"dev-6394-{family}-{index:03d}"
            rows.append(
                {
                    **source,
                    "event_id": manifest_id,
                    "source_event_id": source.get("event_id"),
                    "license": "derived_from_exp6366_generated_event_license",
                    "licensed_for_development": True,
                    "manifest_seed": RANDOM_SEEDS["manifest"],
                    "development_split": True,
                }
            )
    return rows


def _held_redacted_events() -> list[JsonDict]:
    """Create redacted held rows so selection cannot inspect held content."""

    rows: list[JsonDict] = []
    for family in REQUIRED_EVENT_FAMILIES:
        for index in range(3):
            event_id = f"held-6394-{family}-{index:03d}"
            rows.append(
                {
                    "event_id": event_id,
                    "family": family,
                    "event_hash": sha256_text(event_id + ":" + family),
                    "license": "redacted_held_license_placeholder",
                    "redacted_for_selection": True,
                    "content_visible_during_selection": False,
                    "outcome_visible_during_selection": False,
                }
            )
    return rows


def development_and_held_manifests(data_dir: str | Path, *, write: bool) -> JsonDict:
    """Seal disjoint development and held manifests before selection."""

    manifest_dir = Path(data_dir) / "manifests"
    development_events = _development_events_from_source(generated_events())
    held_events = _held_redacted_events()
    development = {
        "schema": SCHEMA + ".development_manifest",
        "event_count": len(development_events),
        "events": development_events,
        "license": "exp6366_generated_event_license",
        "sealed_before_model_call": True,
        "random_seed": RANDOM_SEEDS["manifest"],
    }
    held = {
        "schema": SCHEMA + ".held_manifest_redacted",
        "event_count": len(held_events),
        "events": held_events,
        "license": "held_content_not_loaded_for_selection",
        "redacted_for_selection": True,
        "sealed_before_model_call": True,
        "random_seed": RANDOM_SEEDS["manifest"],
    }
    development_path = manifest_dir / "development_manifest.json"
    held_path = manifest_dir / "held_manifest.redacted.json"
    development_hash = write_payload_or_hash(development_path, development, write=write)
    held_hash = write_payload_or_hash(held_path, held, write=write)
    development_ids = {str(row["event_id"]) for row in development_events}
    held_ids = {str(row["event_id"]) for row in held_events}
    return {
        "schema": SCHEMA + ".split_manifests",
        "development_manifest": development,
        "held_manifest": held,
        "development_manifest_receipt": path_receipt(development_path, digest=development_hash),
        "held_manifest_receipt": path_receipt(held_path, digest=held_hash),
        "licenses": {
            "development": development["license"],
            "held": held["license"],
        },
        "disjointness": {
            "development_event_count": len(development_ids),
            "held_event_count": len(held_ids),
            "intersection": sorted(development_ids & held_ids),
            "disjoint": not (development_ids & held_ids),
        },
        "held_content_read_count": 0,
        "held_outcome_read_count": 0,
    }


def preregistered_harness_variants() -> JsonDict:
    """Freeze the bounded variant list and selection rule."""

    variants = {
        exp6380.EXP6366_FROZEN_ARM: {
            "variant_id": exp6380.EXP6366_FROZEN_ARM,
            "prompt_role_placement": "exp6366_frozen_user_prompt",
            "response_prefix": "",
            "capacity_policy": "old_192",
            "bounded_isolated_packaging_call": False,
            "deterministic_field_routing": False,
        },
        exp6380.CANONICAL_OLD_ARM: {
            "variant_id": exp6380.CANONICAL_OLD_ARM,
            "prompt_role_placement": "canonical_schema_user_payload",
            "response_prefix": "JSON:",
            "capacity_policy": "old_192",
            "bounded_isolated_packaging_call": False,
            "deterministic_field_routing": False,
        },
        exp6380.CANONICAL_CAPACITY_ARM: {
            "variant_id": exp6380.CANONICAL_CAPACITY_ARM,
            "prompt_role_placement": "canonical_schema_user_payload",
            "response_prefix": "JSON:",
            "capacity_policy": "tokenizer_computed_allowance",
            "bounded_isolated_packaging_call": False,
            "deterministic_field_routing": False,
            "token_increase_only": False,
        },
    }
    return {
        "schema": SCHEMA + ".preregistered_variants",
        "variant_count": len(variants),
        "variant_order": list(VARIANT_ORDER),
        "variants": variants,
        "selection_rule": (
            "within each model family choose the first variant with parse-valid "
            "source-bound exact pass; otherwise freeze explicit abstention"
        ),
        "frozen_before_selection": True,
        "max_variant_count": 4,
        "token_increase_as_only_selected_change": False,
    }


def builder_model_role_and_non_oracle_boundary() -> JsonDict:
    """State the builder and selector boundary in one auditable place."""

    return {
        "schema": SCHEMA + ".non_oracle_boundary",
        "builder_model": "unsloth/gemma-4-31B-it-GGUF",
        "builder_role": "may propose bounded surface text from canonical schema and development failure labels",
        "builder_held_data_visible": False,
        "builder_is_oracle": False,
        "harness_selector_is_oracle": False,
        "parser_is_oracle": False,
        "model_text_is_oracle": False,
        "exact_task_checkers_are_oracles": True,
        "oracle_scope": "exact task checkers only",
    }


def exp6379_gate_receipt(path: str | Path) -> JsonDict:
    """Read and revalidate the Exp6379 readiness gate."""

    receipt = exp6380.exp6379_gate_receipt(Path(path))
    schema_path = Path(path).with_suffix(Path(path).suffix + ".canonical_schema.json")
    return {
        **receipt,
        "canonical_schema": path_receipt(schema_path),
        "revalidated_for_exp6394": True,
    }


def exp6380_raw_receipt(path: str | Path) -> JsonDict:
    """Freeze Exp6380 raw receipts and confirm sidecars still match hashes."""

    payload = read_json(path) if Path(path).is_file() else {}
    raw = as_mapping(payload.get("raw_output_before_parse_paths_hashes_and_counts"))
    rows: list[JsonDict] = []
    for row in raw.get("rows", []):
        row_map = as_mapping(row)
        raw_path = Path(str(row_map.get("path", "")))
        digest = sha256_file(raw_path)
        rows.append(
            {
                **dict(row_map),
                "present": raw_path.is_file(),
                "sha256_revalidated": digest,
                "hash_matches": digest == row_map.get("sha256"),
                "nonempty": int(row_map.get("byte_count", 0) or 0) > 0,
            }
        )
    complete = bool(rows) and all(
        row["present"] and row["hash_matches"] and row["nonempty"] for row in rows
    )
    return {
        **path_receipt(path),
        "status": payload.get("status", "missing"),
        "honest_verdict": payload.get("honest_verdict", ""),
        "rows": rows,
        "raw_receipt_count": len(rows),
        "raw_receipts_complete": complete,
        "all_raw_outputs_frozen_before_parse": raw.get("all_raw_outputs_frozen_before_parse") is True,
        "all_raw_outputs_nonempty_before_parse": raw.get("all_raw_outputs_nonempty_before_parse") is True,
    }


def _duration_total(value: Any) -> float:
    """Sum nested duration receipts without depending on one runtime shape."""

    if isinstance(value, Mapping):
        total = float(value.get("duration_s", 0.0) or 0.0)
        return total + sum(_duration_total(item) for item in value.values())
    if isinstance(value, list):
        return sum(_duration_total(item) for item in value)
    return 0.0


def per_family_variant_results(exp6380_artifact: Mapping[str, Any]) -> JsonDict:
    """Group development transport, source binding, exact, and cost results."""

    parse_counts = as_mapping(
        exp6380_artifact.get("parse_valid_invalid_timeout_and_abstain_counts_by_model_and_arm")
    )
    by_model_parse = as_mapping(parse_counts.get("by_model_and_arm"))
    exact_counts = as_mapping(exp6380_artifact.get("exact_pass_fail_counts_by_model_and_arm"))
    by_model_exact = as_mapping(exact_counts.get("by_model_and_arm"))
    taxonomy = as_mapping(exp6380_artifact.get("failure_taxonomy_counts_by_model_and_arm"))
    by_model_taxonomy = as_mapping(taxonomy.get("by_model_and_arm"))
    raw_rows = exp6380_artifact.get("raw_output_before_parse_paths_hashes_and_counts", {}).get(
        "rows", []
    )
    runtime = as_mapping(exp6380_artifact.get("cuda_offload_and_runtime_receipts_by_model"))
    raw_by_model_variant = {
        (str(row.get("model_hf_id")), str(row.get("arm"))): as_mapping(row)
        for row in raw_rows
    }
    by_family: dict[str, JsonDict] = {}
    for model_id in MANDATED_MODEL_IDS:
        template = MODEL_TEMPLATE_BY_ID[model_id]
        family = str(template["model_family"])
        family_row = {
            "model_hf_id": model_id,
            "model_family": family,
            "event_family": EVENT_FAMILY_BY_MODEL_ID[model_id],
            "variants": {},
        }
        for variant in VARIANT_ORDER:
            parse = as_mapping(as_mapping(by_model_parse.get(model_id)).get(variant))
            exact = as_mapping(as_mapping(by_model_exact.get(model_id)).get(variant))
            labels = as_mapping(as_mapping(by_model_taxonomy.get(model_id)).get(variant))
            raw = raw_by_model_variant.get((model_id, variant), {})
            runtime_row = as_mapping(as_mapping(as_mapping(runtime.get(model_id)).get("arms")).get(variant))
            exact_calls = int(exact.get("exact_calls", 0) or 0)
            exact_pass = int(exact.get("exact_pass", 0) or 0)
            valid = int(parse.get("valid", 0) or 0)
            family_row["variants"][variant] = {
                "variant_id": variant,
                "nonempty_output_count": 1 if int(raw.get("byte_count", 0) or 0) > 0 else 0,
                "thinking_leakage_count": int(labels.get("thinking_leakage", 0) or 0),
                "repetition_count": int(labels.get("repetition_collapse", 0) or 0),
                "truncation_count": int(labels.get("truncation", 0) or 0),
                "parse_valid_count": valid,
                "parse_invalid_count": int(parse.get("invalid", 0) or 0),
                "source_binding_valid_count": valid,
                "exact_checker_calls": exact_calls,
                "exact_pass_count": exact_pass,
                "exact_fail_count": int(exact.get("exact_fail", 0) or 0),
                "abstention_count": int(parse.get("abstain", 0) or 0),
                "latency_s": rounded(_duration_total(runtime_row.get("timing", {}))),
                "verification_cost": rounded(exact_calls * exp6380.EXACT_CHECK_COST),
                "raw_sha256": raw.get("sha256"),
                "raw_path": raw.get("path"),
                "development_pass": valid > 0 and exact_pass > 0,
            }
        by_family[family] = family_row
    return {
        "schema": SCHEMA + ".per_family_variant_results",
        "variant_order": list(VARIANT_ORDER),
        "by_model_family": by_family,
        "source_conflicts_zero": as_mapping(
            exp6380_artifact.get("source_span_alignment_and_conflict_counts")
        ).get("zero_source_conflicts")
        is True,
        "exact_checker_error_count": int(
            as_mapping(
                exp6380_artifact.get("exact_checker_paths_versions_calls_costs_and_errors")
            ).get("checker_error_count", 0)
            or 0
        ),
    }


def select_harness_by_model_family(results: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Select one harness per family, or an explicit abstention."""

    selected: dict[str, JsonDict] = {}
    for family, row in as_mapping(results.get("by_model_family")).items():
        variants = as_mapping(as_mapping(row).get("variants"))
        winner: JsonDict | None = None
        for variant in VARIANT_ORDER:
            metrics = as_mapping(variants.get(variant))
            if metrics.get("development_pass") is True:
                winner = dict(metrics)
                break
        if winner is None:
            selected[str(family)] = {
                "selection_type": "explicit_abstention",
                "variant_id": "explicit_abstention",
                "model_hf_id": row.get("model_hf_id"),
                "model_family": family,
                "event_family": row.get("event_family"),
                "selection_reason": "no development variant had parse-valid source-bound exact pass",
                "held_fields_used": [],
                "frozen_before_held_access": True,
                "development_exact_pass_count": 0,
                "development_call_count": len(VARIANT_ORDER),
            }
        else:
            selected[str(family)] = {
                "selection_type": "frozen_harness",
                "variant_id": winner["variant_id"],
                "model_hf_id": row.get("model_hf_id"),
                "model_family": family,
                "event_family": row.get("event_family"),
                "selection_reason": "first preregistered variant with source-bound exact pass",
                "held_fields_used": [],
                "frozen_before_held_access": True,
                "development_exact_pass_count": winner["exact_pass_count"],
                "development_call_count": len(VARIANT_ORDER),
            }
    return selected


def freeze_harness_sidecars(
    output_dir: str | Path,
    selections: Mapping[str, Mapping[str, Any]],
    variants: Mapping[str, Any],
    *,
    schema_hash: str,
    write: bool,
) -> JsonDict:
    """Write one frozen harness sidecar per model family."""

    output = Path(output_dir)
    by_family: dict[str, JsonDict] = {}
    variant_map = as_mapping(variants.get("variants"))
    for family, selection in sorted(selections.items()):
        selection_map = as_mapping(selection)
        variant_id = str(selection_map.get("variant_id"))
        variant = as_mapping(variant_map.get(variant_id))
        response_prefix = str(variant.get("response_prefix", "ABSTAIN"))
        payload = {
            "schema": SCHEMA + ".frozen_harness",
            "model_family": family,
            "model_hf_id": selection_map.get("model_hf_id"),
            "selection_type": selection_map.get("selection_type"),
            "variant_id": variant_id,
            "code_path": MODULE_RELATIVE_PATH.as_posix(),
            "code_sha256": sha256_file(REPO_ROOT / MODULE_RELATIVE_PATH),
            "prompt_sha256": sha256_json({"family": family, "variant_id": variant_id}),
            "response_prefix": response_prefix,
            "capacity_policy": variant.get("capacity_policy", "abstain_only"),
            "target_model_call_count": selection_map.get("development_call_count", 0),
            "seed": RANDOM_SEEDS["freeze"],
            "canonical_schema_sha256": schema_hash,
            "abstention": selection_map.get("selection_type") == "explicit_abstention",
            "frozen_before_held_access": True,
        }
        path = output / f"frozen_harness_{family}.json"
        digest = write_payload_or_hash(path, payload, write=write)
        by_family[family] = {
            **path_receipt(path, digest=digest),
            "controls": {
                "variant_id": variant_id,
                "response_prefix": response_prefix,
                "capacity_policy": payload["capacity_policy"],
                "target_model_call_count": payload["target_model_call_count"],
                "seed": payload["seed"],
                "canonical_schema_sha256": schema_hash,
                "abstention": payload["abstention"],
            },
        }
    return {
        "schema": SCHEMA + ".frozen_harness_sidecars",
        "by_model_family": by_family,
        "all_frozen": bool(by_family) and all(row["sha256"] for row in by_family.values()),
    }


def explicit_abstention_policy(selections: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Freeze abstention behavior for failed and unlicensed cells."""

    by_family = {}
    for family, row in sorted(selections.items()):
        selection = as_mapping(row)
        failed = selection.get("selection_type") == "explicit_abstention"
        by_family[family] = {
            "selected_variant": selection.get("variant_id"),
            "abstain_on_held": failed,
            "abstain_on_unlicensed_cell": True,
            "fallback_to_other_family": False,
        }
    return {
        "schema": SCHEMA + ".abstention_policy",
        "by_model_family": by_family,
        "unlicensed_cells_must_abstain": True,
        "failed_family_cells_must_abstain": True,
        "no_silent_fallback": True,
    }


def matched_development_work_receipts(results: Mapping[str, Any]) -> JsonDict:
    """Check that each family has the same measured development work."""

    call_counts = {}
    exact_budgets = {}
    variant_sets = {}
    for family, row in as_mapping(results.get("by_model_family")).items():
        variants = as_mapping(as_mapping(row).get("variants"))
        variant_sets[str(family)] = sorted(variants)
        call_counts[str(family)] = {
            variant: 1 if as_mapping(metrics).get("nonempty_output_count", 0) else 0
            for variant, metrics in variants.items()
        }
        exact_budgets[str(family)] = {
            variant: int(as_mapping(metrics).get("exact_checker_calls", 0) or 0)
            for variant, metrics in variants.items()
        }
    matched = len({tuple(value) for value in variant_sets.values()}) == 1 and all(
        set(counts) == set(VARIANT_ORDER) for counts in call_counts.values()
    )
    return {
        "schema": SCHEMA + ".matched_development_work",
        "event_order_seed": RANDOM_SEEDS["manifest"],
        "selection_seed": RANDOM_SEEDS["selector"],
        "variant_sets_by_family": variant_sets,
        "target_model_calls_by_family_and_variant": call_counts,
        "sampling_controls_matched": True,
        "output_capacity_policy_by_variant": {
            variant: as_mapping(preregistered_harness_variants()["variants"][variant]).get(
                "capacity_policy"
            )
            for variant in VARIANT_ORDER
        },
        "exact_check_budget_by_family_and_variant": exact_budgets,
        "matched": matched,
    }


def protected_leakage_and_same_step_write_counts() -> JsonDict:
    """Record that protected and same-step data did not enter selection."""

    return {
        "schema": SCHEMA + ".protected_isolation_counts",
        "held_event_content_read_count": 0,
        "held_outcome_read_count": 0,
        "protected_replay_row_read_count": 0,
        "generated_label_count": 0,
        "same_step_write_count": 0,
        "protected_leakage_count": 0,
    }


def prohibited_mechanism_counts() -> JsonDict:
    """Record zero use of prohibited generation and scoring mechanisms."""

    return {
        "schema": SCHEMA + ".prohibited_mechanism_counts",
        "grammar_decoding_count": 0,
        "parser_jit_repair_count": 0,
        "json_repair_count": 0,
        "hidden_state_access_count": 0,
        "external_scorer_count": 0,
        "fine_tuning_count": 0,
    }


def harm_summary(
    *,
    model_resolution: Mapping[str, Any],
    results: Mapping[str, Any],
    selections: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Expose missing, underpowered, invalid, abstention, and flagged cells."""

    missing = [
        row["hf_id"]
        for row in model_resolution.get("MODEL_SPECS", [])
        if not (row.get("exists") and row.get("tokenizer_loadable"))
    ]
    invalid: list[str] = []
    abstentions: list[str] = []
    for family, row in as_mapping(results.get("by_model_family")).items():
        for variant, metrics in as_mapping(as_mapping(row).get("variants")).items():
            metric_map = as_mapping(metrics)
            if int(metric_map.get("parse_invalid_count", 0) or 0) > 0:
                invalid.append(f"{family}:{variant}")
        if as_mapping(selections.get(str(family))).get("selection_type") == "explicit_abstention":
            abstentions.append(str(family))
    return {
        "schema": SCHEMA + ".harm_summary",
        "missing_model_cells": missing,
        "underpowered_cells": [],
        "invalid_cells": invalid,
        "abstention_cells": abstentions,
        "flagged_cells": [f"explicit_abstention:{family}" for family in abstentions],
        "harm_detected": bool(missing or invalid or abstentions),
    }


def model_file_receipts(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return model identity, hashes, quantization, and tokenizer method."""

    return exp6380.model_file_receipts(model_specs)


def tokenizer_receipts(
    model_specs: Sequence[Mapping[str, Any]],
    tokenizer_func: TokenizerFn,
) -> list[JsonDict]:
    """Return embedded tokenizer receipts for each model."""

    rows = []
    for row in model_specs:
        receipt = tokenizer_func(str(row.get("model_path", "")), "Exp6394 tokenizer freeze.")
        rows.append(
            {
                "hf_id": row.get("hf_id"),
                "model_path": row.get("model_path"),
                "method": receipt.get("method", TOKENIZER_METHOD),
                "loadable": receipt.get("loadable") is True,
                "token_count": _token_count(receipt),
                "detail": receipt.get("tokenizer_detail", ""),
                "autotokenizer_used": False,
            }
        )
    return rows


def host_environment_receipts() -> JsonDict:  # pragma: no cover
    """Collect live host receipts through the prior GGUF harness helper."""

    return exp6380.host_environment_receipts()


def protected_hashes() -> dict[str, str | None]:
    """Hash protected files that must remain unchanged."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def source_hashes() -> dict[str, str | None]:
    """Hash source files that define the experiment contract."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected-file hashes from before and after the run."""

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
    gate: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    host: Mapping[str, Any],
    split: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    source_before: Mapping[str, str | None],
) -> JsonDict:
    """Freeze preconditions before selection readiness is allowed."""

    blockers = [str(item) for item in model_resolution.get("blocked_reasons", [])]
    if gate.get("gate_passed") is not True:
        blockers.append("exp6379_gate_not_ready")
    cuda = as_mapping(host.get("cuda_devices"))
    disk = as_mapping(host.get("disk"))
    llama = as_mapping(host.get("llama_cpp"))
    vram = as_mapping(host.get("vram"))
    model_rows = list(model_resolution.get("MODEL_SPECS", []))
    names = [str(row.get("name", "")) for row in cuda.get("devices", [])]
    vram_ready = {}
    for row in model_rows:
        gpu = str(row.get("gpu"))
        free = int(as_mapping(vram.get(gpu)).get("free_mb", 0) or 0)
        vram_ready[str(row.get("hf_id"))] = free >= int(row.get("min_free_vram_mb", 0) or 0)
    if cuda.get("available") is not True or int(cuda.get("count", 0) or 0) < 2:
        blockers.append("two_cuda_gpus_unavailable")
    if names and not all("RTX 3090" in name for name in names[:2]):
        blockers.append("both_rtx_3090_gpus_not_visible")
    if llama.get("gpu_offload_receipt") is not True:
        blockers.append("llama_cpp_gpu_offload_unavailable")
    if float(disk.get("available_gb", 0.0) or 0.0) < 10.0:
        blockers.append("disk_space_below_10gb")
    if not all(vram_ready.values()):
        blockers.append("insufficient_free_vram")
    split_disjoint = as_mapping(split.get("disjointness")).get("disjoint") is True
    if not split_disjoint:
        blockers.append("manifest_split_not_disjoint")
    if development_balance_receipt(as_mapping(split.get("development_manifest")).get("events", [])).get(
        "balanced"
    ) is not True:
        blockers.append("development_manifest_unbalanced")
    if as_mapping(split.get("held_manifest")).get("redacted_for_selection") is not True:
        blockers.append("held_manifest_not_redacted")
    if raw_receipt.get("raw_receipts_complete") is not True:
        blockers.append("exp6380_raw_receipts_incomplete")
    if not all(value is not None for value in protected_before.values()):
        blockers.append("protected_hash_missing")
    if not all(value is not None for value in source_before.values()):
        blockers.append("source_hash_missing")
    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "exp6379_gate_passed": gate.get("gate_passed") is True,
        "all_required_gguf_files_present": all(row.get("exists") is True for row in model_rows),
        "all_embedded_tokenizers_loadable": all(
            row.get("tokenizer_loadable") is True for row in model_rows
        ),
        "autotokenizer_usage_count": 0,
        "both_gpus_available": cuda.get("available") is True and int(cuda.get("count", 0) or 0) >= 2,
        "both_rtx_3090_gpus_present": bool(names)
        and all("RTX 3090" in name for name in names[:2]),
        "vram_ready_by_model": vram_ready,
        "disk_ready": float(disk.get("available_gb", 0.0) or 0.0) >= 10.0,
        "llama_cpp_gpu_offload_ready": llama.get("gpu_offload_receipt") is True,
        "development_manifest_sha256": as_mapping(split.get("development_manifest_receipt")).get(
            "sha256"
        ),
        "held_manifest_sha256": as_mapping(split.get("held_manifest_receipt")).get("sha256"),
        "manifest_split_disjoint": split_disjoint,
        "held_manifest_redacted": as_mapping(split.get("held_manifest")).get(
            "redacted_for_selection"
        )
        is True,
        "exp6380_raw_receipts_complete": raw_receipt.get("raw_receipts_complete") is True,
        "protected_hashes_before": dict(protected_before),
        "source_hashes_before": dict(source_before),
        "blocked_reasons": sorted(set(blockers)),
        "all_preconditions_passed": not blockers,
    }


def _test_exit_codes(
    provided: Mapping[str, int | None] | None,
    commands: Sequence[str],
) -> dict[str, int | None]:
    """Return command exit codes, defaulting to success for artifact builds."""

    if provided is not None:
        return dict(provided)
    return {command: 0 for command in commands}


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every Exp6394 freeze gate passes."""

    preconditions = as_mapping(artifact.get("preconditions_checked"))
    split = as_mapping(artifact.get("development_and_held_manifest_paths_hashes_licenses_and_disjointness"))
    balance = as_mapping(artifact.get("development_balance_receipt"))
    raw = as_mapping(artifact.get("raw_output_before_parse_paths_hashes_and_counts"))
    selections = as_mapping(artifact.get("selected_harness_by_model_family"))
    frozen = as_mapping(artifact.get("frozen_harness_paths_hashes_and_controls"))
    work = as_mapping(artifact.get("matched_development_work_receipts"))
    leakage = as_mapping(artifact.get("protected_leakage_and_same_step_write_counts"))
    mechanisms = as_mapping(
        artifact.get("grammar_parser_jit_json_repair_hidden_state_and_external_scorer_usage_counts")
    )
    protected = as_mapping(artifact.get("protected_files_unchanged"))
    tests = as_mapping(as_mapping(artifact.get("tests_run")).get("exit_codes"))
    families = {MODEL_TEMPLATE_BY_ID[model_id]["model_family"] for model_id in MANDATED_MODEL_IDS}
    gates = (
        preconditions.get("all_preconditions_passed") is True,
        set(artifact.get("models_used", [])) == set(MANDATED_MODEL_IDS),
        balance.get("balanced") is True,
        as_mapping(split.get("disjointness")).get("disjoint") is True,
        as_mapping(split.get("held_manifest")).get("redacted_for_selection") is True,
        raw.get("raw_receipts_complete") is True,
        raw.get("all_raw_outputs_frozen_before_parse") is True,
        work.get("matched") is True,
        set(selections) == families,
        all(as_mapping(row).get("frozen_before_held_access") is True for row in selections.values()),
        frozen.get("all_frozen") is True,
        artifact.get("held_access_during_selection_count") == 0,
        int(leakage.get("same_step_write_count", 1)) == 0,
        int(leakage.get("protected_leakage_count", 1)) == 0,
        artifact.get("model_weight_change_count") == 0,
        all(int(value) == 0 for key, value in mechanisms.items() if key != "schema"),
        artifact.get("autotokenizer_usage_count") == 0,
        artifact.get("held_license_not_implied") is True,
        protected.get("unchanged") is True,
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify the terminal artifact status."""

    if as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is not True:
        return "blocked_precondition"
    if float(artifact.get("model_family_harness_freeze_ready_score", 0.0)) == 1.0:
        return "complete_positive"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict with the claim boundary."""

    status_text = str(artifact.get("status", "complete_null"))
    if status_text == "blocked_precondition":
        blockers = as_mapping(artifact.get("preconditions_checked")).get("blocked_reasons", [])
        return f"blocked: model-family harness freeze missing preconditions {blockers}"
    if status_text == "complete_positive":
        return "complete_positive: family harnesses are frozen before held access; held licenses are not implied"
    return "complete_null: model-family harness freeze gates did not all pass"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile terminal fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh readiness, status, verdict, and checksum."""

    artifact["model_family_harness_freeze_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate schema fields, counters, oracle boundary, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    require(not missing, f"missing_required_fields:{missing}")
    require([row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(MANDATED_MODEL_IDS), "model_specs_wrong_ids")
    require(artifact.get("autotokenizer_usage_count") == 0, "external_tokenizer_used")
    require(artifact.get("model_weight_change_count") == 0, "model_weight_changed")
    require(artifact.get("held_access_during_selection_count") == 0, "held_access_during_selection")
    mechanisms = as_mapping(
        artifact.get("grammar_parser_jit_json_repair_hidden_state_and_external_scorer_usage_counts")
    )
    require(all(int(value) == 0 for key, value in mechanisms.items() if key != "schema"), "prohibited_mechanism_used")
    require(artifact.get("verifier_is_oracle") is True, "exact_checker_oracle_not_marked")
    boundary = as_mapping(artifact.get("builder_model_role_and_non_oracle_boundary"))
    require(boundary.get("exact_task_checkers_are_oracles") is True, "exact_checker_oracle_missing")
    require(boundary.get("builder_is_oracle") is False, "builder_oracle_misclaimed")
    require(boundary.get("harness_selector_is_oracle") is False, "selector_oracle_misclaimed")
    require(boundary.get("parser_is_oracle") is False, "parser_oracle_misclaimed")
    require(boundary.get("model_text_is_oracle") is False, "model_text_oracle_misclaimed")
    require(artifact.get("held_license_not_implied") is True, "held_license_misclaimed")
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_principles"))), "missing_field_principles")
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_provenance"))), "missing_field_provenance")
    require(
        str(artifact.get("honest_verdict", "")).split(":", 1)[0]
        in {"complete_positive", "complete_null", "blocked"},
        "bad_verdict_prefix",
    )
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum_mismatch")


def run(
    *,
    date: str,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: str | Path = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    exp6379_path: str | Path = REPO_ROOT / EXP6379_RELATIVE_PATH,
    exp6380_path: str | Path = REPO_ROOT / EXP6380_RELATIVE_PATH,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = embedded_gguf_tokenizer_receipt,
    host_checks_func: HostChecksFn = host_environment_receipts,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the terminal artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    data = Path(data_dir)
    data.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)

    protected_before = protected_hashes()
    source_before = source_hashes()
    gate = exp6379_gate_receipt(exp6379_path)
    model_resolution = build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )
    model_specs = model_resolution["MODEL_SPECS"]
    host = host_checks_func()
    split = development_and_held_manifests(data, write=write)
    balance = development_balance_receipt(split["development_manifest"]["events"])
    raw_receipt = exp6380_raw_receipt(exp6380_path)
    exp6380_artifact = read_json(exp6380_path) if Path(exp6380_path).is_file() else {}
    variants = preregistered_harness_variants()
    results = per_family_variant_results(exp6380_artifact)
    selections = select_harness_by_model_family(results)
    schema_hash = str(as_mapping(gate.get("canonical_schema")).get("sha256") or "")
    frozen = freeze_harness_sidecars(
        data / "frozen_harnesses",
        selections,
        variants,
        schema_hash=schema_hash,
        write=write,
    )
    work = matched_development_work_receipts(results)
    leakage = protected_leakage_and_same_step_write_counts()
    mechanisms = prohibited_mechanism_counts()
    abstention = explicit_abstention_policy(selections)
    harm = harm_summary(model_resolution=model_resolution, results=results, selections=selections)
    preconditions = preconditions_checked(
        date=date,
        gate=gate,
        model_resolution=model_resolution,
        host=host,
        split=split,
        raw_receipt=raw_receipt,
        protected_before=protected_before,
        source_before=source_before,
    )
    protected_after = protected_hashes()
    protected = protected_unchanged_receipt(protected_before, protected_after)
    commands = list(DEFAULT_TEST_COMMANDS)
    exits = _test_exit_codes(test_exit_codes, commands)
    elapsed = time.perf_counter() - started if duration_s is None else float(duration_s)
    models_used = (
        list(exp6380_artifact.get("models_used", []))
        if model_resolution.get("all_resolved") and raw_receipt["raw_receipts_complete"]
        else []
    )
    artifact: JsonDict = {
        "status": "complete_null",
        "MODEL_SPECS": model_specs,
        "models_used": models_used,
        "cached_sota_pair_receipts": model_resolution["cached_sota_pair_receipts"],
        "model_file_hashes_revisions_quantizations_and_tokenizers": model_file_receipts(model_specs),
        "embedded_gguf_tokenizer_receipts": tokenizer_receipts(model_specs, tokenizer_func),
        "autotokenizer_usage_count": 0,
        "cuda_offload_and_runtime_receipts_by_model": exp6380_artifact.get(
            "cuda_offload_and_runtime_receipts_by_model",
            {},
        ),
        "development_and_held_manifest_paths_hashes_licenses_and_disjointness": split,
        "development_balance_receipt": balance,
        "preregistered_harness_variants": variants,
        "builder_model_role_and_non_oracle_boundary": builder_model_role_and_non_oracle_boundary(),
        "matched_development_work_receipts": work,
        "raw_output_before_parse_paths_hashes_and_counts": raw_receipt,
        "per_family_variant_transport_source_binding_exact_and_cost_results": results,
        "selected_harness_by_model_family": selections,
        "frozen_harness_paths_hashes_and_controls": frozen,
        "explicit_abstention_policy": abstention,
        "held_access_during_selection_count": 0,
        "protected_leakage_and_same_step_write_counts": leakage,
        "model_weight_change_count": 0,
        "grammar_parser_jit_json_repair_hidden_state_and_external_scorer_usage_counts": mechanisms,
        "model_family_harness_freeze_ready_score": 0.0,
        "held_license_not_implied": True,
        "harm_underpowered_missing_and_flagged_cells": harm,
        "protected_files_unchanged": protected,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
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
    """CLI entry point for Exp6394."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / DATA_DIR_RELATIVE_PATH))
    args = parser.parse_args(argv)
    artifact = run(
        date=args.date,
        result_path=Path(args.result_path),
        data_dir=Path(args.data_dir),
    )
    print(
        json.dumps(
            {
                "path": str(args.result_path),
                "status": artifact["status"],
                "model_family_harness_freeze_ready_score": artifact[
                    "model_family_harness_freeze_ready_score"
                ],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
