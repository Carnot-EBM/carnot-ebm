"""Build the V641 versioned source-compilation reuse fixture.

The cache wraps the shipped mention-pointer compiler. It stores public source
compilations only. A separate reducer reads private labels after predictions
exist, which keeps construction authority out of extraction and cache paths.

Spec refs: REQ-VERIFY-7291 and SCENARIO-VERIFY-7291-*.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import yaml

from carnot import experiment_7236_v637_mention_fixture as pointer
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

RUN_DATE = "20260914"
MILESTONE = "2026.09.641"
EXPERIMENT_ID = "exp7291-reuse-fixture"
SCHEMA = "carnot.versioned_source_reuse_fixture.v1"
RANDOM_SEED = 7_291_001
DEVELOPMENT_SEEDS = tuple(7_291_100 + index for index in range(8))
EVALUATION_SEEDS = tuple(7_291_200 + index for index in range(16))
BOOTSTRAP_SEED = 7_291_300
BOOTSTRAP_DRAWS = 10_000

MODEL_SPECS: list[JsonDict] = []
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_in_flight": 0,
    "usable_answers": 0,
}
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

ARMS = ("warm_prefix_direct", "fresh_verifier", "versioned_reuse_verifier")
REQUIRED_CONTROLS = (
    "stale_version_rejection",
    "supported_span_reconstruction",
    "unknown_fields",
    "eviction",
    "fail_closed_extraction",
    "source_shuffle_sensitivity",
    "renamed_entity_invariance",
    "source_id_collision",
    "missing_provenance",
    "deliberately_stale_cache",
)
PARSER_SCHEMA_HASH = (
    "sha256:" + hashlib.sha256(b"carnot.mention_pointer.source_compilation.v1").hexdigest()
)
INJECTED_MODEL_CONFIGURATION = {
    "identity": "cpu_injected_extraction_fixture_v1",
    "temperature": 0.0,
    "schema": "mention_pointer_v1",
    "current_model_invocation": False,
}

RESULT_PATH = Path("results/experiment_7291_v641_reuse_fixture.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7291_v641_reuse_fixture.json")
RAW_DIR = Path("results/raw/experiment_7291_v641_reuse_fixture")
MANIFEST_NAME = "manifest.json"
SCORER_NAME = "scorer-only-labels.json"
ANALYSIS_NAME = "analysis-contract.json"
PREDICTIONS_NAME = "prediction-rows.json"
CONTROLS_NAME = "control-rows.json"
CANDIDATE_NAME = "measured-terminal-candidate.json"
MODULE_PATH = Path("python/carnot/experiment_7291_v641_reuse_fixture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7291_v641_reuse_fixture.py")
TEST_PATH = Path("tests/python/test_experiment_7291_v641_reuse_fixture.py")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("research-references.md"),
    Path("python/carnot/experiment_7275_v640_semantic_replay.py"),
    Path("python/carnot/experiment_7278_v640_source_measurement.py"),
    Path("python/carnot/experiment_7279_v640_source_audit.py"),
    Path("python/carnot/experiment_7236_v637_mention_fixture.py"),
    Path("python/carnot/experiment_7265_v639_mention_heldout.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/check_spec_coverage.py"),
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

REQUIRED_VALIDATION_NAMES = (
    "focused_pytest",
    "affected_suite",
    "full_python_suite",
    "scoped_coverage",
    "scoped_coverage_report",
    "ruff_check",
    "ruff_format",
    "changed_module_mypy",
    "scoped_spec_coverage",
    "independent_raw_reducer",
    "adversarial_verify",
    "verdict_row_consistency",
)

REQUIRED_FIELD_PRINCIPLES = {
    "schema": "Version the artifact; retain ordinary top-level experiment_id and milestone.",
    "status": "Use a terminal complete or blocked record; unfinished own work belongs in separate checkpoints.",
    "run_date": "Use 20260914, real UTC start/end and monotonic timing.",
    "field_principles": "Store explanations here while consumer values remain ordinary top-level values.",
    "preconditions_checked": "Hash actual inputs, authority boundaries, resource ownership and failed checks.",
    "MODEL_SPECS": "Actual executable local model identities; keep historical models in hashed sidecars.",
    "model_invoked": "True for any actual attempted model load or generation, including failed and unusable work.",
    "invocation_counts": "Separate attempted/completed/failed loads and generation; retain in-flight events on timeout.",
    "inference_substrate": "Use the recognized literal for actual computation; never infer from intended task.",
    "inference_substrate_class": "Full generation60s, bounded10s, load-only2s, or actual no-LLM class; never pad elapsed time.",
    "execution_venue": "Host is host; identify actual GPU/native/device execution separately.",
    "duration_s": "Measured monotonic elapsed and disjoint phase spans, including failures and initialization.",
    "random_seed": "Freeze development and independent evaluation seeds before observing outcomes.",
    "reproducibility_checksum": "Bind code, config, inputs, model identity if any and immutable raw evidence.",
    "source_artifact_hashes": "Keep exact producer identities, terminal classes, retirement and quarantine state.",
    "rows": "Every comparative unit/arm/seed with metric, cost, error, abstention and censoring; no aggregate-only claim.",
    "sample_size_budget": "Planned, attempted, complete and censored units plus the frozen stopping rule.",
    "acceptance_gate_results": "Each completeness/value check names expected, observed, passed and principle.",
    "gate_check_summary": "Every blocked_* verdict names upstream/check, exact field, observed and expected value.",
    "verifier_is_oracle": "Expose shared verifier/evaluator authority; same-authority mechanics are not learned correctness.",
    "honest_verdict": "Complete findings start complete_ or complete:; external absence starts blocked_; state the actual finding.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Oracle=true forbids positive; failed efficacy gates forbid positive. Only own unfinished work is partial; unchanged external failure is terminal blocked.",
    "validation_receipts": "Command, exit code, elapsed time and log hash; preserve actual failures.",
    "reuse_fixture_ready_score": "One only for cache, freshness, schema, independent-label and split controls.",
    "fixture_manifest_path": "Immutable development/evaluation source groups and scorer-only label hashes.",
    "cache_key_contract": "Content/version/parser/model keys and conservative dependency invalidation.",
    "comparison_contract": "Exact three arms, fixed call/token ceilings, independent units, costs and preregistered gates.",
    "control_rows": "Every adversarial cache control with observed and expected values.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)

_RELATION = re.compile(
    r"\b(?P<subject>[A-Z][a-z]+)\s+"
    r"(?P<predicate>starts before|precedes|follows)\s+"
    r"(?P<object>[A-Z][a-z]+)\b"
)


def canonical_json(value: Any) -> str:
    """Use one stable JSON spelling for hashes and byte comparisons."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes so parsing cannot hide changed evidence."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks without normalizing its bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind durable evidence while excluding local clock observations."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "timestamps", "phase_spans", "reproducibility_checksum"}
    }
    return sha256_bytes(canonical_json(stable).encode("utf-8"))


def _utc_now() -> str:  # pragma: no cover - a process receipt, not science logic.
    """Record actual UTC time without using it as scientific evidence."""

    return datetime.now(UTC).isoformat()


def _progress(phase: int, event: str, detail: str) -> None:  # pragma: no cover
    """Flush each boundary so an outer monitor can distinguish work from a stall."""

    print(f"[exp7291] phase {phase} {event}: {detail}", flush=True)


def make_document(document_id: str, text: str) -> JsonDict:
    """Build a public document whose mention table comes from its own bytes."""

    return {
        "document_id": document_id,
        "text": text,
        "mentions": pointer.build_mention_table(document_id, text.encode("utf-8")),
    }


def extract_completion(document: Mapping[str, Any], call_type: str) -> JsonDict:
    """Create an injected pointer result from public text only.

    This exact parser stands in for future model extraction. It does not read
    labels. Text outside its narrow grammar becomes an explicit unknown.
    """

    text = str(document.get("text", ""))
    matches = list(_RELATION.finditer(text))
    if call_type == "claim":
        matches = matches[:1]
    elif call_type != "source":
        matches = []
    table = document.get("mentions", [])
    relations: list[JsonDict] = []
    for match in matches:
        by_surface = {
            surface: [row for row in table if row.get("surface_text") == surface]
            for surface in (match["subject"], match["object"])
        }
        if any(len(rows) != 1 for rows in by_surface.values()):
            return {"outcome": "unknown", "relations": []}
        relations.append(
            {
                "subject_pointer": by_surface[match["subject"]][0]["mention_id"],
                "predicate": match["predicate"],
                "object_pointer": by_surface[match["object"]][0]["mention_id"],
                "polarity": "positive",
            }
        )
    if not relations:
        return {"outcome": "unknown", "relations": []}
    return {"outcome": "known", "relations": relations}


def cache_error(error: str, *, invalidated_entries: int = 0) -> JsonDict:
    """Return one stable fail-closed cache result."""

    return {
        "ok": False,
        "cache_hit": False,
        "compiled_source": None,
        "error": error,
        "invalidated_entries": invalidated_entries,
        "evicted_entries": 0,
    }


class SourceCompilationCache:
    """Keep versioned public source compilations behind an opt-in boundary."""

    _REQUIRED = {
        "source_id",
        "source_version",
        "document",
        "completion",
        "parser_schema_hash",
        "model_configuration",
    }

    def __init__(self, *, enabled: bool, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.enabled = enabled
        self.capacity = capacity
        self._entries: OrderedDict[str, JsonDict] = OrderedDict()
        self._current: dict[str, tuple[int, str]] = {}
        self.served_stale_constraints = 0

    def _invalidate(self, source_id: str) -> int:
        keys = [key for key, row in self._entries.items() if row["source_id"] == source_id]
        for key in keys:
            del self._entries[key]
        return len(keys)

    def compile_or_get(self, provenance: Mapping[str, Any]) -> JsonDict:
        """Compile or return one source only when every identity field matches."""

        if not self.enabled:
            return cache_error("cache_disabled")
        if not self._REQUIRED.issubset(provenance):
            return cache_error("missing_provenance")
        source_id = provenance["source_id"]
        version = provenance["source_version"]
        document = provenance["document"]
        configuration = provenance["model_configuration"]
        if (
            not isinstance(source_id, str)
            or not isinstance(version, int)
            or isinstance(version, bool)
            or version < 1
            or not isinstance(document, Mapping)
            or not isinstance(configuration, Mapping)
            or not isinstance(provenance["parser_schema_hash"], str)
        ):
            return cache_error("invalid_provenance")
        text = document.get("text")
        if not isinstance(text, str):
            return cache_error("invalid_provenance")
        digest = sha256_bytes(text.encode("utf-8"))
        current = self._current.get(source_id)
        if current is not None and version < current[0]:
            return cache_error("stale_source_version")
        if current is not None and version == current[0] and digest != current[1]:
            invalidated = self._invalidate(source_id)
            return cache_error("source_id_collision", invalidated_entries=invalidated)
        invalidated = 0
        if current is not None and version > current[0]:
            invalidated = self._invalidate(source_id)
        self._current[source_id] = (version, digest)
        key_payload = {
            "content_digest": digest,
            "source_version": version,
            "parser_schema_hash": provenance["parser_schema_hash"],
            "model_configuration": configuration,
        }
        key = sha256_bytes(canonical_json(key_payload).encode("utf-8"))
        if key in self._entries:
            self._entries.move_to_end(key)
            return {
                "ok": True,
                "cache_hit": True,
                "compiled_source": deepcopy(self._entries[key]["compiled_source"]),
                "error": None,
                "invalidated_entries": invalidated,
                "evicted_entries": 0,
            }
        compiled = pointer.compile_pointer_completion(document, provenance["completion"], "source")
        entry = {
            "cache_key": key_payload,
            "source_id": source_id,
            "source_version": version,
            "compiled_source": deepcopy(compiled),
        }
        self._entries[key] = entry
        evicted = 0
        if len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
            evicted = 1
        return {
            "ok": True,
            "cache_hit": False,
            "compiled_source": deepcopy(compiled),
            "error": None,
            "invalidated_entries": invalidated,
            "evicted_entries": evicted,
        }

    def snapshot(self) -> list[JsonDict]:
        """Expose a copy for schema audits without exposing mutable cache state."""

        return deepcopy(list(self._entries.values()))


def _letters(number: int) -> str:
    """Create stable capitalized entity names without split overlap."""

    chars = []
    for _ in range(5):
        chars.append(chr(ord("a") + number % 26))
        number //= 26
    return "".join(reversed(chars)).capitalize()


def _entity_names(split: str, index: int) -> tuple[str, str, str, str]:
    """Reserve separate name ranges for development and evaluation groups."""

    base = (0 if split == "development" else 10_000) + index * 8
    return tuple(_letters(base + offset) for offset in range(4))  # type: ignore[return-value]


def _gold_span(document: Mapping[str, Any], subject: str, obj: str) -> JsonDict:
    """Record scorer-only spans from construction inputs, not extractor output."""

    text = str(document["text"])
    return {
        "subject_start": len(text[: text.index(subject)].encode("utf-8")),
        "subject_end": len(text[: text.index(subject) + len(subject)].encode("utf-8")),
        "object_start": len(text[: text.index(obj)].encode("utf-8")),
        "object_end": len(text[: text.index(obj) + len(obj)].encode("utf-8")),
    }


def _build_group(split: str, index: int, seed: int) -> tuple[JsonDict, list[JsonDict]]:
    """Build one eight-claim chronology and its separate scorer rows."""

    first, second, third, outside = _entity_names(split, index)
    group_id = f"{'dev' if split == 'development' else 'eval'}-g{index:02d}"
    source_id = f"source-{group_id}"
    source_texts = {
        1: f"{first} precedes {second}.",
        2: f"{second} precedes {first}.",
    }
    sources = {
        version: make_document(f"{source_id}-v{version}", text)
        for version, text in source_texts.items()
    }
    claim_specs = (
        (1, f"at checkpoint one, {first} precedes {second}.", "supported", first, second),
        (
            1,
            f"at checkpoint two, {second} follows {first} by four ticks.",
            "supported",
            second,
            first,
        ),
        (1, f"at checkpoint three, {second} precedes {first}.", "contradicted", second, first),
        (1, f"at checkpoint four, {outside} precedes {first}.", "unknown", outside, first),
        (2, f"at checkpoint five, {second} precedes {first}.", "supported", second, first),
        (2, f"at checkpoint six, {first} precedes {second}.", "contradicted", first, second),
        (
            2,
            f"at checkpoint seven, {second} follows {first} by five ticks.",
            "contradicted",
            second,
            first,
        ),
        (2, f"at checkpoint eight, {outside} follows {first}.", "unknown", outside, first),
    )
    claims = []
    labels = []
    for claim_index, (version, text, label, subject, obj) in enumerate(claim_specs, start=1):
        unit_id = f"{group_id}-u{claim_index:02d}"
        claim = make_document(f"{unit_id}-claim", text)
        claims.append(
            {
                "unit_id": unit_id,
                "chronology_index": claim_index,
                "source_version": version,
                "seed": seed * 100 + claim_index,
                "claim": claim,
            }
        )
        labels.append(
            {
                "unit_id": unit_id,
                "group_id": group_id,
                "split": split,
                "expected_decision": label,
                "source_gold_span": _gold_span(
                    sources[version],
                    first if version == 1 else second,
                    second if version == 1 else first,
                ),
                "claim_gold_span": _gold_span(claim, subject, obj),
            }
        )
    return (
        {
            "group_id": group_id,
            "source_id": source_id,
            "seed": seed,
            "revision_before_claim": 5,
            "source_versions": [
                {"source_version": version, "document": sources[version]} for version in (1, 2)
            ],
            "claims": claims,
        },
        labels,
    )


def build_fixture() -> tuple[JsonDict, JsonDict]:
    """Freeze public groups and keep construction labels in a separate value."""

    development = []
    evaluation = []
    labels: list[JsonDict] = []
    for split, seeds, target in (
        ("development", DEVELOPMENT_SEEDS, development),
        ("evaluation", EVALUATION_SEEDS, evaluation),
    ):
        for index, seed in enumerate(seeds):
            group, group_labels = _build_group(split, index, seed)
            target.append(group)
            labels.extend(group_labels)
    collision_a = make_document("collision-probe-v1", "Xenia precedes Yarrow.")
    collision_b = make_document("collision-probe-v1-mutated", "Yarrow precedes Xenia.")
    public = {
        "schema": "carnot.versioned_source_reuse_public_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "development_groups": development,
        "evaluation_groups": evaluation,
        "collision_probes": [
            {"source_id": "collision-probe", "source_version": 1, "document": collision_a},
            {"source_id": "collision-probe", "source_version": 1, "document": collision_b},
        ],
        "authority_fields_present": False,
    }
    scorer = {
        "schema": "carnot.versioned_source_reuse_scorer_authority.v1",
        "experiment_id": EXPERIMENT_ID,
        "construction_oracle_only": True,
        "readable_by_prediction_path": False,
        "labels": labels,
    }
    return public, scorer


def _source_for(group: Mapping[str, Any], version: int) -> Mapping[str, Any]:
    """Select the one public source version named by a claim."""

    return next(row for row in group["source_versions"] if row["source_version"] == version)


def _provenance(group: Mapping[str, Any], source_row: Mapping[str, Any]) -> JsonDict:
    """Build exact cache provenance from public source data."""

    document = source_row["document"]
    return {
        "source_id": group["source_id"],
        "source_version": source_row["source_version"],
        "document": document,
        "completion": extract_completion(document, "source"),
        "parser_schema_hash": PARSER_SCHEMA_HASH,
        "model_configuration": deepcopy(INJECTED_MODEL_CONFIGURATION),
    }


def _execute_compiled(
    source: Mapping[str, Any], compiled_source: Mapping[str, Any], claim: Mapping[str, Any]
) -> JsonDict:
    """Extract each claim now and execute it against one compiled source."""

    claim_completion = extract_completion(claim, "claim")
    compiled_claim = pointer.compile_pointer_completion(claim, claim_completion, "claim")
    return pointer._execute_compiled_pair(source, compiled_source, compiled_claim)


def _fresh_prediction(source: Mapping[str, Any], claim: Mapping[str, Any]) -> JsonDict:
    """Compile both public documents for one fresh verifier query."""

    compiled_source = pointer.compile_pointer_completion(
        source, extract_completion(source, "source"), "source"
    )
    return _execute_compiled(source, compiled_source, claim)


def reduce_two_draws(first: str, second: str) -> str:
    """Apply the frozen direct rule: disagreement or abstention is unknown."""

    return first if first == second and first in {"supported", "contradicted"} else "unknown"


def _cost(
    arm: str,
    source_bytes: int,
    claim_bytes: int,
    *,
    source_work_charged: bool,
    cache_hit: bool | None,
) -> JsonDict:
    """Count exact public bytes scanned by this CPU fixture arm."""

    if arm == "warm_prefix_direct":
        source_scans, claim_scans, decisions = int(source_work_charged), 2, 2
        lookups = 0
    elif arm == "fresh_verifier":
        source_scans, claim_scans, decisions = 1, 1, 1
        lookups = 0
    else:
        source_scans, claim_scans, decisions = int(source_work_charged), 1, 1
        lookups = 1
    total = source_scans * source_bytes + claim_scans * claim_bytes + decisions + lookups
    return {
        "cost_kind": "cpu_fixture_public_bytes_scanned_plus_operations",
        "source_input_bytes": source_scans * source_bytes,
        "claim_input_bytes": claim_scans * claim_bytes,
        "source_compilations": source_scans if arm != "warm_prefix_direct" else 0,
        "claim_extractions": claim_scans if arm != "warm_prefix_direct" else 0,
        "direct_draws": decisions if arm == "warm_prefix_direct" else 0,
        "cache_lookups": lookups,
        "cache_hit": cache_hit,
        "verification_steps": 0 if arm == "warm_prefix_direct" else 1,
        "total_work_units": total,
        "native_prefix_cache_assumed": False,
    }


def _prediction_row(
    claim_row: Mapping[str, Any],
    group: Mapping[str, Any],
    arm: str,
    result: Mapping[str, Any],
    cost: Mapping[str, Any],
    *,
    draws: Sequence[str] | None = None,
) -> JsonDict:
    """Retain one public prediction without scorer-only fields."""

    return {
        "unit_id": claim_row["unit_id"],
        "group_id": group["group_id"],
        "split": "evaluation",
        "claim_index": claim_row["chronology_index"],
        "source_version": claim_row["source_version"],
        "arm": arm,
        "seed": claim_row["seed"],
        "prediction": result["decision"],
        "draws": list(draws or []),
        "abstention": bool(result["decision"] == "unknown"),
        "errors": list(result.get("errors", [])),
        "censored": False,
        "censoring_reason": None,
        "served_stale_constraints": False,
        "cost": dict(cost),
    }


def _control_rows(public: Mapping[str, Any]) -> list[JsonDict]:
    """Execute cache and semantic attacks outside the primary denominator."""

    group = public["evaluation_groups"][0]
    source_v1 = _source_for(group, 1)["document"]
    source_v2 = _source_for(group, 2)["document"]
    claim_one = group["claims"][0]["claim"]
    cache = SourceCompilationCache(enabled=True, capacity=2)
    first = cache.compile_or_get(_provenance(group, _source_for(group, 1)))
    cache.compile_or_get(_provenance(group, _source_for(group, 2)))
    stale = cache.compile_or_get(_provenance(group, _source_for(group, 1)))

    compiled = first["compiled_source"]
    relation = compiled["relations"][0]
    encoded = source_v1["text"].encode("utf-8")
    spans_ok = (
        encoded[relation["subject_start"] : relation["subject_end"]].decode("utf-8")
        == relation["subject_surface"]
        and encoded[relation["object_start"] : relation["object_end"]].decode("utf-8")
        == relation["object_surface"]
    )

    completion = extract_completion(source_v1, "source")
    extra = deepcopy(completion)
    extra["unknown_field"] = True
    unknown_result = pointer.compile_pointer_completion(source_v1, extra, "source")
    malformed = pointer.compile_pointer_completion(source_v1, {"outcome": "known"}, "source")

    small = SourceCompilationCache(enabled=True, capacity=1)
    small.compile_or_get(_provenance(group, _source_for(group, 1)))
    other_group = public["evaluation_groups"][1]
    eviction = small.compile_or_get(_provenance(other_group, _source_for(other_group, 1)))

    original = _fresh_prediction(source_v1, claim_one)["decision"]
    shuffled_source = _source_for(other_group, 1)["document"]
    shuffled = _fresh_prediction(shuffled_source, claim_one)["decision"]

    old_names = _entity_names("evaluation", 0)
    new_names = ("Xenia", "Yarrow", "Zorin", "Willa")
    renamed_source_text = source_v1["text"]
    renamed_claim_text = claim_one["text"]
    for old, new in zip(old_names, new_names, strict=True):
        renamed_source_text = renamed_source_text.replace(old, new)
        renamed_claim_text = renamed_claim_text.replace(old, new)
    renamed = _fresh_prediction(
        make_document("renamed-source", renamed_source_text),
        make_document("renamed-claim", renamed_claim_text),
    )["decision"]

    collision_cache = SourceCompilationCache(enabled=True, capacity=2)
    collision_rows = public["collision_probes"]
    first_collision = {
        **collision_rows[0],
        "completion": extract_completion(collision_rows[0]["document"], "source"),
        "parser_schema_hash": PARSER_SCHEMA_HASH,
        "model_configuration": deepcopy(INJECTED_MODEL_CONFIGURATION),
    }
    second_collision = {
        **collision_rows[1],
        "completion": extract_completion(collision_rows[1]["document"], "source"),
        "parser_schema_hash": PARSER_SCHEMA_HASH,
        "model_configuration": deepcopy(INJECTED_MODEL_CONFIGURATION),
    }
    collision_cache.compile_or_get(first_collision)
    collision = collision_cache.compile_or_get(second_collision)
    missing = deepcopy(first_collision)
    del missing["parser_schema_hash"]

    stale_claim = group["claims"][5]["claim"]
    stale_prediction = _execute_compiled(source_v1, compiled, stale_claim)["decision"]
    fresh_prediction = _fresh_prediction(source_v2, stale_claim)["decision"]
    values = (
        ("stale_version_rejection", "stale_source_version", stale["error"]),
        ("supported_span_reconstruction", True, spans_ok),
        ("unknown_fields", "completion_shape", unknown_result["errors"][0]),
        ("eviction", 1, eviction["evicted_entries"]),
        ("fail_closed_extraction", "unknown", malformed["outcome"]),
        ("source_shuffle_sensitivity", True, original != shuffled),
        ("renamed_entity_invariance", original, renamed),
        ("source_id_collision", "source_id_collision", collision["error"]),
        (
            "missing_provenance",
            "missing_provenance",
            collision_cache.compile_or_get(missing)["error"],
        ),
        (
            "deliberately_stale_cache",
            True,
            stale_prediction != fresh_prediction,
        ),
    )
    rows = []
    for control, expected, observed in values:
        row = {
            "control": control,
            "expected": expected,
            "observed": observed,
            "passed": observed == expected,
            "labelled_negative_control": control == "deliberately_stale_cache",
            "included_in_primary_rows": False,
        }
        if control == "deliberately_stale_cache":
            row["stale_prediction"] = stale_prediction
            row["fresh_prediction"] = fresh_prediction
        rows.append(row)
    return rows


def execute_public_fixture(public: Mapping[str, Any]) -> JsonDict:
    """Run three arms using public bytes only; labels are not an argument."""

    cache = SourceCompilationCache(enabled=True, capacity=32)
    rows: list[JsonDict] = []
    started = time.monotonic()
    last_progress = started
    cache_hits = cache_misses = invalidations = evictions = 0
    for group_index, group in enumerate(public["evaluation_groups"], start=1):
        direct_seen_versions: set[int] = set()
        for claim_row in group["claims"]:
            version = claim_row["source_version"]
            source_row = _source_for(group, version)
            source = source_row["document"]
            claim = claim_row["claim"]
            source_bytes = len(source["text"].encode("utf-8"))
            claim_bytes = len(claim["text"].encode("utf-8"))

            direct_result = _fresh_prediction(source, claim)
            draws = [direct_result["decision"], direct_result["decision"]]
            direct_decision = reduce_two_draws(*draws)
            rows.append(
                _prediction_row(
                    claim_row,
                    group,
                    "warm_prefix_direct",
                    {"decision": direct_decision, "errors": []},
                    _cost(
                        "warm_prefix_direct",
                        source_bytes,
                        claim_bytes,
                        source_work_charged=version not in direct_seen_versions,
                        cache_hit=None,
                    ),
                    draws=draws,
                )
            )
            direct_seen_versions.add(version)

            fresh = _fresh_prediction(source, claim)
            rows.append(
                _prediction_row(
                    claim_row,
                    group,
                    "fresh_verifier",
                    fresh,
                    _cost(
                        "fresh_verifier",
                        source_bytes,
                        claim_bytes,
                        source_work_charged=True,
                        cache_hit=None,
                    ),
                )
            )

            cached = cache.compile_or_get(_provenance(group, source_row))
            cache_hits += int(cached["cache_hit"])
            cache_misses += int(not cached["cache_hit"])
            invalidations += cached["invalidated_entries"]
            evictions += cached["evicted_entries"]
            reused = _execute_compiled(source, cached["compiled_source"], claim)
            rows.append(
                _prediction_row(
                    claim_row,
                    group,
                    "versioned_reuse_verifier",
                    reused,
                    _cost(
                        "versioned_reuse_verifier",
                        source_bytes,
                        claim_bytes,
                        source_work_charged=not cached["cache_hit"],
                        cache_hit=cached["cache_hit"],
                    ),
                )
            )
        now = time.monotonic()
        if now - last_progress >= 60.0:  # pragma: no cover - fixture finishes sooner.
            _progress(
                3,
                "loop_progress",
                f"completed_groups={group_index}/16 elapsed_s={now - started:.3f}",
            )
            last_progress = now
    return {
        "prediction_rows": rows,
        "control_rows": _control_rows(public),
        "cache_stats": {
            "source_compilations": cache_misses,
            "cache_hits": cache_hits,
            "invalidated_entries": invalidations,
            "evicted_entries": evictions,
            "claim_extractions": 128,
            "served_stale_constraints": cache.served_stale_constraints,
        },
    }


def reduce_rows(
    predictions: Sequence[Mapping[str, Any]], scorer: Mapping[str, Any]
) -> list[JsonDict]:
    """Join predictions to scorer labels only after public execution finishes."""

    labels = {row["unit_id"]: row for row in scorer["labels"] if row["split"] == "evaluation"}
    rows = []
    for prediction in predictions:
        expected = labels[prediction["unit_id"]]["expected_decision"]
        predicted = prediction["prediction"]
        row = deepcopy(dict(prediction))
        row.update(
            {
                "expected_decision": expected,
                "metric": int(predicted == expected),
                "error": None if predicted == expected else "semantic_mismatch",
                "coverage": int(predicted != "unknown"),
                "false_accept": int(
                    predicted in {"supported", "contradicted"} and predicted != expected
                ),
            }
        )
        rows.append(row)
    return rows


def _percentile(values: Sequence[float], probability: float) -> float:
    """Select one deterministic lower empirical percentile."""

    ordered = sorted(values)
    return ordered[int(probability * (len(ordered) - 1))]


def paired_group_bootstrap(rows: Sequence[Mapping[str, Any]], *, seed: int, draws: int) -> JsonDict:
    """Resample sixteen source groups while keeping their claims paired."""

    groups = sorted({str(row["group_id"]) for row in rows})
    by_group_arm: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for group in groups:
        for arm in ARMS:
            by_group_arm[(group, arm)] = sorted(
                [row for row in rows if row["group_id"] == group and row["arm"] == arm],
                key=lambda row: int(row["claim_index"]),
            )

    def mean_metric(sample: Sequence[str], arm: str, metric: str) -> float:
        values = [float(row[metric]) for group in sample for row in by_group_arm[(group, arm)]]
        return sum(values) / len(values)

    def speedup(sample: Sequence[str], claims: int) -> float:
        direct = sum(
            float(row["cost"]["total_work_units"])
            for group in sample
            for row in by_group_arm[(group, "warm_prefix_direct")][:claims]
        )
        reuse = sum(
            float(row["cost"]["total_work_units"])
            for group in sample
            for row in by_group_arm[(group, "versioned_reuse_verifier")][:claims]
        )
        return direct / reuse

    rng = random.Random(seed)
    accuracy_deltas = []
    coverage_deltas = []
    speedups = {claims: [] for claims in (1, 2, 4, 8)}
    for _ in range(draws):
        sample = [rng.choice(groups) for _ in groups]
        accuracy_deltas.append(
            mean_metric(sample, "versioned_reuse_verifier", "metric")
            - mean_metric(sample, "warm_prefix_direct", "metric")
        )
        coverage_deltas.append(
            mean_metric(sample, "versioned_reuse_verifier", "coverage")
            - mean_metric(sample, "warm_prefix_direct", "coverage")
        )
        for claims in speedups:
            speedups[claims].append(speedup(sample, claims))

    original_accuracy = mean_metric(groups, "versioned_reuse_verifier", "metric") - mean_metric(
        groups, "warm_prefix_direct", "metric"
    )
    original_coverage = mean_metric(groups, "versioned_reuse_verifier", "coverage") - mean_metric(
        groups, "warm_prefix_direct", "coverage"
    )
    false_accepts = {
        "reuse": sum(
            int(row["false_accept"]) for row in rows if row["arm"] == "versioned_reuse_verifier"
        ),
        "direct": sum(
            int(row["false_accept"]) for row in rows if row["arm"] == "warm_prefix_direct"
        ),
    }
    return {
        "method": "paired_nonparametric_bootstrap_over_source_groups",
        "resampling_unit": "source_group",
        "independent_groups": len(groups),
        "claims_per_group": 8,
        "draws": draws,
        "seed": seed,
        "interval": "one_sided_95_percent_empirical_lower",
        "accuracy_delta": {
            "estimate": original_accuracy,
            "one_sided_95_lower": _percentile(accuracy_deltas, 0.05),
        },
        "coverage_delta": {
            "estimate": original_coverage,
            "one_sided_95_lower": _percentile(coverage_deltas, 0.05),
        },
        "false_accept_counts": false_accepts,
        "amortization": [
            {
                "claims": claims,
                "paired_total_cost_speedup": speedup(groups, claims),
                "one_sided_95_lower_speedup": _percentile(values, 0.05),
            }
            for claims, values in speedups.items()
        ],
    }


def comparison_contract() -> JsonDict:
    """Freeze the future live comparison without claiming backend cache support."""

    return {
        "arms": [
            {
                "arm": "warm_prefix_direct",
                "calls_per_claim": 2,
                "draw_seeds": [7_291_401, 7_291_402],
                "tie_or_abstention_rule": "disagreement_or_unknown_yields_unknown",
                "current_full_source": True,
                "instruction_order": ["instruction", "source", "claim"],
            },
            {
                "arm": "fresh_verifier",
                "source_extractions_per_claim": 1,
                "claim_extractions_per_claim": 1,
                "verification_per_claim": 1,
            },
            {
                "arm": "versioned_reuse_verifier",
                "source_compilations_per_group": 2,
                "claim_extractions_per_claim": 1,
                "verification_per_claim": 1,
            },
        ],
        "identical_claim_units": 128,
        "independent_source_groups": 16,
        "source_revision_before_claim": 5,
        "max_output_tokens_per_call": 128,
        "call_ceilings": {"direct": 256, "fresh_verifier": 256, "reuse_verifier": 160},
        "total_call_ceiling": 672,
        "retry_count": 0,
        "native_prefix_or_kv_reuse": "allowed_for_both_eligible_paths_if_observed",
        "cost_rule": "report backend counters and complete observed costs; never infer a cache hit",
        "cost_components": [
            "initialization",
            "prefill",
            "source_compilation",
            "claim_extraction",
            "cache_lookup",
            "invalidation",
            "verification",
            "generation",
            "failed_calls",
        ],
    }


def analysis_contract() -> JsonDict:
    """Freeze group-level analysis and its falsifiable pilot boundaries."""

    return {
        "schema": "carnot.versioned_source_reuse_analysis_contract.v1",
        "frozen_before_predictions": True,
        "bootstrap": {
            "method": "paired_nonparametric_bootstrap_over_source_groups",
            "seed": BOOTSTRAP_SEED,
            "draws": BOOTSTRAP_DRAWS,
            "one_sided_confidence": 0.95,
        },
        "gates": {
            "cached_fresh_semantic_mismatches": {"operator": "==", "threshold": 0},
            "served_stale_constraints": {"operator": "==", "threshold": 0},
            "accuracy_delta_lower": {"operator": ">=", "threshold": -0.02},
            "coverage_delta_lower": {"operator": ">=", "threshold": -0.02},
            "reuse_false_accepts_minus_direct": {"operator": "<=", "threshold": 0},
            "eight_claim_cost_speedup_lower": {"operator": ">=", "threshold": 1.5},
        },
        "amortization_claim_counts": [1, 2, 4, 8],
        "rare_error_safety_certified": False,
        "general_source_extraction_correctness_claimed": False,
    }


def _gate(name: str, expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    """Retain both sides and the reason for one frozen gate."""

    return {
        "gate": name,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "principle": principle,
    }


def acceptance_gates(
    rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    cache_stats: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    public: Mapping[str, Any],
) -> list[JsonDict]:
    """Apply readiness and preregistered parity/cost gates separately."""

    by_unit: dict[str, dict[str, str]] = {}
    for row in rows:
        by_unit.setdefault(row["unit_id"], {})[row["arm"]] = row["prediction"]
    mismatches = sum(
        predictions["fresh_verifier"] != predictions["versioned_reuse_verifier"]
        for predictions in by_unit.values()
    )
    control_failures = [row["control"] for row in controls if not row["passed"]]
    public_text = canonical_json(public)
    authority_leaks = [
        field
        for field in ("expected_decision", "gold_span", "construction_label")
        if field in public_text
    ]
    amortized = bootstrap["amortization"][-1]
    return [
        _gate(
            "cache_controls",
            [],
            control_failures,
            not control_failures,
            "Every attack must have its frozen disposition.",
        ),
        _gate(
            "authority_separation",
            [],
            authority_leaks,
            not authority_leaks,
            "Labels cannot define cache or extractor outputs.",
        ),
        _gate(
            "split_counts",
            [8, 16],
            [len(public["development_groups"]), len(public["evaluation_groups"])],
            len(public["development_groups"]) == 8 and len(public["evaluation_groups"]) == 16,
            "Development and evaluation groups must remain disjoint.",
        ),
        _gate(
            "cached_fresh_semantic_parity",
            0,
            mismatches,
            mismatches == 0,
            "Reuse must not change a verifier decision.",
        ),
        _gate(
            "served_stale_constraints",
            0,
            cache_stats["served_stale_constraints"],
            cache_stats["served_stale_constraints"] == 0,
            "Changed sources must never serve old constraints.",
        ),
        _gate(
            "accuracy_delta_lower",
            -0.02,
            bootstrap["accuracy_delta"]["one_sided_95_lower"],
            bootstrap["accuracy_delta"]["one_sided_95_lower"] >= -0.02,
            "Reuse must retain direct accuracy within the preregistered margin.",
        ),
        _gate(
            "coverage_delta_lower",
            -0.02,
            bootstrap["coverage_delta"]["one_sided_95_lower"],
            bootstrap["coverage_delta"]["one_sided_95_lower"] >= -0.02,
            "Reuse must retain direct coverage within the preregistered margin.",
        ),
        _gate(
            "false_accept_count",
            "reuse <= direct",
            bootstrap["false_accept_counts"],
            bootstrap["false_accept_counts"]["reuse"] <= bootstrap["false_accept_counts"]["direct"],
            "A speed result cannot hide more confident errors.",
        ),
        _gate(
            "eight_claim_cost_speedup_lower",
            1.5,
            amortized["one_sided_95_lower_speedup"],
            amortized["one_sided_95_lower_speedup"] >= 1.5,
            "Reuse must repay compilation and lookup cost at eight claims.",
        ),
    ]


def _path_text(path: Path, root: Path) -> str:
    """Use repository-relative paths when possible and exact absolute paths otherwise."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _resolve_recorded_path(value: str, root: Path) -> Path:
    """Resolve either a repository-relative or exact absolute evidence path."""

    path = Path(value)
    return path if path.is_absolute() else root / path


def _write_json(path: Path, value: Mapping[str, Any], *, immutable: bool = False) -> None:
    """Atomically write JSON, or require exact bytes for frozen raw evidence."""

    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable evidence changed: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):  # pragma: no cover - os.replace removes it.
            os.unlink(temporary)


def _manifest_lists_experiment(value: Any, experiment_id: str) -> bool:
    """Find exact identifier fields without matching unrelated prose."""

    if isinstance(value, Mapping):
        if value.get("experiment_id") == experiment_id or value.get("id") == experiment_id:
            return True
        return any(_manifest_lists_experiment(item, experiment_id) for item in value.values())
    if isinstance(value, list):
        return any(_manifest_lists_experiment(item, experiment_id) for item in value)
    return False


def authenticate_inputs(root: Path) -> list[JsonDict]:
    """Authenticate required files and explicit exclusion state before execution."""

    checks = []
    for relative in SOURCE_PATHS:
        path = root / relative
        present = path.is_file()
        checks.append(
            {
                "check": "source_input",
                "upstream": relative.as_posix(),
                "field": "file_state",
                "expected": "present regular file",
                "observed": "present regular file" if present else "missing",
                "passed": present,
                "sha256": sha256_file(path) if present else None,
            }
        )
    exclusion = root / "ops/exclusion_manifest.yaml"
    excluded = False
    if exclusion.is_file():
        parsed = yaml.safe_load(exclusion.read_text(encoding="utf-8"))
        excluded = _manifest_lists_experiment(parsed, EXPERIMENT_ID)
    checks.append(
        {
            "check": "exclusion_state",
            "upstream": "ops/exclusion_manifest.yaml",
            "field": "experiment_id",
            "expected": "not listed",
            "observed": "listed" if excluded else "not listed",
            "passed": not excluded,
            "sha256": sha256_file(exclusion) if exclusion.is_file() else None,
        }
    )
    return checks


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    """Project the first failed prerequisite without discarding its exact values."""

    failed = next((row for row in checks if not row["passed"]), None)
    if failed is None:
        return None
    return {
        "failed_check": failed["check"],
        "upstream": failed["upstream"],
        "artifact_field": failed["field"],
        "expected": failed["expected"],
        "observed": failed["observed"],
    }


def _base_artifact(run_date: str) -> JsonDict:
    """Create required terminal fields before any fallible input read."""

    return {
        "schema": SCHEMA,
        "status": "partial",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "field_principles": deepcopy(REQUIRED_FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "timestamps": {"started_at_utc": None, "completed_at_utc": None},
        "phase_spans": [],
        "random_seed": RANDOM_SEED,
        "split_seeds": {
            "development": list(DEVELOPMENT_SEEDS),
            "evaluation": list(EVALUATION_SEEDS),
            "bootstrap": BOOTSTRAP_SEED,
        },
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_groups": 16,
            "claims_per_group": 8,
            "planned_claim_units": 128,
            "planned_arm_rows": 384,
            "attempted_claim_units": 0,
            "complete_claim_units": 0,
            "censored_claim_units": 0,
            "stopping_rule": "execute each fixed public claim once per arm; no adaptive stop",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": None,
        "verifier_is_oracle": True,
        "honest_verdict": "partial_exp7291_current_work_unfinished",
        "verdict_class": "partial",
        "validation_receipts": [],
        "reuse_fixture_ready_score": 0,
        "fixture_manifest_path": None,
        "cache_key_contract": {},
        "comparison_contract": comparison_contract(),
        "control_rows": [],
    }


def blocked_artifact(
    run_date: str, checks: Sequence[Mapping[str, Any]], duration_s: float
) -> JsonDict:
    """Finish an external absence as blocked rather than unfinished work."""

    artifact = _base_artifact(run_date)
    artifact.update(
        {
            "status": "blocked",
            "preconditions_checked": list(checks),
            "duration_s": duration_s,
            "gate_check_summary": _gate_summary(checks),
            "honest_verdict": "blocked_exp7291_external_prerequisite_missing_or_excluded",
            "verdict_class": "blocked",
        }
    )
    artifact["source_artifact_hashes"] = {
        row["upstream"]: row["sha256"] for row in checks if row.get("sha256")
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def fixture_validation_receipts() -> list[JsonDict]:
    """Create deterministic passing receipts for isolated artifact unit tests."""

    return [
        {
            "name": name,
            "command": f"fixture:{name}",
            "exit_code": 0,
            "passed": True,
            "timed_out": False,
            "duration_s": 0.0,
            "log_path": f"fixture/{name}.log",
            "log_sha256": sha256_bytes(name.encode("utf-8")),
        }
        for name in REQUIRED_VALIDATION_NAMES
    ]


def _validations_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require each named validation exactly once and require it to pass."""

    names = [row.get("name") for row in receipts]
    return sorted(names) == sorted(REQUIRED_VALIDATION_NAMES) and all(
        row.get("passed") is True and row.get("exit_code") == 0 for row in receipts
    )


def _validations_accounted(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one terminal, internally consistent receipt for every command."""

    names = [row.get("name") for row in receipts]
    return sorted(names) == sorted(REQUIRED_VALIDATION_NAMES) and all(
        isinstance(row.get("exit_code"), int) and row.get("passed") is (row.get("exit_code") == 0)
        for row in receipts
    )


def classify_terminal(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Separate a clean fixture from the known repository collection failure."""

    if _validations_pass(receipts):
        return {
            "status": "complete",
            "verdict_class": "circular_positive",
            "honest_verdict": "complete_circular_positive_versioned_reuse_fixture_ready",
            "validation_complete": True,
        }
    failures = [row for row in receipts if row.get("passed") is not True]
    if (
        _validations_accounted(receipts)
        and len(failures) == 1
        and failures[0].get("name") == "full_python_suite"
        and failures[0].get("baseline_failure") is True
    ):
        return {
            "status": "complete",
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_reuse_fixture_ready_but_repository_full_suite_failed",
            "validation_complete": False,
        }
    return {
        "status": "partial",
        "verdict_class": "partial",
        "honest_verdict": "partial_exp7291_validation_failed",
        "validation_complete": False,
    }


def _pending_receipts() -> list[JsonDict]:  # pragma: no cover - runtime candidate only.
    """Show that subprocesses have not yet produced receipts."""

    return [
        {
            "name": name,
            "command": "pending",
            "exit_code": None,
            "passed": False,
            "timed_out": False,
            "duration_s": 0.0,
            "log_path": f"results/raw/experiment_7291_v641_reuse_fixture/validation/{name}.log",
            "log_sha256": "pending",
        }
        for name in REQUIRED_VALIDATION_NAMES
    ]


def build_artifact(
    root: Path,
    run_date: str,
    *,
    raw_dir: Path,
    output_path: Path,
    validation_receipts: Sequence[Mapping[str, Any]],
    started_at_utc: str = "2026-09-14T00:00:00+00:00",
    completed_at_utc: str = "2026-09-14T00:00:01+00:00",
    duration_s: float = 1.0,
) -> JsonDict:
    """Build immutable raw evidence, reduce it, and write one terminal artifact."""

    checks = authenticate_inputs(root)
    if any(not row["passed"] for row in checks):
        artifact = blocked_artifact(run_date, checks, duration_s)
        _write_json(output_path, artifact)
        return artifact

    public, scorer = build_fixture()
    scorer_path = raw_dir / SCORER_NAME
    analysis_path = raw_dir / ANALYSIS_NAME
    manifest_path = raw_dir / MANIFEST_NAME
    predictions_path = raw_dir / PREDICTIONS_NAME
    controls_path = raw_dir / CONTROLS_NAME
    analysis = analysis_contract()
    _write_json(scorer_path, scorer, immutable=True)
    _write_json(analysis_path, analysis, immutable=True)
    public["scorer_authority"] = {
        "path": _path_text(scorer_path, root),
        "sha256": sha256_file(scorer_path),
        "readable_by_prediction_path": False,
    }
    public["analysis_contract"] = {
        "path": _path_text(analysis_path, root),
        "sha256": sha256_file(analysis_path),
        "frozen_before_predictions": True,
    }
    _write_json(manifest_path, public, immutable=True)

    raw = execute_public_fixture(public)
    _write_json(predictions_path, {"rows": raw["prediction_rows"]}, immutable=True)
    _write_json(controls_path, {"rows": raw["control_rows"]}, immutable=True)
    reduced = independent_reduce_raw(raw_dir)
    rows = reduced["rows"]
    bootstrap = paired_group_bootstrap(rows, seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS)
    gates = acceptance_gates(rows, raw["control_rows"], raw["cache_stats"], bootstrap, public)
    readiness_names = {"cache_controls", "authority_separation", "split_counts"}
    ready = all(row["passed"] for row in gates if row["gate"] in readiness_names)

    artifact = _base_artifact(run_date)
    artifact.update(
        {
            "status": "complete",
            "preconditions_checked": checks,
            "duration_s": duration_s,
            "timestamps": {
                "started_at_utc": started_at_utc,
                "completed_at_utc": completed_at_utc,
            },
            "phase_spans": [{"phase": "fixture_and_reduction", "duration_s": duration_s}],
            "source_artifact_hashes": {
                **{row["upstream"]: row["sha256"] for row in checks if row.get("sha256")},
                _path_text(manifest_path, root): sha256_file(manifest_path),
                _path_text(scorer_path, root): sha256_file(scorer_path),
                _path_text(analysis_path, root): sha256_file(analysis_path),
                _path_text(predictions_path, root): sha256_file(predictions_path),
                _path_text(controls_path, root): sha256_file(controls_path),
            },
            "source_provenance": {
                "producer_identities": ["exp7236-mention-fixture", "exp7275-semantic-replay"],
                "terminal_classes": ["complete", "complete"],
                "retired": False,
                "quarantined": False,
                "historical_model_receipts_used_for_current_invocation": False,
            },
            "rows": rows,
            "sample_size_budget": {
                **artifact["sample_size_budget"],
                "attempted_claim_units": 128,
                "complete_claim_units": 128,
                "censored_claim_units": 0,
                "complete_arm_rows": len(rows),
            },
            "acceptance_gate_results": gates,
            "gate_check_summary": None,
            "honest_verdict": "complete_circular_positive_versioned_reuse_fixture_ready",
            "verdict_class": "circular_positive",
            "validation_receipts": list(validation_receipts),
            "reuse_fixture_ready_score": int(ready),
            "fixture_manifest_path": _path_text(manifest_path, root),
            "fixture_manifest_sha256": sha256_file(manifest_path),
            "scorer_authority_path": _path_text(scorer_path, root),
            "scorer_authority_sha256": sha256_file(scorer_path),
            "analysis_contract_path": _path_text(analysis_path, root),
            "analysis_contract_sha256": sha256_file(analysis_path),
            "raw_prediction_rows_path": _path_text(predictions_path, root),
            "raw_prediction_rows_sha256": sha256_file(predictions_path),
            "cache_key_contract": {
                "opt_in_default": False,
                "key_fields": [
                    "content_digest",
                    "source_version",
                    "parser_schema_hash",
                    "exact_model_configuration",
                ],
                "source_id_is_provenance_scope": True,
                "stored_payload": "compiled_source_records_only",
                "labels_cached": False,
                "query_answers_cached": False,
                "claim_compilations_cached": False,
                "changed_source_action": "invalidate_complete_dependent_compilation",
                "stale_version_action": "reject_without_serving",
            },
            "comparison_contract": comparison_contract(),
            "control_rows": raw["control_rows"],
            "cache_statistics": raw["cache_stats"],
            "bootstrap_results": bootstrap,
            "authority_separation": {
                "prediction_function_inputs": ["public_manifest"],
                "scorer_function_inputs": ["prediction_rows", "scorer_only_labels"],
                "construction_oracle_can_feed_cache": False,
                "construction_oracle_can_feed_extractor": False,
            },
            "pilot_limitations": {
                "rare_error_safety_certified": False,
                "arbitrary_prose_compilation_correctness_claimed": False,
                "native_model_cost_claimed": False,
                "external_paper_correctness_imported": False,
            },
            "methodology_note": "All exact outcomes are deterministic construction controls with shared verifier authority, not a learned accuracy claim.",
            "null_delta_methodology_note": "Zero parity deltas are deterministic cache invariants over the fixed CPU fixture.",
            "validation_complete": _validations_pass(validation_receipts),
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact, root=root, require_validations=False)
    if errors:
        raise ValueError(f"terminal artifact invalid: {errors}")
    _write_json(output_path, artifact)
    return artifact


def independent_reduce_raw(raw_dir: Path) -> JsonDict:
    """Rebuild graded rows from immutable public predictions and private labels."""

    manifest = json.loads((raw_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    scorer_path = raw_dir / SCORER_NAME
    analysis_path = raw_dir / ANALYSIS_NAME
    predictions_path = raw_dir / PREDICTIONS_NAME
    errors = []
    if manifest["scorer_authority"]["sha256"] != sha256_file(scorer_path):
        errors.append("scorer_authority_hash")
    if manifest["analysis_contract"]["sha256"] != sha256_file(analysis_path):
        errors.append("analysis_contract_hash")
    scorer = json.loads(scorer_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))["rows"]
    rows = reduce_rows(predictions, scorer)
    return {"rows": rows, "errors": errors}


def validate_artifact(
    artifact: Mapping[str, Any], *, root: Path, require_validations: bool = True
) -> list[str]:
    """Cold-check schema, hashes, denominators, controls, and terminal semantics."""

    errors = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": ZERO_INVOCATION_COUNTS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
    }
    for field, value in expected.items():
        if artifact.get(field) != value:
            errors.append(field)
    if artifact.get("status") == "complete":
        if artifact.get("reuse_fixture_ready_score") != 1:
            errors.append("reuse_fixture_ready_score")
        allowed_classes = {"circular_positive", "disqualified"}
        if artifact.get("verdict_class") not in allowed_classes:
            errors.append("verdict_class")
        rows = artifact.get("rows", [])
        if len(rows) != 384:
            errors.append("rows")
        controls = artifact.get("control_rows", [])
        if [row.get("control") for row in controls] != list(REQUIRED_CONTROLS):
            errors.append("control_rows")
        manifest_value = artifact.get("fixture_manifest_path")
        if not isinstance(manifest_value, str):
            errors.append("fixture_manifest_path")
        else:
            manifest_path = _resolve_recorded_path(manifest_value, root)
            if not manifest_path.is_file() or sha256_file(manifest_path) != artifact.get(
                "fixture_manifest_sha256"
            ):
                errors.append("fixture_manifest_sha256")
        if require_validations:
            terminal = classify_terminal(artifact.get("validation_receipts", []))
            if terminal["status"] != "complete":
                errors.append("validation_receipts")
            for field in ("status", "verdict_class", "honest_verdict", "validation_complete"):
                if artifact.get(field) != terminal[field]:
                    errors.append(field)
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    return list(dict.fromkeys(errors))


def validation_commands(
    root: Path, candidate: Path, raw_dir: Path
) -> list[tuple[str, list[str]]]:  # pragma: no cover
    """Return the exact focused, coverage, full-suite, and raw-evidence checks."""

    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    coverage_file = "/tmp/.coverage-exp7291-v641"
    changed = [MODULE_PATH.as_posix(), WRAPPER_PATH.as_posix(), TEST_PATH.as_posix()]
    return [
        (
            "focused_pytest",
            [
                pytest,
                "-o",
                "addopts=",
                "-n",
                "0",
                "--basetemp=/tmp/exp7291-focused",
                TEST_PATH.as_posix(),
                "-q",
            ],
        ),
        (
            "affected_suite",
            [
                pytest,
                "-o",
                "addopts=",
                "-n",
                "0",
                "--basetemp=/tmp/exp7291-affected",
                "tests/python/test_experiment_7236_v637_mention_fixture.py",
                "-q",
            ],
        ),
        (
            "full_python_suite",
            [
                pytest,
                "-o",
                "addopts=",
                "-n",
                "0",
                "--basetemp=/tmp/exp7291-full",
                "tests/python",
                "-q",
            ],
        ),
        (
            "scoped_coverage",
            [
                python,
                "-u",
                "-m",
                "coverage",
                "run",
                f"--data-file={coverage_file}",
                f"--include=*/{MODULE_PATH.name}",
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "-n",
                "0",
                "--basetemp=/tmp/exp7291-coverage",
                TEST_PATH.as_posix(),
                "-q",
            ],
        ),
        (
            "scoped_coverage_report",
            [
                python,
                "-u",
                "-m",
                "coverage",
                "report",
                f"--data-file={coverage_file}",
                f"--include=*/{MODULE_PATH.name}",
                "--show-missing",
                "--fail-under=100",
            ],
        ),
        ("ruff_check", [python, "-u", "-m", "ruff", "check", *changed]),
        ("ruff_format", [python, "-u", "-m", "ruff", "format", "--check", *changed]),
        ("changed_module_mypy", [python, "-u", "-m", "mypy", MODULE_PATH.as_posix()]),
        (
            "scoped_spec_coverage",
            [python, "-u", "scripts/check_spec_coverage.py", TEST_PATH.as_posix()],
        ),
        (
            "independent_raw_reducer",
            [
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--replay-raw",
                str(raw_dir),
            ],
        ),
        ("adversarial_verify", [python, "-u", "scripts/adversarial_verify.py", str(candidate)]),
        (
            "verdict_row_consistency",
            [python, "-u", "scripts/verdict_row_consistency_lint.py", str(candidate)],
        ),
    ]


def _run_validations(
    root: Path,
    candidate: Path,
    raw_dir: Path,
    preserved: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[JsonDict]:  # pragma: no cover
    """Stream each child and emit a heartbeat while a subprocess remains active."""

    validation_dir = raw_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = f"{root / 'python'}:{root}"
    receipts = []
    commands = validation_commands(root, candidate, raw_dir)
    for index, (name, command) in enumerate(commands, start=1):
        _progress(6, "subprocess_before", f"{name} completed={index - 1}/{len(commands)}")
        if preserved is not None and name in preserved:
            receipts.append(deepcopy(dict(preserved[name])))
            _progress(
                6,
                "subprocess_after",
                f"{name} preserved_receipt=true completed={index}/{len(commands)}",
            )
            continue
        started = time.monotonic()
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        stop = threading.Event()

        def heartbeat() -> None:
            while not stop.wait(45.0):
                _progress(
                    6, "subprocess_heartbeat", f"{name} elapsed_s={time.monotonic() - started:.1f}"
                )

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        lines = []
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            print(f"[exp7291:{name}] {line.rstrip()}", flush=True)
        exit_code = process.wait()
        stop.set()
        thread.join(timeout=1.0)
        log_path = validation_dir / f"{name}.log"
        log_path.write_text("".join(lines), encoding="utf-8")
        receipts.append(
            {
                "name": name,
                "command": shlex.join(command),
                "exit_code": exit_code,
                "passed": exit_code == 0,
                "timed_out": False,
                "duration_s": time.monotonic() - started,
                "log_path": _path_text(log_path, root),
                "log_sha256": sha256_file(log_path),
                "baseline_failure": bool(
                    name == "full_python_suite"
                    and exit_code != 0
                    and "17 errors during collection" in "".join(lines)
                    and "Qwen3.6-35B-A3B-GGUF" in "".join(lines)
                    and "experiment_6966_gguf_gguf_gg_openai" in "".join(lines)
                ),
            }
        )
        _progress(
            6, "subprocess_after", f"{name} exit_code={exit_code} completed={index}/{len(commands)}"
        )
    return receipts


def _preserved_full_suite(
    root: Path, checkpoint_path: Path, raw_dir: Path, candidate: Path
) -> dict[str, JsonDict]:  # pragma: no cover
    """Reuse one exact failed suite receipt so the known baseline runs only once."""

    if not checkpoint_path.is_file():
        return {}
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    matches = [
        row
        for row in checkpoint.get("validation_receipts", [])
        if row.get("name") == "full_python_suite" and row.get("exit_code") == 2
    ]
    if len(matches) != 1:
        return {}
    receipt = deepcopy(matches[0])
    log_path = _resolve_recorded_path(receipt["log_path"], root)
    expected_command = dict(validation_commands(root, candidate, raw_dir))["full_python_suite"]
    if (
        not log_path.is_file()
        or sha256_file(log_path) != receipt.get("log_sha256")
        or receipt.get("command") != shlex.join(expected_command)
    ):
        return {}
    content = log_path.read_text(encoding="utf-8")
    markers = (
        "17 errors during collection",
        "Qwen3.6-35B-A3B-GGUF",
        "experiment_6966_gguf_gguf_gg_openai",
    )
    if not all(marker in content for marker in markers):
        return {}
    receipt["baseline_failure"] = True
    receipt["preserved_from_first_mandated_run"] = True
    return {"full_python_suite": receipt}


def run_experiment(
    root: Path | None = None, run_date: str = RUN_DATE
) -> JsonDict:  # pragma: no cover
    """Authenticate, build raw evidence, validate once, and publish atomically."""

    repository = root or find_repo_root(start=__file__)
    started = time.monotonic()
    started_utc = _utc_now()
    raw_dir = repository / RAW_DIR
    result_path = repository / RESULT_PATH
    checkpoint_path = repository / CHECKPOINT_PATH
    candidate = raw_dir / CANDIDATE_NAME
    preserved = _preserved_full_suite(repository, checkpoint_path, raw_dir, candidate)
    _progress(0, "start", "authenticating inputs and declared output paths")
    checks = authenticate_inputs(repository)
    if any(not row["passed"] for row in checks):
        artifact = blocked_artifact(run_date, checks, time.monotonic() - started)
        _write_json(result_path, artifact)
        _progress(0, "blocked", str(artifact["gate_check_summary"]))
        return artifact
    checkpoint = _base_artifact(run_date)
    checkpoint["preconditions_checked"] = checks
    checkpoint["reproducibility_checksum"] = artifact_checksum(checkpoint)
    _write_json(checkpoint_path, checkpoint)

    _progress(1, "before", "building immutable public and scorer fixtures")
    artifact = build_artifact(
        repository,
        run_date,
        raw_dir=raw_dir,
        output_path=candidate,
        validation_receipts=_pending_receipts(),
        started_at_utc=started_utc,
        completed_at_utc=_utc_now(),
        duration_s=time.monotonic() - started,
    )
    _progress(1, "after", "raw fixture, controls, predictions, and reduction complete")

    _progress(
        2,
        "before",
        "running focused, full-suite, coverage, lint, type, replay, and artifact checks",
    )
    receipts = _run_validations(repository, candidate, raw_dir, preserved)
    _progress(
        2,
        "after",
        f"validation subprocesses complete passed={sum(row['passed'] for row in receipts)}/{len(receipts)}",
    )
    artifact["validation_receipts"] = receipts
    artifact.update(classify_terminal(receipts))
    artifact["duration_s"] = time.monotonic() - started
    artifact["timestamps"]["completed_at_utc"] = _utc_now()
    artifact["phase_spans"] = [{"phase": "complete_run", "duration_s": artifact["duration_s"]}]
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact, root=repository, require_validations=True)
    if errors or artifact["status"] != "complete":
        artifact.update(
            {
                "status": "partial",
                "honest_verdict": "partial_exp7291_validation_failed",
                "verdict_class": "partial",
                "validation_complete": False,
            }
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        _write_json(checkpoint_path, artifact)
        raise RuntimeError(f"validation failed; terminal artifact not published: {errors}")
    _write_json(result_path, artifact)
    _progress(3, "complete", f"published {RESULT_PATH.as_posix()}")
    return artifact


def _date_argument(value: str) -> str:
    """Accept only the fixed V641 execution date."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the CPU fixture or independently replay its raw evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=_date_argument, default=RUN_DATE)
    parser.add_argument("--replay-raw", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.replay_raw is not None:
        _progress(0, "before", "independent raw reduction")
        reduced = independent_reduce_raw(arguments.replay_raw)
        print(
            canonical_json({"rows": len(reduced["rows"]), "errors": reduced["errors"]}), flush=True
        )
        _progress(0, "after", "independent raw reduction")
        return int(bool(reduced["errors"]))
    run_experiment(run_date=arguments.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
