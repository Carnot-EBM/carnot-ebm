"""Build the V637 public mention-pointer fixture without model inference.

Public tables bind stable mention IDs to exact UTF-8 bytes. A narrow adapter
turns predicted pointers, direction, and polarity into the existing typed
relation executor input. Private rows hold construction seeds and labels.

Spec refs: REQ-VERIFY-7236 and SCENARIO-VERIFY-7236-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any

from carnot import experiment_7208_v635_span_fixture as v635
from carnot.experiment_artifacts import atomic_write_json
from carnot.paths import repo_root as find_repo_root
from carnot.verify.experiment_7195_source_relation_executor import (
    EntityBinding,
    TypedRelation,
    execute_relation,
)


JsonDict = dict[str, Any]

RUN_DATE = "20260912"
MILESTONE = "2026.09.637"
EXPERIMENT_ID = "exp7236-mention-fixture"
RANDOM_SEED = 7_236_001
MODEL_SPECS: list[JsonDict] = []
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

ARMS = ("original_offset", "explicit_schema_offset_control", "mention_pointer")
CONDITIONS = ("supported", "reversed", "joint_support", "missing_support")
SPLIT_COUNTS = {"calibration": 8, "held_out": 64}
RELATION_PHRASES = {"calibration": "precedes", "held_out": "starts before"}
TOKEN_CAPS = {"source": 384, "claim": 128}

RESULT_PATH = Path("results/experiment_7236_v637_mention_fixture.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7236_v637_mention_fixture.json")
RAW_DIR = Path("results/raw/experiment_7236")
PUBLIC_MANIFEST_PATH = RAW_DIR / "public_manifest.json"
AUTHORITY_MANIFEST_PATH = RAW_DIR / "authority_manifest.json"
MODULE_PATH = Path("python/carnot/experiment_7236_v637_mention_fixture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7236_v637_mention_fixture.py")
TEST_PATH = Path("tests/python/test_experiment_7236_v637_mention_fixture.py")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")

SOURCE_PATHS = {
    "agents": Path("AGENTS.md"),
    "claude": Path("CLAUDE.md"),
    "codex": Path("CODEX.md"),
    "research_program": Path("research-program.md"),
    "research_references": Path("research-references.md"),
    "exclusion_manifest": Path("ops/exclusion_manifest.yaml"),
    "e2e_test_plan": Path("ops/e2e-test-plan.md"),
    "v636_fixture_module": Path("python/carnot/experiment_7222_v636_span_fixture.py"),
    "v636_canary_module": Path("python/carnot/experiment_7223_v636_span_canary.py"),
    "v635_fixture_module": Path("python/carnot/experiment_7208_v635_span_fixture.py"),
    "typed_executor": Path("python/carnot/verify/experiment_7195_source_relation_executor.py"),
    "exp7222_artifact": Path("results/experiment_7222_v636_span_fixture.json"),
    "exp7223_artifact": Path("results/experiment_7223_v636_span_canary.json"),
    "spec": SPEC_PATH,
    "module": MODULE_PATH,
    "entrypoint": WRAPPER_PATH,
    "focused_tests": TEST_PATH,
    "adversarial_verifier": Path("scripts/adversarial_verify.py"),
    "row_consistency_lint": Path("scripts/verdict_row_consistency_lint.py"),
}

REQUIRED_FIELD_PRINCIPLES = {
    "schema": "Version the artifact and bind experiment_id and milestone to this task.",
    "status": "Terminal complete or blocked only; unfinished work uses a separate checkpoint path.",
    "run_date": "Use 20260912; retain actual UTC start and end timestamps.",
    "field_principles": "Keep ordinary values at top level; put their explanations in this map.",
    "preconditions_checked": "Observed paths, resources, model identity and upstream checks before expensive work.",
    "inference_substrate": "Use the recognized literal for the operation actually executed.",
    "inference_substrate_class": "Actual class determines the duration floor; never pad duration or relabel to pass.",
    "execution_venue": "Top-level orchestration is host; board rows name kv260, gatemate or polarfire.",
    "execution_host": "Actual hostname, distinct from execution_venue.",
    "duration_s": "Measured monotonic elapsed work; record phase spans separately.",
    "MODEL_SPECS": "Models actually invoked; [] for tasks with no LLM.",
    "model_invoked": "Current task execution only; historical sources and injected fixtures are separate.",
    "source_artifact_hashes": "Hash source code, public inputs, private evaluator inputs and raw output files.",
    "rows": "Every comparison retains one row per independent unit and arm, with errors and abstentions.",
    "sample_size_budget": "Predeclared independent units, attempted/completed/censored units, and stopping rule.",
    "random_seed": "Freeze seeds and schedules before observing evaluation labels.",
    "reproducibility_checksum": "Hash the exact settings, inputs and raw rows supporting the result.",
    "gate_check_summary": "Every blocked_* verdict names check, upstream, artifact_field, expected and observed value.",
    "verifier_is_oracle": "True when the verification authority also defines correctness; separate code is insufficient independence.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. External incompleteness is blocked.",
    "honest_verdict": "Completed findings start complete_ or complete:. External absence starts blocked_. Failed acceptance forbids positive.",
    "acceptance_gate_results": "Preserve each frozen criterion, actual value and pass/fail independently of task completion.",
    "mention_fixture_ready_score": "One requires deterministic mention resolution, sealed splits, leakage guards and failing semantic mutants.",
    "public_manifest_path": "results/raw/experiment_7236/public_manifest.json with public rows and hashes.",
    "authority_manifest_path": "results/raw/experiment_7236/authority_manifest.json inaccessible to model/controller input.",
    "arm_contract": "Three fixed interfaces with identical calibration inputs, budget, examples and unknown support.",
    "mutation_rows": "Each structural and semantic counterexample retains its expected and actual decision.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_POINTER_RELATION_FIELDS = {
    "subject_pointer",
    "predicate",
    "object_pointer",
    "polarity",
}


def canonical_json(value: Any) -> str:
    """Use one stable Unicode JSON spelling for all byte receipts."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Name SHA-256 beside the digest so its algorithm stays explicit."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_file(path: Path) -> str:
    """Hash file bytes without changing text encoding or line endings."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def unwrap_principle_value(value: Any) -> Any:
    """Unwrap only the exact annotation shape, never an ordinary data map."""

    if isinstance(value, Mapping) and set(value) == {"principle", "value"}:
        return value["value"]
    return value


def _is_quarantined(value: Mapping[str, Any]) -> bool:
    """Recognize explicit quarantine fields without treating any map as true."""

    return any(
        unwrap_principle_value(value.get(field)) is True
        for field in ("flagged_adversarial", "quarantined", "fabricated")
    )


def upstream_acceptable(value: Mapping[str, Any]) -> bool:
    """Accept a ready upstream only when no quarantine marker overrides it."""

    score = unwrap_principle_value(
        value.get("mention_fixture_ready_score", value.get("span_fixture_ready_score"))
    )
    return score == 1 and not _is_quarantined(value)


def _progress(phase: int, event: str, detail: str) -> None:
    """Flush each observed boundary so external monitoring sees real progress."""

    print(f"[exp7236] phase {phase} {event}: {detail}", flush=True)


def _utc_now() -> str:
    """Return one actual UTC observation for the execution receipt."""

    return datetime.now(UTC).isoformat()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash reproducible evidence while excluding process-local clock values."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key
        not in {
            "duration_s",
            "timestamps",
            "phase_spans",
            "reproducibility_checksum",
        }
    }
    return sha256_bytes(canonical_json(stable).encode("utf-8"))


def build_mention_table(document_id: str, public_bytes: bytes) -> list[JsonDict]:
    """Derive stable mention IDs and exact spans from public UTF-8 bytes only."""

    text = public_bytes.decode("utf-8")
    mentions: list[JsonDict] = []
    for match in _WORD.finditer(text):
        surface = match.group(0)
        first = surface[0]
        if not (first.isupper() or (ord(first) > 127 and first.isalpha())):
            continue
        start = len(text[: match.start()].encode("utf-8"))
        end = len(text[: match.end()].encode("utf-8"))
        mentions.append(
            {
                "document_id": document_id,
                "mention_id": f"m{len(mentions):03d}",
                "byte_start": start,
                "byte_end": end,
                "surface_text": surface,
            }
        )
    return mentions


def resolve_pointer(
    public_bytes: bytes, mention_table: Sequence[Mapping[str, Any]], pointer: str
) -> JsonDict | None:
    """Resolve one unique valid pointer or return unknown without guessing."""

    matches = [row for row in mention_table if row.get("mention_id") == pointer]
    if len(matches) != 1:
        return None
    row = matches[0]
    start, end, surface = row.get("byte_start"), row.get("byte_end"), row.get("surface_text")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not isinstance(surface, str)
        or not 0 <= start < end <= len(public_bytes)
    ):
        return None
    try:
        observed = public_bytes[start:end].decode("utf-8")
    except UnicodeDecodeError:
        return None
    return dict(row) if observed == surface else None


def _unknown(*errors: str) -> JsonDict:
    """Return one typed abstention and retain each compiler reason once."""

    return {"outcome": "unknown", "relations": [], "errors": list(dict.fromkeys(errors))}


def compile_pointer_completion(
    document: Mapping[str, Any], completion: Any, call_type: str
) -> JsonDict:
    """Resolve untrusted public pointers while leaving semantics model-selected."""

    if call_type not in TOKEN_CAPS:
        return _unknown("call_type")
    if not isinstance(completion, Mapping) or set(completion) != {"outcome", "relations"}:
        return _unknown("completion_shape")
    outcome, relations = completion.get("outcome"), completion.get("relations")
    if outcome not in {"known", "unknown"} or not isinstance(relations, list):
        return _unknown("completion_shape")
    if outcome == "unknown":
        return _unknown("unknown_with_relations" if relations else "explicit_unknown")
    limit = 4 if call_type == "source" else 1
    if not 1 <= len(relations) <= limit:
        return _unknown("relation_count")
    text = document.get("text")
    table = document.get("mentions")
    document_id = document.get("document_id")
    if not isinstance(text, str) or not isinstance(table, list) or not isinstance(document_id, str):
        return _unknown("document_shape")
    encoded = text.encode("utf-8")
    ranges = v635._sentence_ranges(encoded)
    compiled: list[JsonDict] = []
    for relation in relations:
        if not isinstance(relation, Mapping) or set(relation) != _POINTER_RELATION_FIELDS:
            return _unknown("relation_shape")
        if relation["predicate"] not in v635.PREDICATES:
            return _unknown("unsupported_predicate")
        if relation["polarity"] not in {"positive", "negative"}:
            return _unknown("invalid_polarity")
        subject = resolve_pointer(encoded, table, str(relation["subject_pointer"]))
        obj = resolve_pointer(encoded, table, str(relation["object_pointer"]))
        if subject is None or obj is None:
            return _unknown("unresolved_pointer")
        if subject.get("document_id") != document_id or obj.get("document_id") != document_id:
            return _unknown("cross_document_pointer")
        sentence_indexes = [
            index
            for index, (start, end) in enumerate(ranges)
            if start <= subject["byte_start"] < subject["byte_end"] <= end
            and start <= obj["byte_start"] < obj["byte_end"] <= end
        ]
        if len(sentence_indexes) != 1:
            return _unknown("pointer_sentence_mismatch")
        sentence_index = sentence_indexes[0]
        start, end = ranges[sentence_index]
        compiled.append(
            {
                "sentence_index": sentence_index,
                "subject_start": subject["byte_start"],
                "subject_end": subject["byte_end"],
                "subject_surface": subject["surface_text"],
                "predicate": relation["predicate"],
                "object_start": obj["byte_start"],
                "object_end": obj["byte_end"],
                "object_surface": obj["surface_text"],
                "polarity": relation["polarity"],
                "relation_start": start,
                "relation_end": end,
            }
        )
    return {"outcome": "known", "relations": compiled, "errors": []}


def _execute_compiled_pair(
    source: Mapping[str, Any], compiled_source: Mapping[str, Any], compiled_claim: Mapping[str, Any]
) -> JsonDict:
    """Send compiled mention surfaces through the existing typed executor."""

    if compiled_source.get("outcome") != "known" or compiled_claim.get("outcome") != "known":
        errors = [*compiled_source.get("errors", []), *compiled_claim.get("errors", [])]
        return {"decision": "unknown", "abstention": True, "errors": errors or ["unknown"]}
    source_bytes = str(source["text"]).encode("utf-8")
    source_relations = list(compiled_source["relations"])
    binding_rows: dict[str, JsonDict] = {}
    for relation in source_relations:
        for role in ("subject", "object"):
            surface = relation[f"{role}_surface"]
            row = {
                "surface": surface,
                "start": relation[f"{role}_start"],
                "end": relation[f"{role}_end"],
            }
            binding_rows.setdefault(surface, row)
    bindings = tuple(
        EntityBinding(surface, surface, row["start"], row["end"])
        for surface, row in sorted(binding_rows.items())
    )

    edges: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for relation in source_relations:
        family, subject, obj = v635._canonical_relation(
            relation["predicate"], relation["subject_surface"], relation["object_surface"]
        )
        if relation["polarity"] == "positive":
            edges[family].add((subject, obj))
    derived: list[JsonDict] = []
    for family, family_edges in edges.items():
        changed = True
        while changed:
            additions = {
                (left, right)
                for left, middle in family_edges
                for other_middle, right in family_edges
                if middle == other_middle and left != right
            } - family_edges
            changed = bool(additions)
            family_edges.update(additions)
        existing = {
            (relation["subject_surface"], relation["object_surface"])
            for relation in source_relations
            if relation["predicate"] == family and relation["polarity"] == "positive"
        }
        derived.extend(
            {
                "subject_surface": subject,
                "predicate": family,
                "object_surface": obj,
                "polarity": "positive",
            }
            for subject, obj in sorted(family_edges - existing)
        )
    typed_source = tuple(
        TypedRelation(
            relation["subject_surface"],
            relation["predicate"],
            relation["object_surface"],
            relation["polarity"],
            0,
            len(source_bytes),
        )
        for relation in [*source_relations, *derived]
    )
    claim = compiled_claim["relations"][0]
    typed_claim = TypedRelation(
        claim["subject_surface"],
        claim["predicate"],
        claim["object_surface"],
        claim["polarity"],
        0,
        max(1, len(source_bytes)),
    )
    result = execute_relation(source_bytes, bindings, typed_source, typed_claim)
    return {
        "decision": result.decision,
        "abstention": result.abstention,
        "errors": list(result.uncertainty_reasons),
    }


def execute_pointer_pair(
    source: Mapping[str, Any],
    claim: Mapping[str, Any],
    source_completion: Any,
    claim_completion: Any,
) -> JsonDict:
    """Compile two separately visible calls and execute their typed relation."""

    compiled_source = compile_pointer_completion(source, source_completion, "source")
    compiled_claim = compile_pointer_completion(claim, claim_completion, "claim")
    return _execute_compiled_pair(source, compiled_source, compiled_claim)


def _pointer_to_offsets(document: Mapping[str, Any], completion: Mapping[str, Any]) -> JsonDict:
    """Reconstruct the retired numeric interface for a fixed diagnostic control."""

    compiled = compile_pointer_completion(document, completion, "source")
    call_type = "source"
    if len(completion.get("relations", [])) == 1 and str(document.get("document_id", "")).endswith(
        "claim"
    ):
        compiled = compile_pointer_completion(document, completion, "claim")
        call_type = "claim"
    if compiled["outcome"] != "known":
        return {"outcome": "unknown", "relations": []}
    relations = [
        {
            "sentence_index": row["sentence_index"],
            "subject_start": row["subject_start"],
            "subject_end": row["subject_end"],
            "predicate": row["predicate"],
            "object_start": row["object_start"],
            "object_end": row["object_end"],
            "polarity": row["polarity"],
        }
        for row in compiled["relations"]
    ]
    raw = {"outcome": "known", "relations": relations}
    encoded = str(document["text"]).encode("utf-8")
    grammar = v635.compile_grammar(encoded, call_type, "reference")
    return v635.compile_completion(encoded, raw, call_type, grammar["grammar_sha256"])


def _execute_arm(
    arm: str,
    source: Mapping[str, Any],
    claim: Mapping[str, Any],
    source_completion: Mapping[str, Any],
    claim_completion: Mapping[str, Any],
) -> JsonDict:
    """Run one frozen representation without changing its semantic prediction."""

    if arm == "mention_pointer":
        return execute_pointer_pair(source, claim, source_completion, claim_completion)
    compiled_source = _pointer_to_offsets(source, source_completion)
    compiled_claim = _pointer_to_offsets(claim, claim_completion)
    return _execute_compiled_pair(source, compiled_source, compiled_claim)


def _letters(number: int) -> str:
    """Encode an integer as fixed lowercase letters for parser-safe entity names."""

    chars = []
    for _ in range(5):
        chars.append(chr(ord("a") + number % 26))
        number //= 26
    return "".join(reversed(chars))


def _entity_names(split: str, index: int, twin: bool) -> tuple[str, str, str]:
    """Create disjoint deterministic names for one base and its audit twin."""

    prefix = {
        ("calibration", False): "C",
        ("calibration", True): "D",
        ("held_out", False): "H",
        ("held_out", True): "J",
    }[(split, twin)]
    start = index * 3
    return tuple(prefix + _letters(start + role) for role in range(3))  # type: ignore[return-value]


def _unit_id(split: str, index: int) -> str:
    """Create an opaque stable row identity without exposing its condition."""

    digest = hashlib.sha256(f"{RANDOM_SEED}:{split}:{index}".encode()).hexdigest()[:20]
    return f"u-{digest}"


def _document(document_id: str, text: str) -> JsonDict:
    """Package public text with its table derived only from those same bytes."""

    return {
        "document_id": document_id,
        "text": text,
        "mentions": build_mention_table(document_id, text.encode("utf-8")),
    }


def _surface_pointer(document: Mapping[str, Any], surface: str, sentence: int) -> str:
    """Select the unique surface occurrence in one generated sentence."""

    encoded = str(document["text"]).encode("utf-8")
    start, end = v635._sentence_ranges(encoded)[sentence]
    matches = [
        row
        for row in document["mentions"]
        if row["surface_text"] == surface and start <= row["byte_start"] < row["byte_end"] <= end
    ]
    if len(matches) != 1:
        raise ValueError("generated mention is not unique inside its sentence")
    return str(matches[0]["mention_id"])


def _gold_completion(
    document: Mapping[str, Any], relations: Sequence[tuple[str, str, str, str, int]]
) -> JsonDict:
    """Store generated positive-control relations only in the private authority."""

    return {
        "outcome": "known",
        "relations": [
            {
                "subject_pointer": _surface_pointer(document, subject, sentence),
                "predicate": predicate,
                "object_pointer": _surface_pointer(document, obj, sentence),
                "polarity": polarity,
            }
            for subject, predicate, obj, polarity, sentence in relations
        ],
    }


def _texts(
    condition: str, names: tuple[str, str, str], predicate: str
) -> tuple[str, str, list[tuple[str, str, str, str, int]], str]:
    """Render one controlled condition and its private expected decision."""

    subject, middle, obj = names
    if condition == "joint_support":
        return (
            f"{subject} {predicate} {middle}. {middle} {predicate} {obj}.",
            f"{subject} {predicate} {obj}.",
            [(subject, predicate, middle, "positive", 0), (middle, predicate, obj, "positive", 1)],
            "supported",
        )
    source = f"{subject} {predicate} {middle}."
    if condition == "supported":
        claim, expected = f"{subject} {predicate} {middle}.", "supported"
    elif condition == "reversed":
        claim, expected = f"{middle} {predicate} {subject}.", "contradicted"
    else:
        claim, expected = f"{subject} {predicate} {obj}.", "unknown"
    return source, claim, [(subject, predicate, middle, "positive", 0)], expected


def arm_contract() -> JsonDict:
    """Freeze equal calibration inputs and budgets across three interfaces."""

    unit_ids = [_unit_id("calibration", index) for index in range(SPLIT_COUNTS["calibration"])]
    examples = [{"outcome": "unknown", "relations": []}]
    rows: JsonDict = {}
    for arm in ARMS:
        rows[arm] = {
            "calibration_unit_ids": unit_ids,
            "examples": examples,
            "temperature": 0.0,
            "token_caps": deepcopy(TOKEN_CAPS),
            "unknown_allowed": True,
            "reprompt_allowed": False,
            "model_predicts_numeric_offsets": arm != "mention_pointer",
            "interface": "mention_ids" if arm == "mention_pointer" else "numeric_byte_offsets",
        }
    return rows


def build_fixture() -> tuple[JsonDict, JsonDict]:
    """Build 72 sealed bases and paired twins before any outcome observation."""

    public_rows: list[JsonDict] = []
    authority_rows: list[JsonDict] = []
    counts = {split: {condition: 0 for condition in CONDITIONS} for split in SPLIT_COUNTS}
    for split, count in SPLIT_COUNTS.items():
        predicate = RELATION_PHRASES[split]
        for index in range(count):
            condition = CONDITIONS[index % len(CONDITIONS)]
            unit_id = _unit_id(split, index)
            public_variants: list[JsonDict] = []
            private_variants: list[JsonDict] = []
            vocabulary: list[str] = []
            for twin in (False, True):
                variant = "twin" if twin else "original"
                names = _entity_names(split, index, twin)
                vocabulary.extend(names)
                source_text, claim_text, source_relations, expected = _texts(
                    condition, names, predicate
                )
                source = _document(f"{unit_id}-{variant}-source", source_text)
                claim = _document(f"{unit_id}-{variant}-claim", claim_text)
                public_variants.append({"variant": variant, "source": source, "claim": claim})
                claim_relation = [(names[0], predicate, names[1], "positive", 0)]
                if condition == "reversed":
                    claim_relation = [(names[1], predicate, names[0], "positive", 0)]
                elif condition in {"joint_support", "missing_support"}:
                    claim_relation = [(names[0], predicate, names[2], "positive", 0)]
                private_variants.append(
                    {
                        "variant": variant,
                        "exact_label": expected,
                        "gold_source_completion": _gold_completion(source, source_relations),
                        "gold_claim_completion": _gold_completion(claim, claim_relation),
                    }
                )
            public_rows.append({"unit_id": unit_id, "split": split, "variants": public_variants})
            authority_rows.append(
                {
                    "unit_id": unit_id,
                    "split": split,
                    "condition_key": condition,
                    "generation_seed": RANDOM_SEED
                    + index
                    + (0 if split == "calibration" else 10_000),
                    "relation_phrase": predicate,
                    "entity_vocabulary": vocabulary,
                    "variants": private_variants,
                }
            )
            counts[split][condition] += 1
    public = {
        "schema": "carnot.exp7236.public_mentions.v1",
        "split_base_counts": deepcopy(SPLIT_COUNTS),
        "independent_base_count": sum(SPLIT_COUNTS.values()),
        "surface_variants_per_base": 2,
        "arm_contract": arm_contract(),
        "rows": public_rows,
    }
    authority = {
        "schema": "carnot.exp7236.private_authority.v1",
        "random_seed": RANDOM_SEED,
        "condition_counts": counts,
        "rows": authority_rows,
    }
    return public, authority


def execute_fixture(public: Mapping[str, Any], authority: Mapping[str, Any]) -> list[JsonDict]:
    """Run every independent base and arm while keeping twin evidence paired."""

    private_by_id = {row["unit_id"]: row for row in authority["rows"]}
    rows: list[JsonDict] = []
    started = time.monotonic()
    last_report = started
    for base_index, public_row in enumerate(public["rows"], 1):
        private = private_by_id[public_row["unit_id"]]
        public_variants = {row["variant"]: row for row in public_row["variants"]}
        private_variants = {row["variant"]: row for row in private["variants"]}
        for arm in ARMS:
            results: dict[str, JsonDict] = {}
            permuted_results: dict[str, JsonDict] = {}
            for variant in ("original", "twin"):
                visible = public_variants[variant]
                hidden = private_variants[variant]
                results[variant] = _execute_arm(
                    arm,
                    visible["source"],
                    visible["claim"],
                    hidden["gold_source_completion"],
                    hidden["gold_claim_completion"],
                )
                permuted = deepcopy(visible)
                permuted["source"]["mentions"].reverse()
                permuted["claim"]["mentions"].reverse()
                permuted_results[variant] = _execute_arm(
                    arm,
                    permuted["source"],
                    permuted["claim"],
                    hidden["gold_source_completion"],
                    hidden["gold_claim_completion"],
                )
            expected = private_variants["original"]["exact_label"]
            passed = (
                results["original"]["decision"] == results["twin"]["decision"] == expected
                and permuted_results == results
            )
            rows.append(
                {
                    "unit_id": public_row["unit_id"],
                    "base_id": public_row["unit_id"],
                    "split": public_row["split"],
                    "arm": arm,
                    "seed": RANDOM_SEED,
                    "metric": int(passed),
                    "error": None if passed else "positive_control_or_equivariance_failure",
                    "abstention": results["original"]["abstention"],
                    "prediction": results["original"]["decision"],
                    "twin_prediction": results["twin"]["decision"],
                    "expected_prediction": expected,
                    "permutation_equivariant": permuted_results == results,
                }
            )
        now = time.monotonic()
        if now - last_report >= 60:
            _progress(4, "heartbeat", f"completed={base_index}/72 elapsed_s={now - started:.1f}")
            last_report = now
    return rows


def mutation_rows() -> list[JsonDict]:
    """Execute structural and semantic counterexamples through the real adapter."""

    source = _document("mutation-source", "Aster precedes Brin.")
    claim = _document("mutation-claim", "Aster precedes Brin.")
    source_gold = _gold_completion(source, [("Aster", "precedes", "Brin", "positive", 0)])
    claim_gold = _gold_completion(claim, [("Aster", "precedes", "Brin", "positive", 0)])

    def known(subject: str, predicate: str, obj: str, polarity: str = "positive") -> JsonDict:
        return {
            "outcome": "known",
            "relations": [
                {
                    "subject_pointer": subject,
                    "predicate": predicate,
                    "object_pointer": obj,
                    "polarity": polarity,
                }
            ],
        }

    cases: list[tuple[str, JsonDict, JsonDict, JsonDict, JsonDict, str]] = []
    reverse_claim = _document("reverse-claim", "Brin precedes Aster.")
    cases.append(
        (
            "wrong_direction",
            source,
            reverse_claim,
            source_gold,
            _gold_completion(reverse_claim, [("Brin", "precedes", "Aster", "positive", 0)]),
            "contradicted",
        )
    )
    cases.append(
        (
            "reversed_polarity",
            source,
            claim,
            source_gold,
            known("m000", "precedes", "m001", "negative"),
            "contradicted",
        )
    )
    cases.append(
        (
            "dangling_pointer",
            source,
            claim,
            known("missing", "precedes", "m001"),
            claim_gold,
            "unknown",
        )
    )
    ambiguous = deepcopy(source)
    ambiguous["mentions"].append(deepcopy(ambiguous["mentions"][0]))
    cases.append(("ambiguous_pointer", ambiguous, claim, source_gold, claim_gold, "unknown"))
    outside = deepcopy(source)
    outside["mentions"][0]["byte_end"] = len(source["text"].encode("utf-8")) + 1
    cases.append(("out_of_range_pointer", outside, claim, source_gold, claim_gold, "unknown"))
    missing_claim = _document("missing-claim", "Aster precedes Cora.")
    cases.append(
        (
            "missing_support",
            source,
            missing_claim,
            source_gold,
            _gold_completion(missing_claim, [("Aster", "precedes", "Cora", "positive", 0)]),
            "unknown",
        )
    )
    rows: list[JsonDict] = []
    for name, source_doc, claim_doc, source_completion, claim_completion, expected in cases:
        result = execute_pointer_pair(source_doc, claim_doc, source_completion, claim_completion)
        rows.append(
            {
                "mutation": name,
                "expected_decision": expected,
                "actual_decision": result["decision"],
                "error": None if result["decision"] == expected else "unexpected_decision",
                "abstention": result["abstention"],
                "passed": result["decision"] == expected,
            }
        )
    permuted = deepcopy(source)
    permuted["mentions"].reverse()
    permuted_result = execute_pointer_pair(permuted, claim, source_gold, claim_gold)
    rows.append(
        {
            "mutation": "mention_permutation",
            "expected_decision": "supported",
            "actual_decision": permuted_result["decision"],
            "error": None if permuted_result["decision"] == "supported" else "not_equivariant",
            "abstention": permuted_result["abstention"],
            "passed": permuted_result["decision"] == "supported",
        }
    )
    compiled = compile_pointer_completion(source, source_gold, "source")
    first = compiled["relations"][0]
    exact = (
        source["text"].encode("utf-8")[first["subject_start"] : first["subject_end"]] == b"Aster"
        and source["text"].encode("utf-8")[first["object_start"] : first["object_end"]] == b"Brin"
    )
    rows.append(
        {
            "mutation": "exact_offset_reconstruction",
            "expected_decision": "supported",
            "actual_decision": "supported" if exact else "unknown",
            "error": None if exact else "offset_mismatch",
            "abstention": not exact,
            "passed": exact,
        }
    )
    return rows


def _public_privacy_ok(public: Mapping[str, Any]) -> bool:
    """Prove model-visible bytes contain no private evaluator field name."""

    text = canonical_json(public)
    forbidden = ("condition_key", "exact_label", "generation_seed", "gold_relation")
    return not any(token in text for token in forbidden)


def _acceptance_rows(
    public: Mapping[str, Any],
    authority: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    mutations: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Keep every readiness criterion separate from the terminal verdict."""

    calibration_vocab = {
        name
        for row in authority["rows"]
        if row["split"] == "calibration"
        for name in row["entity_vocabulary"]
    }
    held_vocab = {
        name
        for row in authority["rows"]
        if row["split"] == "held_out"
        for name in row["entity_vocabulary"]
    }
    expected_counts = {
        "calibration": {condition: 2 for condition in CONDITIONS},
        "held_out": {condition: 16 for condition in CONDITIONS},
    }
    checks = {
        "public_privacy": _public_privacy_ok(public),
        "sealed_balanced_splits": authority["condition_counts"] == expected_counts,
        "joint_phrase_vocabulary_holdout": calibration_vocab.isdisjoint(held_vocab)
        and RELATION_PHRASES["calibration"] != RELATION_PHRASES["held_out"],
        "complete_rows": len(rows) == 216 and all(row["metric"] == 1 for row in rows),
        "permutation_equivariance": all(row["permutation_equivariant"] for row in rows),
        "exact_offset_reconstruction": next(
            row for row in mutations if row["mutation"] == "exact_offset_reconstruction"
        )["passed"],
        "semantic_mutants": all(row["passed"] for row in mutations),
        "arm_contract": set(public["arm_contract"]) == set(ARMS),
        "no_model_invocation": MODEL_SPECS == [],
    }
    return [
        {"criterion": criterion, "expected_value": True, "actual_value": passed, "passed": passed}
        for criterion, passed in checks.items()
    ]


def _gate(
    check: str, upstream: str | None, field: str, expected: Any, observed: Any, passed: bool
) -> JsonDict:
    """Retain both compared values for every precondition observation."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def _gate_summary(failure: Mapping[str, Any] | None) -> JsonDict:
    """Project the first failed check into one stable terminal summary."""

    if failure is None:
        return {
            "passed": True,
            "failed_check": None,
            "upstream": None,
            "artifact_field": None,
            "expected_value": "all_preconditions_pass",
            "observed_value": "all_preconditions_pass",
        }
    return {
        "passed": False,
        "failed_check": failure["check"],
        "upstream": failure["upstream"],
        "artifact_field": failure["artifact_field"],
        "expected_value": failure["expected_value"],
        "observed_value": failure["observed_value"],
    }


def _paths(output_root: Path) -> dict[str, Path]:
    """Resolve task-owned raw, checkpoint, and terminal destinations."""

    return {
        "public": output_root / PUBLIC_MANIFEST_PATH,
        "authority": output_root / AUTHORITY_MANIFEST_PATH,
        "checkpoint": output_root / CHECKPOINT_PATH,
        "result": output_root / RESULT_PATH,
    }


def _source_paths(root: Path, overrides: Mapping[str, Path] | None) -> dict[str, Path]:
    """Resolve exact sources while tests can isolate one missing prerequisite."""

    paths = {name: root / relative for name, relative in SOURCE_PATHS.items()}
    for name, path in (overrides or {}).items():
        paths[name] = path if path.is_absolute() else root / path
    return paths


def _source_hashes(paths: Mapping[str, Path]) -> JsonDict:
    """Bind every cited source path before CPU fixture execution starts."""

    return {
        name: sha256_file(path) if path.is_file() else "missing" for name, path in paths.items()
    }


def _preconditions(
    root: Path, run_date: str, output_root: Path, overrides: Mapping[str, Path] | None
) -> tuple[list[JsonDict], dict[str, Path]]:
    """Authenticate sources, historical quarantine, imports, and output paths."""

    checks: list[JsonDict] = []
    sources = _source_paths(root, overrides)
    checks.append(_gate("run_date", None, "run_date", RUN_DATE, run_date, run_date == RUN_DATE))
    for name, path in sources.items():
        exists = path.is_file() and os.access(path, os.R_OK)
        checks.append(_gate("source_exists", name, "path", "readable_file", str(path), exists))
        if not exists:
            return checks, sources
    try:
        fixture = json.loads(sources["exp7222_artifact"].read_text(encoding="utf-8"))
        canary = json.loads(sources["exp7223_artifact"].read_text(encoding="utf-8"))
        spec = sources["spec"].read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        checks.append(
            _gate("source_parse", None, "json_or_utf8", "valid", type(exc).__name__, False)
        )
        return checks, sources
    checks.append(
        _gate(
            "driving_spec",
            "verification_spec",
            "REQ-VERIFY-7236",
            True,
            "REQ-VERIFY-7236" in spec,
            "REQ-VERIFY-7236" in spec,
        )
    )
    checks.append(
        _gate(
            "upstream_fixture_authentic",
            "exp7222",
            "span_fixture_ready_score",
            1,
            unwrap_principle_value(fixture.get("span_fixture_ready_score")),
            upstream_acceptable(fixture),
        )
    )
    historical_rejected = _is_quarantined(canary) and not upstream_acceptable(canary)
    checks.append(
        _gate(
            "historical_quarantine_rejected",
            "exp7223",
            "flagged_adversarial",
            True,
            unwrap_principle_value(canary.get("flagged_adversarial")),
            historical_rejected,
        )
    )
    for name, path in _paths(output_root).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        writable = os.access(path.parent, os.W_OK)
        checks.append(
            _gate(
                "output_destination",
                name,
                "parent",
                "writable_directory",
                str(path.parent),
                writable,
            )
        )
    return checks, sources


def _base_artifact(run_date: str) -> JsonDict:
    """Create a schema-complete value before any fallible source check."""

    return {
        "schema": {
            "name": "carnot.experiment_7236_v637_mention_fixture",
            "version": 1,
            "experiment_id": EXPERIMENT_ID,
            "milestone": MILESTONE,
        },
        "status": "running",
        "run_date": run_date,
        "field_principles": deepcopy(REQUIRED_FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_independent_units": 72,
            "planned_rows": 216,
            "attempted_units": 0,
            "completed_units": 0,
            "censored_units": 72,
            "surface_variants_per_base": 2,
            "stopping_rule": "execute all 8 calibration and 64 held-out bases once in each arm",
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "pending",
        "gate_check_summary": _gate_summary(None),
        "verifier_is_oracle": True,
        "verdict_class": "partial",
        "honest_verdict": "partial_exp7236_running",
        "acceptance_gate_results": [],
        "mention_fixture_ready_score": 0,
        "public_manifest_path": PUBLIC_MANIFEST_PATH.as_posix(),
        "authority_manifest_path": AUTHORITY_MANIFEST_PATH.as_posix(),
        "arm_contract": arm_contract(),
        "mutation_rows": [],
        "timestamps": {"started_at_utc": _utc_now(), "completed_at_utc": None},
        "phase_spans": {},
        "validation_command_rows": [],
        "scope_limit": "controlled_corpus_fixture_only_no_free_form_benchmark_generalization",
    }


def _seal(artifact: JsonDict, path: Path, started: float) -> None:
    """Refresh measured duration and checksum before one atomic JSON write."""

    artifact["duration_s"] = time.monotonic() - started
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    atomic_write_json(path, artifact, allow_override=True, env={}, sort_keys=True)


def validate_artifact(
    artifact: object, root: Path | None = None, *, output_root: Path | None = None
) -> list[str]:
    """Cold-replay required fields, raw bytes, semantics, gates, and checksum."""

    if not isinstance(artifact, Mapping):
        return ["artifact_mapping"]
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if artifact.get("field_principles") != REQUIRED_FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_contract")
    if artifact.get("execution_venue") != EXECUTION_VENUE or not artifact.get("execution_host"):
        errors.append("execution_identity")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    duration = artifact.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    if artifact.get("status") == "blocked":
        summary = artifact.get("gate_check_summary")
        if (
            artifact.get("inference_substrate") != "blocked_no_run"
            or artifact.get("inference_substrate_class") != "blocked_no_run"
            or artifact.get("verdict_class") != "blocked"
            or artifact.get("mention_fixture_ready_score") != 0
        ):
            errors.append("blocked_terminal_state")
        if not isinstance(summary, Mapping) or summary.get("passed") is not False:
            errors.append("gate_check_summary")
        return errors
    if artifact.get("status") != "complete":
        errors.append("status")
        return errors
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS:
        errors.append("inference_substrate_class")
    if (
        artifact.get("verdict_class") != "circular_positive"
        or artifact.get("mention_fixture_ready_score") != 1
    ):
        errors.append("readiness_terminal_state")
    budget = artifact.get("sample_size_budget")
    if not isinstance(budget, Mapping) or (
        budget.get("completed_units"),
        budget.get("censored_units"),
        budget.get("planned_rows"),
    ) != (72, 0, 216):
        errors.append("sample_size_budget")
    destination = output_root or root or find_repo_root(start=__file__)
    paths = _paths(destination)
    try:
        public = json.loads(paths["public"].read_text(encoding="utf-8"))
        authority = json.loads(paths["authority"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        errors.append("sealed_fixture_files")
        return errors
    expected_public, expected_authority = build_fixture()
    if public != expected_public:
        errors.append("public_manifest")
    if authority != expected_authority:
        errors.append("authority_manifest")
    expected_rows = execute_fixture(public, authority)
    if artifact.get("rows") != expected_rows:
        errors.append("rows")
    expected_mutations = mutation_rows()
    if artifact.get("mutation_rows") != expected_mutations:
        errors.append("mutation_rows")
    expected_acceptance = _acceptance_rows(public, authority, expected_rows, expected_mutations)
    if artifact.get("acceptance_gate_results") != expected_acceptance or not all(
        row["passed"] for row in expected_acceptance
    ):
        errors.append("acceptance_gate_results")
    if artifact.get("arm_contract") != arm_contract():
        errors.append("arm_contract")
    hashes = artifact.get("source_artifact_hashes")
    repo = root or find_repo_root(start=__file__)
    source_paths = _source_paths(repo, None)
    expected_hashes = _source_hashes(source_paths)
    expected_hashes.update(
        {
            "public_manifest": sha256_file(paths["public"]),
            "authority_manifest": sha256_file(paths["authority"]),
        }
    )
    if hashes != expected_hashes:
        errors.append("source_artifact_hashes")
    return errors


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_root: Path | None = None,
    path_overrides: Mapping[str, Path] | None = None,
) -> JsonDict:
    """Execute one CPU fixture or preserve one exact external prerequisite block."""

    started = time.monotonic()
    destination = output_root or root
    paths = _paths(destination)
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    phase = time.monotonic()
    _progress(0, "start", "write schema-complete checkpoint before source checks")
    artifact = _base_artifact(run_date)
    _seal(artifact, paths["checkpoint"], started)
    artifact["phase_spans"]["phase_0_s"] = time.monotonic() - phase
    _progress(0, "end", str(paths["checkpoint"]))

    phase = time.monotonic()
    _progress(1, "start", "authenticate exact sources, historical quarantine, and destinations")
    checks, sources = _preconditions(root, run_date, destination, path_overrides)
    artifact["preconditions_checked"] = checks
    artifact["source_artifact_hashes"] = _source_hashes(sources)
    failure = next((row for row in checks if row["passed"] is not True), None)
    artifact["phase_spans"]["phase_1_s"] = time.monotonic() - phase
    if failure is not None:
        artifact["status"] = "blocked"
        artifact["verdict_class"] = "blocked"
        artifact["honest_verdict"] = f"blocked_exp7236_{failure['check']}"
        artifact["gate_check_summary"] = _gate_summary(failure)
        artifact["timestamps"]["completed_at_utc"] = _utc_now()
        _seal(artifact, paths["checkpoint"], started)
        _progress(1, "end", f"blocked={failure['check']}")
        _progress(7, "validation_start", "cold-check exact external block")
        validation_errors = validate_artifact(artifact, root, output_root=destination)
        _progress(7, "validation_end", f"errors={validation_errors}")
        if validation_errors:
            raise ValueError(f"invalid blocked Exp7236 artifact: {validation_errors}")
        _progress(8, "write_start", "atomically write blocked terminal artifact")
        _seal(artifact, paths["result"], started)
        _progress(8, "write_end", str(paths["result"]))
        return artifact
    _seal(artifact, paths["checkpoint"], started)
    _progress(1, "end", "all exact preconditions passed; quarantined history rejected")

    phase = time.monotonic()
    _progress(2, "start", "freeze three equal calibration interfaces")
    artifact["arm_contract"] = arm_contract()
    artifact["phase_spans"]["phase_2_s"] = time.monotonic() - phase
    _seal(artifact, paths["checkpoint"], started)
    _progress(2, "end", "arms=3 calibration_bases=8 unknown_allowed=true")

    phase = time.monotonic()
    _progress(3, "start", "build public mention bytes and private evaluator authority")
    public, authority = build_fixture()
    atomic_write_json(paths["public"], public, allow_override=True, env={}, sort_keys=True)
    atomic_write_json(paths["authority"], authority, allow_override=True, env={}, sort_keys=True)
    artifact["source_artifact_hashes"].update(
        {
            "public_manifest": sha256_file(paths["public"]),
            "authority_manifest": sha256_file(paths["authority"]),
        }
    )
    artifact["phase_spans"]["phase_3_s"] = time.monotonic() - phase
    _seal(artifact, paths["checkpoint"], started)
    _progress(3, "end", "sealed calibration=8 held_out=64 twins=72")

    phase = time.monotonic()
    _progress(4, "benchmark_start", "run pointer compiler, offset controls, and typed executor")
    artifact["rows"] = execute_fixture(public, authority)
    artifact["sample_size_budget"].update(
        {"attempted_units": 72, "completed_units": 72, "censored_units": 0}
    )
    artifact["inference_substrate"] = INFERENCE_SUBSTRATE
    artifact["inference_substrate_class"] = INFERENCE_SUBSTRATE_CLASS
    artifact["phase_spans"]["phase_4_s"] = time.monotonic() - phase
    _seal(artifact, paths["checkpoint"], started)
    _progress(4, "benchmark_end", "completed_units=72 comparison_rows=216")

    phase = time.monotonic()
    _progress(5, "start", "inject structural and semantic counterexamples")
    artifact["mutation_rows"] = mutation_rows()
    artifact["phase_spans"]["phase_5_s"] = time.monotonic() - phase
    _seal(artifact, paths["checkpoint"], started)
    _progress(5, "end", f"mutations={len(artifact['mutation_rows'])}")

    phase = time.monotonic()
    _progress(6, "start", "reduce privacy, split, reconstruction, and semantic gates")
    artifact["acceptance_gate_results"] = _acceptance_rows(
        public, authority, artifact["rows"], artifact["mutation_rows"]
    )
    artifact["mention_fixture_ready_score"] = int(
        all(row["passed"] for row in artifact["acceptance_gate_results"])
    )
    artifact["status"] = "complete"
    artifact["verdict_class"] = (
        "circular_positive" if artifact["mention_fixture_ready_score"] else "disqualified"
    )
    artifact["honest_verdict"] = (
        "complete_circular_positive_mention_fixture_ready_oracle_control_no_generalization_claim"
        if artifact["mention_fixture_ready_score"]
        else "complete_disqualified_mention_fixture_not_ready"
    )
    artifact["gate_check_summary"] = _gate_summary(None)
    artifact["timestamps"]["completed_at_utc"] = _utc_now()
    artifact["phase_spans"]["phase_6_s"] = time.monotonic() - phase
    _seal(artifact, paths["checkpoint"], started)
    _progress(6, "end", f"ready={artifact['mention_fixture_ready_score']}")

    _progress(7, "validation_start", "cold-replay raw bytes, semantic rows, gates, and checksum")
    validation_errors = validate_artifact(artifact, root, output_root=destination)
    _progress(7, "validation_end", f"errors={validation_errors}")
    if validation_errors:
        raise ValueError(f"invalid Exp7236 artifact: {validation_errors}")
    _progress(8, "write_start", "atomically write stable checkpoint and terminal deliverable")
    _seal(artifact, paths["checkpoint"], started)
    _seal(artifact, paths["result"], started)
    _progress(8, "write_end", str(paths["result"]))
    return artifact


def _date_argument(value: str) -> str:
    """Accept only the execution date fixed by the V637 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Build the fixture and return success only after cold validation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    args = parser.parse_args(argv)
    root = find_repo_root(start=__file__)
    artifact = build_artifact(root, args.date)
    errors = validate_artifact(artifact, root)
    if errors:
        print(f"[exp7236] invalid artifact: {errors}", file=sys.stderr, flush=True)
        return 1
    print(
        f"[exp7236] complete verdict={artifact['honest_verdict']} "
        f"score={artifact['mention_fixture_ready_score']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin wrapper exercises main.
    raise SystemExit(main())
