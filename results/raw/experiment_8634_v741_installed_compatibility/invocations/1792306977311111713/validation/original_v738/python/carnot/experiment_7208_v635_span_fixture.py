"""Build the finite V635 source-span relation fixture.

The module turns controlled public text into bounded span tuples, validates two
decoding contracts, and then executes exact relation semantics. A separate
text interpreter supplies authority labels so the candidate path cannot grade
itself merely by returning its own output.

Spec refs: REQ-VERIFY-7208 and SCENARIO-VERIFY-7208-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any

import yaml

from carnot.experiment_artifacts import atomic_write_bytes, atomic_write_json
from carnot.experiment_7196_v634_qwen_atomic_capture import (
    artifact_checksum as exp7196_checksum,
)
from carnot.experiment_7197_v634_grounding_value_audit import (
    artifact_checksum as exp7197_checksum,
)
from carnot.paths import repo_root as find_repo_root
from carnot.verify.experiment_7195_source_relation_executor import (
    EntityBinding,
    TypedRelation,
    execute_relation,
)


JsonDict = dict[str, Any]
Tokenize = Callable[[bytes], Sequence[int]]
VocabularyLoader = Callable[..., Any]

RUN_DATE = "20260911"
RANDOM_SEED = 7_208_001
SURFACE_SEED = 7_208_002
MODEL_SPECS: list[JsonDict] = []
INFERENCE_SUBSTRATE = "exact_span_compilation_and_relation_closure_with_upstream_aggregation"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"
AUTHORITY_IMPORTS_CANDIDATE = False

RESULT_PATH = Path("results/experiment_7208_v635_span_fixture.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7208_v635_span_fixture.json")
FIXTURE_DIR = Path("results/fixtures/experiment_7208")
PUBLIC_VIEW_PATH = FIXTURE_DIR / "public.jsonl"
AUTHORITY_SIDECAR_PATH = FIXTURE_DIR / "authority.jsonl"
FIXTURE_MANIFEST_PATH = FIXTURE_DIR / "manifest.json"
MODULE_PATH = Path("python/carnot/experiment_7208_v635_span_fixture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7208_v635_span_fixture.py")
TEST_PATH = Path("tests/python/test_experiment_7208_v635_span_fixture.py")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
EXP7196_PATH = Path("results/experiment_7196_v634_qwen_atomic_capture.json")
EXP7197_PATH = Path("results/experiment_7197_v634_grounding_value_audit.json")
RAW_MANIFEST_PATH = Path("results/raw/experiment_7196_v634_qwen_atomic_capture/raw_manifest.json")

FAMILIES = ("precedes", "starts before", "ends before", "occurs before")
INVERSE_PREDICATES = ("follows", "starts after", "ends after", "occurs after")
PREDICATES = FAMILIES + INVERSE_PREDICATES
VARIANTS = ("supported", "reversal", "joint_support", "support_removed")
SPLIT_BASE_COUNTS = {"canary": 8, "development": 8, "test": 64}
TOKEN_BUDGETS = {"source": 160, "claim": 80}
MODEL_SETTINGS = {
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "seed": 7208001,
    "cache_prompt": False,
}

RELATION_FIELDS = (
    "sentence_index",
    "subject_start",
    "subject_end",
    "predicate",
    "object_start",
    "object_end",
    "polarity",
)
TUPLE_SCHEMA = {
    "outcome": ["known", "unknown"],
    "relations": {
        "item_fields": list(RELATION_FIELDS),
        "source_min_max": [1, 4],
        "claim_min_max": [1, 1],
        "unknown_requires_empty": True,
        "offset_unit": "utf8_bytes_half_open",
        "predicates": list(PREDICATES),
        "polarities": ["positive", "negative"],
    },
}

REQUIRED_FIELD_PRINCIPLES = {
    "field_principles": "Echo the reason for each field beside its actual evidence.",
    "status": "Write a terminal artifact only after completion or a diagnosed external block.",
    "run_date": "Use 20260911; do not substitute an upstream experiment date.",
    "preconditions_checked": "Record the actual resource, code and gate observations.",
    "inference_substrate": "Describe executed computation, not the intended workload.",
    "inference_substrate_class": "The actual operation determines its duration floor.",
    "execution_venue": "Use exactly host, kv260, gatemate or polarfire; these tasks execute on host.",
    "execution_host": "Put the actual hostname here, never inside execution_venue.",
    "duration_s": "Measure monotonic work time; do not pad it to pass a floor.",
    "source_artifact_hashes": "Bind source code, input data and frozen contracts to the claim.",
    "rows": "Keep unit_id, arm, seed, metric, error and abstention for each comparison; do not replace numeric rows with a task roster.",
    "sample_size_budget": "Retain planned, attempted, completed, censored and independent-unit counts.",
    "random_seed": "Freeze stochastic choices before held-out outcomes are read.",
    "reproducibility_checksum": "Hash the inputs, code, settings and raw rows.",
    "gate_check_summary": "Every blocked_* verdict names failed check, upstream, field, expected and observed value.",
    "verifier_is_oracle": "Same correctness authority remains circular even with a separate implementation.",
    "verdict_class": "Use exactly positive | circular_positive | null | blocked | disqualified | partial; only incomplete own work can be partial.",
    "honest_verdict": "Use complete_ or complete: for completed findings; blocked_* for external absence. Readiness is not scientific value.",
    "span_fixture_ready_score": "One means the complete sealed panel and representation contract are usable.",
    "public_view_path": "The model reads only source and claim text plus allowed span references.",
    "authority_sidecar_path": "Independent labels stay outside inference and decoding.",
    "split_manifest": "Related variants stay together and no held-out label tunes the method.",
    "grammar_contract": "The same bounded schema supports controlled syntax and reference arms.",
    "lexical_control_rows": "A frozen lexical control exposes easy or saturated benchmark cells.",
    "span_mutation_rows": "Adversarial references and semantics test the actual compiler/executor boundary.",
    "MODEL_SPECS": "Empty because this task invokes no LLM.",
    "model_invoked": "False; upstream inference is only cited.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)

SOURCE_PATHS = {
    "agents": Path("AGENTS.md"),
    "codex": Path("CODEX.md"),
    "claude": Path("CLAUDE.md"),
    "research_program": Path("research-program.md"),
    "research_references": Path("research-references.md"),
    "exclusion_manifest": EXCLUSION_PATH,
    "e2e_test_plan": Path("ops/e2e-test-plan.md"),
    "constraint_spec": SPEC_PATH,
    "exp7195_module": Path("python/carnot/experiment_7195_v634_typed_grounding.py"),
    "exp7195_executor": Path("python/carnot/verify/experiment_7195_source_relation_executor.py"),
    "exp7196_module": Path("python/carnot/experiment_7196_v634_qwen_atomic_capture.py"),
    "exp7197_module": Path("python/carnot/experiment_7197_v634_grounding_value_audit.py"),
    "exp7196_artifact": EXP7196_PATH,
    "exp7197_artifact": EXP7197_PATH,
    "raw_manifest": RAW_MANIFEST_PATH,
    "module": MODULE_PATH,
    "entrypoint": WRAPPER_PATH,
    "focused_tests": TEST_PATH,
    "spec_coverage_tool": Path("scripts/check_spec_coverage.py"),
    "adversarial_tool": Path("scripts/adversarial_verify.py"),
    "row_lint_tool": Path("scripts/verdict_row_consistency_lint.py"),
}

_CLAUSE = re.compile(
    rb"(?P<subject>[A-Z][a-z]+) (?P<negative>does not )?"
    rb"(?P<predicate>precedes|precede|follows|follow|starts before|start before|"
    rb"starts after|start after|ends before|end before|ends after|end after|"
    rb"occurs before|occur before|occurs after|occur after) (?P<object>[A-Z][a-z]+)\."
)
_AUTHORITY_CLAUSE = re.compile(
    r"^(?P<subject>[A-Z][a-z]+) (?P<negative>does not )?"
    r"(?P<predicate>precedes|precede|follows|follow|starts before|start before|"
    r"starts after|start after|ends before|end before|ends after|end after|"
    r"occurs before|occur before|occurs after|occur after) (?P<object>[A-Z][a-z]+)\.$"
)
_INVERSE = dict(zip(INVERSE_PREDICATES, FAMILIES, strict=True))
_SURFACE_TO_PREDICATE = {
    "precede": "precedes",
    "follow": "follows",
    "start before": "starts before",
    "start after": "starts after",
    "end before": "ends before",
    "end after": "ends after",
    "occur before": "occurs before",
    "occur after": "occurs after",
}


def canonical_json(value: Any) -> str:
    """Use one compact JSON spelling so every byte receipt can be replayed."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize exact rows with one final newline for stable public files."""

    return ("\n".join(canonical_json(row) for row in rows) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Name the digest algorithm beside the exact byte hash."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_file(path: Path) -> str:
    """Hash source bytes without changing their encoding or line endings."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind scientific evidence while excluding process-local duration."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_bytes(canonical_json(stable).encode("utf-8"))


def _progress(phase: int, event: str, detail: str) -> None:
    """Flush observed phase state so an outer monitor can distinguish work from a stall."""

    print(f"[exp7208] phase {phase} {event}: {detail}", flush=True)


def _unwrap(value: Any) -> Any:
    """Unwrap only the exact principle/value form, never an arbitrary mapping."""

    if isinstance(value, Mapping) and set(value) == {"principle", "value"}:
        return value["value"]
    return value


def _is_quarantined(artifact: Mapping[str, Any]) -> bool:
    """Reject explicit structured quarantine without treating mappings as truthy flags."""

    for field in ("flagged_adversarial", "quarantined", "fabricated"):
        observed = _unwrap(artifact.get(field))
        if observed is True or (
            isinstance(observed, str) and observed.lower() in {"true", "quarantined"}
        ):
            return True
    return False


def _manifest_hits(value: Any, wanted: set[str]) -> bool:
    """Find an excluded upstream identifier in nested manifest records."""

    if isinstance(value, Mapping):
        return any(_manifest_hits(item, wanted) for item in value.values())
    if isinstance(value, list):
        return any(_manifest_hits(item, wanted) for item in value)
    return isinstance(value, str) and value in wanted


def _gate(
    check: str,
    upstream: str | None,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Retain both sides of each precondition instead of hiding a failure."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def _gate_summary(failure: Mapping[str, Any] | None) -> JsonDict:
    """Project a failed check into the required stable terminal shape."""

    if failure is None:
        return {
            "passed": True,
            "failed_check": None,
            "upstream": None,
            "field": None,
            "expected_value": "all_preconditions_and_recomputations_pass",
            "observed_value": "all_preconditions_and_recomputations_pass",
        }
    return {
        "passed": False,
        "failed_check": failure["check"],
        "upstream": failure["upstream"],
        "field": failure["field"],
        "expected_value": failure["expected_value"],
        "observed_value": failure["observed_value"],
    }


def _sentence_ranges(text: bytes) -> list[tuple[int, int]]:
    """Return non-empty sentence byte ranges without converting byte offsets to characters."""

    ranges: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(rb"[.!?]", text):
        end = match.end()
        while start < end and text[start : start + 1].isspace():
            start += 1
        if start < end:
            ranges.append((start, end))
        start = end
    return ranges


def _entity_spans(text: bytes) -> list[tuple[int, int, int, str]]:
    """Derive public entity spans from the controlled-language surface form only."""

    spans: list[tuple[int, int, int, str]] = []
    for sentence_index, (start, end) in enumerate(_sentence_ranges(text)):
        for match in re.finditer(rb"\b[A-Z][a-z]+\b", text[start:end]):
            left, right = start + match.start(), start + match.end()
            spans.append((sentence_index, left, right, text[left:right].decode("ascii")))
    return spans


def _grammar_literal(value: str) -> str:
    """Quote one exact terminal with JSON escaping accepted by llama.cpp GBNF."""

    return json.dumps(value)


def _enum_rule(values: Sequence[str]) -> str:
    """Make one finite GBNF choice from already serialized terminal text."""

    return " | ".join(_grammar_literal(value) for value in values)


def _grammar_text(text: bytes, call_type: str, mode: str) -> str:
    """Compile the common tuple grammar with optional public-reference choices."""

    if call_type not in TOKEN_BUDGETS or mode not in {"grammar_only", "reference"}:
        raise ValueError("unsupported grammar contract")
    relation_list = (
        'relation | relation "," relation | relation "," relation "," relation | '
        'relation "," relation "," relation "," relation'
        if call_type == "source"
        else "relation"
    )
    if mode == "grammar_only":
        sentence_rule = '"0" | nonzero digits'
        offset_rule = '"0" | nonzero digits'
        numeric_rules = ["nonzero ::= [1-9]", "digits ::= [0-9]*"]
    else:
        spans = _entity_spans(text)
        sentences = sorted({str(index) for index, _, _, _ in spans}) or ["0"]
        offsets = sorted({str(value) for _, left, right, _ in spans for value in (left, right)})
        sentence_rule = _enum_rule(sentences)
        offset_rule = _enum_rule(offsets or ["0"])
        numeric_rules = []
    predicate_rule = _enum_rule([json.dumps(value) for value in PREDICATES])
    polarity_rule = _enum_rule([json.dumps("positive"), json.dumps("negative")])
    lines = [
        "root ::= unknown | known",
        f"unknown ::= {_grammar_literal(canonical_json({'outcome': 'unknown', 'relations': []}))}",
        f'known ::= "{{\\"outcome\\":\\"known\\",\\"relations\\":[" relation-list "]}}"',
        f"relation-list ::= {relation_list}",
        'relation ::= "{\\"sentence_index\\":" sentence '
        '",\\"subject_start\\":" offset '
        '",\\"subject_end\\":" offset '
        '",\\"predicate\\":" predicate '
        '",\\"object_start\\":" offset '
        '",\\"object_end\\":" offset '
        '",\\"polarity\\":" polarity "}"',
        f"sentence ::= {sentence_rule}",
        f"offset ::= {offset_rule}",
        f"predicate ::= {predicate_rule}",
        f"polarity ::= {polarity_rule}",
        *numeric_rules,
    ]
    return "\n".join(lines) + "\n"


def compile_grammar(text: bytes, call_type: str, mode: str) -> JsonDict:
    """Return a request-bound llama.cpp grammar without evaluator information."""

    grammar = _grammar_text(text, call_type, mode)
    return {
        "arm": mode,
        "call_type": call_type,
        "public_input_sha256": sha256_bytes(text),
        "tuple_schema_sha256": sha256_bytes(canonical_json(TUPLE_SCHEMA).encode("utf-8")),
        "token_budget": TOKEN_BUDGETS[call_type],
        "model_settings": deepcopy(MODEL_SETTINGS),
        "grammar": grammar,
        "grammar_sha256": sha256_bytes(grammar.encode("utf-8")),
    }


def grammar_errors(grammar: str) -> list[str]:
    """Check deterministic GBNF serialization before any native decoder sees it."""

    errors: list[str] = []
    lines = grammar.splitlines()
    names = [line.split(" ::=", 1)[0] for line in lines if " ::=" in line]
    if not lines or names[:1] != ["root"] or any(" ::=" not in line for line in lines):
        errors.append("gbnf_rule_shape")
    if len(names) != len(set(names)):
        errors.append("gbnf_duplicate_rule")
    if "expected_decision" in grammar or "supported" in grammar or "contradicted" in grammar:
        errors.append("authority_leak")
    try:
        grammar.encode("utf-8").decode("utf-8")
    except UnicodeError:
        errors.append("grammar_utf8")
    return errors


def extract_public_completion(text: bytes, call_type: str) -> JsonDict:
    """Extract controlled clauses using only the bytes visible to one independent call."""

    relations: list[JsonDict] = []
    ranges = _sentence_ranges(text)
    for sentence_index, (start, end) in enumerate(ranges):
        match = _CLAUSE.fullmatch(text[start:end])
        if match is None:
            continue
        relations.append(
            {
                "sentence_index": sentence_index,
                "subject_start": start + match.start("subject"),
                "subject_end": start + match.end("subject"),
                "predicate": _SURFACE_TO_PREDICATE.get(
                    match.group("predicate").decode("ascii"),
                    match.group("predicate").decode("ascii"),
                ),
                "object_start": start + match.start("object"),
                "object_end": start + match.end("object"),
                "polarity": "negative" if match.group("negative") else "positive",
            }
        )
    limit = 4 if call_type == "source" else 1
    if not relations or len(relations) > limit:
        return {"outcome": "unknown", "relations": []}
    return {"outcome": "known", "relations": relations}


def _valid_int(value: Any) -> bool:
    """Accept integers but reject Boolean values that Python treats as integers."""

    return isinstance(value, int) and not isinstance(value, bool)


def compile_completion(
    text: bytes,
    completion: Any,
    call_type: str,
    expected_grammar_sha256: str,
) -> JsonDict:
    """Resolve untrusted tuple offsets against exactly one public document."""

    errors: list[str] = []
    try:
        reference = compile_grammar(text, call_type, "reference")
    except (KeyError, ValueError):
        return {"outcome": "unknown", "relations": [], "errors": ["call_type"]}
    if expected_grammar_sha256 != reference["grammar_sha256"]:
        errors.append("grammar_request_mismatch")
    if not isinstance(completion, Mapping) or set(completion) != {"outcome", "relations"}:
        return {"outcome": "unknown", "relations": [], "errors": errors + ["completion_shape"]}
    outcome = completion.get("outcome")
    relations = completion.get("relations")
    if outcome not in {"known", "unknown"} or not isinstance(relations, list):
        return {"outcome": "unknown", "relations": [], "errors": errors + ["completion_shape"]}
    if outcome == "unknown":
        if relations:
            errors.append("unknown_with_relations")
        return {"outcome": "unknown", "relations": [], "errors": errors}
    limit = 4 if call_type == "source" else 1
    if not 1 <= len(relations) <= limit:
        errors.append("relation_count")

    ranges = _sentence_ranges(text)
    allowed = {(index, left, right) for index, left, right, _ in _entity_spans(text)}
    compiled: list[JsonDict] = []
    for relation in relations:
        if not isinstance(relation, Mapping) or set(relation) != set(RELATION_FIELDS):
            errors.append("relation_shape")
            continue
        sentence = relation["sentence_index"]
        offsets = tuple(
            relation[field] for field in RELATION_FIELDS if field.endswith(("start", "end"))
        )
        if not _valid_int(sentence):
            errors.append("sentence_index")
            continue
        if any(not _valid_int(value) for value in offsets):
            errors.append("span_out_of_range")
            continue
        subject_start = relation["subject_start"]
        subject_end = relation["subject_end"]
        object_start = relation["object_start"]
        object_end = relation["object_end"]
        if not (
            0 <= subject_start < subject_end <= len(text)
            and 0 <= object_start < object_end <= len(text)
        ):
            errors.append("span_out_of_range")
            continue
        if not 0 <= sentence < len(ranges):
            errors.append("sentence_index")
            continue
        left, right = ranges[sentence]
        if not (
            left <= subject_start < subject_end <= right
            and left <= object_start < object_end <= right
        ):
            errors.append("span_outside_sentence")
            continue
        if (sentence, subject_start, subject_end) not in allowed or (
            sentence,
            object_start,
            object_end,
        ) not in allowed:
            errors.append("type_mismatch")
            continue
        if relation["predicate"] not in PREDICATES:
            errors.append("unsupported_predicate")
        if relation["polarity"] not in {"positive", "negative"}:
            errors.append("invalid_polarity")
        compiled.append(
            {
                **dict(relation),
                "subject_surface": text[subject_start:subject_end].decode("ascii"),
                "object_surface": text[object_start:object_end].decode("ascii"),
            }
        )
    if errors:
        return {"outcome": "unknown", "relations": [], "errors": list(dict.fromkeys(errors))}
    return {"outcome": "known", "relations": compiled, "errors": []}


def _canonical_relation(predicate: str, subject: str, obj: str) -> tuple[str, str, str]:
    """Normalize inverse words before relation closure or exact execution."""

    if predicate in _INVERSE:
        return _INVERSE[predicate], obj, subject
    return predicate, subject, obj


def _candidate_decision(source_text: str, claim_text: str) -> JsonDict:
    """Run public extraction, reference compilation, closure, and the shipped executor."""

    source = source_text.encode("utf-8")
    claim_bytes = claim_text.encode("utf-8")
    source_completion = extract_public_completion(source, "source")
    claim_completion = extract_public_completion(claim_bytes, "claim")
    source_grammar = compile_grammar(source, "source", "reference")
    claim_grammar = compile_grammar(claim_bytes, "claim", "reference")
    compiled_source = compile_completion(
        source, source_completion, "source", source_grammar["grammar_sha256"]
    )
    compiled_claim = compile_completion(
        claim_bytes, claim_completion, "claim", claim_grammar["grammar_sha256"]
    )
    compile_errors = [*compiled_source["errors"], *compiled_claim["errors"]]
    if compiled_source["outcome"] != "known" or compiled_claim["outcome"] != "known":
        return {
            "decision": "unknown",
            "abstention": True,
            "errors": compile_errors or ["explicit_unknown"],
        }

    relations = list(compiled_source["relations"])
    surfaces = {
        relation[field] for relation in relations for field in ("subject_surface", "object_surface")
    }
    bindings = []
    for surface in sorted(surfaces):
        encoded = surface.encode("ascii")
        start = source.index(encoded)
        bindings.append(EntityBinding(surface, surface, start, start + len(encoded)))

    positive_edges: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for relation in relations:
        family, subject, obj = _canonical_relation(
            relation["predicate"], relation["subject_surface"], relation["object_surface"]
        )
        if relation["polarity"] == "positive" and family in FAMILIES:
            positive_edges[family].add((subject, obj))
    for family, edges in positive_edges.items():
        changed = True
        while changed:
            additions = {
                (left, right)
                for left, middle in edges
                for other_middle, right in edges
                if middle == other_middle and left != right
            } - edges
            changed = bool(additions)
            edges.update(additions)
        existing = {
            _canonical_relation(
                relation["predicate"],
                relation["subject_surface"],
                relation["object_surface"],
            )
            for relation in relations
            if relation["polarity"] == "positive"
        }
        for subject, obj in sorted(edges):
            if (family, subject, obj) not in existing:
                relations.append(
                    {
                        "predicate": family,
                        "subject_surface": subject,
                        "object_surface": obj,
                        "polarity": "positive",
                    }
                )

    typed_source = tuple(
        TypedRelation(
            relation["subject_surface"],
            relation["predicate"],
            relation["object_surface"],
            relation["polarity"],
            0,
            len(source),
        )
        for relation in relations
    )
    claim = compiled_claim["relations"][0]
    typed_claim = TypedRelation(
        claim["subject_surface"],
        claim["predicate"],
        claim["object_surface"],
        claim["polarity"],
        0,
        max(1, len(source)),
    )
    result = execute_relation(source, tuple(bindings), typed_source, typed_claim)
    return {
        "decision": result.decision,
        "abstention": result.abstention,
        "errors": list(result.uncertainty_reasons),
    }


def _authority_literal(clause: str) -> tuple[str, str, str, bool] | None:
    """Parse one controlled clause in the independent label implementation."""

    match = _AUTHORITY_CLAUSE.fullmatch(clause.strip())
    if match is None:
        return None
    subject = match.group("subject")
    obj = match.group("object")
    predicate = _SURFACE_TO_PREDICATE.get(match.group("predicate"), match.group("predicate"))
    if predicate in _INVERSE:
        predicate, subject, obj = _INVERSE[predicate], obj, subject
    return predicate, subject, obj, match.group("negative") is None


def authority_decision(source_text: str, claim_text: str) -> str:
    """Interpret public controlled language without calling the candidate path."""

    source_clauses = [item.strip() + "." for item in source_text.split(".") if item.strip()]
    literals = [_authority_literal(clause) for clause in source_clauses]
    claim = _authority_literal(claim_text)
    if claim is None or not literals or any(literal is None for literal in literals):
        return "unknown"
    assignments: dict[tuple[str, str, str], set[bool]] = defaultdict(set)
    positive: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for literal in literals:
        assert literal is not None
        family, subject, obj, truth = literal
        assignments[(family, subject, obj)].add(truth)
        if truth:
            positive[family].add((subject, obj))
            assignments[(family, obj, subject)].add(False)
    for family, edges in positive.items():
        changed = True
        while changed:
            additions = {
                (left, right)
                for left, middle in edges
                for other_middle, right in edges
                if middle == other_middle and left != right
            } - edges
            changed = bool(additions)
            edges.update(additions)
        for subject, obj in edges:
            assignments[(family, subject, obj)].add(True)
            assignments[(family, obj, subject)].add(False)
    family, subject, obj, expected_truth = claim
    values = assignments.get((family, subject, obj), set())
    if len(values) != 1:
        return "unknown"
    return "supported" if expected_truth in values else "contradicted"


def _surface(base_number: int, role: int) -> str:
    """Render a deterministic ASCII entity name from the separate surface seed."""

    digest = hashlib.sha256(f"{SURFACE_SEED}:{base_number}:{role}".encode()).hexdigest()
    consonants = "bcdfghjklmnprstv"
    vowels = "aeiou"
    name = "".join(
        consonants[int(digest[index * 2], 16) % len(consonants)]
        + vowels[int(digest[index * 2 + 1], 16) % len(vowels)]
        for index in range(3)
    )
    return name.capitalize()


def _opaque(prefix: str, *parts: Any) -> str:
    """Create a seed-bound opaque identity without leaking split or label text."""

    payload = canonical_json([RANDOM_SEED, *parts]).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:24]}"


def _relation_dict(text: str, subject: str, predicate: str, obj: str) -> JsonDict:
    """Create one exact positive tuple for a generated controlled sentence."""

    return extract_public_completion(text.encode("utf-8"), "claim")["relations"][0]


def build_panel() -> JsonDict:
    """Materialize 80 balanced bases without reading any model outcome."""

    public_rows: list[JsonDict] = []
    authority_rows: list[JsonDict] = []
    base_rows: list[JsonDict] = []
    base_number = 0
    for split, split_count in SPLIT_BASE_COUNTS.items():
        per_family = split_count // len(FAMILIES)
        for family in FAMILIES:
            family_bases: list[str] = []
            pending: list[tuple[str, int, tuple[str, str, str]]] = []
            for family_index in range(per_family):
                base_id = _opaque("b", split, family, family_index)
                names = tuple(_surface(base_number, role) for role in range(3))
                family_bases.append(base_id)
                pending.append((base_id, base_number, names))
                base_number += 1
            for family_index, (base_id, number, names) in enumerate(pending):
                alpha_id = family_bases[family_index ^ 1]
                subject, middle, obj = names
                definitions = {
                    "supported": (
                        f"{subject} {family} {middle}.",
                        f"{subject} {family} {middle}.",
                    ),
                    "reversal": (
                        f"{subject} {family} {middle}.",
                        f"{middle} {family} {subject}.",
                    ),
                    "joint_support": (
                        f"{subject} {family} {middle}. {middle} {family} {obj}.",
                        f"{subject} {family} {obj}.",
                    ),
                    "support_removed": (
                        f"{subject} {family} {middle}.",
                        f"{subject} {family} {obj}.",
                    ),
                }
                base_rows.append(
                    {
                        "base_id": base_id,
                        "split": split,
                        "relation_family": family,
                        "alpha_rename_base_id": alpha_id,
                        "surface_render_index": number,
                    }
                )
                for variant, (source_text, claim_text) in definitions.items():
                    unit_id = _opaque("u", base_id, variant)
                    expected = authority_decision(source_text, claim_text)
                    public_rows.append(
                        {
                            "unit_id": unit_id,
                            "source_text": source_text,
                            "claim_text": claim_text,
                        }
                    )
                    authority_rows.append(
                        {
                            "unit_id": unit_id,
                            "base_id": base_id,
                            "split": split,
                            "relation_family": family,
                            "variant": variant,
                            "expected_decision": expected,
                            "alpha_rename_base_id": alpha_id,
                            "alpha_rename_variant": variant,
                            "source_relation_count": len(
                                extract_public_completion(source_text.encode(), "source")[
                                    "relations"
                                ]
                            ),
                            "claim_relation": _relation_dict(claim_text, subject, family, middle),
                        }
                    )
    split_bases = {
        split: sorted(row["base_id"] for row in base_rows if row["split"] == split)
        for split in SPLIT_BASE_COUNTS
    }
    split_hashes = {
        split: sha256_bytes(canonical_json(values).encode("utf-8"))
        for split, values in split_bases.items()
    }
    split_manifest = {
        "construction_seed": RANDOM_SEED,
        "surface_seed": SURFACE_SEED,
        "base_count": 80,
        "row_count": 320,
        "variants_per_base": 4,
        "split_base_counts": deepcopy(SPLIT_BASE_COUNTS),
        "split_row_counts": {key: value * 4 for key, value in SPLIT_BASE_COUNTS.items()},
        "family_base_counts_per_split": {
            split: {family: count // 4 for family in FAMILIES}
            for split, count in SPLIT_BASE_COUNTS.items()
        },
        "split_hashes": split_hashes,
        "split_hashes_disjoint": len(set(split_hashes.values())) == 3,
        "held_out_used_for_selection": False,
        "model_outcome_used_for_selection": False,
        "base_rows": base_rows,
    }
    return {
        "public_rows": public_rows,
        "authority_rows": authority_rows,
        "split_manifest": split_manifest,
    }


def execute_panel(
    public_rows: Sequence[Mapping[str, Any]], authority_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Run public bytes through the candidate path, then compare independent labels."""

    authority = {str(row["unit_id"]): row for row in authority_rows}
    rows: list[JsonDict] = []
    started = time.monotonic()
    last_report = started
    for index, public in enumerate(public_rows, 1):
        private = authority[str(public["unit_id"])]
        result = _candidate_decision(str(public["source_text"]), str(public["claim_text"]))
        expected = private["expected_decision"]
        passed = result["decision"] == expected
        rows.append(
            {
                "unit_id": public["unit_id"],
                "base_id": private["base_id"],
                "split": private["split"],
                "relation_family": private["relation_family"],
                "variant": private["variant"],
                "arm": "span_compiler_exact_executor",
                "seed": RANDOM_SEED,
                "metric": int(passed),
                "error": None if passed else "independent_label_disagreement",
                "abstention": result["abstention"],
                "prediction": result["decision"],
                "expected_prediction": expected,
                "compiler_or_executor_errors": result["errors"],
            }
        )
        now = time.monotonic()
        if now - last_report >= 60:
            _progress(
                4, "heartbeat", f"executed={index}/{len(public_rows)} elapsed_s={now - started:.1f}"
            )
            last_report = now
    return rows


def _lexical_tokens(value: str) -> Counter[str]:
    """Count public word tokens without direction, order, or relation semantics."""

    return Counter(re.findall(r"[a-z]+", value.lower()))


def lexical_control(
    public_rows: Sequence[Mapping[str, Any]], authority_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Apply one frozen token-membership rule to all easy and matched rows."""

    authority = {str(row["unit_id"]): row for row in authority_rows}
    rows: list[JsonDict] = []
    for public in public_rows:
        private = authority[str(public["unit_id"])]
        source_tokens = _lexical_tokens(str(public["source_text"]))
        claim_tokens = _lexical_tokens(str(public["claim_text"]))
        prediction = (
            "supported"
            if all(source_tokens[token] >= count for token, count in claim_tokens.items())
            else "unknown"
        )
        expected = private["expected_decision"]
        matched = private["variant"] in {"supported", "reversal"}
        rows.append(
            {
                "unit_id": public["unit_id"],
                "base_id": private["base_id"],
                "split": private["split"],
                "variant": private["variant"],
                "arm": "frozen_lexical_token_membership",
                "seed": RANDOM_SEED,
                "metric": int(prediction == expected),
                "error": None if prediction == expected else "semantic_order_blind",
                "abstention": prediction == "unknown",
                "prediction": prediction,
                "expected_prediction": expected,
                "matched_semantic_pair_indistinguishable": matched,
                "claim_token_multiset_sha256": sha256_bytes(
                    canonical_json(sorted(claim_tokens.items())).encode("utf-8")
                ),
            }
        )
    return rows


def _stored_completion(path: Path, manifest_row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Authenticate one raw call before its model output bytes enter diagnosis."""

    document = json.loads(path.read_text(encoding="utf-8"))
    completion = document.get("completion")
    if not isinstance(completion, Mapping):
        raise ValueError(f"raw completion missing: {path}")
    if sha256_bytes(canonical_json(completion).encode("utf-8")) != manifest_row["row_sha256"]:
        raise ValueError(f"raw row hash mismatch: {path}")
    raw = str(completion.get("raw_output", "")).encode("utf-8")
    if sha256_bytes(raw) != manifest_row["raw_output_sha256"]:
        raise ValueError(f"raw output hash mismatch: {path}")
    return completion


def diagnose_exp7196(root: Path) -> list[JsonDict]:
    """Reproduce old transport failures from manifest-authenticated raw bytes."""

    artifact = json.loads((root / EXP7196_PATH).read_text(encoding="utf-8"))
    manifest_path = root / RAW_MANIFEST_PATH
    if sha256_file(manifest_path) != artifact["source_artifact_hashes"]["raw_manifest"]:
        raise ValueError("Exp7196 raw manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schedule = {str(row["call_id"]): row for row in manifest["schedule"]}
    grouped: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    raw_root = manifest_path.parent.resolve()
    for manifest_row in manifest["raw_rows"]:
        if manifest_row["call_type"] not in {"source", "claim"}:
            continue
        path = Path(str(manifest_row["path"])).resolve()
        if path.parent != raw_root:
            raise ValueError("raw manifest path escaped its owned directory")
        completion = _stored_completion(path, manifest_row)
        grouped[str(manifest_row["call_type"])].append(
            (completion, schedule[str(manifest_row["call_id"])])
        )
    rows: list[JsonDict] = []
    for call_type in ("source", "claim"):
        calls = grouped[call_type]
        invalid = [item for item in calls if item[0].get("parse_status") != "valid"]
        truncated = [item for item in calls if item[0].get("truncated") is True]
        missing_terminators = []
        repeated = 0
        semantic = 0
        overhead = 0
        for completion, request in calls:
            raw = str(completion.get("raw_output", ""))
            overhead += sum(character in '{}[]":,\n\r\t ' for character in raw)
            try:
                json.loads(raw)
            except json.JSONDecodeError:
                if completion.get("truncated") is True:
                    missing_terminators.append(completion)
            lines = [line.strip() for line in raw.splitlines() if len(line.strip()) >= 12]
            repeated += int(len(lines) != len(set(lines)))
            model_input = request.get("model_input", {})
            visible = str(model_input.get(f"{call_type}_text", ""))
            parsed = completion.get("parsed_output")
            if (
                completion.get("parse_status") == "valid"
                and visible
                and isinstance(parsed, Mapping)
            ):
                semantic += int(not parsed.get("relations"))
        rows.append(
            {
                "call_type": call_type,
                "denominator": len(calls),
                "invalid_count": len(invalid),
                "truncated_count": len(truncated),
                "missing_terminator_count": len(missing_terminators),
                "format_overhead_bytes": overhead,
                "repeated_output_count": repeated,
                "semantic_error_count": semantic,
                "manifest_rows_authenticated": len(calls),
                "diagnosis_source": "exact_raw_output_bytes",
                "larger_budget_assumed": False,
            }
        )
    return rows


def span_mutation_checks() -> list[JsonDict]:
    """Exercise the reference compiler and executor boundary with adverse inputs."""

    source = "Aster precedes Brin. Cora precedes Dain."
    encoded = source.encode()
    valid_relation = extract_public_completion(encoded, "source")["relations"][0]
    grammar = compile_grammar(encoded, "source", "reference")

    def compile_mutation(
        name: str, completion: JsonDict, expected_error: str, grammar_hash: str | None = None
    ) -> JsonDict:
        result = compile_completion(
            encoded,
            completion,
            "source",
            grammar_hash or grammar["grammar_sha256"],
        )
        return {
            "mutation": name,
            "expected": expected_error,
            "observed": result["errors"],
            "passed": expected_error in result["errors"],
        }

    mutations: list[JsonDict] = []
    relation = deepcopy(valid_relation)
    relation["subject_end"] = len(encoded) + 1
    mutations.append(
        compile_mutation(
            "out_of_range", {"outcome": "known", "relations": [relation]}, "span_out_of_range"
        )
    )
    other_hash = compile_grammar(b"Eris precedes Fenn.", "source", "reference")["grammar_sha256"]
    mutations.append(
        compile_mutation(
            "cross_document",
            {"outcome": "known", "relations": [valid_relation]},
            "grammar_request_mismatch",
            other_hash,
        )
    )
    relation = deepcopy(valid_relation)
    relation["object_start"] = encoded.index(b"Cora")
    relation["object_end"] = relation["object_start"] + 4
    mutations.append(
        compile_mutation(
            "wrong_sentence", {"outcome": "known", "relations": [relation]}, "span_outside_sentence"
        )
    )
    relation = deepcopy(valid_relation)
    relation["subject_start"] = encoded.index(b"precedes")
    relation["subject_end"] = relation["subject_start"] + len(b"precedes")
    mutations.append(
        compile_mutation(
            "type_mismatch", {"outcome": "known", "relations": [relation]}, "type_mismatch"
        )
    )
    relation = deepcopy(valid_relation)
    relation["predicate"] = "touches"
    mutations.append(
        compile_mutation(
            "unsupported_predicate",
            {"outcome": "known", "relations": [relation]},
            "unsupported_predicate",
        )
    )
    relation = deepcopy(valid_relation)
    relation["polarity"] = "maybe"
    mutations.append(
        compile_mutation(
            "invalid_polarity", {"outcome": "known", "relations": [relation]}, "invalid_polarity"
        )
    )
    mutations.append(
        compile_mutation(
            "excess_source_relations",
            {"outcome": "known", "relations": [valid_relation] * 5},
            "relation_count",
        )
    )

    semantic_cases = (
        ("direction_reversal", "Aster precedes Brin.", "Brin precedes Aster.", "contradicted"),
        ("negation", "Aster precedes Brin.", "Aster does not precede Brin.", "contradicted"),
        ("support_removed", "Aster precedes Brin.", "Aster precedes Cora.", "unknown"),
    )
    for name, source_text, claim_text, expected in semantic_cases:
        result = _candidate_decision(source_text, claim_text)
        mutations.append(
            {
                "mutation": name,
                "expected": expected,
                "observed": result["decision"],
                "passed": result["decision"] == expected,
            }
        )
    grammar_text = grammar["grammar"]
    mutations.append(
        {
            "mutation": "grammar_serialization",
            "expected": "stable_utf8_gbnf",
            "observed": sha256_bytes(grammar_text.encode().decode().encode()),
            "passed": not grammar_errors(grammar_text)
            and sha256_bytes(grammar_text.encode()) == grammar["grammar_sha256"],
        }
    )
    return mutations


def measure_completion_tokens(completions: Sequence[bytes], tokenizer: Tokenize | None) -> JsonDict:
    """Measure with an injected embedded tokenizer or explicitly defer the values."""

    byte_sizes = [len(value) for value in completions]
    if tokenizer is None:
        return {
            "measurement_status": "deferred_to_canary",
            "minimum_completion_tokens": None,
            "maximum_completion_tokens": None,
            "minimum_completion_bytes": min(byte_sizes),
            "maximum_completion_bytes": max(byte_sizes),
        }
    token_sizes = [len(tokenizer(value)) for value in completions]
    return {
        "measurement_status": "measured_embedded_gguf_tokenizer",
        "minimum_completion_tokens": min(token_sizes),
        "maximum_completion_tokens": max(token_sizes),
        "minimum_completion_bytes": min(byte_sizes),
        "maximum_completion_bytes": max(byte_sizes),
    }


def resolve_tokenizer_loader(
    importer: Callable[[str], Any] = importlib.import_module,
) -> VocabularyLoader | None:
    """Find llama.cpp lazily so a missing optional tokenizer can defer cleanly."""

    try:
        module = importer("llama_cpp")
    except ImportError:
        return None
    return module.Llama


def load_embedded_tokenizer(
    upstream: Mapping[str, Any], *, loader: VocabularyLoader
) -> tuple[Any | None, Tokenize | None, JsonDict]:
    """Load only GGUF vocabulary metadata and return a byte-token counter."""

    specs = upstream.get("model_specs")
    spec = specs[0] if isinstance(specs, list) and specs else {}
    path = Path(str(spec.get("path", "")))
    if not path.is_file():
        return (
            None,
            None,
            {
                "embedded_tokenizer_available": False,
                "tokenizer_model_path": str(path),
                "tokenizer_model_sha256": spec.get("sha256"),
                "defer_reason": "gguf_path_missing",
            },
        )
    try:
        owner = loader(model_path=str(path), vocab_only=True, n_ctx=8, verbose=False)
    except Exception as exc:
        return (
            None,
            None,
            {
                "embedded_tokenizer_available": False,
                "tokenizer_model_path": str(path),
                "tokenizer_model_sha256": spec.get("sha256"),
                "defer_reason": f"vocabulary_load_failed:{type(exc).__name__}",
            },
        )

    def tokenize(value: bytes) -> Sequence[int]:
        return owner.tokenize(value, add_bos=False)

    return (
        owner,
        tokenize,
        {
            "embedded_tokenizer_available": True,
            "tokenizer_model_path": str(path),
            "tokenizer_model_sha256": spec.get("sha256"),
            "vocab_only": True,
            "model_invoked": False,
        },
    )


def _grammar_manifest(public_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Deduplicate grammar bytes while retaining a receipt for every logical request."""

    catalog: dict[str, JsonDict] = {}
    requests: list[JsonDict] = []
    for public in public_rows:
        for call_type in ("source", "claim"):
            text = str(public[f"{call_type}_text"]).encode("utf-8")
            request_id = sha256_bytes(call_type.encode() + b":" + text)
            if request_id not in catalog:
                syntax = compile_grammar(text, call_type, "grammar_only")
                reference = compile_grammar(text, call_type, "reference")
                catalog[request_id] = {
                    "request_id": request_id,
                    "call_type": call_type,
                    "public_input_sha256": sha256_bytes(text),
                    "grammar_only": syntax,
                    "reference": reference,
                }
            requests.append(
                {
                    "unit_id": public["unit_id"],
                    "call_type": call_type,
                    "request_id": request_id,
                    "public_input_sha256": sha256_bytes(text),
                    "grammar_only_sha256": catalog[request_id]["grammar_only"]["grammar_sha256"],
                    "reference_sha256": catalog[request_id]["reference"]["grammar_sha256"],
                }
            )
    return {
        "schema": "carnot.exp7208.grammar_manifest.v1",
        "request_count": len(requests),
        "unique_request_count": len(catalog),
        "requests": requests,
        "grammar_catalog": list(catalog.values()),
    }


def _fixture_manifest(
    panel: Mapping[str, Any],
    public_bytes: bytes,
    authority_bytes: bytes,
) -> JsonDict:
    """Bind rows, related bases, independent labels, and both grammar contracts."""

    public_rows = panel["public_rows"]
    authority_rows = panel["authority_rows"]
    grammar_manifest = _grammar_manifest(public_rows)
    row_receipts = [
        {
            "unit_id": public["unit_id"],
            "public_row_sha256": sha256_bytes(canonical_json(public).encode("utf-8")),
            "authority_row_sha256": sha256_bytes(canonical_json(authority).encode("utf-8")),
        }
        for public, authority in zip(public_rows, authority_rows, strict=True)
    ]
    return {
        "schema": "carnot.exp7208.span_fixture.v1",
        "construction_frozen_before_outcomes": True,
        "model_outcome_selection": False,
        "public_view_sha256": sha256_bytes(public_bytes),
        "authority_sidecar_sha256": sha256_bytes(authority_bytes),
        "split_manifest": deepcopy(panel["split_manifest"]),
        "row_receipts": row_receipts,
        "grammar_manifest": grammar_manifest,
        "authority_implementation": "independent_controlled_language_interpreter",
        "candidate_implementation": "public_regex_to_span_compiler_to_shipped_typed_executor",
        "authority_imports_or_calls_candidate": AUTHORITY_IMPORTS_CANDIDATE,
    }


def _resolved_paths(root: Path, overrides: Mapping[str, Path] | None) -> dict[str, Path]:
    """Resolve fixed sources while allowing tests to replace one external input."""

    paths = {name: root / path for name, path in SOURCE_PATHS.items()}
    for name, path in (overrides or {}).items():
        paths[name] = path if path.is_absolute() else root / path
    return paths


def _preconditions(
    root: Path,
    run_date: str,
    output_root: Path,
    overrides: Mapping[str, Path] | None,
) -> tuple[list[JsonDict], dict[str, Path], JsonDict]:
    """Check source bytes, exact gates, quarantine, tools, imports, and destinations."""

    paths = _resolved_paths(root, overrides)
    checks: list[JsonDict] = []

    def record(row: JsonDict) -> bool:
        _progress(1, "check_start", row["check"])
        checks.append(row)
        _progress(1, "check_end", f"{row['check']} passed={row['passed']}")
        return bool(row["passed"])

    if not record(_gate("run_date", None, "run_date", RUN_DATE, run_date, run_date == RUN_DATE)):
        return checks, paths, {}
    for name, path in paths.items():
        if not record(
            _gate(
                "source_exists",
                name,
                "path",
                "existing_file",
                str(path),
                path.is_file() and os.access(path, os.R_OK),
            )
        ):
            return checks, paths, {}
    spec_text = paths["constraint_spec"].read_text(encoding="utf-8")
    if not record(
        _gate(
            "driving_spec",
            "constraint_spec",
            "REQ-VERIFY-7208",
            True,
            "REQ-VERIFY-7208" in spec_text,
            "REQ-VERIFY-7208" in spec_text,
        )
    ):
        return checks, paths, {}
    import_ok = (
        callable(execute_relation) and callable(exp7196_checksum) and callable(exp7197_checksum)
    )
    if not record(_gate("required_imports", "repository", "callables", True, import_ok, import_ok)):
        return checks, paths, {}
    output_root.mkdir(parents=True, exist_ok=True)
    destination_ok = os.access(output_root, os.W_OK)
    if not record(
        _gate(
            "output_destination",
            "repository",
            "output_root",
            "writable_directory",
            str(output_root),
            destination_ok,
        )
    ):
        return checks, paths, {}
    try:
        exp7196 = json.loads(paths["exp7196_artifact"].read_text(encoding="utf-8"))
        exp7197 = json.loads(paths["exp7197_artifact"].read_text(encoding="utf-8"))
        raw_manifest = json.loads(paths["raw_manifest"].read_text(encoding="utf-8"))
        exclusion = yaml.safe_load(paths["exclusion_manifest"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        record(
            _gate(
                "source_parse",
                "upstream",
                "json_or_yaml",
                "valid",
                f"{type(exc).__name__}:{exc}",
                False,
            )
        )
        return checks, paths, {}
    documents = {"exp7196": exp7196, "exp7197": exp7197}
    for name, document in documents.items():
        if not record(
            _gate(
                "structured_quarantine",
                name,
                "flagged_adversarial|quarantined|fabricated",
                False,
                _is_quarantined(document),
                not _is_quarantined(document),
            )
        ):
            return checks, paths, {}
    excluded = _manifest_hits(
        exclusion,
        {
            "exp7196-qwen-atomic-capture",
            "experiment_7196_v634_qwen_atomic_capture",
            "exp7197-grounding-value-audit",
            "experiment_7197_v634_grounding_value_audit",
        },
    )
    if not record(
        _gate("exclusion_manifest", "upstreams", "experiment_ids", False, excluded, not excluded)
    ):
        return checks, paths, {}
    producer = {
        "exp7196": {
            "status": _unwrap(exp7196.get("status")),
            "atomic_capture_complete_score": _unwrap(exp7196.get("atomic_capture_complete_score")),
            "verdict_class": _unwrap(exp7196.get("verdict_class")),
        },
        "exp7197": {
            "status": _unwrap(exp7197.get("status")),
            "grounding_audit_complete_score": _unwrap(
                exp7197.get("grounding_audit_complete_score")
            ),
            "grounding_value_score": _unwrap(exp7197.get("grounding_value_score")),
        },
    }
    expected_producer = {
        "exp7196": {
            "status": "complete",
            "atomic_capture_complete_score": 1,
            "verdict_class": "null",
        },
        "exp7197": {
            "status": "complete",
            "grounding_audit_complete_score": 1,
            "grounding_value_score": 0,
        },
    }
    if not record(
        _gate(
            "producer_gate_fields",
            "exp7196|exp7197",
            "terminal_fields",
            expected_producer,
            producer,
            producer == expected_producer,
        )
    ):
        return checks, paths, {}
    checksums = {
        "exp7196": exp7196_checksum(exp7196) == exp7196.get("reproducibility_checksum"),
        "exp7197": exp7197_checksum(exp7197) == exp7197.get("reproducibility_checksum"),
    }
    if not record(
        _gate(
            "upstream_authentication",
            "exp7196|exp7197",
            "reproducibility_checksum",
            {"exp7196": True, "exp7197": True},
            checksums,
            all(checksums.values()),
        )
    ):
        return checks, paths, {}
    raw_expected = exp7196.get("source_artifact_hashes", {}).get("raw_manifest")
    raw_observed = sha256_file(paths["raw_manifest"])
    raw_state = {
        "hash": raw_observed,
        "status": raw_manifest.get("status"),
        "schema": raw_manifest.get("schema"),
        "raw_row_count": len(raw_manifest.get("raw_rows", [])),
        "forbidden_input_open_count": raw_manifest.get("forbidden_input_open_count"),
    }
    raw_expected_state = {
        "hash": raw_expected,
        "status": "complete",
        "schema": "carnot.exp7196.raw_manifest.v1",
        "raw_row_count": 576,
        "forbidden_input_open_count": 0,
    }
    record(
        _gate(
            "raw_manifest_authentication",
            "exp7196",
            "raw_manifest",
            raw_expected_state,
            raw_state,
            raw_state == raw_expected_state,
        )
    )
    return checks, paths, {"exp7196": exp7196, "exp7197": exp7197, "raw": raw_manifest}


def _source_hashes(paths: Mapping[str, Path]) -> JsonDict:
    """Bind every cited repository source and frozen contract to the claim."""

    hashes = {
        name: sha256_file(path) if path.is_file() else "missing" for name, path in paths.items()
    }
    hashes["tuple_schema"] = sha256_bytes(canonical_json(TUPLE_SCHEMA).encode("utf-8"))
    hashes["model_settings"] = sha256_bytes(canonical_json(MODEL_SETTINGS).encode("utf-8"))
    return hashes


def _base_artifact(run_date: str) -> JsonDict:
    """Create every required field before a fallible prerequisite is checked."""

    return {
        "field_principles": deepcopy(REQUIRED_FIELD_PRINCIPLES),
        "status": "running",
        "run_date": run_date,
        "preconditions_checked": [],
        "inference_substrate": "blocked_before_qualifying_computation",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_rows": 320,
            "attempted_rows": 0,
            "completed_rows": 0,
            "censored_rows": 320,
            "independent_base_cases": 80,
            "held_out_rows": 256,
            "variants_per_base": 4,
        },
        "random_seed": RANDOM_SEED,
        "surface_render_seed": SURFACE_SEED,
        "reproducibility_checksum": "pending",
        "gate_check_summary": _gate_summary(None),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_exp7208_running_preconditions",
        "span_fixture_ready_score": 0,
        "public_view_path": PUBLIC_VIEW_PATH.as_posix(),
        "authority_sidecar_path": AUTHORITY_SIDECAR_PATH.as_posix(),
        "fixture_manifest_path": FIXTURE_MANIFEST_PATH.as_posix(),
        "split_manifest": {},
        "grammar_contract": {},
        "lexical_control_rows": [],
        "span_mutation_rows": [],
        "upstream_diagnosis_rows": [],
        "upstream_history": {},
        "MODEL_SPECS": [],
        "model_invoked": False,
    }


def _write_checkpoint(path: Path, artifact: Mapping[str, Any], started: float) -> None:
    """Keep nonterminal progress below results/checkpoints, never at the result path."""

    value = deepcopy(dict(artifact))
    value["duration_s"] = time.monotonic() - started
    value["reproducibility_checksum"] = artifact_checksum(value)
    atomic_write_json(path, value, allow_override=False, sort_keys=True)


def _grammar_contract(
    manifest: Mapping[str, Any], completions: Sequence[bytes], tokenizer: Tokenize | None
) -> JsonDict:
    """Summarize the shared schema, settings, bounded arms, and measured sizes."""

    catalog = manifest["grammar_manifest"]["grammar_catalog"]
    serialization_ok = all(
        not grammar_errors(item[arm]["grammar"])
        and sha256_bytes(item[arm]["grammar"].encode("utf-8")) == item[arm]["grammar_sha256"]
        for item in catalog
        for arm in ("grammar_only", "reference")
    )
    return {
        "schema": deepcopy(TUPLE_SCHEMA),
        "tuple_schema_sha256": sha256_bytes(canonical_json(TUPLE_SCHEMA).encode("utf-8")),
        "arms": ["grammar_only", "reference"],
        "source_relation_limit": 4,
        "claim_relation_limit": 1,
        "unknown_is_explicit": True,
        "same_public_bytes": True,
        "same_token_budgets": deepcopy(TOKEN_BUDGETS),
        "same_model_settings": deepcopy(MODEL_SETTINGS),
        "semantic_executor_separate": True,
        "hidden_label_constraints": False,
        "finite_semantic_pruning_adaptation_not_chopchop_reproduction": True,
        "grammar_request_count": manifest["grammar_manifest"]["request_count"],
        "unique_grammar_request_count": manifest["grammar_manifest"]["unique_request_count"],
        "serialization_checks_passed": serialization_ok,
        "completion_size_receipt": measure_completion_tokens(completions, tokenizer),
    }


def _blocked_artifact(
    artifact: JsonDict,
    checks: Sequence[Mapping[str, Any]],
    paths: Mapping[str, Path],
    started: float,
) -> JsonDict:
    """Finish an external block without claiming panel execution occurred."""

    failure = next((row for row in checks if row["passed"] is not True), None)
    artifact["status"] = "blocked"
    artifact["preconditions_checked"] = list(checks)
    artifact["source_artifact_hashes"] = _source_hashes(paths)
    artifact["gate_check_summary"] = _gate_summary(failure)
    artifact["honest_verdict"] = f"blocked_exp7208_{failure['check'] if failure else 'unknown'}"
    artifact["duration_s"] = time.monotonic() - started
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _fixture_paths(output_root: Path) -> tuple[Path, Path, Path, Path, Path]:
    """Resolve fixture, checkpoint, and terminal paths under one owned output root."""

    return (
        output_root / PUBLIC_VIEW_PATH,
        output_root / AUTHORITY_SIDECAR_PATH,
        output_root / FIXTURE_MANIFEST_PATH,
        output_root / CHECKPOINT_PATH,
        output_root / RESULT_PATH,
    )


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_root: Path | None = None,
    path_overrides: Mapping[str, Path] | None = None,
    tokenizer: Tokenize | None = None,
    tokenizer_loader: VocabularyLoader | None = None,
) -> JsonDict:
    """Build one ready fixture or a diagnosed external-block artifact."""

    started = time.monotonic()
    destination = output_root or root
    public_path, authority_path, manifest_path, checkpoint_path, result_path = _fixture_paths(
        destination
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    _progress(0, "start", "write schema-complete checkpoint before checks")
    artifact = _base_artifact(run_date)
    _write_checkpoint(checkpoint_path, artifact, started)
    _progress(0, "end", str(checkpoint_path))

    _progress(1, "start", "verify sources, producer gates, quarantine, imports, and paths")
    checks, paths, upstream = _preconditions(root, run_date, destination, path_overrides)
    artifact["preconditions_checked"] = checks
    failure = next((row for row in checks if row["passed"] is not True), None)
    if failure is not None:
        artifact = _blocked_artifact(artifact, checks, paths, started)
        _progress(1, "end", f"blocked={failure['check']}")
        _progress(7, "validation_start", "validate terminal external block")
        errors = validate_artifact(artifact, root, output_root=destination)
        _progress(7, "validation_end", f"errors={errors}")
        if errors:
            raise ValueError(f"invalid blocked Exp7208 artifact: {errors}")
        _progress(8, "write_start", "atomically write blocked terminal artifact")
        _write_checkpoint(checkpoint_path, artifact, started)
        atomic_write_json(result_path, artifact, allow_override=False, sort_keys=True)
        _progress(8, "write_end", str(result_path))
        return artifact
    artifact["source_artifact_hashes"] = _source_hashes(paths)
    artifact["upstream_history"] = {
        "exp7196_honest_verdict": upstream["exp7196"]["honest_verdict"],
        "exp7197_honest_verdict": upstream["exp7197"]["honest_verdict"],
        "exp7197_grounding_value_score": upstream["exp7197"]["grounding_value_score"],
        "known_failed_value_promoted_to_readiness": False,
        "old_saturated_panel_preserved": True,
    }
    _write_checkpoint(checkpoint_path, artifact, started)
    _progress(1, "end", "all exact preconditions passed")

    _progress(2, "start", "diagnose authenticated Exp7196 source and claim bytes")
    artifact["upstream_diagnosis_rows"] = diagnose_exp7196(root)
    _write_checkpoint(checkpoint_path, artifact, started)
    _progress(2, "end", "reproduced source and claim invalid and truncated counts")

    _progress(3, "start", "build sealed 80-base public and authority panel")
    panel = build_panel()
    public_bytes = jsonl_bytes(panel["public_rows"])
    authority_bytes = jsonl_bytes(panel["authority_rows"])
    public_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(public_path, public_bytes, allow_override=False)
    atomic_write_bytes(authority_path, authority_bytes, allow_override=False)
    artifact["split_manifest"] = deepcopy(panel["split_manifest"])
    _write_checkpoint(checkpoint_path, artifact, started)
    _progress(3, "end", "public=320 authority=320 held_out=256")

    _progress(4, "start", "compile public spans and execute exact relation closure")
    artifact["rows"] = execute_panel(panel["public_rows"], panel["authority_rows"])
    artifact["lexical_control_rows"] = lexical_control(
        panel["public_rows"], panel["authority_rows"]
    )
    artifact["span_mutation_rows"] = span_mutation_checks()
    artifact["sample_size_budget"].update(
        {
            "attempted_rows": len(artifact["rows"]),
            "completed_rows": len(artifact["rows"]),
            "censored_rows": 0,
        }
    )
    artifact["inference_substrate"] = INFERENCE_SUBSTRATE
    artifact["inference_substrate_class"] = INFERENCE_SUBSTRATE_CLASS
    _write_checkpoint(checkpoint_path, artifact, started)
    _progress(4, "end", "candidate and independent authority rows complete")

    tokenizer_owner = None
    tokenizer_load_receipt: JsonDict = {
        "embedded_tokenizer_available": tokenizer is not None,
        "defer_reason": None if tokenizer is not None else "loader_not_requested",
    }
    if tokenizer is None and tokenizer_loader is not None:
        _progress(5, "model_load_start", "load embedded GGUF vocabulary only; no inference")
        tokenizer_owner, tokenizer, tokenizer_load_receipt = load_embedded_tokenizer(
            upstream["exp7196"], loader=tokenizer_loader
        )
        _progress(
            5,
            "model_load_end",
            f"embedded_tokenizer_available={tokenizer_load_receipt['embedded_tokenizer_available']}",
        )
    _progress(5, "start", "serialize both GBNF contracts for every public request")
    manifest = _fixture_manifest(panel, public_bytes, authority_bytes)
    atomic_write_json(manifest_path, manifest, allow_override=False, sort_keys=True)
    completions = [
        canonical_json(extract_public_completion(str(row[field]).encode(), call_type)).encode()
        for row in panel["public_rows"]
        for call_type, field in (("source", "source_text"), ("claim", "claim_text"))
    ]
    try:
        artifact["grammar_contract"] = _grammar_contract(manifest, completions, tokenizer)
    finally:
        if tokenizer_owner is not None:
            tokenizer_owner.close()
    artifact["grammar_contract"]["embedded_tokenizer_receipt"] = tokenizer_load_receipt
    artifact["source_artifact_hashes"].update(
        {
            "public_view": sha256_file(public_path),
            "authority_sidecar": sha256_file(authority_path),
            "fixture_manifest": sha256_file(manifest_path),
        }
    )
    _write_checkpoint(checkpoint_path, artifact, started)
    _progress(5, "end", "grammar bytes and per-request hashes are sealed")

    _progress(6, "start", "apply readiness conjunction without reading model outcomes")
    diagnosis = {row["call_type"]: row for row in artifact["upstream_diagnosis_rows"]}
    ready_checks = {
        "all_rows": len(artifact["rows"]) == 320
        and all(row["metric"] == 1 for row in artifact["rows"]),
        "test_rows": sum(row["split"] == "test" for row in artifact["rows"]) == 256,
        "split_hashes": panel["split_manifest"]["split_hashes_disjoint"],
        "mutations": all(row["passed"] for row in artifact["span_mutation_rows"]),
        "grammar_serialization": artifact["grammar_contract"]["serialization_checks_passed"],
        "raw_diagnosis": diagnosis["claim"]["invalid_count"] == 192
        and diagnosis["claim"]["truncated_count"] == 192
        and diagnosis["source"]["invalid_count"] == 144
        and diagnosis["source"]["truncated_count"] == 144,
        "sealed_sidecars": manifest["public_view_sha256"] == sha256_file(public_path)
        and manifest["authority_sidecar_sha256"] == sha256_file(authority_path),
    }
    artifact["readiness_checks"] = ready_checks
    artifact["span_fixture_ready_score"] = int(all(ready_checks.values()))
    artifact["status"] = "complete"
    artifact["verdict_class"] = (
        "circular_positive" if artifact["span_fixture_ready_score"] else "disqualified"
    )
    artifact["honest_verdict"] = (
        "complete_circular_positive_span_fixture_ready_no_verifier_value_claim"
        if artifact["span_fixture_ready_score"]
        else "complete_disqualified_span_fixture_not_ready"
    )
    artifact["gate_check_summary"] = _gate_summary(None)
    artifact["duration_s"] = time.monotonic() - started
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _write_checkpoint(checkpoint_path, artifact, started)
    _progress(6, "end", f"ready={artifact['span_fixture_ready_score']}")

    _progress(7, "validation_start", "cold replay terminal fields and sealed fixture bytes")
    errors = validate_artifact(artifact, root, output_root=destination)
    _progress(7, "validation_end", f"errors={errors}")
    if errors:
        raise ValueError(f"invalid Exp7208 artifact: {errors}")
    _progress(8, "write_start", "atomically write checkpoint and terminal deliverable")
    artifact["duration_s"] = time.monotonic() - started
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _write_checkpoint(checkpoint_path, artifact, started)
    atomic_write_json(result_path, artifact, allow_override=False, sort_keys=True)
    _progress(8, "write_end", str(result_path))
    return artifact


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read exact object rows from one sealed JSONL sidecar."""

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("JSONL row must be an object")
        rows.append(value)
    return rows


def validate_artifact(
    artifact: object,
    root: Path | None = None,
    *,
    output_root: Path | None = None,
) -> list[str]:
    """Cold-check terminal state, fields, fixtures, semantics, and checksum."""

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
    if artifact.get("execution_venue") != "host" or not artifact.get("execution_host"):
        errors.append("execution_identity")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("oracle_declaration")
    duration = artifact.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    status = artifact.get("status")
    if status == "blocked":
        if (
            artifact.get("verdict_class") != "blocked"
            or artifact.get("span_fixture_ready_score") != 0
        ):
            errors.append("blocked_terminal_state")
        summary = artifact.get("gate_check_summary")
        if not isinstance(summary, Mapping) or summary.get("passed") is not False:
            errors.append("gate_check_summary")
        return errors
    if status != "complete":
        errors.append("status")
        return errors
    if artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS:
        errors.append("inference_substrate_class")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    budget = artifact.get("sample_size_budget")
    if not isinstance(budget, Mapping) or budget.get("completed_rows") != 320:
        errors.append("sample_size_budget")
    if len(artifact.get("rows", [])) != 320 or not all(
        row.get("metric") == 1 for row in artifact.get("rows", [])
    ):
        errors.append("rows")
    if len(artifact.get("lexical_control_rows", [])) != 320:
        errors.append("lexical_control_rows")
    if not artifact.get("span_mutation_rows") or not all(
        row.get("passed") is True for row in artifact.get("span_mutation_rows", [])
    ):
        errors.append("span_mutation_rows")
    if (
        artifact.get("span_fixture_ready_score") != 1
        or artifact.get("verdict_class") != "circular_positive"
    ):
        errors.append("readiness_terminal_state")

    repo = root or find_repo_root(start=__file__)
    destination = output_root or repo
    public_path, authority_path, manifest_path, _, _ = _fixture_paths(destination)
    try:
        public_rows = _read_jsonl(public_path)
        authority_rows = _read_jsonl(authority_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        errors.append("sealed_fixture_files")
        return errors
    expected = build_panel()
    if public_rows != expected["public_rows"]:
        errors.append("public_view")
    if authority_rows != expected["authority_rows"]:
        errors.append("authority_sidecar")
    if manifest.get("public_view_sha256") != sha256_file(public_path) or manifest.get(
        "authority_sidecar_sha256"
    ) != sha256_file(authority_path):
        errors.append("fixture_manifest_hashes")
    if artifact.get("split_manifest") != expected["split_manifest"]:
        errors.append("split_manifest")
    if artifact.get("rows") != execute_panel(public_rows, authority_rows):
        errors.append("semantic_replay")
    if artifact.get("lexical_control_rows") != lexical_control(public_rows, authority_rows):
        errors.append("lexical_replay")
    grammar = artifact.get("grammar_contract")
    if not isinstance(grammar, Mapping) or grammar.get("serialization_checks_passed") is not True:
        errors.append("grammar_contract")
    hashes = artifact.get("source_artifact_hashes")
    if not isinstance(hashes, Mapping) or any(
        hashes.get(name) != sha256_file(path)
        for name, path in {
            "public_view": public_path,
            "authority_sidecar": authority_path,
            "fixture_manifest": manifest_path,
        }.items()
    ):
        errors.append("fixture_source_hashes")
    return errors


def _date_argument(value: str) -> str:
    """Accept only the fixed V635 execution date."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Build the fixture and accept any valid terminal artifact class."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    args = parser.parse_args(argv)
    root = find_repo_root(start=__file__)
    artifact = build_artifact(root, args.date, tokenizer_loader=resolve_tokenizer_loader())
    errors = validate_artifact(artifact, root)
    if errors:
        print(f"[exp7208] invalid artifact: {errors}", file=sys.stderr, flush=True)
        return 1
    print(
        f"[exp7208] complete verdict={artifact['honest_verdict']} "
        f"score={artifact['span_fixture_ready_score']}",
        flush=True,
    )
    return 0
