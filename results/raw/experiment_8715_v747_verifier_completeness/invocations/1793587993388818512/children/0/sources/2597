"""Build a deterministic span-first relation fixture.

Spec refs: REQ-VERIFY-6926 and SCENARIO-VERIFY-6926-*.

The parser first copies exact source bytes into two span receipts. It reads the
relation phrase only after both spans have one unambiguous source location.
This order keeps anchoring independent from any model claim about semantics.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from carnot import experiment_6913_relation_source_tuple_qualification as source_checker
from carnot import experiment_6914_relation_asp_isomorphic_qualification as exact_checker


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
RESULT_PATH = Path("results/experiment_6926_span_first_relation_fixture.json")
RANDOM_SEED = 2609036926
SCHEMA = "carnot.exp6926.span_first_relation_fixture.v1"
INFERENCE_SUBSTRATE = "deterministic_cpu_span_first_fixture_no_llm"
READY_VERDICT = "complete_circular_positive_span_first_relation_fixture"
BLOCKED_VERDICT = "complete_blocked_span_first_relation_fixture"
FAMILIES = ("graph_coloring", "scheduling", "allocation", "precedence", "exclusion")
LABELS = ("positive", "negative", "unknown")

SOURCE_PATHS = {
    "clean_relation_rows": Path("results/experiment_6912_alias_safe_relation_corpus_reducer.json"),
    "prior_relation_bank": Path("results/experiment_6915_qualified_relation_event_bank.json"),
    "relation_checker": Path(
        "python/carnot/experiment_6913_relation_source_tuple_qualification.py"
    ),
    "asp_isomorphism_checker": Path(
        "python/carnot/experiment_6914_relation_asp_isomorphic_qualification.py"
    ),
    "asp_compiler": Path("python/carnot/asp_energy.py"),
}
EXPECTED_SOURCE_HASHES = {
    "clean_relation_rows": "sha256:5122cd3e95c59ec116dad0d64c03484171a79fe977f9891872da03a8a7a7c2c6",
    "prior_relation_bank": "sha256:530c884a68ba5a46e72586d5a9e844d6ff209af548a4cf204c46e52bf7302470",
    "relation_checker": "sha256:2af255ed40236fc472530795944924a66c61bfbeb0c6209cea0b28e99d01d3ad",
    "asp_isomorphism_checker": "sha256:6e0b14555eaceba7f78e55638c62e4b5bca1f93f591d2d6598caab948ba1f610",
    "asp_compiler": "sha256:0f6077bcd49aa93a6cdbde72422ecf97d905b76b31cadbc0cd401c494af015e1",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "fixture_rows",
    "protocol_rows",
    "utf8_offset_rows",
    "ambiguity_rows",
    "relation_family_rows",
    "label_balance_rows",
    "asp_effect_rows",
    "isomorphism_rows",
    "mutation_rows",
    "partition_rows",
    "heldout_hash_manifest",
    "parser_rejection_rows",
    "random_seed",
    "reproducibility_checksum",
    "span_relation_fixture_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each required field states why its evidence is present.",
    "preconditions_checked": "Input and checker failures stop fixture qualification.",
    "inference_substrate": "The fixed value states that no model inference occurs.",
    "duration_s": "Measured wall time shows that the fixture builder ran.",
    "source_artifact_hashes": "Pinned hashes bind the builder to qualified source tools.",
    "rows": "Per-example and per-mutation rows prevent aggregate-only claims.",
    "fixture_rows": "One row freezes each source, tuple, and expected effect.",
    "protocol_rows": "Prompt rows expose the three-line interface without held answers.",
    "utf8_offset_rows": "Exact byte receipts detect Unicode and boundary drift.",
    "ambiguity_rows": "Repeated-span candidates remain explicit evidence.",
    "relation_family_rows": "Family rows prove that no relation family is absent.",
    "label_balance_rows": "Label rows prove equal positive, negative, and unknown counts.",
    "asp_effect_rows": "Two exact engines record every bounded ASP consequence.",
    "isomorphism_rows": "Injective renamings must preserve projected exact effects.",
    "mutation_rows": "Every replay and invalid mutation has one terminal row.",
    "partition_rows": "Seed and content hashes freeze calibration and held-out groups.",
    "heldout_hash_manifest": "Hashes bind held outputs without placing them in prompts.",
    "parser_rejection_rows": "Every invalid proposal retains its exact rejection reason.",
    "random_seed": "One fixed seed makes partition assignment reproducible.",
    "reproducibility_checksum": "A stable digest detects deterministic evidence drift.",
    "span_relation_fixture_ready_score": "One requires complete balanced exact evidence.",
    "gate_check_summary": "Failed gates retain their expected and observed values.",
    "verifier_is_oracle": "True states that exact code decides bounded correctness.",
    "verdict_class": "The closed class prevents an oracle result from claiming a moat.",
    "honest_verdict": "A complete prefix records a terminal result.",
}

RELATION_FORMS: dict[str, dict[str, tuple[str, str]]] = {
    "graph_coloring": {
        "positive": ("has color", "is colored"),
        "negative": ("does not have color", "is not colored"),
        "unknown": ("color relation unknown", "color is not stated"),
    },
    "scheduling": {
        "positive": ("is scheduled at", "takes place at"),
        "negative": ("is not scheduled at", "does not take place at"),
        "unknown": ("schedule relation unknown", "schedule is not stated"),
    },
    "allocation": {
        "positive": ("is allocated to", "is assigned to"),
        "negative": ("is not allocated to", "is not assigned to"),
        "unknown": ("allocation relation unknown", "allocation is not stated"),
    },
    "precedence": {
        "positive": ("precedes", "comes before"),
        "negative": ("does not precede", "does not come before"),
        "unknown": ("precedence relation unknown", "ordering is not stated"),
    },
    "exclusion": {
        "positive": ("excludes", "rules out"),
        "negative": ("does not exclude", "does not rule out"),
        "unknown": ("exclusion relation unknown", "exclusion is not stated"),
    },
}
PREDICATES = {
    "graph_coloring": "has_color",
    "scheduling": "scheduled_at",
    "allocation": "allocated_to",
    "precedence": "precedes",
    "exclusion": "excludes",
}
PHRASE_INDEX = {
    phrase: (family, PREDICATES[family], label)
    for family, labels in RELATION_FORMS.items()
    for label, phrases in labels.items()
    for phrase in phrases
}

SOURCE_CASES: dict[str, dict[str, tuple[str, str, str]]] = {
    "graph_coloring": {
        "positive": ("Node Café-α has color blue.", "Node Café-α", "blue"),
        "negative": ("Node Café-β does not have color green.", "Node Café-β", "green"),
        "unknown": (
            "The source does not state whether Node Café-γ has color amber.",
            "Node Café-γ",
            "amber",
        ),
    },
    "scheduling": {
        "positive": ("Task Ω-α is scheduled at dawn.", "Task Ω-α", "dawn"),
        "negative": ("Task Ω-β is not scheduled at noon.", "Task Ω-β", "noon"),
        "unknown": (
            "The schedule for Task Ω-γ and slot twilight is unspecified.",
            "Task Ω-γ",
            "twilight",
        ),
    },
    "allocation": {
        "positive": (
            "Server Zürich-α is allocated to Team North.",
            "Server Zürich-α",
            "Team North",
        ),
        "negative": (
            "Server Zürich-β is not allocated to Team South.",
            "Server Zürich-β",
            "Team South",
        ),
        "unknown": (
            "No allocation record connects Server Zürich-γ to Team East.",
            "Server Zürich-γ",
            "Team East",
        ),
    },
    "precedence": {
        "positive": (
            "Task πρώτος-α precedes Task δεύτερος-α.",
            "Task πρώτος-α",
            "Task δεύτερος-α",
        ),
        "negative": (
            "Task πρώτος-β does not precede Task δεύτερος-β.",
            "Task πρώτος-β",
            "Task δεύτερος-β",
        ),
        "unknown": (
            "The ordering between Task πρώτος-γ and Task δεύτερος-γ is unknown.",
            "Task πρώτος-γ",
            "Task δεύτερος-γ",
        ),
    },
    "exclusion": {
        "positive": ("Rule ξ-α excludes Option rouge-α.", "Rule ξ-α", "Option rouge-α"),
        "negative": (
            "Rule ξ-β does not exclude Option rouge-β.",
            "Rule ξ-β",
            "Option rouge-β",
        ),
        "unknown": (
            "The exclusion status of Rule ξ-γ and Option rouge-γ is unstated.",
            "Rule ξ-γ",
            "Option rouge-γ",
        ),
    },
}


def canonical_json(value: Any) -> str:
    """Use stable UTF-8 JSON so hashes do not depend on process ordering."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Return one prefixed SHA-256 digest for an exact byte sequence."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-safe value after deterministic serialization."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash one local input without changing it."""

    try:
        return sha256_bytes(path.read_bytes())
    except OSError:
        return "missing"


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Keep both sides of a gate so a blocked result is actionable."""

    return {
        "check": check,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": expected == observed,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return every failed gate and the first stable failure."""

    copied = [deepcopy(dict(row)) for row in checks]
    failed = [row for row in copied if row.get("passed") is not True]
    first = failed[0] if failed else None
    return {
        "passed": not failed,
        "failed_check": first.get("check") if first else None,
        "expected": first.get("expected") if first else True,
        "observed": first.get("observed") if first else True,
        "failed_checks": failed,
        "checks": copied,
    }


def _rejected(reason: str, **evidence: Any) -> JsonDict:
    """Build one uniform parser rejection without losing prior evidence."""

    return {
        "status": "rejected",
        "reason": reason,
        "relation_interpreted": False,
        **evidence,
    }


def _span_candidates(source_bytes: bytes, span_text: str) -> list[JsonDict]:
    """Find all byte-level occurrences, including overlapping occurrences."""

    needle = span_text.encode("utf-8")
    candidates: list[JsonDict] = []
    start = 0
    while needle and (offset := source_bytes.find(needle, start)) >= 0:
        candidates.append({"start_utf8": offset, "end_utf8": offset + len(needle)})
        start = offset + 1
    return candidates


def parse_proposal(source_bytes: bytes, proposal_text: str) -> JsonDict:
    """Resolve two exact spans before reading one directed relation phrase."""

    try:
        source_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return _rejected("source_invalid_utf8")
    if "\r" in proposal_text:
        return _rejected("carriage_return_not_allowed")
    lines = proposal_text.split("\n")
    if len(lines) != 3:
        return _rejected("line_count_not_three", observed_line_count=len(lines))
    span_a_match = re.fullmatch(r"SPAN_A: (.+)", lines[0])
    if span_a_match is None:
        return _rejected("malformed_span_a_line")
    span_b_match = re.fullmatch(r"SPAN_B: (.+)", lines[1])
    if span_b_match is None:
        return _rejected("malformed_span_b_line")

    span_a_text = span_a_match.group(1)
    span_b_text = span_b_match.group(1)
    span_a_candidates = _span_candidates(source_bytes, span_a_text)
    span_b_candidates = _span_candidates(source_bytes, span_b_text)
    shared = {
        "span_a_candidates": span_a_candidates,
        "span_b_candidates": span_b_candidates,
    }
    if not span_a_candidates:
        return _rejected("span_a_absent", **shared)
    if len(span_a_candidates) != 1:
        return _rejected("span_a_ambiguous", **shared)
    if not span_b_candidates:
        return _rejected("span_b_absent", **shared)
    if len(span_b_candidates) != 1:
        return _rejected("span_b_ambiguous", **shared)

    span_a = {**span_a_candidates[0], "text": span_a_text}
    span_b = {**span_b_candidates[0], "text": span_b_text}
    resolved = {**shared, "span_a": span_a, "span_b": span_b}
    if max(span_a["start_utf8"], span_b["start_utf8"]) < min(
        span_a["end_utf8"], span_b["end_utf8"]
    ):
        return _rejected("spans_overlap", **resolved)
    if span_a["start_utf8"] > span_b["start_utf8"]:
        return _rejected("span_direction_reversed", **resolved)

    relation_match = re.fullmatch(r"RELATION: A -> (.+) -> B", lines[2])
    if relation_match is None:
        return _rejected("malformed_relation_line", **resolved)
    phrase = relation_match.group(1)
    relation = PHRASE_INDEX.get(phrase)
    if relation is None:
        result = _rejected("unsupported_relation_phrase", **resolved)
        result["relation_interpreted"] = True
        result["relation_phrase"] = phrase
        return result
    family, predicate, label = relation
    return {
        "status": "accepted",
        "reason": "accepted",
        "relation_interpreted": True,
        **resolved,
        "relation_phrase": phrase,
        "family": family,
        "canonical_predicate": predicate,
        "label": label,
        "canonical_tuple": [span_a_text, predicate, span_b_text, label],
        "protocol_sha256": sha256_bytes(proposal_text.encode("utf-8")),
    }


def _protocol(span_a: str, span_b: str, phrase: str) -> str:
    """Format exactly the public three-line protocol."""

    return f"SPAN_A: {span_a}\nSPAN_B: {span_b}\nRELATION: A -> {phrase} -> B"


def _effect(models: Sequence[Sequence[str]]) -> JsonDict:
    """Normalize exact models before any engine comparison."""

    canonical = sorted([sorted({str(atom) for atom in model}) for model in models])
    return {
        "satisfiable": bool(canonical),
        "model_count": len(canonical),
        "models": canonical,
        "models_sha256": sha256_json(canonical),
    }


def asp_program_for_tuple(canonical_tuple: Sequence[str], fixture: Mapping[str, Any]) -> str:
    """Compile a canonical tuple to one bounded explicit-polarity program."""

    if len(canonical_tuple) != 4:
        raise ValueError("canonical_tuple_arity")
    label = str(canonical_tuple[3])
    positive_atom = str(fixture["positive_atom"])
    negative_atom = str(fixture["negative_atom"])
    program = f":- {positive_atom}, {negative_atom}.\n"
    if label == "positive":
        return program + f"{positive_atom}.\n"
    if label == "negative":
        return program + f"{negative_atom}.\n"
    if label == "unknown":
        return program
    raise ValueError("unsupported_relation_label")


def generate_fixture_rows(seed: int = RANDOM_SEED) -> list[JsonDict]:
    """Create one immutable source row for every family and label cell."""

    rows: list[JsonDict] = []
    for family_index, family in enumerate(FAMILIES):
        for label_index, label in enumerate(LABELS):
            source_text, span_a, span_b = SOURCE_CASES[family][label]
            source_bytes = source_text.encode("utf-8")
            fixture_id = f"{family}_{label}"
            entity_a_id = f"a{family_index}_{label_index}"
            entity_b_id = f"b{family_index}_{label_index}"
            predicate = PREDICATES[family]
            positive_atom = f"{predicate}_{entity_a_id}_{entity_b_id}_positive"
            negative_atom = f"{predicate}_{entity_a_id}_{entity_b_id}_negative"
            canonical_tuple = [span_a, predicate, span_b, label]
            canonical_phrase, paraphrase = RELATION_FORMS[family][label]
            fixture: JsonDict = {
                "row_type": "fixture_example",
                "fixture_id": fixture_id,
                "group_id": fixture_id,
                "family": family,
                "label": label,
                "seed": seed,
                "source_text": source_text,
                "source_bytes": source_bytes,
                "source_bytes_b64": base64.b64encode(source_bytes).decode("ascii"),
                "source_text_hash": sha256_bytes(source_bytes),
                "span_a_text": span_a,
                "span_b_text": span_b,
                "entity_a_id": entity_a_id,
                "entity_b_id": entity_b_id,
                "canonical_predicate": predicate,
                "canonical_tuple": canonical_tuple,
                "canonical_relation_phrase": canonical_phrase,
                "paraphrase_relation_phrase": paraphrase,
                "protocol_text": _protocol(span_a, span_b, canonical_phrase),
                "paraphrase_protocol_text": _protocol(span_a, span_b, paraphrase),
                "positive_atom": positive_atom,
                "negative_atom": negative_atom,
            }
            fixture["asp_program"] = asp_program_for_tuple(canonical_tuple, fixture)
            expected_models = (
                [[positive_atom]]
                if label == "positive"
                else [[negative_atom]]
                if label == "negative"
                else [[]]
            )
            fixture["expected_asp_effect"] = _effect(expected_models)
            content = {key: value for key, value in fixture.items() if key != "source_bytes"}
            fixture["content_hash"] = sha256_json(content)
            rows.append(fixture)
    return rows


def check_relation_semantics(parsed: Mapping[str, Any], fixture: Mapping[str, Any]) -> JsonDict:
    """Compare a parsed tuple with the source row's exact directed meaning."""

    if parsed.get("status") != "accepted":
        return {
            "passed": False,
            "reason": parsed.get("reason", "proposal_not_accepted"),
            "canonical_tuple": parsed.get("canonical_tuple"),
        }
    observed = list(parsed.get("canonical_tuple", []))
    expected = list(fixture["canonical_tuple"])
    if parsed.get("family") != fixture.get("family"):
        reason = "relation_family_conflicts_source"
    elif observed != expected:
        reason = "relation_label_conflicts_source"
    else:
        reason = "accepted"
    return {
        "passed": reason == "accepted",
        "reason": reason,
        "canonical_tuple": observed,
        "expected_tuple": expected,
    }


def evaluate_asp_effect(fixture: Mapping[str, Any]) -> JsonDict:
    """Run the qualified energy compiler and independent clingo solver."""

    program = str(fixture["asp_program"])
    fixture_id = str(fixture["fixture_id"])
    primary = exact_checker.primary_exact_engine(program, fixture_id)
    independent = exact_checker.independent_exact_engine(program, fixture_id)
    primary_effect = _effect(primary["models"])
    independent_effect = _effect(independent["models"])
    expected = deepcopy(fixture["expected_asp_effect"])
    return {
        "row_type": "asp_effect",
        "fixture_id": fixture_id,
        "program_sha256": sha256_bytes(program.encode("utf-8")),
        "primary_effect": primary_effect,
        "independent_effect": independent_effect,
        "expected_effect": expected,
        "solver_parity": primary_effect == independent_effect,
        "expected_effect_met": primary_effect == expected,
        "primary_solver_receipt": deepcopy(primary["receipt"]),
        "independent_solver_receipt": deepcopy(independent["receipt"]),
        "terminal": True,
    }


def rename_asp_program(program: str, atom_map: Mapping[str, str]) -> str:
    """Apply an injective atom map and expose a plain failure type."""

    if len(set(atom_map.values())) != len(atom_map):
        raise ValueError("non_injective_renaming")
    try:
        return exact_checker.rename_program(program, atom_map)
    except exact_checker.QualificationError as exc:
        raise ValueError(str(exc)) from exc


def evaluate_isomorphism(fixture: Mapping[str, Any]) -> JsonDict:
    """Rename both entity-bearing atoms and project exact models back."""

    positive_atom = str(fixture["positive_atom"])
    negative_atom = str(fixture["negative_atom"])
    atom_map = {
        positive_atom: positive_atom + "_renamed",
        negative_atom: negative_atom + "_renamed",
    }
    renamed_program = rename_asp_program(str(fixture["asp_program"]), atom_map)
    program_id = str(fixture["fixture_id"]) + "_isomorphic"
    primary = exact_checker.primary_exact_engine(renamed_program, program_id)
    independent = exact_checker.independent_exact_engine(renamed_program, program_id)
    inverse = {value: key for key, value in atom_map.items()}
    projected_models = [
        [inverse.get(str(atom), str(atom)) for atom in model] for model in primary["models"]
    ]
    projected_effect = _effect(projected_models)
    expected = deepcopy(fixture["expected_asp_effect"])
    return {
        "row_type": "isomorphism",
        "fixture_id": fixture["fixture_id"],
        "entity_map": {
            fixture["entity_a_id"]: str(fixture["entity_a_id"]) + "_renamed",
            fixture["entity_b_id"]: str(fixture["entity_b_id"]) + "_renamed",
        },
        "atom_map": atom_map,
        "renamed_tuple": [
            str(fixture["span_a_text"]) + " renamed",
            fixture["canonical_predicate"],
            str(fixture["span_b_text"]) + " renamed",
            fixture["label"],
        ],
        "renamed_program": renamed_program,
        "renamed_program_sha256": sha256_bytes(renamed_program.encode("utf-8")),
        "injective": len(set(atom_map.values())) == len(atom_map),
        "solver_parity": _effect(primary["models"]) == _effect(independent["models"]),
        "projected_effect": projected_effect,
        "expected_effect": expected,
        "invariant": projected_effect == expected,
        "primary_solver_receipt": deepcopy(primary["receipt"]),
        "independent_solver_receipt": deepcopy(independent["receipt"]),
        "terminal": True,
    }


def _json_fixture(fixture: Mapping[str, Any], partition: str) -> JsonDict:
    """Remove the in-memory bytes while retaining their exact base64 form."""

    row = {key: deepcopy(value) for key, value in fixture.items() if key != "source_bytes"}
    row["partition"] = partition
    row["terminal"] = True
    return row


def build_partitions(
    fixtures: Sequence[Mapping[str, Any]], seed: int
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:
    """Assign one third of content-hashed rows to the held-out partition."""

    ranked = sorted(
        (
            sha256_bytes(f"{seed}:{row['content_hash']}".encode()),
            str(row["fixture_id"]),
        )
        for row in fixtures
    )
    held_count = max(1, len(ranked) // 3)
    held_ids = {fixture_id for _, fixture_id in ranked[:held_count]}
    partition_rows: list[JsonDict] = []
    protocol_rows: list[JsonDict] = []
    held_hashes: list[JsonDict] = []
    for row in fixtures:
        fixture_id = str(row["fixture_id"])
        assignment_hash = sha256_bytes(f"{seed}:{row['content_hash']}".encode())
        partition = "heldout" if fixture_id in held_ids else "calibration"
        partition_rows.append(
            {
                "row_type": "partition",
                "fixture_id": fixture_id,
                "group_id": row["group_id"],
                "partition": partition,
                "seed": seed,
                "content_hash": row["content_hash"],
                "assignment_hash": assignment_hash,
                "terminal": True,
            }
        )
        prompt = (
            f"SOURCE:\n{row['source_text']}\n\n"
            "Copy two verbatim source spans. Then name their directed relation.\n"
            "Return exactly these three lines:\n"
            "SPAN_A: <verbatim source span>\n"
            "SPAN_B: <verbatim source span>\n"
            "RELATION: A -> <relation phrase> -> B"
        )
        protocol_rows.append(
            {
                "row_type": "live_prompt",
                "fixture_id": fixture_id,
                "partition": partition,
                "source_text_hash": row["source_text_hash"],
                "live_prompt": prompt,
                "live_prompt_hash": sha256_bytes(prompt.encode("utf-8")),
                "expected_output_exposed": False,
                "terminal": True,
            }
        )
        if partition == "heldout":
            held_hashes.append(
                {
                    "fixture_id": fixture_id,
                    "content_hash": row["content_hash"],
                    "expected_output_hash": sha256_json(
                        {
                            "tuple": row["canonical_tuple"],
                            "effect": row["expected_asp_effect"],
                        }
                    ),
                }
            )
    partition_rows.sort(key=lambda row: row["fixture_id"])
    protocol_rows.sort(key=lambda row: row["fixture_id"])
    held_hashes.sort(key=lambda row: row["fixture_id"])
    manifest = {
        "seed": seed,
        "count": len(held_hashes),
        "rows": held_hashes,
        "manifest_hash": sha256_json(held_hashes),
        "expected_outputs_exposed_in_live_prompts": False,
    }
    return partition_rows, protocol_rows, manifest


def _finalize_mutation(row: JsonDict) -> JsonDict:
    """Hash one terminal mutation after its exact outcome is known."""

    row["terminal"] = True
    row["content_hash"] = sha256_json(row)
    return row


def _proposal_mutation(
    mutation_id: str,
    source_bytes: bytes,
    proposal: str,
    expected_reason: str,
    fixture: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Run one invalid proposal and retain its exact parser or semantic reason."""

    parsed = parse_proposal(source_bytes, proposal)
    semantic = check_relation_semantics(parsed, fixture) if fixture is not None else None
    reason = str(semantic["reason"] if semantic is not None else parsed["reason"])
    return _finalize_mutation(
        {
            "row_type": "mutation_replay",
            "mutation_id": mutation_id,
            "mutation": mutation_id,
            "fixture_id": fixture.get("fixture_id") if fixture is not None else None,
            "proposal": proposal,
            "parser_result": parsed,
            "semantic_result": semantic,
            "expected_reason": expected_reason,
            "reason": reason,
            "valid": False,
            "expected_reason_met": reason == expected_reason,
        }
    )


def build_mutation_rows(fixtures: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Emit base, paraphrase, isomorphism, and adversarial mutation rows."""

    rows: list[JsonDict] = []
    for fixture in fixtures:
        for mutation, protocol_key in (
            ("protocol_replay", "protocol_text"),
            ("paraphrase", "paraphrase_protocol_text"),
        ):
            parsed = parse_proposal(fixture["source_bytes"], str(fixture[protocol_key]))
            semantic = check_relation_semantics(parsed, fixture)
            asp = evaluate_asp_effect(fixture)
            rows.append(
                _finalize_mutation(
                    {
                        "row_type": "mutation_replay",
                        "mutation_id": f"{fixture['fixture_id']}:{mutation}",
                        "mutation": mutation,
                        "fixture_id": fixture["fixture_id"],
                        "parser_result": parsed,
                        "semantic_result": semantic,
                        "asp_result": asp,
                        "reason": "accepted" if semantic["passed"] else semantic["reason"],
                        "valid": bool(
                            semantic["passed"]
                            and asp["solver_parity"]
                            and asp["expected_effect_met"]
                        ),
                    }
                )
            )
        isomorphism = evaluate_isomorphism(fixture)
        rows.append(
            _finalize_mutation(
                {
                    "row_type": "mutation_replay",
                    "mutation_id": f"{fixture['fixture_id']}:entity_renaming",
                    "mutation": "entity_renaming",
                    "fixture_id": fixture["fixture_id"],
                    "isomorphism_result": isomorphism,
                    "reason": "accepted" if isomorphism["invariant"] else "isomorphism_failed",
                    "valid": bool(isomorphism["invariant"] and isomorphism["solver_parity"]),
                }
            )
        )

    rows.extend(
        [
            _proposal_mutation(
                "span_a_ambiguous",
                b"A links B and A links C",
                _protocol("A", "B", "precedes"),
                "span_a_ambiguous",
            ),
            _proposal_mutation(
                "spans_overlap",
                b"tokenAB",
                _protocol("tokenA", "AB", "precedes"),
                "spans_overlap",
            ),
            _proposal_mutation(
                "span_direction_reversed",
                b"B appears before A",
                _protocol("A", "B", "precedes"),
                "span_direction_reversed",
            ),
            _proposal_mutation(
                "span_a_absent",
                b"A then B",
                _protocol("missing", "B", "precedes"),
                "span_a_absent",
            ),
            _proposal_mutation(
                "line_count_not_three",
                b"A then B",
                "SPAN_A: A\nSPAN_B: B",
                "line_count_not_three",
            ),
            _proposal_mutation(
                "malformed_span_a_line",
                b"A then B",
                "SPAN_B: B\nSPAN_A: A\nRELATION: A -> precedes -> B",
                "malformed_span_a_line",
            ),
            _proposal_mutation(
                "malformed_relation_line",
                b"A then B",
                "SPAN_A: A\nSPAN_B: B\nREL: A -> precedes -> B",
                "malformed_relation_line",
            ),
            _proposal_mutation(
                "unsupported_relation_phrase",
                b"A then B",
                _protocol("A", "B", "invented relation"),
                "unsupported_relation_phrase",
            ),
        ]
    )
    negative = next(
        row for row in fixtures if row["family"] == "allocation" and row["label"] == "negative"
    )
    rows.append(
        _proposal_mutation(
            "relation_label_conflicts_source",
            negative["source_bytes"],
            _protocol(
                str(negative["span_a_text"]),
                str(negative["span_b_text"]),
                "is allocated to",
            ),
            "relation_label_conflicts_source",
            negative,
        )
    )
    return rows


def _source_receipts() -> dict[str, JsonDict]:
    """Read current hashes for every pinned source input."""

    return {
        name: {
            "path": str(path),
            "expected_sha256": EXPECTED_SOURCE_HASHES[name],
            "observed_sha256": sha256_file(REPO_ROOT / path),
        }
        for name, path in SOURCE_PATHS.items()
    }


def _read_json(path: Path) -> Any:
    """Read one precondition artifact without accepting non-JSON data."""

    return json.loads(path.read_text(encoding="utf-8"))


def check_preconditions() -> tuple[list[JsonDict], dict[str, JsonDict]]:
    """Check clean rows, exact engines, source hashes, and UTF-8 stability."""

    receipts = _source_receipts()
    checks = [
        gate_check(f"source_hash:{name}", row["expected_sha256"], row["observed_sha256"])
        for name, row in receipts.items()
    ]
    try:
        clean = _read_json(REPO_ROOT / SOURCE_PATHS["clean_relation_rows"])
        prior = _read_json(REPO_ROOT / SOURCE_PATHS["prior_relation_bank"])
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        checks.append(gate_check("source_rows_readable", True, f"{type(exc).__name__}:{exc}"))
        return checks, receipts
    clean_rows = clean.get("rows") if isinstance(clean, Mapping) else None
    prior_rows = prior.get("rows") if isinstance(prior, Mapping) else None
    checks.extend(
        [
            gate_check("source_rows_readable", True, bool(clean_rows) and bool(prior_rows)),
            gate_check(
                "clean_relation_corpus_ready_score",
                1,
                clean.get("clean_relation_corpus_ready_score")
                if isinstance(clean, Mapping)
                else None,
            ),
            gate_check(
                "prior_qualified_event_count",
                8,
                sum(row.get("eligible") is True for row in prior_rows if isinstance(row, Mapping))
                if isinstance(prior_rows, list)
                else None,
            ),
            gate_check(
                "relation_checker_callable", True, callable(source_checker.check_source_span)
            ),
            gate_check("asp_checker_callable", True, callable(exact_checker.primary_exact_engine)),
            gate_check(
                "isomorphism_checker_callable", True, callable(exact_checker.rename_program)
            ),
        ]
    )
    sentinel = "éΩB".encode()
    checks.append(
        gate_check(
            "stable_utf8_byte_handling",
            {"omega_start": 2, "omega_end": 4, "round_trip": True},
            {
                "omega_start": sentinel.index("Ω".encode()),
                "omega_end": sentinel.index("Ω".encode()) + len("Ω".encode()),
                "round_trip": sentinel.decode("utf-8").encode("utf-8") == sentinel,
            },
        )
    )
    try:
        import clingo

        solver_observed: Any = bool(clingo.__version__)
    except ImportError:
        solver_observed = False
    checks.append(gate_check("independent_clingo_available", True, solver_observed))
    return checks, receipts


def _derived_rows(
    fixtures: Sequence[Mapping[str, Any]],
    partition_rows: Sequence[Mapping[str, Any]],
    mutation_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Derive every required evidence table from terminal rows."""

    partitions = {str(row["fixture_id"]): str(row["partition"]) for row in partition_rows}
    fixture_rows = [_json_fixture(row, partitions[str(row["fixture_id"])]) for row in fixtures]
    utf8_rows: list[JsonDict] = []
    asp_rows: list[JsonDict] = []
    isomorphism_rows: list[JsonDict] = []
    for fixture in fixtures:
        parsed = parse_proposal(fixture["source_bytes"], str(fixture["protocol_text"]))
        utf8_rows.append(
            {
                "row_type": "utf8_offset",
                "fixture_id": fixture["fixture_id"],
                "source_text_hash": fixture["source_text_hash"],
                "span_a": parsed.get("span_a"),
                "span_b": parsed.get("span_b"),
                "span_a_bytes_b64": base64.b64encode(
                    str(fixture["span_a_text"]).encode("utf-8")
                ).decode("ascii"),
                "span_b_bytes_b64": base64.b64encode(
                    str(fixture["span_b_text"]).encode("utf-8")
                ).decode("ascii"),
                "byte_identity": parsed.get("status") == "accepted",
                "terminal": True,
            }
        )
        asp_rows.append(evaluate_asp_effect(fixture))
        isomorphism_rows.append(evaluate_isomorphism(fixture))
    family_rows = [
        {
            "family": family,
            "fixture_count": sum(row["family"] == family for row in fixture_rows),
            "labels": sorted(row["label"] for row in fixture_rows if row["family"] == family),
            "terminal": True,
        }
        for family in FAMILIES
    ]
    label_rows = [
        {
            "label": label,
            "fixture_count": sum(row["label"] == label for row in fixture_rows),
            "families": sorted(row["family"] for row in fixture_rows if row["label"] == label),
            "terminal": True,
        }
        for label in LABELS
    ]
    rejection_rows = [deepcopy(row) for row in mutation_rows if row.get("valid") is False]
    ambiguity_rows = [
        deepcopy(row) for row in rejection_rows if str(row.get("reason", "")).endswith("_ambiguous")
    ]
    return {
        "fixture_rows": fixture_rows,
        "utf8_offset_rows": utf8_rows,
        "ambiguity_rows": ambiguity_rows,
        "relation_family_rows": family_rows,
        "label_balance_rows": label_rows,
        "asp_effect_rows": asp_rows,
        "isomorphism_rows": isomorphism_rows,
        "parser_rejection_rows": rejection_rows,
    }


def _readiness_checks(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Compute readiness only from stored per-row evidence."""

    expected_cells = sorted(
        [list(cell) for cell in ((f, label) for f in FAMILIES for label in LABELS)]
    )
    observed_cells = sorted(
        [
            [str(row.get("family")), str(row.get("label"))]
            for row in artifact.get("fixture_rows", [])
            if isinstance(row, Mapping)
        ]
    )
    mutations = [row for row in artifact.get("mutation_rows", []) if isinstance(row, Mapping)]
    asp_rows = [row for row in artifact.get("asp_effect_rows", []) if isinstance(row, Mapping)]
    iso_rows = [row for row in artifact.get("isomorphism_rows", []) if isinstance(row, Mapping)]
    partitions = [row for row in artifact.get("partition_rows", []) if isinstance(row, Mapping)]
    return [
        gate_check(
            "preconditions_passed", True, artifact.get("preconditions_checked", {}).get("passed")
        ),
        gate_check("family_label_cells", expected_cells, observed_cells),
        gate_check("fixture_row_count", 15, len(observed_cells)),
        gate_check("partition_row_count", 15, len(partitions)),
        gate_check(
            "partition_set",
            ["calibration", "heldout"],
            sorted({row.get("partition") for row in partitions}),
        ),
        gate_check("ambiguity_explicit", True, bool(artifact.get("ambiguity_rows"))),
        gate_check(
            "asp_solver_disagreement_count",
            0,
            sum(row.get("solver_parity") is not True for row in asp_rows),
        ),
        gate_check(
            "asp_expected_effect_failure_count",
            0,
            sum(row.get("expected_effect_met") is not True for row in asp_rows),
        ),
        gate_check(
            "isomorphism_failure_count",
            0,
            sum(
                row.get("invariant") is not True or row.get("solver_parity") is not True
                for row in iso_rows
            ),
        ),
        gate_check(
            "valid_mutation_failure_count",
            0,
            sum(
                row.get("valid") is not True
                for row in mutations
                if row.get("mutation") in {"protocol_replay", "paraphrase", "entity_renaming"}
            ),
        ),
        gate_check(
            "invalid_reason_mismatch_count",
            0,
            sum(
                row.get("expected_reason_met") is not True
                for row in mutations
                if row.get("valid") is False
            ),
        ),
        gate_check(
            "row_union_count",
            len(artifact.get("fixture_rows", [])) + len(mutations),
            len(artifact.get("rows", [])),
        ),
        gate_check(
            "held_outputs_exposed",
            False,
            any(
                row.get("expected_output_exposed") is not False
                for row in artifact.get("protocol_rows", [])
            ),
        ),
    ]


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic content while excluding measured wall time."""

    payload = deepcopy(dict(artifact))
    payload.pop("duration_s", None)
    payload.pop("reproducibility_checksum", None)
    return sha256_json(payload)


def _blocked_artifact(
    date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
) -> JsonDict:
    """Return the complete fail-closed shape without building fixture rows."""

    summary = gate_summary(checks)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 6926,
        "run_date": date,
        "status": "blocked",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": summary,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [],
        "fixture_rows": [],
        "protocol_rows": [],
        "utf8_offset_rows": [],
        "ambiguity_rows": [],
        "relation_family_rows": [],
        "label_balance_rows": [],
        "asp_effect_rows": [],
        "isomorphism_rows": [],
        "mutation_rows": [],
        "partition_rows": [],
        "heldout_hash_manifest": {"seed": RANDOM_SEED, "count": 0, "rows": []},
        "parser_rejection_rows": [],
        "random_seed": RANDOM_SEED,
        "span_relation_fixture_ready_score": 0,
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact(
    *,
    date: str,
    duration_s: float,
    precondition_checks: Sequence[Mapping[str, Any]] | None = None,
    source_hashes: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build a complete ready artifact or a complete blocked artifact."""

    if precondition_checks is None or source_hashes is None:
        real_checks, real_hashes = check_preconditions()
        checks = list(precondition_checks) if precondition_checks is not None else real_checks
        hashes = dict(source_hashes) if source_hashes is not None else real_hashes
    else:
        checks = list(precondition_checks)
        hashes = dict(source_hashes)
    preconditions = gate_summary(checks)
    if not preconditions["passed"]:
        return _blocked_artifact(date, duration_s, checks, hashes)

    fixtures = generate_fixture_rows(RANDOM_SEED)
    partition_rows, protocol_rows, held_manifest = build_partitions(fixtures, RANDOM_SEED)
    mutations = build_mutation_rows(fixtures)
    derived = _derived_rows(fixtures, partition_rows, mutations)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 6926,
        "run_date": date,
        "status": "complete",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": hashes,
        **derived,
        "protocol_rows": protocol_rows,
        "mutation_rows": mutations,
        "partition_rows": partition_rows,
        "heldout_hash_manifest": held_manifest,
        "random_seed": RANDOM_SEED,
        "span_relation_fixture_ready_score": 0,
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "verdict_class": "partial",
        "honest_verdict": "complete_partial_span_first_relation_fixture",
    }
    artifact["rows"] = deepcopy(artifact["fixture_rows"]) + deepcopy(mutations)
    readiness = gate_summary(_readiness_checks(artifact))
    artifact["gate_check_summary"] = readiness
    if readiness["passed"]:
        artifact["span_relation_fixture_ready_score"] = 1
        artifact["verdict_class"] = "circular_positive"
        artifact["honest_verdict"] = READY_VERDICT
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


REPLAY_FIELDS = (
    "rows",
    "fixture_rows",
    "protocol_rows",
    "utf8_offset_rows",
    "ambiguity_rows",
    "relation_family_rows",
    "label_balance_rows",
    "asp_effect_rows",
    "isomorphism_rows",
    "mutation_rows",
    "partition_rows",
    "heldout_hash_manifest",
    "parser_rejection_rows",
    "random_seed",
    "span_relation_fixture_ready_score",
    "gate_check_summary",
    "verdict_class",
    "honest_verdict",
)


def replay_artifact(artifact: Mapping[str, Any]) -> JsonDict:
    """Rebuild all deterministic evidence for a fresh-process comparison."""

    rebuilt = build_artifact(
        date=str(artifact.get("run_date", "")),
        duration_s=float(artifact.get("duration_s", 0.0) or 0.0),
        precondition_checks=artifact.get("preconditions_checked", {}).get("checks", []),
        source_hashes=artifact.get("source_artifact_hashes", {}),
    )
    mismatched = [field for field in REPLAY_FIELDS if artifact.get(field) != rebuilt.get(field)]
    return {
        "agreement": not mismatched,
        "mismatched_fields": mismatched,
        "stored_checksum": artifact.get("reproducibility_checksum"),
        "recomputed_checksum": artifact_checksum(artifact),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Reject missing fields, stale hashes, and inconsistent terminal verdicts."""

    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    principles = artifact.get("field_principles", {})
    if not isinstance(principles, Mapping) or any(
        field not in principles for field in REQUIRED_ARTIFACT_FIELDS
    ):
        errors.append("field_principles")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    if artifact.get("verdict_class") not in {
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    blocked = artifact.get("verdict_class") == "blocked"
    if blocked:
        if artifact.get("span_relation_fixture_ready_score") != 0:
            errors.append("blocked_ready_score")
        if artifact.get("gate_check_summary", {}).get("passed") is not False:
            errors.append("blocked_gate_summary")
        if artifact.get("honest_verdict") != BLOCKED_VERDICT:
            errors.append("blocked_honest_verdict")
    else:
        readiness = gate_summary(_readiness_checks(artifact))
        expected_score = int(readiness["passed"])
        if artifact.get("span_relation_fixture_ready_score") != expected_score:
            errors.append("ready_score_drift")
        if artifact.get("gate_check_summary") != readiness:
            errors.append("gate_summary_drift")
        if not replay_artifact(artifact)["agreement"]:
            errors.append("deterministic_replay")
    return list(dict.fromkeys(errors))


def _write_json(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace the result only after the complete JSON is on disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    temporary.replace(path)


def run_experiment(*, date: str, output_path: Path = RESULT_PATH) -> JsonDict:
    """Check preconditions, build the fixture, validate it, and write once."""

    started = time.perf_counter()
    checks, hashes = check_preconditions()
    artifact = build_artifact(
        date=date,
        duration_s=time.perf_counter() - started,
        precondition_checks=checks,
        source_hashes=hashes,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError("artifact_validation_failed:" + ",".join(errors))
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _write_json(output_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run the required builder command or verify an existing artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260903")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify is not None:
        try:
            artifact = _read_json(args.verify)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            print(canonical_json({"agreement": False, "error": f"{type(exc).__name__}:{exc}"}))
            return 1
        replay = replay_artifact(artifact if isinstance(artifact, Mapping) else {})
        errors = validate_artifact(artifact if isinstance(artifact, Mapping) else {})
        print(canonical_json({"replay": replay, "validation_errors": errors}))
        return int(bool(errors) or not replay["agreement"])
    artifact = run_experiment(date=args.date, output_path=args.output)
    print(
        canonical_json(
            {
                "output": str(args.output),
                "ready_score": artifact["span_relation_fixture_ready_score"],
                "honest_verdict": artifact["honest_verdict"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository uses the thin script entrypoint.
    raise SystemExit(main())
