"""Acquire span-first relation proposals from three local GGUF families.

The model copies two source spans before it names their directed relation.
Exact byte grounding runs before alias normalization. This ordering prevents a
plausible relation label from rescuing spans that are absent or ambiguous.

Spec refs: REQ-REPORT-6929 and SCENARIO-REPORT-6929-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot.inference.sota_models import resolve_cached_gguf


JsonDict = dict[str, Any]
Resolver = Callable[[str, str], str | None]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_6929_three_family_span_acquisition.json")
SCRIPT_PATH = REPO_ROOT / "scripts/experiments/experiment_6929_three_family_span_acquisition.py"
EXP6926_PATH = Path("results/experiment_6926_span_first_relation_fixture.json")
EXP6926_SHA256 = "sha256:a33fef76714bc9cbe502048457d951e26f3e1819ced1e3410cc0a3f977a3d58b"
EXP6926_SCHEMA = "carnot.exp6926.span_first_relation_fixture.v1"
SCHEMA = "carnot.exp6929.three_family_span_acquisition.v1"
INFERENCE_SUBSTRATE = "live_local_gguf_span_first_relation_acquisition"
RANDOM_SEED = 2609036929
EXAMPLES_PER_CELL = 6
MODEL_TIMEOUT_S = 7200.0
CONTEXT_SIZE = 1024
OFFLOAD_LAYERS = -1

MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "name": "Qwen3.6-35B-A3B",
        "model_family": "qwen3.6_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "quantization": "Q4_K_M",
    },
    {
        "name": "Gemma4-31B-it",
        "model_family": "gemma4_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "quantization": "Q4_K_M",
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "model_family": "gemma4_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "quantization": "Q4_K_M",
    },
)
FAMILIES = ("graph_coloring", "scheduling", "allocation", "precedence", "exclusion")
DECODING_SETTINGS: JsonDict = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 40,
    "repeat_penalty": 1.0,
    "max_tokens": 128,
}


def _case(
    family: str,
    index: int,
    source_text: str,
    span_a: str,
    span_b: str,
    hidden_label: str,
) -> JsonDict:
    """Build one immutable case while keeping hidden fields out of prompts."""

    return {
        "prompt_id": f"heldout_{family}_{index:02d}",
        "family": family,
        "seed": RANDOM_SEED + index,
        "source_text": source_text,
        "span_a": span_a,
        "span_b": span_b,
        "hidden_label": hidden_label,
        "split": "heldout",
    }


HELDOUT_CASES: tuple[JsonDict, ...] = (
    _case(
        "graph_coloring",
        0,
        "Vertex Montréal-7 has color cobalt.",
        "Vertex Montréal-7",
        "cobalt",
        "positive",
    ),
    _case(
        "graph_coloring",
        1,
        "Node Łódź-8 is colored with amber.",
        "Node Łódź-8",
        "amber",
        "positive",
    ),
    _case(
        "graph_coloring",
        2,
        "Vertex 東京-9 does not have color green.",
        "Vertex 東京-9",
        "green",
        "negative",
    ),
    _case("graph_coloring", 3, "Node Γ-10 is not colored red.", "Node Γ-10", "red", "negative"),
    _case(
        "graph_coloring",
        4,
        "No color is stated for Vertex Zürich-11 and violet.",
        "Vertex Zürich-11",
        "violet",
        "unknown",
    ),
    _case(
        "graph_coloring",
        5,
        "The source gives no color relation for Node Київ-12 and ochre.",
        "Node Київ-12",
        "ochre",
        "unknown",
    ),
    _case(
        "scheduling", 0, "Task Café-21 is scheduled at dawn.", "Task Café-21", "dawn", "positive"
    ),
    _case("scheduling", 1, "Job Ω-22 takes place at noon.", "Job Ω-22", "noon", "positive"),
    _case(
        "scheduling",
        2,
        "Task 東京-23 is not scheduled at dusk.",
        "Task 東京-23",
        "dusk",
        "negative",
    ),
    _case(
        "scheduling",
        3,
        "Job Δ-24 does not take place at midnight.",
        "Job Δ-24",
        "midnight",
        "negative",
    ),
    _case(
        "scheduling",
        4,
        "No schedule is stated for Task Zürich-25 and sunrise.",
        "Task Zürich-25",
        "sunrise",
        "unknown",
    ),
    _case(
        "scheduling",
        5,
        "The source gives no timing for Job Київ-26 and twilight.",
        "Job Київ-26",
        "twilight",
        "unknown",
    ),
    _case(
        "allocation",
        0,
        "Server Montréal-31 is allocated to Team Cedar.",
        "Server Montréal-31",
        "Team Cedar",
        "positive",
    ),
    _case(
        "allocation",
        1,
        "Unit Łódź-32 is assigned to Group Birch.",
        "Unit Łódź-32",
        "Group Birch",
        "positive",
    ),
    _case(
        "allocation",
        2,
        "Server 東京-33 is not allocated to Team Maple.",
        "Server 東京-33",
        "Team Maple",
        "negative",
    ),
    _case(
        "allocation",
        3,
        "Unit Γ-34 is not assigned to Group Pine.",
        "Unit Γ-34",
        "Group Pine",
        "negative",
    ),
    _case(
        "allocation",
        4,
        "No allocation is stated for Server Zürich-35 and Team Elm.",
        "Server Zürich-35",
        "Team Elm",
        "unknown",
    ),
    _case(
        "allocation",
        5,
        "The source gives no assignment for Unit Київ-36 and Group Ash.",
        "Unit Київ-36",
        "Group Ash",
        "unknown",
    ),
    _case(
        "precedence",
        0,
        "Task Café-41 precedes Task Cedar-41.",
        "Task Café-41",
        "Task Cedar-41",
        "positive",
    ),
    _case(
        "precedence",
        1,
        "Job Ω-42 comes before Job Birch-42.",
        "Job Ω-42",
        "Job Birch-42",
        "positive",
    ),
    _case(
        "precedence",
        2,
        "Task 東京-43 does not precede Task Maple-43.",
        "Task 東京-43",
        "Task Maple-43",
        "negative",
    ),
    _case(
        "precedence",
        3,
        "Job Δ-44 does not come before Job Pine-44.",
        "Job Δ-44",
        "Job Pine-44",
        "negative",
    ),
    _case(
        "precedence",
        4,
        "No ordering is stated between Task Zürich-45 and Task Elm-45.",
        "Task Zürich-45",
        "Task Elm-45",
        "unknown",
    ),
    _case(
        "precedence",
        5,
        "The source gives no precedence relation for Job Київ-46 and Job Ash-46.",
        "Job Київ-46",
        "Job Ash-46",
        "unknown",
    ),
    _case(
        "exclusion",
        0,
        "Rule Café-51 excludes Option Cedar-51.",
        "Rule Café-51",
        "Option Cedar-51",
        "positive",
    ),
    _case(
        "exclusion",
        1,
        "Policy Ω-52 rules out Choice Birch-52.",
        "Policy Ω-52",
        "Choice Birch-52",
        "positive",
    ),
    _case(
        "exclusion",
        2,
        "Rule 東京-53 does not exclude Option Maple-53.",
        "Rule 東京-53",
        "Option Maple-53",
        "negative",
    ),
    _case(
        "exclusion",
        3,
        "Policy Δ-54 does not rule out Choice Pine-54.",
        "Policy Δ-54",
        "Choice Pine-54",
        "negative",
    ),
    _case(
        "exclusion",
        4,
        "No exclusion is stated for Rule Zürich-55 and Option Elm-55.",
        "Rule Zürich-55",
        "Option Elm-55",
        "unknown",
    ),
    _case(
        "exclusion",
        5,
        "The source gives no exclusion relation for Policy Київ-56 and Choice Ash-56.",
        "Policy Київ-56",
        "Choice Ash-56",
        "unknown",
    ),
)
EXPECTED_ATTEMPTS = len(MODEL_SPECS) * len(HELDOUT_CASES)
EXPECTED_HELDOUT_SPLIT_HASH = (
    "sha256:a85456bc8d048b16f7ab1fdf4522ff40a680dfdcceba4ef9ad96701c90437d81"
)

RELATION_ALIASES: dict[str, tuple[str, str, str]] = {
    "has color": ("graph_coloring", "has_color", "positive"),
    "is colored": ("graph_coloring", "has_color", "positive"),
    "is colored with": ("graph_coloring", "has_color", "positive"),
    "does not have color": ("graph_coloring", "has_color", "negative"),
    "is not colored": ("graph_coloring", "has_color", "negative"),
    "color relation not stated": ("graph_coloring", "has_color", "unknown"),
    "no color relation": ("graph_coloring", "has_color", "unknown"),
    "is scheduled at": ("scheduling", "scheduled_at", "positive"),
    "takes place at": ("scheduling", "scheduled_at", "positive"),
    "is not scheduled at": ("scheduling", "scheduled_at", "negative"),
    "does not take place at": ("scheduling", "scheduled_at", "negative"),
    "schedule relation not stated": ("scheduling", "scheduled_at", "unknown"),
    "no schedule relation": ("scheduling", "scheduled_at", "unknown"),
    "is allocated to": ("allocation", "allocated_to", "positive"),
    "is assigned to": ("allocation", "allocated_to", "positive"),
    "is not allocated to": ("allocation", "allocated_to", "negative"),
    "is not assigned to": ("allocation", "allocated_to", "negative"),
    "allocation relation not stated": ("allocation", "allocated_to", "unknown"),
    "no allocation relation": ("allocation", "allocated_to", "unknown"),
    "precedes": ("precedence", "precedes", "positive"),
    "comes before": ("precedence", "precedes", "positive"),
    "does not precede": ("precedence", "precedes", "negative"),
    "does not come before": ("precedence", "precedes", "negative"),
    "precedence relation not stated": ("precedence", "precedes", "unknown"),
    "no precedence relation": ("precedence", "precedes", "unknown"),
    "excludes": ("exclusion", "excludes", "positive"),
    "rules out": ("exclusion", "excludes", "positive"),
    "does not exclude": ("exclusion", "excludes", "negative"),
    "does not rule out": ("exclusion", "excludes", "negative"),
    "exclusion relation not stated": ("exclusion", "excludes", "unknown"),
    "no exclusion relation": ("exclusion", "excludes", "unknown"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "per_game_results",
    "model_specs",
    "model_rows",
    "model_family_cell_rows",
    "heldout_split_hashes",
    "source_text_rows",
    "raw_output_rows",
    "span_grounding_rows",
    "offset_rows",
    "alias_rows",
    "direction_rows",
    "parse_failure_rows",
    "timeout_rows",
    "hidden_label_isolation_rows",
    "task_runtime_receipt",
    "random_seed",
    "reproducibility_checksum",
    "span_acquisition_bank_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "field_principles": "Each required field explains the scientific reason for its evidence.",
    "preconditions_checked": "Resource checks prevent missing models or hardware from becoming invented data.",
    "inference_substrate": "The substrate distinguishes live local generation from a deterministic fixture.",
    "duration_s": "Measured wall time makes implausibly short live-inference claims visible.",
    "source_artifact_hashes": "Hashes bind the run to its exact fixture contract and source code.",
    "rows": "One row per scheduled tuple prevents malformed and failed attempts from disappearing.",
    "per_game_results": "Prompt summaries reveal missing model coverage for any held-out source.",
    "model_specs": "The fixed model set prevents legacy models from filling required cells.",
    "model_rows": "Per-model summaries preserve load and generation failures.",
    "model_family_cell_rows": "Cell counts prove equal acquisition coverage across models and relation families.",
    "heldout_split_hashes": "Frozen source hashes detect prompt drift after inference begins.",
    "source_text_rows": "Exact held-out source text makes copied spans independently groundable.",
    "raw_output_rows": "Raw generations preserve evidence that strict parsing rejects.",
    "span_grounding_rows": "Candidate receipts show whether both copied spans exist uniquely.",
    "offset_rows": "Byte offsets make Unicode grounding replayable without character-count assumptions.",
    "alias_rows": "Raw and canonical relation values expose every normalization decision.",
    "direction_rows": "Direction evidence prevents a reversed tuple from passing as grounded.",
    "parse_failure_rows": "Failure rows prevent survivorship bias toward well-formed outputs.",
    "timeout_rows": "Timeout rows keep resource failures in the attempted population.",
    "hidden_label_isolation_rows": "Isolation receipts prove expected answers never entered model context.",
    "task_runtime_receipt": "Process and device receipts bind rows to this task-owned live run.",
    "random_seed": "Frozen seeds support repeatable decoding requests.",
    "reproducibility_checksum": "A digest detects changes to inputs, outputs, and run receipts.",
    "span_acquisition_bank_ready_score": "One measures terminal row coverage only, not relation accuracy.",
    "gate_check_summary": "Failed checks retain expected and observed values for diagnosis.",
    "verifier_is_oracle": "False prevents acquisition completeness from claiming semantic authority.",
    "verdict_class": "The closed verdict class separates readiness from a capability win.",
    "honest_verdict": "A terminal prefix makes completion or blockage machine-readable.",
}


def canonical_json(value: Any) -> str:
    """Serialize stable UTF-8 JSON so byte hashes do not depend on ordering."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Return a visibly typed SHA-256 digest for exact bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file or return an explicit missing marker."""

    try:
        return sha256_bytes(path.read_bytes())
    except OSError:
        return "missing"


def check_row(check: str, expected: Any, observed: Any) -> JsonDict:
    """Retain both sides of one falsifiable precondition or readiness gate."""

    return {
        "check": check,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": expected == observed,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return all checks and the first stable failure with both values."""

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


def heldout_split_hash() -> str:
    """Hash the full frozen prompt identity, source, family, seed, and hidden metadata."""

    return sha256_bytes(canonical_json(HELDOUT_CASES).encode("utf-8"))


PUBLIC_INSTRUCTION = """Read the source below.
Copy exactly two non-overlapping source spans that name the left item and the right item. Keep their source order.
Then state, in short words, the one directed relationship from the first span to the second span.
Return one JSON object only with exactly these string fields:
{"span_a":"...","span_b":"...","relation":"..."}
SOURCE:
"""


def build_prompt(case: Mapping[str, Any]) -> str:
    """Build model context from public instructions and source text only."""

    return PUBLIC_INSTRUCTION + str(case["source_text"])


def hidden_label_isolation(
    case: Mapping[str, Any], prompt: str, reprompts: Sequence[str]
) -> JsonDict:
    """Audit non-source context for answer labels and private checker metadata."""

    public_context = prompt.replace(str(case["source_text"]), "", 1)
    public_context += "\n".join(str(value) for value in reprompts)
    folded = public_context.casefold()
    exposures: list[str] = []
    forbidden_markers = (
        "offset",
        "alias",
        "expected answer",
        "checker output",
        "prior failure",
    )
    for marker in forbidden_markers:
        if marker in folded:
            exposures.append(f"private_marker:{marker}")
    for label in ("positive", "negative", "unknown"):
        if label in folded:
            exposures.append(f"hidden_label:{label}")
    return {
        "prompt_id": case["prompt_id"],
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "reprompt_count": len(reprompts),
        "forbidden_exposures": exposures,
        "passed": not exposures,
        "terminal": True,
    }


def span_candidates(source_bytes: bytes, span_text: str) -> list[JsonDict]:
    """Find all byte matches, including overlaps, so duplicates remain ambiguous."""

    needle = span_text.encode("utf-8")
    candidates: list[JsonDict] = []
    start = 0
    while needle and (offset := source_bytes.find(needle, start)) >= 0:
        candidates.append({"start_utf8": offset, "end_utf8": offset + len(needle)})
        start = offset + 1
    return candidates


def _parse_base(state: str, reason: str) -> JsonDict:
    """Create a uniform terminal parser row before adding available evidence."""

    return {
        "parse_state": state,
        "failure_reason": reason,
        "span_a": None,
        "span_b": None,
        "span_a_candidates": [],
        "span_b_candidates": [],
        "span_a_offsets": None,
        "span_b_offsets": None,
        "raw_relation": None,
        "alias_normalized": False,
        "directed_relation": None,
        "direction_valid": None,
        "terminal": True,
    }


def parse_model_output(source_text: str, raw_output: str) -> JsonDict:
    """Ground two unique UTF-8 spans before interpreting the relation text."""

    row = _parse_base("invalid_json", "invalid_json")
    try:
        payload = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return row
    if not isinstance(payload, Mapping):
        row.update(parse_state="json_not_object", failure_reason="json_not_object")
        return row
    required = ("span_a", "span_b", "relation")
    if any(not isinstance(payload.get(field), str) or not payload.get(field) for field in required):
        row.update(parse_state="missing_required_field", failure_reason="missing_required_field")
        return row

    span_a = str(payload["span_a"])
    span_b = str(payload["span_b"])
    raw_relation = str(payload["relation"])
    source_bytes = source_text.encode("utf-8")
    candidates_a = span_candidates(source_bytes, span_a)
    candidates_b = span_candidates(source_bytes, span_b)
    row.update(
        span_a=span_a,
        span_b=span_b,
        raw_relation=raw_relation,
        span_a_candidates=candidates_a,
        span_b_candidates=candidates_b,
    )
    if not candidates_a:
        row.update(parse_state="span_a_absent", failure_reason="span_a_absent")
        return row
    if len(candidates_a) != 1:
        row.update(parse_state="span_a_ambiguous", failure_reason="span_a_ambiguous")
        return row
    row["span_a_offsets"] = candidates_a[0]
    if not candidates_b:
        row.update(parse_state="span_b_absent", failure_reason="span_b_absent")
        return row
    if len(candidates_b) != 1:
        row.update(parse_state="span_b_ambiguous", failure_reason="span_b_ambiguous")
        return row
    row["span_b_offsets"] = candidates_b[0]

    offset_a = candidates_a[0]
    offset_b = candidates_b[0]
    if max(offset_a["start_utf8"], offset_b["start_utf8"]) < min(
        offset_a["end_utf8"], offset_b["end_utf8"]
    ):
        row.update(
            parse_state="spans_overlap", failure_reason="spans_overlap", direction_valid=False
        )
        return row
    if offset_a["start_utf8"] > offset_b["start_utf8"]:
        row.update(
            parse_state="span_direction_reversed",
            failure_reason="span_direction_reversed",
            direction_valid=False,
        )
        return row
    row["direction_valid"] = True

    normalized_alias = " ".join(raw_relation.casefold().strip().rstrip(".").split())
    normalized = RELATION_ALIASES.get(normalized_alias)
    if normalized is None:
        row.update(
            parse_state="unsupported_relation_alias",
            failure_reason="unsupported_relation_alias",
        )
        return row
    family, predicate, polarity = normalized
    row.update(
        parse_state="accepted",
        failure_reason=None,
        alias_normalized=True,
        directed_relation={
            "family": family,
            "predicate": predicate,
            "polarity": polarity,
            "source": "A",
            "target": "B",
        },
    )
    return row


def build_attempt_row(
    model: Mapping[str, Any], case: Mapping[str, Any], inference: Mapping[str, Any]
) -> JsonDict:
    """Join one scheduled tuple to raw inference and terminal parse evidence."""

    prompt = build_prompt(case)
    isolation = hidden_label_isolation(case, prompt, [])
    timed_out = inference.get("timed_out") is True
    runner_failure = inference.get("runner_failure")
    if timed_out:
        parsed = _parse_base("timeout", str(runner_failure or "model_timeout"))
    elif runner_failure:
        parsed = _parse_base("runner_failure", str(runner_failure))
    else:
        parsed = parse_model_output(str(case["source_text"]), str(inference.get("raw_output", "")))
    attempt_id = f"{model['hf_id']}::{case['prompt_id']}::{case['seed']}"
    return {
        "row_type": "model_prompt_seed_attempt",
        "attempt_id": attempt_id,
        "model_id": model["hf_id"],
        "model_name": model["name"],
        "model_family": model["model_family"],
        "prompt_id": case["prompt_id"],
        "family": case["family"],
        "split": "heldout",
        "seed": case["seed"],
        "source_text_hash": sha256_bytes(str(case["source_text"]).encode("utf-8")),
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "raw_output": str(inference.get("raw_output", "")),
        "timed_out": timed_out,
        "runner_failure": runner_failure,
        "token_budget": {
            **deepcopy(DECODING_SETTINGS),
            "usage": deepcopy(inference.get("usage", {})),
        },
        "generation_duration_s": inference.get("generation_duration_s"),
        "runner_receipt": deepcopy(inference.get("runner_receipt", {})),
        "hidden_label_isolation": isolation,
        **parsed,
        "terminal": True,
    }


def _expected_attempt_ids() -> set[str]:
    """Return the exact scheduled model, prompt, and seed identities."""

    return {
        f"{model['hf_id']}::{case['prompt_id']}::{case['seed']}"
        for model in MODEL_SPECS
        for case in HELDOUT_CASES
    }


def readiness_report(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute bank readiness from unique terminal attempts, never accuracy."""

    identities = [str(row.get("attempt_id", "")) for row in rows]
    counts = Counter(identities)
    expected = _expected_attempt_ids()
    cell_rows: list[JsonDict] = []
    for model in MODEL_SPECS:
        for family in FAMILIES:
            cell = [
                row
                for row in rows
                if row.get("model_id") == model["hf_id"] and row.get("family") == family
            ]
            cell_rows.append(
                {
                    "model_id": model["hf_id"],
                    "model_family": model["model_family"],
                    "relation_family": family,
                    "expected_attempt_count": EXAMPLES_PER_CELL,
                    "attempt_count": len(cell),
                    "terminal_count": sum(row.get("terminal") is True for row in cell),
                    "complete": len(cell) == EXAMPLES_PER_CELL
                    and all(row.get("terminal") is True for row in cell),
                }
            )
    duplicate_ids = sorted(identity for identity, count in counts.items() if count != 1)
    missing_ids = sorted(expected - set(identities))
    unexpected_ids = sorted(set(identities) - expected)
    isolation_failures = sorted(
        str(row.get("attempt_id"))
        for row in rows
        if row.get("hidden_label_isolation", {}).get("passed") is not True
    )
    checks = [
        check_row("attempt_count", EXPECTED_ATTEMPTS, len(rows)),
        check_row("attempt_identity_set", sorted(expected), sorted(set(identities))),
        check_row("duplicate_attempt_ids", [], duplicate_ids),
        check_row("missing_attempt_ids", [], missing_ids),
        check_row("unexpected_attempt_ids", [], unexpected_ids),
        check_row(
            "nonterminal_attempt_count", 0, sum(row.get("terminal") is not True for row in rows)
        ),
        check_row("model_family_cell_count", 15, len(cell_rows)),
        check_row(
            "incomplete_model_family_cells",
            [],
            [
                f"{row['model_id']}::{row['relation_family']}"
                for row in cell_rows
                if row["complete"] is not True
            ],
        ),
        check_row("hidden_label_isolation_failures", [], isolation_failures),
    ]
    return {
        **gate_summary(checks),
        "expected_attempt_count": EXPECTED_ATTEMPTS,
        "observed_attempt_count": len(rows),
        "model_family_cell_rows": cell_rows,
    }


def chat_content(response: Any) -> str:
    """Read text from the llama.cpp chat shape without trusting optional keys."""

    if not isinstance(response, Mapping):
        return ""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return ""
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        return ""
    return str(message.get("content") or "")


def worker_acquire(
    *,
    model_file: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]] = HELDOUT_CASES,
    llama_factory: Callable[..., Any] | None = None,
    pid: int | None = None,
) -> list[JsonDict]:
    """Load one GGUF on both GPUs and retain one result for every model call."""

    if llama_factory is None:
        from llama_cpp import Llama

        llama_factory = Llama
    worker_pid = pid if pid is not None else os.getpid()
    model = llama_factory(
        model_path=str(model_file["model_path"]),
        n_ctx=CONTEXT_SIZE,
        n_batch=256,
        n_ubatch=128,
        n_gpu_layers=OFFLOAD_LAYERS,
        tensor_split=[0.5, 0.5],
        seed=RANDOM_SEED,
        verbose=False,
    )
    output: list[JsonDict] = []
    try:
        for case in cases:
            started = time.perf_counter()
            try:
                response = model.create_chat_completion(
                    messages=[{"role": "user", "content": build_prompt(case)}],
                    **DECODING_SETTINGS,
                    seed=int(case["seed"]),
                    response_format={"type": "json_object"},
                )
                output.append(
                    {
                        "prompt_id": case["prompt_id"],
                        "raw_output": chat_content(response),
                        "timed_out": False,
                        "runner_failure": None,
                        "usage": deepcopy(response.get("usage", {}))
                        if isinstance(response, Mapping)
                        else {},
                        "generation_duration_s": round(time.perf_counter() - started, 6),
                        "runner_receipt": {
                            "runner": "llama_cpp.Llama.create_chat_completion",
                            "worker_pid": worker_pid,
                            "model_path": model_file["model_path"],
                            "tensor_split": [0.5, 0.5],
                            "n_gpu_layers": OFFLOAD_LAYERS,
                        },
                    }
                )
            except Exception as exc:
                output.append(
                    {
                        "prompt_id": case["prompt_id"],
                        "raw_output": "",
                        "timed_out": False,
                        "runner_failure": f"{type(exc).__name__}:{exc}",
                        "usage": {},
                        "generation_duration_s": round(time.perf_counter() - started, 6),
                        "runner_receipt": {
                            "runner": "llama_cpp.Llama.create_chat_completion",
                            "worker_pid": worker_pid,
                            "model_path": model_file["model_path"],
                        },
                    }
                )
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()
    return output


def _worker_command(model: Mapping[str, Any], index: int) -> list[str]:
    """Build one task-owned child command for a sequential model lifecycle."""

    return [
        sys.executable,
        str(SCRIPT_PATH),
        "--worker",
        "--model-index",
        str(index),
        "--model-path",
        str(model["model_path"]),
    ]


def execute_worker(model: Mapping[str, Any], index: int) -> JsonDict:
    """Run one model worker with a hard bound and parse its flushed JSON rows."""

    command = _worker_command(model, index)
    started = time.perf_counter()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["CARNOT_FORCE_LIVE"] = "1"
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=MODEL_TIMEOUT_S,
            env=env,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        raw_stdout = exc.stdout or ""
        raw_stderr = exc.stderr or ""
        stdout = raw_stdout.decode() if isinstance(raw_stdout, bytes) else raw_stdout
        stderr = raw_stderr.decode() if isinstance(raw_stderr, bytes) else raw_stderr
        returncode = None
        timed_out = True
    except OSError as exc:
        stdout = ""
        stderr = f"{type(exc).__name__}:{exc}"
        returncode = None
        timed_out = False
    outputs: list[JsonDict] = []
    malformed_lines: list[str] = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            malformed_lines.append(line)
            continue
        if isinstance(value, dict) and value.get("prompt_id"):
            outputs.append(value)
        else:
            malformed_lines.append(line)
    status = "timeout" if timed_out else "complete" if returncode == 0 else "failed"
    return {
        "model_id": model["hf_id"],
        "status": status,
        "timed_out": timed_out,
        "returncode": returncode,
        "stderr": stderr,
        "stdout_sha256": sha256_bytes(stdout.encode("utf-8")),
        "malformed_worker_lines": malformed_lines,
        "duration_s": round(time.perf_counter() - started, 6),
        "command": command,
        "outputs": outputs,
    }


def execute_models(
    model_files: Sequence[Mapping[str, Any]],
    *,
    executor: Callable[[dict[str, Any], int], Mapping[str, Any]] = execute_worker,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Run models sequentially and fill every missing scheduled result explicitly."""

    rows: list[JsonDict] = []
    process_rows: list[JsonDict] = []
    for index, model in enumerate(model_files):
        process = dict(executor(dict(model), index))
        process_rows.append(
            {key: deepcopy(value) for key, value in process.items() if key != "outputs"}
        )
        outputs_by_prompt: dict[str, list[Mapping[str, Any]]] = {}
        for output in process.get("outputs", []):
            if isinstance(output, Mapping):
                outputs_by_prompt.setdefault(str(output.get("prompt_id", "")), []).append(output)
        for case in HELDOUT_CASES:
            candidates = outputs_by_prompt.get(str(case["prompt_id"]), [])
            if len(candidates) == 1:
                inference = dict(candidates[0])
            else:
                timed_out = process.get("timed_out") is True
                reason = (
                    "model_timeout"
                    if timed_out
                    else "duplicate_worker_outputs"
                    if len(candidates) > 1
                    else f"worker_missing_terminal_output:{process.get('status')}"
                )
                inference = {
                    "raw_output": "",
                    "timed_out": timed_out,
                    "runner_failure": reason,
                    "usage": {},
                    "generation_duration_s": None,
                    "runner_receipt": {},
                }
            receipt = dict(inference.get("runner_receipt", {}))
            receipt.update(
                {
                    "worker_status": process.get("status"),
                    "worker_returncode": process.get("returncode"),
                    "worker_command": deepcopy(process.get("command", [])),
                }
            )
            inference["runner_receipt"] = receipt
            rows.append(build_attempt_row(model, case, inference))
    return rows, process_rows


def query_gpus() -> list[JsonDict]:
    """Read stable identities for the two local CUDA devices."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    output: list[JsonDict] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            output.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "uuid": parts[2],
                    "memory_total_mb": int(float(parts[3])),
                }
            )
        except ValueError:
            continue
    return output


def llama_cpp_status() -> JsonDict:
    """Confirm that the installed llama.cpp binding supports CUDA offload."""

    try:
        import llama_cpp
        from llama_cpp import llama_cpp as backend

        return {
            "importable": True,
            "version": getattr(llama_cpp, "__version__", "unknown"),
            "supports_gpu_offload": bool(backend.llama_supports_gpu_offload()),
        }
    except Exception as exc:
        return {
            "importable": False,
            "version": None,
            "supports_gpu_offload": False,
            "error": f"{type(exc).__name__}:{exc}",
        }


def _fixture_contract(repo_root: Path) -> tuple[Any, JsonDict]:
    """Read Exp6926 and verify its schema, score, pinned hash, and source receipts."""

    path = repo_root / EXP6926_PATH
    source_hashes: JsonDict = {"exp6926_fixture": sha256_file(path)}
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return f"{type(exc).__name__}:{exc}", source_hashes
    receipts = artifact.get("source_artifact_hashes", {})
    source_valid = isinstance(receipts, Mapping) and bool(receipts)
    if source_valid:
        for name, receipt in receipts.items():
            if not isinstance(receipt, Mapping):
                source_valid = False
                continue
            current = sha256_file(repo_root / str(receipt.get("path", "")))
            source_hashes[str(name)] = current
            if not (receipt.get("expected_sha256") == receipt.get("observed_sha256") == current):
                source_valid = False
    observed = {
        "artifact_sha256": source_hashes["exp6926_fixture"],
        "schema": artifact.get("schema"),
        "ready_score": artifact.get("span_relation_fixture_ready_score"),
        "source_hashes_valid": source_valid,
    }
    expected = {
        "artifact_sha256": EXP6926_SHA256,
        "schema": EXP6926_SCHEMA,
        "ready_score": 1,
        "source_hashes_valid": True,
    }
    return observed if observed == expected else observed, source_hashes


def check_preconditions(
    *,
    repo_root: Path = REPO_ROOT,
    output_path: Path | None = None,
    resolver: Resolver = resolve_cached_gguf,
    gpu_probe: Callable[[], list[JsonDict]] = query_gpus,
    llama_probe: Callable[[], JsonDict] = llama_cpp_status,
) -> JsonDict:
    """Check fixture identity, held-out freeze, models, exact bytes, and dual CUDA."""

    target = output_path or repo_root / RESULT_PATH
    expected_fixture = {
        "artifact_sha256": EXP6926_SHA256,
        "schema": EXP6926_SCHEMA,
        "ready_score": 1,
        "source_hashes_valid": True,
    }
    fixture_observed, source_hashes = _fixture_contract(repo_root)
    checks = [check_row("exp6926_fixture_contract", expected_fixture, fixture_observed)]
    checks.append(
        check_row("heldout_source_text_frozen", EXPECTED_HELDOUT_SPLIT_HASH, heldout_split_hash())
    )
    sentinel = "éΩB".encode()
    checks.append(
        check_row(
            "exact_utf8_offset_utility",
            [{"start_utf8": 2, "end_utf8": 4}],
            span_candidates(sentinel, "Ω"),
        )
    )
    model_files: list[JsonDict] = []
    for spec in MODEL_SPECS:
        error = None
        try:
            resolved = resolver(str(spec["hf_id"]), str(spec["quantization"]))
        except Exception as exc:
            resolved = None
            error = f"{type(exc).__name__}:{exc}"
        path = Path(resolved).absolute() if resolved else None
        hit = bool(path and path.is_file())
        row = {
            **dict(spec),
            "model_path": str(path) if hit and path is not None else None,
            "model_sha256": sha256_file(path) if hit and path is not None else None,
            "size_bytes": path.stat().st_size if hit and path is not None else 0,
            "cache_state": "hit" if hit else "miss",
            "resolver_error": error,
        }
        model_files.append(row)
        checks.append(
            check_row(
                f"model_cache:{spec['hf_id']}",
                "cached_or_resolvable_gguf",
                "cached_or_resolvable_gguf" if hit else error or "missing",
            )
        )
    gpus = gpu_probe()
    selected_gpus = sorted(gpus, key=lambda row: int(row.get("index", 0)))[:2]
    valid_dual = (
        len(selected_gpus) == 2 and len({str(row.get("uuid", "")) for row in selected_gpus}) == 2
    )
    checks.append(check_row("dual_cuda_devices", True, valid_dual))
    llama = llama_probe()
    checks.append(
        check_row(
            "llama_cpp_cuda_offload",
            True,
            llama.get("importable") is True and llama.get("supports_gpu_offload") is True,
        )
    )
    writable = target.parent.is_dir() and os.access(target.parent, os.W_OK)
    checks.append(check_row("result_path_writable", True, writable))
    summary = gate_summary(checks)
    return {
        "checks": checks,
        "passed": summary["passed"],
        "model_files": model_files,
        "gpus": selected_gpus,
        "llama_cpp": llama,
        "source_artifact_hashes": source_hashes,
    }


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stored evidence while excluding measured wall time and the digest itself."""

    payload = deepcopy(dict(artifact))
    payload.pop("duration_s", None)
    payload.pop("reproducibility_checksum", None)
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def _source_text_rows() -> list[JsonDict]:
    """Expose exact held-out sources and hashes for independent byte replay."""

    return [
        {
            "prompt_id": case["prompt_id"],
            "family": case["family"],
            "split": case["split"],
            "seed": case["seed"],
            "source_text": case["source_text"],
            "source_text_hash": sha256_bytes(str(case["source_text"]).encode("utf-8")),
            "terminal": True,
        }
        for case in HELDOUT_CASES
    ]


def _empty_artifact(
    *,
    date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    model_files: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Create a schema-complete artifact before adding live attempt rows."""

    return {
        "schema": SCHEMA,
        "experiment_id": 6929,
        "run_date": date,
        "status": "blocked",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": gate_summary(checks),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [],
        "per_game_results": [],
        "model_specs": [deepcopy(dict(row)) for row in model_files],
        "model_rows": [],
        "model_family_cell_rows": [],
        "heldout_split_hashes": {
            "split": "heldout",
            "prompt_count": len(HELDOUT_CASES),
            "sha256": EXPECTED_HELDOUT_SPLIT_HASH,
        },
        "source_text_rows": _source_text_rows(),
        "raw_output_rows": [],
        "span_grounding_rows": [],
        "offset_rows": [],
        "alias_rows": [],
        "direction_rows": [],
        "parse_failure_rows": [],
        "timeout_rows": [],
        "hidden_label_isolation_rows": [],
        "task_runtime_receipt": {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "span_acquisition_bank_ready_score": 0,
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_three_family_span_acquisition",
    }


def blocked_artifact(
    *,
    date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    model_files: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Return the exact terminal artifact for failed preconditions."""

    artifact = _empty_artifact(
        date=date,
        duration_s=duration_s,
        checks=checks,
        source_hashes=source_hashes,
        model_files=model_files,
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact(
    *,
    date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    model_files: Sequence[Mapping[str, Any]],
    gpus: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    process_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Derive every required evidence table from preserved attempt rows."""

    attempts = [deepcopy(dict(row)) for row in rows]
    readiness = readiness_report(attempts)
    artifact = _empty_artifact(
        date=date,
        duration_s=duration_s,
        checks=checks,
        source_hashes=source_hashes,
        model_files=model_files,
    )
    model_rows = [
        {
            "model_id": model["hf_id"],
            "model_family": model["model_family"],
            "attempt_count": sum(row.get("model_id") == model["hf_id"] for row in attempts),
            "terminal_count": sum(
                row.get("model_id") == model["hf_id"] and row.get("terminal") is True
                for row in attempts
            ),
            "accepted_parse_count": sum(
                row.get("model_id") == model["hf_id"] and row.get("parse_state") == "accepted"
                for row in attempts
            ),
            "terminal": True,
        }
        for model in MODEL_SPECS
    ]
    per_game = [
        {
            "prompt_id": case["prompt_id"],
            "family": case["family"],
            "attempt_count": sum(row.get("prompt_id") == case["prompt_id"] for row in attempts),
            "terminal_count": sum(
                row.get("prompt_id") == case["prompt_id"] and row.get("terminal") is True
                for row in attempts
            ),
            "terminal": True,
        }
        for case in HELDOUT_CASES
    ]
    artifact.update(
        {
            "status": "complete" if readiness["passed"] else "partial",
            "rows": attempts,
            "per_game_results": per_game,
            "model_rows": model_rows,
            "model_family_cell_rows": readiness["model_family_cell_rows"],
            "raw_output_rows": [
                {
                    "attempt_id": row["attempt_id"],
                    "raw_output": row["raw_output"],
                    "parse_state": row["parse_state"],
                    "terminal": True,
                }
                for row in attempts
            ],
            "span_grounding_rows": [
                {
                    "attempt_id": row["attempt_id"],
                    "span_a": row["span_a"],
                    "span_b": row["span_b"],
                    "span_a_candidates": deepcopy(row["span_a_candidates"]),
                    "span_b_candidates": deepcopy(row["span_b_candidates"]),
                    "parse_state": row["parse_state"],
                    "terminal": True,
                }
                for row in attempts
            ],
            "offset_rows": [
                {
                    "attempt_id": row["attempt_id"],
                    "span_a_offsets": deepcopy(row["span_a_offsets"]),
                    "span_b_offsets": deepcopy(row["span_b_offsets"]),
                    "source_text_hash": row["source_text_hash"],
                    "terminal": True,
                }
                for row in attempts
            ],
            "alias_rows": [
                {
                    "attempt_id": row["attempt_id"],
                    "raw_relation": row["raw_relation"],
                    "alias_normalized": row["alias_normalized"],
                    "directed_relation": deepcopy(row["directed_relation"]),
                    "terminal": True,
                }
                for row in attempts
            ],
            "direction_rows": [
                {
                    "attempt_id": row["attempt_id"],
                    "direction_valid": row["direction_valid"],
                    "parse_state": row["parse_state"],
                    "terminal": True,
                }
                for row in attempts
            ],
            "parse_failure_rows": [
                {
                    "attempt_id": row["attempt_id"],
                    "parse_state": row["parse_state"],
                    "failure_reason": row["failure_reason"],
                    "raw_output": row["raw_output"],
                    "terminal": True,
                }
                for row in attempts
                if row.get("parse_state") != "accepted"
            ],
            "timeout_rows": [deepcopy(row) for row in attempts if row.get("timed_out") is True],
            "hidden_label_isolation_rows": [
                deepcopy(row["hidden_label_isolation"]) for row in attempts
            ],
            "task_runtime_receipt": {
                "task_id": "exp6929-three-family-span-acquisition",
                "task_pid": os.getpid(),
                "gpu_uuids": [row.get("uuid") for row in gpus],
                "model_process_rows": [deepcopy(dict(row)) for row in process_rows],
                "scheduled_attempt_count": EXPECTED_ATTEMPTS,
                "terminal_attempt_count": sum(row.get("terminal") is True for row in attempts),
                "models_sequential": True,
            },
            "span_acquisition_bank_ready_score": int(readiness["passed"]),
            "gate_check_summary": readiness,
            "verdict_class": "null" if readiness["passed"] else "partial",
            "honest_verdict": "complete_span_acquisition_bank_ready"
            if readiness["passed"]
            else "complete_partial_span_acquisition_bank",
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Reject missing evidence, false readiness, bad checksums, and verdict drift."""

    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    principles = artifact.get("field_principles", {})
    if not isinstance(principles, Mapping) or any(
        field not in principles for field in REQUIRED_ARTIFACT_FIELDS
    ):
        errors.append("field_principles")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(("complete_", "blocked_")):
        errors.append("honest_verdict")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    if artifact.get("verdict_class") == "blocked":
        if artifact.get("honest_verdict") != "blocked_three_family_span_acquisition":
            errors.append("blocked_honest_verdict")
        if artifact.get("span_acquisition_bank_ready_score") != 0:
            errors.append("blocked_ready_score")
        if artifact.get("gate_check_summary", {}).get("passed") is not False:
            errors.append("blocked_gate_summary")
    else:
        readiness = readiness_report(
            [row for row in artifact.get("rows", []) if isinstance(row, Mapping)]
        )
        if artifact.get("span_acquisition_bank_ready_score") != int(readiness["passed"]):
            errors.append("ready_score_drift")
        if artifact.get("model_family_cell_rows") != readiness["model_family_cell_rows"]:
            errors.append("cell_rows_drift")
        if artifact.get("gate_check_summary") != readiness:
            errors.append("gate_summary_drift")
    return list(dict.fromkeys(errors))


def write_json_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace the deliverable only after a complete JSON file is durable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    temporary.replace(path)


def run(
    *,
    date: str,
    output_path: Path = REPO_ROOT / RESULT_PATH,
    repo_root: Path = REPO_ROOT,
    preflight_fn: Callable[..., Mapping[str, Any]] = check_preconditions,
    execute_fn: Callable[
        [Sequence[Mapping[str, Any]]], tuple[list[JsonDict], list[JsonDict]]
    ] = execute_models,
) -> JsonDict:
    """Run preflight, sequential model workers, validation, and one atomic write."""

    started = time.perf_counter()
    preflight = dict(preflight_fn(repo_root=repo_root, output_path=output_path))
    if preflight.get("passed") is not True:
        artifact = blocked_artifact(
            date=date,
            duration_s=round(time.perf_counter() - started, 6),
            checks=preflight.get("checks", []),
            source_hashes=preflight.get("source_artifact_hashes", {}),
            model_files=preflight.get("model_files", []),
        )
    else:
        rows, process_rows = execute_fn(preflight.get("model_files", []))
        artifact = build_artifact(
            date=date,
            duration_s=round(time.perf_counter() - started, 6),
            checks=preflight.get("checks", []),
            source_hashes=preflight.get("source_artifact_hashes", {}),
            model_files=preflight.get("model_files", []),
            gpus=preflight.get("gpus", []),
            rows=rows,
            process_rows=process_rows,
        )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError("artifact_validation_failed:" + ",".join(errors))
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    write_json_atomic(output_path, artifact)
    return artifact


def worker_main(model_index: int, model_path: str) -> int:
    """Emit one flushed JSON line per attempted prompt from a model child."""

    model_file = {**dict(MODEL_SPECS[model_index]), "model_path": model_path}
    for row in worker_acquire(model_file=model_file):
        print(canonical_json(row), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the dated experiment or its internal task-owned model worker."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_PATH)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model-index", type=int)
    parser.add_argument("--model-path")
    args = parser.parse_args(argv)
    if args.worker:
        if args.model_index is None or not args.model_path:
            parser.error("--worker requires --model-index and --model-path")
        return worker_main(args.model_index, args.model_path)
    if not args.date:
        parser.error("--date is required")
    artifact = run(date=args.date, output_path=args.output)
    print(
        canonical_json(
            {
                "output": str(args.output),
                "ready_score": artifact["span_acquisition_bank_ready_score"],
                "honest_verdict": artifact["honest_verdict"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin script is the public command surface.
    raise SystemExit(main())
