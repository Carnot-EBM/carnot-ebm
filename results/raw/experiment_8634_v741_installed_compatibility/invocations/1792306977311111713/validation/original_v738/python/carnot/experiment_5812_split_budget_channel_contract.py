"""Exp5812 reusable split-budget reasoning/finalization channel contract.

Spec refs: REQ-VERIFY-5812, SCENARIO-VERIFY-5812-CONTRACT,
SCENARIO-VERIFY-5812-CONTROLS, SCENARIO-VERIFY-5812-GRAMMAR-BOUNDARY,
SCENARIO-VERIFY-5812-REPLAY.

This module implements the transport contract that Exp5799 was missing.  It is
offline and deterministic: no headline model is loaded, no GGUF template is
edited, and no Hugging Face tokenizer is used.  The executable surface is a
two-call split between bounded reasoning and bounded finalization, with replay
receipts proving that the frozen transcript, candidate environment, and exact
fixture parser still agree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any

from carnot import experiment_5785_hardness_surface_fixture as fixture
from carnot import experiment_5798_sota_answer_channel_diagnostic as exp5798
from carnot import experiment_5811_exp5799_event_provenance_audit as exp5811


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5812_split_budget_channel_contract.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5812_split_budget_channel_contract.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5812_split_budget_channel_contract.py")
EXP5811_ARTIFACT_RELATIVE_PATH = exp5811.RESULT_RELATIVE_PATH
EXP5798_ARTIFACT_RELATIVE_PATH = exp5798.RESULT_RELATIVE_PATH
EXP5785_ARTIFACT_RELATIVE_PATH = fixture.RESULT_RELATIVE_PATH
EXP5785_ROWS_RELATIVE_PATH = fixture.ROW_FILE_RELATIVE_PATH
EXP5785_MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5785_hardness_surface_fixture.py")
EXP5799_PRODUCER_RELATIVE_PATH = Path(
    "python/carnot/experiment_5799_sota_answer_channel_canary.py"
)
EXP5799_TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5799_sota_answer_channel_canary.py")
SOTA_MODEL_METADATA_RELATIVE_PATH = Path("python/carnot/inference/sota_models.py")
EXPERIMENT_TEMPLATE_RELATIVE_PATH = Path("scripts/experiment_template.py")
VERIFY_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")

SCHEMA = "carnot.experiment_5812.split_budget_channel_contract.v1"
CONTRACT_VERSION = "split_budget_reasoning_finalization_contract_v1"
EXPERIMENT = 5812
EXPERIMENT_ID = "experiment_5812_split_budget_channel_contract"
MILESTONE = "2026.07.518"
RUN_DATE = "20260722"
RANDOM_SEED = 5812
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
GRAMMAR_CLAIM_BOUNDARY = (
    "candidate_membership_and_syntax_only_exact_parser_solver_remain_semantic_authority"
)
SEMANTIC_CONTRACT_ID = "exp5812_frozen_reasoning_then_candidate_id_final_v1"
EXACT_PARSER_ID = "exp5785_candidate_id_to_row_id_label_exact_parser"
STOP_STRINGS = ("<|eot_id|>", "<stop>")

SPEC_REFS = (
    "REQ-VERIFY-5812",
    "SCENARIO-VERIFY-5812-CONTRACT",
    "SCENARIO-VERIFY-5812-CONTROLS",
    "SCENARIO-VERIFY-5812-GRAMMAR-BOUNDARY",
    "SCENARIO-VERIFY-5812-REPLAY",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "contract_version_and_code_hashes",
    "reasoning_stage_contract",
    "finalization_stage_contract",
    "sealed_candidate_environment",
    "grammar_claim_boundary",
    "preregistered_mode_matrix",
    "adversarial_control_results",
    "replay_receipts",
    "split_budget_contract_ready_score",
    "llm_calls_made",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal contract state distinguishes an executable transport from design prose.",
    "preconditions_checked": "Gate and dependency checks prevent building against quarantined or missing evidence.",
    "contract_version_and_code_hashes": "Versioned implementation identity prevents silent transport drift.",
    "reasoning_stage_contract": "An independent cap, stop policy, timeout, and transcript hash remove the shared-budget ambiguity.",
    "finalization_stage_contract": "A separate cap and parser boundary guarantee room for a final answer without claiming truth.",
    "sealed_candidate_environment": "Only fixture-declared IDs may be referenced; hidden labels never enter prompts.",
    "grammar_claim_boundary": "Grammar may establish membership and syntax only; exact validation remains semantic authority.",
    "preregistered_mode_matrix": "Fixed modes prevent post-hoc tuning on canary outcomes.",
    "adversarial_control_results": "Negative controls prove empty, injected, ghost, truncated, and wrong outputs fail closed.",
    "replay_receipts": "Frozen transcript and deterministic parser replay make the two-stage boundary auditable.",
    "split_budget_contract_ready_score": "A bare scalar gates expensive inference only when the contract and attacks pass.",
    "llm_calls_made": "Zero headline calls keeps protocol implementation separate from capability evidence.",
    "duration_s": "Measured implementation/test time exposes a bootstrap-only artifact.",
    "inference_substrate": "`aggregation_from_upstream_artifacts` declares offline implementation and tests with no headline model load.",
    "verifier_is_oracle": "The fixture parser/solver defines correctness, so channel success is execution-grounded rather than a verifier moat.",
    "field_provenance": "Every contract field points to code, test, or sealed fixture evidence.",
    "test_commands": "Commands document focused, adversarial, replay, and coverage execution.",
    "test_exit_codes": "Exit codes prevent failed controls from being narrated as passing.",
    "reproducibility_checksum": "A checksum detects later contract, fixture, or test drift.",
    "honest_verdict": "A `complete:` or `blocked:` prefix makes protocol readiness terminal.",
}

FIELD_PRINCIPLE_EXTRAS: dict[str, str] = {
    "schema": "Versioned schema id for the split-budget contract.",
    "experiment": "Numeric experiment id binds the artifact to Exp5812.",
    "experiment_id": "Stable local slug ties the artifact to the implementation.",
    "milestone": "Binds the contract to V518.",
    "run_date": "Operator-requested contract date.",
    "random_seed": "Deterministic metadata for an offline contract artifact.",
    "spec_refs": "OpenSpec anchors for the split-budget contract and controls.",
    "result_path": "Declares the intended terminal JSON path.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5812_split_budget_channel_contract.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5812_split_budget_channel_contract.py "
    "-m pytest tests/python/test_experiment_5812_split_budget_channel_contract.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5812_split_budget_channel_contract.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)

SHARED_BUDGET_CONTROL: JsonDict = {
    "mode_id": "shared_budget_control_128",
    "mode_type": "shared_budget_control",
    "max_tokens": 128,
    "timeout_s": 900,
    "stops": list(STOP_STRINGS),
    "finalizer": "legacy single-call reasoning and final line share one output cap",
    "parser": EXACT_PARSER_ID,
    "bounded": True,
    "executable": False,
    "preserved_as_control": True,
    "retirement_rules_preregistered": True,
    "fail_closed_conditions": [
        "empty_final",
        "truncation",
        "invalid_or_ghost_candidate_id",
        "exact_validator_mismatch",
    ],
}

SPLIT_BUDGET_MODES: list[JsonDict] = [
    {
        "mode_id": "split_reasoning_96_final_32_candidate_id",
        "mode_type": "split_budget",
        "reasoning": {"max_tokens": 96, "timeout_s": 900, "stops": list(STOP_STRINGS)},
        "finalization": {"max_tokens": 32, "timeout_s": 180, "stops": list(STOP_STRINGS)},
        "finalizer": "candidate-id-only finalizer over frozen reasoning transcript",
        "parser": EXACT_PARSER_ID,
        "bounded": True,
        "executable": True,
        "runtime_supports_environment_indexed_grammar": False,
        "retirement_rules_preregistered": True,
        "fail_closed_conditions": [
            "empty_reasoning",
            "empty_final",
            "reasoning_truncation",
            "final_truncation",
            "stop_collision",
            "unclosed_thinking",
            "timeout",
            "invalid_or_ghost_candidate_id",
            "hidden_label_leakage",
            "replay_mismatch",
            "exact_validator_mismatch",
        ],
    },
    {
        "mode_id": "split_reasoning_192_final_48_env_indexed_ids",
        "mode_type": "split_budget",
        "reasoning": {"max_tokens": 192, "timeout_s": 900, "stops": list(STOP_STRINGS)},
        "finalization": {"max_tokens": 48, "timeout_s": 180, "stops": list(STOP_STRINGS)},
        "finalizer": "candidate-id-only finalizer with optional finite-ID grammar support",
        "parser": EXACT_PARSER_ID,
        "bounded": True,
        "executable": True,
        "runtime_supports_environment_indexed_grammar": True,
        "retirement_rules_preregistered": True,
        "fail_closed_conditions": [
            "empty_reasoning",
            "empty_final",
            "reasoning_truncation",
            "final_truncation",
            "stop_collision",
            "unclosed_thinking",
            "timeout",
            "invalid_or_ghost_candidate_id",
            "hidden_label_leakage",
            "replay_mismatch",
            "exact_validator_mismatch",
        ],
    },
]

EXPECTED_ADVERSARIAL_CONTROLS = (
    "empty_reasoning",
    "empty_final",
    "overlong_reasoning",
    "stop_collision",
    "unclosed_thinking",
    "duplicate_candidate_id",
    "invalid_candidate_id",
    "ghost_candidate_id",
    "schema_control_plane_injection",
    "candidate_label_leakage",
    "timeout",
    "replay_mismatch",
    "exact_wrong_answer",
)


class SplitBudgetReplayError(ValueError):
    """Raised when a frozen split-budget receipt no longer replays."""


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence deterministically before hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a local file in chunks so large artifacts remain streamable."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")  # pragma: no cover
    return dict(payload)


def _hash_path(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.is_file() else "missing"


def _memory_probe() -> JsonDict:
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:  # pragma: no cover - host fallback.
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {"available_mb": available_mb, "required_mb": 512, "ok": available_mb >= 512}


def _disk_probe(root: Path) -> JsonDict:
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": 512, "ok": available_mb >= 512}


def _row_id_slug(row_id: str) -> str:
    return hashlib.sha256(row_id.encode("utf-8")).hexdigest()[:10].upper()


def build_candidate_environment(row: Mapping[str, Any]) -> JsonDict:
    """Build the sealed finite-ID environment from a fixture row."""

    row_id = str(row["row_id"])
    slug = _row_id_slug(row_id)
    candidates: JsonDict = {}
    label_to_candidate_id: JsonDict = {}
    for index, item in enumerate(row["label_mapping"]):
        candidate_id = f"CID_{slug}_{index:02d}"
        candidate = str(item["candidate"])
        label = str(item["label"])
        candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "candidate": candidate,
            "candidate_hash": sha256_text(candidate),
            "label": label,
            "source_row_hash": str(row["row_hash"]),
            "source_label_mapping_hash": sha256_json(row["label_mapping"]),
        }
        label_to_candidate_id[label] = candidate_id
    candidate_ids = list(candidates)
    environment = {
        "row_id": row_id,
        "fixture_row_hash": str(row["row_hash"]),
        "environment_id": f"env::{row_id}::{sha256_json(candidate_ids)[7:23]}",
        "candidate_ids": candidate_ids,
        "candidate_by_id": candidates,
        "label_to_candidate_id": label_to_candidate_id,
        "hidden_labels_exposed_to_prompt": False,
        "exact_label_exposed_to_prompt": False,
        "exact_answer_exposed_to_prompt": False,
    }
    environment["environment_hash"] = sha256_json(environment)
    return environment


def prompt_leakage_scan(row: Mapping[str, Any], prompt_text: str) -> JsonDict:
    """Detect hidden labels or exact answers in a prompt payload."""

    lowered = prompt_text.lower()
    leaks: list[str] = []
    if "exact_label" in lowered or "exact answer" in lowered or "exact_answer" in lowered:
        leaks.append("exact_metadata_key")
    exact_answer = str(row.get("exact_answer") or "")
    if exact_answer and exact_answer in prompt_text:
        leaks.append("exact_answer_value")
    for item in row["label_mapping"]:
        label = str(item["label"])
        if f"{label}:" in prompt_text or f"label {label.lower()}" in lowered:
            leaks.append(f"candidate_label:{label}")
    return {
        "hidden_label_leakage_detected": bool(leaks),
        "leakage_markers": sorted(set(leaks)),
        "prompt_sha256": sha256_text(prompt_text),
    }


def build_finalizer_prompt(
    row: Mapping[str, Any],
    environment: Mapping[str, Any],
    reasoning: Mapping[str, Any],
    *,
    include_hidden_labels: bool = False,
) -> JsonDict:
    """Build the finalizer prompt without exposing hidden fixture labels."""

    candidate_ids = " ".join(str(item) for item in environment["candidate_ids"])
    text = (
        f"Today is {RUN_DATE}.\n"
        "Finalize the sealed finite-choice fixture using candidate IDs only.\n"
        f"Fixture surface: {row['surface_text']}\n"
        f"Frozen reasoning hash: {reasoning['transcript_hash']}\n"
        f"Frozen reasoning transcript:\n{reasoning['raw_text']}\n"
        f"Allowed candidate IDs for {row['row_id']}: {candidate_ids}\n"
        f"Return exactly: {row['row_id']}: <candidate_id>"
    )
    if include_hidden_labels:
        text += f"\nexact_label={row['exact_label']}\n{row['exact_label']}: leaked"
    leakage = prompt_leakage_scan(row, text)
    return {"prompt_text": text, "prompt_sha256": sha256_text(text), "leakage": leakage}


def _has_stop_collision(raw_text: str, stops: Sequence[str]) -> bool:
    lowered = raw_text.lower()
    return any(str(stop).lower() in lowered for stop in stops)


def _has_unclosed_thinking(raw_text: str) -> bool:
    lowered = raw_text.lower()
    return ("<think>" in lowered and "</think>" not in lowered) or (
        "<analysis>" in lowered and "</analysis>" not in lowered
    )


def classify_reasoning_stage(
    raw_text: str,
    *,
    finish_reason: str,
    output_tokens: int,
    config: Mapping[str, Any],
    timeout: bool = False,
) -> JsonDict:
    """Classify the bounded reasoning call and freeze its transcript."""

    max_tokens = int(config["max_tokens"])
    stops = list(config.get("stops") or STOP_STRINGS)
    empty = raw_text.strip() == ""
    truncation = finish_reason == "length" or output_tokens >= max_tokens
    stop_collision = _has_stop_collision(raw_text, stops)
    unclosed = _has_unclosed_thinking(raw_text)
    if timeout:
        failure_mode = "timeout"
    elif empty:
        failure_mode = "empty_reasoning"
    elif truncation:
        failure_mode = "reasoning_truncation"
    elif stop_collision:
        failure_mode = "stop_collision"
    elif unclosed:
        failure_mode = "unclosed_thinking"
    else:
        failure_mode = "valid_reasoning"
    valid = failure_mode == "valid_reasoning"
    return {
        "stage": "reasoning",
        "call_ordinal": 1,
        "raw_text": raw_text,
        "raw_sha256": sha256_text(raw_text),
        "transcript_hash": sha256_text(raw_text),
        "immutable": True,
        "finish_reason": finish_reason,
        "output_tokens": int(output_tokens),
        "max_tokens": max_tokens,
        "timeout_s": int(config["timeout_s"]),
        "stops": stops,
        "timeout": timeout,
        "empty_reasoning": empty,
        "truncation": truncation,
        "stop_collision": stop_collision,
        "unclosed_thinking": unclosed,
        "valid_reasoning": valid,
        "failure_mode": failure_mode,
    }


def _parse_candidate_final(
    raw_text: str,
    row: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> JsonDict:
    lowered = raw_text.lower()
    if "<|eot_id|>" in raw_text or "<stop>" in lowered:
        return _parse_failure("stop_token")
    if "ignore previous" in lowered or "<script" in lowered or "```" in raw_text:
        return _parse_failure("adversarial_payload")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return _parse_failure("missing_answer")
    parsed: JsonDict = {}
    for line in lines:
        if ":" not in line:
            return _parse_failure("truncation")
        row_id, candidate_id = [part.strip() for part in line.split(":", 1)]
        if not row_id or not candidate_id:
            return _parse_failure("truncation")
        if row_id in parsed:
            return _parse_failure("duplicate_id")
        if row_id != row["row_id"]:
            return _parse_failure("invalid_id")
        if candidate_id not in environment["candidate_by_id"]:
            reason = "ghost_candidate_id" if candidate_id.startswith("CID_") else "invalid_candidate_id"
            return _parse_failure(reason)
        parsed[row_id] = candidate_id
    if str(row["row_id"]) not in parsed:  # pragma: no cover - duplicate guard after row-id check.
        return _parse_failure("missing_answer")
    candidate_id = str(parsed[str(row["row_id"])])
    candidate = dict(environment["candidate_by_id"][candidate_id])
    hidden_line = f"{row['row_id']}: {candidate['label']}"
    fixture_receipt = fixture.parse_response(hidden_line, {str(row["row_id"]): row})
    parse_ok = fixture_receipt["parse_ok"] is True
    exact_answer_error = bool(parse_ok and candidate["label"] != row["exact_label"])
    return {
        "parse_ok": parse_ok,
        "parser_failure_reason": str(fixture_receipt["parser_failure_reason"]),
        "parsed_candidate_ids": parsed,
        "selected_candidate_id": candidate_id,
        "selected_candidate": str(candidate["candidate"]),
        "selected_candidate_hash": str(candidate["candidate_hash"]),
        "selected_hidden_label": str(candidate["label"]),
        "exact_answer_error": exact_answer_error,
        "fixture_parser_receipt": fixture_receipt,
    }


def _parse_failure(reason: str) -> JsonDict:
    return {
        "parse_ok": False,
        "parser_failure_reason": reason,
        "parsed_candidate_ids": {},
        "selected_candidate_id": "",
        "selected_candidate": "",
        "selected_candidate_hash": "",
        "selected_hidden_label": "",
        "exact_answer_error": False,
        "fixture_parser_receipt": fixture._parse_failure(reason),
    }


def classify_finalization_stage(
    row: Mapping[str, Any],
    environment: Mapping[str, Any],
    raw_text: str,
    *,
    finish_reason: str,
    output_tokens: int,
    config: Mapping[str, Any],
    timeout: bool = False,
    reasoning_transcript_hash: str = "",
    prompt_hash: str = "",
) -> JsonDict:
    """Classify the bounded finalization call at the candidate-ID parser boundary."""

    max_tokens = int(config["max_tokens"])
    stops = list(config.get("stops") or STOP_STRINGS)
    truncation = finish_reason == "length" or output_tokens >= max_tokens
    parser = _parse_candidate_final(raw_text, row, environment)
    stop_collision = parser["parser_failure_reason"] == "stop_token" or _has_stop_collision(
        raw_text,
        stops,
    )
    empty_final = raw_text.strip() == ""
    grammar_membership_ok = bool(
        parser["parse_ok"] is True
        and parser["selected_candidate_id"] in environment["candidate_by_id"]
    )
    parser_failure = parser["parse_ok"] is not True
    exact_answer_error = parser["exact_answer_error"] is True
    if timeout:
        failure_mode = "timeout"
    elif empty_final:
        failure_mode = "empty_final"
    elif truncation:
        failure_mode = "final_truncation"
    elif stop_collision:
        failure_mode = "stop_collision"
    elif parser_failure:
        failure_mode = str(parser["parser_failure_reason"])
    elif exact_answer_error:
        failure_mode = "exact_answer_error"
    else:
        failure_mode = "valid_exact_output"
    valid = failure_mode == "valid_exact_output"
    result = {
        "stage": "finalization",
        "call_ordinal": 2,
        "raw_text": raw_text,
        "raw_sha256": sha256_text(raw_text),
        "finish_reason": finish_reason,
        "output_tokens": int(output_tokens),
        "max_tokens": max_tokens,
        "timeout_s": int(config["timeout_s"]),
        "stops": stops,
        "timeout": timeout,
        "truncation": truncation,
        "empty_final": empty_final,
        "stop_collision": stop_collision,
        "grammar_membership_ok": grammar_membership_ok,
        "parser_failure": parser_failure,
        "exact_answer_error": exact_answer_error,
        "valid_exact_output": valid,
        "failure_mode": failure_mode,
        "reasoning_transcript_hash": reasoning_transcript_hash,
        "prompt_hash": prompt_hash,
        "candidate_environment_hash": str(environment["environment_hash"]),
        **parser,
    }
    result["parser_receipt_hash"] = sha256_json(
        {
            "parse_ok": result["parse_ok"],
            "parser_failure_reason": result["parser_failure_reason"],
            "parsed_candidate_ids": result["parsed_candidate_ids"],
            "selected_candidate_hash": result["selected_candidate_hash"],
            "exact_answer_error": result["exact_answer_error"],
        }
    )
    return result


def environment_indexed_grammar_receipt(
    environment: Mapping[str, Any],
    *,
    runtime_supports: bool,
) -> JsonDict:
    """Record the finite-ID grammar support boundary without claiming truth."""

    supported = runtime_supports is True
    return {
        "runtime_supports_environment_indexed_grammar": supported,
        "grammar_runtime": "finite_id_environment_mask" if supported else "unsupported",
        "environment_hash": str(environment["environment_hash"]),
        "enforced_candidate_ids": list(environment["candidate_ids"]) if supported else [],
        "parser_remains_fail_closed": True,
        "claim_boundary": GRAMMAR_CLAIM_BOUNDARY,
        "candidate_membership_claimed": supported,
        "syntax_safety_claimed": supported,
        "semantic_correctness_claimed": False,
        "exact_validation_required": True,
    }


def execute_two_stage_contract(
    row: Mapping[str, Any],
    mode: Mapping[str, Any],
    *,
    reasoning_text: str,
    final_text: str,
    reasoning_finish_reason: str = "stop",
    final_finish_reason: str = "stop",
    reasoning_tokens: int = 8,
    final_tokens: int = 2,
    reasoning_timeout: bool = False,
    final_timeout: bool = False,
    runtime_supports_grammar: bool = False,
    include_hidden_labels: bool = False,
) -> JsonDict:
    """Run the deterministic two-stage contract over supplied raw receipts."""

    environment = build_candidate_environment(row)
    reasoning = classify_reasoning_stage(
        reasoning_text,
        finish_reason=reasoning_finish_reason,
        output_tokens=reasoning_tokens,
        config=mode["reasoning"],
        timeout=reasoning_timeout,
    )
    prompt = build_finalizer_prompt(
        row,
        environment,
        reasoning,
        include_hidden_labels=include_hidden_labels,
    )
    finalization = classify_finalization_stage(
        row,
        environment,
        final_text,
        finish_reason=final_finish_reason,
        output_tokens=final_tokens,
        config=mode["finalization"],
        timeout=final_timeout,
        reasoning_transcript_hash=str(reasoning["transcript_hash"]),
        prompt_hash=str(prompt["prompt_sha256"]),
    )
    valid = bool(
        reasoning["valid_reasoning"] is True
        and finalization["valid_exact_output"] is True
        and prompt["leakage"]["hidden_label_leakage_detected"] is False
    )
    failure_mode = "valid_exact_output"
    if not valid:
        failure_mode = (
            str(reasoning["failure_mode"])
            if reasoning["valid_reasoning"] is not True
            else (
                "hidden_label_leakage"
                if prompt["leakage"]["hidden_label_leakage_detected"] is True
                else str(finalization["failure_mode"])
            )
        )
    receipt = {
        "schema": SCHEMA + ".split_receipt",
        "mode_id": str(mode["mode_id"]),
        "semantic_contract_id": SEMANTIC_CONTRACT_ID,
        "exact_parser_id": EXACT_PARSER_ID,
        "fixture_row": _copy_json(row),
        "candidate_environment": environment,
        "grammar_receipt": environment_indexed_grammar_receipt(
            environment,
            runtime_supports=runtime_supports_grammar,
        ),
        "reasoning": reasoning,
        "finalizer_prompt": prompt,
        "finalization": finalization,
        "prompt_leakage": prompt["leakage"],
        "budget_accounting": {
            "shared_budget_used": False,
            "reasoning_output_tokens": int(reasoning["output_tokens"]),
            "finalization_output_tokens": int(finalization["output_tokens"]),
            "reasoning_max_tokens": int(reasoning["max_tokens"]),
            "finalization_max_tokens": int(finalization["max_tokens"]),
            "measured_separately": True,
        },
        "call_count": 2,
        "valid_exact_output": valid,
        "failure_mode": failure_mode,
        "receipt_hash": "",
        "replay_ok": False,
    }
    receipt["receipt_hash"] = _split_receipt_hash(receipt)
    receipt["replay_ok"] = replay_split_receipt(receipt)
    return receipt


def _split_receipt_hash(receipt: Mapping[str, Any]) -> str:
    stable = _copy_json(receipt)
    stable["receipt_hash"] = ""
    stable["replay_ok"] = False
    return sha256_json(stable)


def replay_split_receipt(receipt: Mapping[str, Any]) -> bool:
    """Replay frozen reasoning, final raw bytes, parser receipt, and environment."""

    row = dict(receipt["fixture_row"])
    environment = dict(receipt["candidate_environment"])
    reasoning = dict(receipt["reasoning"])
    finalization = dict(receipt["finalization"])
    if sha256_text(str(reasoning["raw_text"])) != reasoning.get("transcript_hash"):
        raise SplitBudgetReplayError("reasoning_transcript_hash")
    if sha256_text(str(finalization["raw_text"])) != finalization.get("raw_sha256"):
        raise SplitBudgetReplayError("final_raw_sha256")
    if finalization.get("candidate_environment_hash") != environment.get("environment_hash"):
        raise SplitBudgetReplayError("candidate_environment_hash")
    if finalization.get("reasoning_transcript_hash") != reasoning.get("transcript_hash"):
        raise SplitBudgetReplayError("finalizer_reasoning_hash")
    replayed = classify_finalization_stage(
        row,
        environment,
        str(finalization["raw_text"]),
        finish_reason=str(finalization["finish_reason"]),
        output_tokens=int(finalization["output_tokens"]),
        config={
            "max_tokens": int(finalization["max_tokens"]),
            "timeout_s": int(finalization["timeout_s"]),
            "stops": list(finalization["stops"]),
        },
        timeout=finalization.get("timeout") is True,
        reasoning_transcript_hash=str(reasoning["transcript_hash"]),
        prompt_hash=str(finalization.get("prompt_hash") or ""),
    )
    if replayed["parser_receipt_hash"] != finalization.get("parser_receipt_hash"):
        raise SplitBudgetReplayError("parser_receipt_hash")
    if receipt.get("receipt_hash") and _split_receipt_hash(receipt) != receipt.get("receipt_hash"):
        raise SplitBudgetReplayError("receipt_hash")
    return True


def positive_control_receipt(
    row: Mapping[str, Any],
    mode: Mapping[str, Any],
    *,
    runtime_supports_grammar: bool,
) -> JsonDict:
    """Build the clean two-stage receipt used to gate contract readiness."""

    environment = build_candidate_environment(row)
    candidate_id = str(environment["label_to_candidate_id"][row["exact_label"]])
    return execute_two_stage_contract(
        row,
        mode,
        reasoning_text="bounded reasoning transcript over sealed fixture facts",
        final_text=f"{row['row_id']}: {candidate_id}",
        reasoning_tokens=7,
        final_tokens=2,
        runtime_supports_grammar=runtime_supports_grammar,
    )


def _control_receipt(passed: bool, **fields: Any) -> JsonDict:
    return {"passed": passed, **fields}


def adversarial_control_results(
    row: Mapping[str, Any],
    other_row: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Run parser, budget, leakage, timeout, ghost, and replay controls."""

    mode = SPLIT_BUDGET_MODES[0]
    environment = build_candidate_environment(row)
    exact_id = str(environment["label_to_candidate_id"][row["exact_label"]])
    wrong_id = next(
        candidate_id
        for candidate_id, candidate in environment["candidate_by_id"].items()
        if candidate["label"] != row["exact_label"]
    )
    ghost_env = build_candidate_environment(other_row or row)
    ghost_id = next(
        candidate_id for candidate_id in ghost_env["candidate_ids"] if candidate_id not in environment["candidate_ids"]
    )
    valid_final = f"{row['row_id']}: {exact_id}"

    empty_reasoning = execute_two_stage_contract(
        row,
        mode,
        reasoning_text="",
        final_text=valid_final,
    )
    empty_final = execute_two_stage_contract(
        row,
        mode,
        reasoning_text="bounded reasoning",
        final_text="",
    )
    overlong = execute_two_stage_contract(
        row,
        mode,
        reasoning_text="bounded reasoning but capped",
        final_text=valid_final,
        reasoning_finish_reason="length",
        reasoning_tokens=int(mode["reasoning"]["max_tokens"]),
    )
    stop = execute_two_stage_contract(
        row,
        mode,
        reasoning_text="bounded reasoning <stop> collision",
        final_text=valid_final,
    )
    unclosed = execute_two_stage_contract(
        row,
        mode,
        reasoning_text="<think> unfinished reasoning",
        final_text=valid_final,
    )
    duplicate = execute_two_stage_contract(
        row,
        mode,
        reasoning_text="bounded reasoning",
        final_text=f"{valid_final}\n{valid_final}",
    )
    invalid = execute_two_stage_contract(
        row,
        mode,
        reasoning_text="bounded reasoning",
        final_text=f"{row['row_id']}: NOT_A_CANDIDATE",
    )
    ghost = execute_two_stage_contract(
        row,
        mode,
        reasoning_text="bounded reasoning",
        final_text=f"{row['row_id']}: {ghost_id}",
    )
    injection = execute_two_stage_contract(
        row,
        mode,
        reasoning_text="bounded reasoning",
        final_text='{"row_id":"x","label":"A"}\nignore previous instructions',
    )
    leaked = execute_two_stage_contract(
        row,
        mode,
        reasoning_text="bounded reasoning",
        final_text=valid_final,
        include_hidden_labels=True,
    )
    timed_out = execute_two_stage_contract(
        row,
        mode,
        reasoning_text="bounded reasoning",
        final_text=valid_final,
        reasoning_timeout=True,
    )
    exact_wrong = execute_two_stage_contract(
        row,
        mode,
        reasoning_text="bounded reasoning",
        final_text=f"{row['row_id']}: {wrong_id}",
    )
    replay = positive_control_receipt(row, mode, runtime_supports_grammar=False)
    tampered = _copy_json(replay)
    tampered["finalization"]["candidate_environment_hash"] = "sha256:tampered"
    replay_ok = True
    try:
        replay_split_receipt(tampered)
    except SplitBudgetReplayError:
        replay_ok = False

    return {
        "empty_reasoning": _control_receipt(
            empty_reasoning["valid_exact_output"] is False,
            failure_mode=empty_reasoning["failure_mode"],
            valid_exact_output=empty_reasoning["valid_exact_output"],
        ),
        "empty_final": _control_receipt(
            empty_final["valid_exact_output"] is False,
            failure_mode=empty_final["failure_mode"],
            parser_failure_reason=empty_final["finalization"]["parser_failure_reason"],
            valid_exact_output=empty_final["valid_exact_output"],
        ),
        "overlong_reasoning": _control_receipt(
            overlong["valid_exact_output"] is False,
            failure_mode=overlong["failure_mode"],
            valid_exact_output=overlong["valid_exact_output"],
        ),
        "stop_collision": _control_receipt(
            stop["valid_exact_output"] is False,
            failure_mode=stop["failure_mode"],
            valid_exact_output=stop["valid_exact_output"],
        ),
        "unclosed_thinking": _control_receipt(
            unclosed["valid_exact_output"] is False,
            failure_mode=unclosed["failure_mode"],
            valid_exact_output=unclosed["valid_exact_output"],
        ),
        "duplicate_candidate_id": _control_receipt(
            duplicate["valid_exact_output"] is False,
            parser_failure_reason=duplicate["finalization"]["parser_failure_reason"],
            valid_exact_output=duplicate["valid_exact_output"],
        ),
        "invalid_candidate_id": _control_receipt(
            invalid["valid_exact_output"] is False,
            parser_failure_reason=invalid["finalization"]["parser_failure_reason"],
            valid_exact_output=invalid["valid_exact_output"],
        ),
        "ghost_candidate_id": _control_receipt(
            ghost["valid_exact_output"] is False,
            parser_failure_reason=ghost["finalization"]["parser_failure_reason"],
            valid_exact_output=ghost["valid_exact_output"],
        ),
        "schema_control_plane_injection": _control_receipt(
            injection["valid_exact_output"] is False,
            parser_failure_reason=injection["finalization"]["parser_failure_reason"],
            valid_exact_output=injection["valid_exact_output"],
        ),
        "candidate_label_leakage": _control_receipt(
            leaked["valid_exact_output"] is False,
            hidden_label_leakage_detected=leaked["prompt_leakage"][
                "hidden_label_leakage_detected"
            ],
            leakage_markers=leaked["prompt_leakage"]["leakage_markers"],
        ),
        "timeout": _control_receipt(
            timed_out["valid_exact_output"] is False,
            failure_mode=timed_out["failure_mode"],
            valid_exact_output=timed_out["valid_exact_output"],
        ),
        "replay_mismatch": _control_receipt(
            replay_ok is False,
            replay_ok=replay_ok,
            failure_mode="replay_mismatch",
        ),
        "exact_wrong_answer": _control_receipt(
            exact_wrong["finalization"]["parse_ok"] is True
            and exact_wrong["finalization"]["exact_answer_error"] is True
            and exact_wrong["valid_exact_output"] is False,
            parse_ok=exact_wrong["finalization"]["parse_ok"],
            exact_answer_error=exact_wrong["finalization"]["exact_answer_error"],
            valid_exact_output=exact_wrong["valid_exact_output"],
        ),
    }


def _fixture_rows() -> list[JsonDict]:
    rows = fixture.generate_fixture_rows()
    return [dict(row) for row in rows]


def _representative_rows() -> tuple[JsonDict, JsonDict]:
    rows = _fixture_rows()
    primary = next(
        row
        for row in rows
        if row["surface_kind"] == "canonical" and row["split"] == "train"
    )
    other = next(row for row in rows if row["row_id"] != primary["row_id"])
    return primary, other


def _structured_gate_replay(root: Path) -> JsonDict:
    exp5811_artifact = _read_json(root / EXP5811_ARTIFACT_RELATIVE_PATH)
    exp5798_artifact = _read_json(root / EXP5798_ARTIFACT_RELATIVE_PATH)
    fixture_artifact = _read_json(root / EXP5785_ARTIFACT_RELATIVE_PATH)
    receipts = {
        "exp5811": {
            "status": exp5811_artifact.get("status"),
            "ready_score": exp5811_artifact.get("canary_evidence_ready_score"),
            "passed": exp5811_artifact.get("status") == "complete"
            and exp5811_artifact.get("canary_evidence_ready_score") == 1.0,
        },
        "exp5798": {
            "status": exp5798_artifact.get("status"),
            "ready_score": exp5798_artifact.get("channel_diagnostic_ready_score"),
            "passed": exp5798_artifact.get("status") == "complete"
            and exp5798_artifact.get("channel_diagnostic_ready_score") == 1.0,
        },
        "exp5785_fixture": {
            "status": fixture_artifact.get("status"),
            "ready_score": fixture_artifact.get("fixture_ready_score"),
            "parser_control_pass_rate": fixture_artifact.get("parser_control_pass_rate"),
            "passed": fixture_artifact.get("status") == "complete"
            and fixture_artifact.get("fixture_ready_score") == 1.0
            and fixture_artifact.get("parser_control_pass_rate") == 1.0,
        },
    }
    receipts["all_passed"] = all(dict(row).get("passed") is True for row in receipts.values())
    return receipts


def _input_hashes(root: Path) -> JsonDict:
    exp5798_artifact = _read_json(root / EXP5798_ARTIFACT_RELATIVE_PATH)
    return {
        "exp5811_audit": _hash_path(root, EXP5811_ARTIFACT_RELATIVE_PATH),
        "exp5798_diagnostic": _hash_path(root, EXP5798_ARTIFACT_RELATIVE_PATH),
        "sealed_fixture_artifact": _hash_path(root, EXP5785_ARTIFACT_RELATIVE_PATH),
        "sealed_fixture_rows": _hash_path(root, EXP5785_ROWS_RELATIVE_PATH),
        "sealed_fixture_parser": _hash_path(root, EXP5785_MODULE_RELATIVE_PATH),
        "current_channel_producer": _hash_path(root, EXP5799_PRODUCER_RELATIVE_PATH),
        "current_channel_producer_tests": _hash_path(root, EXP5799_TEST_RELATIVE_PATH),
        "embedded_template_metadata_fixture": sha256_json(
            exp5798_artifact.get("embedded_template_metadata") or {}
        ),
        "sota_model_metadata_registry": _hash_path(root, SOTA_MODEL_METADATA_RELATIVE_PATH),
        "experiment_template": _hash_path(root, EXPERIMENT_TEMPLATE_RELATIVE_PATH),
        "verification_spec": _hash_path(root, VERIFY_SPEC_RELATIVE_PATH),
        "research_references": _hash_path(root, RESEARCH_REFERENCES_RELATIVE_PATH),
        "codex_instructions": _hash_path(root, CODEX_RELATIVE_PATH),
        "claude_instructions": _hash_path(root, CLAUDE_RELATIVE_PATH),
        "split_budget_module": _hash_path(root, MODULE_RELATIVE_PATH),
        "split_budget_tests": _hash_path(root, TEST_RELATIVE_PATH),
    }


def collect_preconditions(root: Path = REPO_ROOT) -> JsonDict:
    """Replay upstream gates and record dependency hashes without model loads."""

    root = Path(root)
    hashes = _input_hashes(root)
    gates = _structured_gate_replay(root)
    missing = sorted(name for name, value in hashes.items() if value == "missing")
    blocked = []
    if gates["all_passed"] is not True:
        blocked.append("structured_gate_replay_failed")
    if missing:
        blocked.append("missing_hashed_inputs:" + ",".join(missing))
    memory = _memory_probe()
    disk = _disk_probe(root)
    if memory["ok"] is not True:
        blocked.append("insufficient_free_ram")
    if disk["ok"] is not True:
        blocked.append("insufficient_free_disk")
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "structured_gate_replay": gates,
        "input_hashes": hashes,
        "memory": memory,
        "disk": disk,
        "headline_model_loaded": False,
        "llm_calls_made": 0,
        "autotokenizer_used_on_gguf": False,
        "gguf_templates_modified": False,
        "research_conductor_modified": False,
        "preconditions_ready": not blocked,
        "blocked_reasons": blocked,
    }


def _preregistered_mode_matrix() -> list[JsonDict]:
    modes = [_copy_json(SHARED_BUDGET_CONTROL)]
    for mode in SPLIT_BUDGET_MODES:
        row = _copy_json(mode)
        row["canary_mode_preregistered"] = True
        row["mode_retirement_rules"] = {
            "retire_on_empty_reasoning": True,
            "retire_on_empty_final": True,
            "retire_on_truncation": True,
            "retire_on_stop_collision": True,
            "retire_on_unclosed_thinking": True,
            "retire_on_timeout": True,
            "retire_on_hidden_label_leakage": True,
            "retire_on_replay_mismatch": True,
            "retire_on_exact_answer_mismatch_control": True,
        }
        modes.append(row)
    return modes


def _stage_contracts(mode: Mapping[str, Any], receipt: Mapping[str, Any]) -> tuple[JsonDict, JsonDict]:
    reasoning = dict(receipt["reasoning"])
    finalization = dict(receipt["finalization"])
    reasoning_contract = {
        "stage_id": "bounded_reasoning_stage_v1",
        "call_ordinal": 1,
        "max_tokens": int(mode["reasoning"]["max_tokens"]),
        "timeout_s": int(mode["reasoning"]["timeout_s"]),
        "stops": list(mode["reasoning"]["stops"]),
        "raw_receipt_hash": reasoning["raw_sha256"],
        "transcript_hash": reasoning["transcript_hash"],
        "immutable_after_capture": True,
        "finish_reason": reasoning["finish_reason"],
        "output_tokens": reasoning["output_tokens"],
        "budget_accounting": {
            "measured_separately": True,
            "output_tokens": reasoning["output_tokens"],
            "max_tokens": int(mode["reasoning"]["max_tokens"]),
        },
    }
    final_contract = {
        "stage_id": "bounded_finalization_stage_v1",
        "call_ordinal": 2,
        "max_tokens": int(mode["finalization"]["max_tokens"]),
        "timeout_s": int(mode["finalization"]["timeout_s"]),
        "stops": list(mode["finalization"]["stops"]),
        "raw_receipt_hash": finalization["raw_sha256"],
        "parser": EXACT_PARSER_ID,
        "candidate_environment_hash": finalization["candidate_environment_hash"],
        "reasoning_transcript_hash": finalization["reasoning_transcript_hash"],
        "hidden_label_leakage_detected": receipt["prompt_leakage"][
            "hidden_label_leakage_detected"
        ],
        "finish_reason": finalization["finish_reason"],
        "output_tokens": finalization["output_tokens"],
        "budget_accounting": {
            "measured_separately": True,
            "output_tokens": finalization["output_tokens"],
            "max_tokens": int(mode["finalization"]["max_tokens"]),
        },
    }
    return reasoning_contract, final_contract


def _field_provenance() -> JsonDict:
    provenance: JsonDict = {
        field: {
            "principle": REQUIRED_FIELD_PRINCIPLES[field],
            "sources": [
                "task_prompt",
                VERIFY_SPEC_RELATIVE_PATH.as_posix(),
                TEST_RELATIVE_PATH.as_posix(),
                MODULE_RELATIVE_PATH.as_posix(),
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    provenance.update(
        {
            field: {"principle": principle, "sources": ["local_metadata"]}
            for field, principle in FIELD_PRINCIPLE_EXTRAS.items()
        }
    )
    provenance["sealed_fixture_sources"] = {
        "fixture_artifact": EXP5785_ARTIFACT_RELATIVE_PATH.as_posix(),
        "fixture_rows": EXP5785_ROWS_RELATIVE_PATH.as_posix(),
        "parser_module": EXP5785_MODULE_RELATIVE_PATH.as_posix(),
    }
    return provenance


def split_budget_contract_ready_score_from_artifact(artifact: Mapping[str, Any]) -> float:
    """Recompute the bare readiness scalar from contract receipts."""

    controls = dict(artifact.get("adversarial_control_results") or {})
    modes = list(artifact.get("preregistered_mode_matrix") or [])
    positive = dict(dict(artifact.get("replay_receipts") or {}).get("positive_control") or {})
    reasoning = dict(artifact.get("reasoning_stage_contract") or {})
    finalization = dict(artifact.get("finalization_stage_contract") or {})
    ready = bool(
        artifact.get("status") == "complete"
        and dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and artifact.get("llm_calls_made") == 0
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and len(controls) == len(EXPECTED_ADVERSARIAL_CONTROLS)
        and all(dict(row).get("passed") is True for row in controls.values())
        and any(dict(mode).get("mode_type") == "shared_budget_control" for mode in modes)
        and sum(dict(mode).get("mode_type") == "split_budget" for mode in modes) >= 2
        and all(dict(mode).get("retirement_rules_preregistered") is True for mode in modes)
        and positive.get("replay_ok") is True
        and positive.get("call_count") == 2
        and dict(positive.get("budget_accounting") or {}).get("measured_separately") is True
        and dict(positive.get("prompt_leakage") or {}).get("hidden_label_leakage_detected")
        is False
        and dict(reasoning.get("budget_accounting") or {}).get("measured_separately") is True
        and dict(finalization.get("budget_accounting") or {}).get("measured_separately")
        is True
        and finalization.get("hidden_label_leakage_detected") is False
    )
    return 1.0 if ready else 0.0


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum blanked."""

    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    duration_s: float | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    """Build the terminal Exp5812 contract artifact."""

    started = time.perf_counter()
    root = Path(root)
    preconditions = collect_preconditions(root)
    row, other = _representative_rows()
    mode = SPLIT_BUDGET_MODES[0]
    positive = positive_control_receipt(row, mode, runtime_supports_grammar=True)
    controls = adversarial_control_results(row, other)
    reasoning_contract, finalization_contract = _stage_contracts(mode, positive)
    measured_duration = (
        float(duration_s)
        if duration_s is not None
        else round(time.perf_counter() - started, 6)
    )
    status = "complete" if preconditions["preconditions_ready"] is True else "blocked"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "status": status,
        "preconditions_checked": preconditions,
        "contract_version_and_code_hashes": {
            "contract_version": CONTRACT_VERSION,
            "schema": SCHEMA,
            "code_hash": _hash_path(root, MODULE_RELATIVE_PATH),
            "test_hash": _hash_path(root, TEST_RELATIVE_PATH),
            "hashed_inputs": preconditions["input_hashes"],
            "transport_drift_allowed": False,
        },
        "reasoning_stage_contract": reasoning_contract,
        "finalization_stage_contract": finalization_contract,
        "sealed_candidate_environment": {
            "environment_hash": positive["candidate_environment"]["environment_hash"],
            "row_id": positive["candidate_environment"]["row_id"],
            "fixture_row_hash": positive["candidate_environment"]["fixture_row_hash"],
            "candidate_ids": positive["candidate_environment"]["candidate_ids"],
            "candidate_count": len(positive["candidate_environment"]["candidate_ids"]),
            "hidden_labels_exposed_to_prompt": False,
            "exact_label_exposed_to_prompt": False,
            "exact_answer_exposed_to_prompt": False,
        },
        "grammar_claim_boundary": {
            "claim_boundary": GRAMMAR_CLAIM_BOUNDARY,
            "supported_runtime_receipt": positive["grammar_receipt"],
            "unsupported_runtime_receipt": environment_indexed_grammar_receipt(
                positive["candidate_environment"],
                runtime_supports=False,
            ),
            "semantic_correctness_claimed": False,
            "exact_validation_required": True,
        },
        "preregistered_mode_matrix": _preregistered_mode_matrix(),
        "adversarial_control_results": controls,
        "replay_receipts": {
            "positive_control": positive,
            "frozen_transcript_replays": positive["replay_ok"],
            "deterministic_parser_replay": True,
            "replay_mismatch_control_passed": controls["replay_mismatch"]["passed"],
        },
        "split_budget_contract_ready_score": 0.0,
        "llm_calls_made": 0,
        "duration_s": measured_duration,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands or []),
        "test_exit_codes": dict(test_exit_codes or {}),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["split_budget_contract_ready_score"] = split_budget_contract_ready_score_from_artifact(
        artifact
    )
    artifact["honest_verdict"] = (
        "complete: split_budget_channel_contract_ready_no_headline_model_calls"
        if artifact["split_budget_contract_ready_score"] == 1.0
        else "blocked: " + ",".join(preconditions["blocked_reasons"] or ["contract_gates"])
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate required fields, field provenance, readiness, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact["status"] not in {"complete", "blocked"}:
        raise ValueError("status")  # pragma: no cover
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact["llm_calls_made"] != 0:
        raise ValueError("llm_calls_made")
    provenance = artifact["field_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")  # pragma: no cover
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        receipt = dict(provenance.get(field) or {})
        if receipt.get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    expected_score = split_budget_contract_ready_score_from_artifact(artifact)
    if artifact["split_budget_contract_ready_score"] != expected_score:
        raise ValueError("split_budget_contract_ready_score")
    verdict = str(artifact["honest_verdict"])
    if not verdict.startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def build_and_write_artifact(
    *,
    root: Path = REPO_ROOT,
    result_path: Path | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    """Write the Exp5812 terminal artifact."""

    artifact = build_artifact(
        root=root,
        duration_s=duration_s,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    output = Path(result_path or Path(root) / RESULT_RELATIVE_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = build_and_write_artifact(test_commands=DEFAULT_TEST_COMMANDS)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
