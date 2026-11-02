"""Replay the V639 mention capture and seal the next comparator contract.

This module does no model work. It rebuilds requests and reductions from stored
bytes so transport defects stay separate from model quality. Historical invalid
answers remain invalid; a separate diagnostic explains the direct-label defect.

Spec refs: REQ-VERIFY-7275 and SCENARIO-VERIFY-7275-*.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Callable

from carnot import experiment_7238_v637_mention_capture as capture
from carnot import experiment_7265_v639_mention_heldout as prior
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.llama_server_supervisor import utc_now
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]
RUN_DATE = "20260913"
MILESTONE = "2026.09.640"
EXPERIMENT_ID = "exp7275-semantic-replay"
SCHEMA = "carnot.exp7275.v640_semantic_replay.v1"
RANDOM_SEED = 727_520_260_913
MODEL_SPECS: list[JsonDict] = []
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "usable_answers": 0,
}

SOURCE_ARTIFACT_PATH = Path("results/experiment_7265_v639_mention_heldout.json")
SOURCE_RAW_DIR = Path("results/raw/experiment_7265")
SOURCE_SCHEDULE_PATH = SOURCE_RAW_DIR / "schedule.json"
SOURCE_MANIFEST_PATH = SOURCE_RAW_DIR / "raw_call_manifest.json"
PUBLIC_PATH = prior.PUBLIC_PATH
AUTHORITY_PATH = prior.AUTHORITY_PATH
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7275_v640_semantic_replay.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7275_v640_semantic_replay.py")
TEST_PATH = Path("tests/python/test_experiment_7275_v640_semantic_replay.py")
RESULT_PATH = Path("results/experiment_7275_v640_semantic_replay.json")
RAW_DIR = Path("results/raw/experiment_7275")
RAW_CANDIDATE_PATH = RAW_DIR / "measured-terminal-candidate.json"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7275_v640_semantic_replay.json")

PINNED_INPUT_HASHES = {
    SOURCE_ARTIFACT_PATH: "sha256:712335d44e39a49b1325f044a14dea84472e4deee9f354b6821c4ec764c10cfc",
    SOURCE_SCHEDULE_PATH: "sha256:adc122a11f66b99400d7ab81df4c22f8b0cf96fef7651e041da7366662c7ebb5",
    SOURCE_MANIFEST_PATH: "sha256:50d21d16d37f8952273fa3d1b1aa4e3faa1c663316b077315855f38a0aeab9b6",
    PUBLIC_PATH: prior.PINNED_PUBLIC_SHA256,
    AUTHORITY_PATH: prior.PINNED_AUTHORITY_SHA256,
    EXCLUSION_PATH: "sha256:28fa92633ce4692ad7fcfa2db69cdfa2266fd75d2c8ef3284d6adee2d82ade92",
}

DIRECT_DECISIONS = ("supported", "contradicted", "unknown")
DIRECT_SYMBOLS = {"a": "supported", "b": "contradicted", "c": "unknown"}
SYSTEM_PROMPT = "Return only one JSON object accepted by the supplied grammar."
DIRECT_PROMPT = (
    "Judge the claim from the source. Return one full decision string: supported, "
    "contradicted, or unknown. Return only one JSON object with the decision key."
    "\nSOURCE:\n{source}\nCLAIM:\n{claim}"
)
DIRECT_GRAMMAR = (
    'root ::= "{\\"decision\\":" decision "}"\n'
    'decision ::= "\\"supported\\"" | "\\"contradicted\\"" | '
    '"\\"unknown\\""\n'
)

REQUIRED_VALIDATION_NAMES = (
    "focused_pytest",
    "affected_suites",
    "scoped_coverage",
    "scoped_coverage_report",
    "ruff_check",
    "ruff_format",
    "mypy",
    "scoped_spec_coverage",
    "independent_raw_replay",
    "adversarial_verify",
    "verdict_row_consistency",
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Version the result and retain ordinary top-level experiment_id and milestone.",
    "status": "Use complete or blocked for terminal evidence; keep unfinished work in separate checkpoints.",
    "run_date": "Use 20260913 and actual UTC start and end times.",
    "field_principles": "Store explanations here; consumer values remain ordinary top-level fields.",
    "preconditions_checked": "Record actual input hashes, authority separation, resource ownership, and failures.",
    "MODEL_SPECS": "Declare models executable now; keep historical identities in hashed sidecars.",
    "model_invoked": "Derive this from current calls, including failed or unusable generation.",
    "invocation_counts": "Separate attempted and completed loads and generation from usable answers.",
    "inference_substrate": "Use the recognized literal for the actual computation.",
    "inference_substrate_class": "Use the correct no-LLM class and never pad duration.",
    "execution_venue": "Host orchestration is host; identify device work separately.",
    "duration_s": "Measure monotonic elapsed time and disjoint phase spans.",
    "random_seed": "Freeze independent-unit seeds before observing results.",
    "reproducibility_checksum": "Bind code, configuration, manifests, and raw evidence.",
    "source_artifact_hashes": "Preserve exact input identity, retirement, and quarantine status.",
    "rows": "Keep each unit, arm, seed, error, abstention, cost, metric, and censoring state.",
    "sample_size_budget": "Record planned, attempted, completed, censored units, and stopping rule.",
    "acceptance_gate_results": "Record expected, observed, passed, and principle for every criterion.",
    "gate_check_summary": "For blocked work name upstream, exact check, observed value, and expected value.",
    "verifier_is_oracle": "Expose shared evaluator authority; conformance is not learned correctness.",
    "honest_verdict": "Completed findings start complete_; external absence starts blocked_.",
    "verdict_class": "Use the closed verdict class set; only unfinished current work is partial.",
    "validation_receipts": "Retain command, exit code, timing, and log hash without hiding failures.",
    "semantic_replay_ready_score": "One means exact reconstruction and fail-closed malformed controls, without an accuracy threshold.",
    "first_divergence_rows": "Locate producer and consumer mismatches without repairing historical replies.",
    "comparator_contract_path": "Hash the public decision schema and exact native request settings for the next canary.",
    "historical_arm_reduction": "Report each arm separately with all invalid and abstained outputs.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

GATE_PRINCIPLES = {
    "authenticated_inputs": "Every upstream byte string must match its authenticated identity.",
    "raw_reconstruction": "All 320 stored completion rows must rebuild exactly.",
    "semantic_reconstruction": "All 192 semantic rows must rebuild exactly.",
    "corrupted_controls": "Every malformed or mutated fixture must fail closed.",
    "known_root_cause": "Unknown cause prevents readiness and fresh inference.",
    "validation": "Every focused test, lint, replay, and artifact check must pass.",
}


def canonical_json(value: Any) -> str:
    """Use the stable JSON spelling that the native request transport used."""

    return prior.canonical_json(value)


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the repository's prefixed SHA-256 spelling."""

    return prior.sha256_bytes(value)


def sha256_file(path: Path) -> str:
    """Hash one file without parsing or normalizing its evidence."""

    return sha256_bytes(path.read_bytes())


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind durable evidence while excluding process-local clock observations."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "timestamps", "phase_spans", "reproducibility_checksum"}
    }
    return sha256_bytes(canonical_json(stable).encode("utf-8"))


def _gate_row(
    check: str,
    expected: Any,
    observed: Any,
    passed: bool,
    *,
    upstream: str,
    field: str,
) -> JsonDict:
    """Keep each precondition failure precise enough for a blocked receipt."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
        "upstream": upstream,
        "field": field,
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Project the first failed prerequisite without hiding later failures."""

    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is None:
        return {
            "failed_check": None,
            "upstream": None,
            "field": None,
            "expected_value": None,
            "observed_value": None,
        }
    return {
        "failed_check": failed.get("check"),
        "upstream": failed.get("upstream"),
        "field": failed.get("field"),
        "expected_value": failed.get("expected_value"),
        "observed_value": failed.get("observed_value"),
    }


def _manifest_lists_experiment(manifest: Any, experiment_id: int) -> bool:
    """Find an exact retired experiment ID without matching unrelated prose."""

    if isinstance(manifest, Mapping):
        if manifest.get("experiment_id") == experiment_id:
            return True
        ids = manifest.get("experiment_ids")
        if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes)):
            if experiment_id in ids:
                return True
        return any(_manifest_lists_experiment(value, experiment_id) for value in manifest.values())
    if isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes)):
        return any(_manifest_lists_experiment(value, experiment_id) for value in manifest)
    return False


def authenticate_inputs(
    root: Path, *, expected_hashes: Mapping[Path, str] | None = None
) -> list[JsonDict]:
    """Authenticate upstream evidence before reducer work or output creation."""

    wanted = dict(expected_hashes or PINNED_INPUT_HASHES)
    checks: list[JsonDict] = []
    for relative, expected in wanted.items():
        path = root / relative
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            _gate_row(
                "authenticated_input",
                expected,
                observed,
                observed == expected,
                upstream=relative.as_posix(),
                field="sha256",
            )
        )
    source_path = root / SOURCE_ARTIFACT_PATH
    if source_path.is_file() and checks[0]["passed"]:
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source_errors = prior.validate_artifact(source)
        checks.append(
            _gate_row(
                "source_terminal_contract",
                [],
                source_errors,
                not source_errors,
                upstream="exp7265-mention-heldout",
                field="terminal_schema_and_checksum",
            )
        )
        expected_calls = source.get("source_artifact_hashes") or {}
        for index in range(prior.PLANNED_CALLS):
            relative = SOURCE_RAW_DIR / f"call_{index:02d}.json"
            expected = expected_calls.get(f"call_{index:02d}")
            path = root / relative
            observed = sha256_file(path) if path.is_file() else None
            checks.append(
                _gate_row(
                    "authenticated_raw_call",
                    expected,
                    observed,
                    isinstance(expected, str) and observed == expected,
                    upstream=relative.as_posix(),
                    field="sha256",
                )
            )
    else:
        checks.append(
            _gate_row(
                "source_terminal_contract",
                "authenticated source artifact",
                "unavailable",
                False,
                upstream="exp7265-mention-heldout",
                field="terminal_schema_and_checksum",
            )
        )
    exclusion_path = root / EXCLUSION_PATH
    exclusion = prior.load_yaml(exclusion_path) if exclusion_path.is_file() else {}
    excluded = any(_manifest_lists_experiment(exclusion, value) for value in (7265, 7275))
    checks.append(
        _gate_row(
            "retirement_and_quarantine",
            False,
            excluded,
            not excluded,
            upstream=EXCLUSION_PATH.as_posix(),
            field="experiment_id",
        )
    )
    ownership = {
        "execution_venue": "host",
        "pid": os.getpid(),
        "uid": os.getuid(),
        "result_parent_writable": os.access(root / RESULT_PATH.parent, os.W_OK),
        "raw_parent_writable": os.access(root / RAW_DIR.parent, os.W_OK),
    }
    checks.append(
        _gate_row(
            "resource_ownership",
            {
                "execution_venue": "host",
                "result_parent_writable": True,
                "raw_parent_writable": True,
            },
            ownership,
            ownership["result_parent_writable"] and ownership["raw_parent_writable"],
            upstream="host_process",
            field="output_ownership",
        )
    )
    return checks


def _decode_b64(value: Any) -> bytes:
    """Decode retained bytes strictly so corruption cannot be normalized away."""

    if not isinstance(value, str):
        raise ValueError("base64_type")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("base64_invalid") from exc


def _response_content(response_bytes: bytes) -> tuple[JsonDict, str]:
    """Decode the native response and select exactly one assistant content field."""

    try:
        decoded = json.loads(response_bytes)
        choices = decoded["choices"]
        content = choices[0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ValueError("response_contract") from exc
    if not isinstance(decoded, dict) or not isinstance(content, str) or len(choices) != 1:
        raise ValueError("response_contract")
    return decoded, content


def _direct_mapping_authorized(sealed: Mapping[str, Any]) -> bool:
    """Allow neutral-symbol diagnosis only when both public contracts define it."""

    prompt = str(sealed.get("prompt") or "")
    grammar = str(sealed.get("grammar") or "")
    return (
        "decision a when the claim is supported" in prompt
        and "b when it is contradicted" in prompt
        and "c when the source is insufficient" in prompt
        and grammar == prior._direct_grammar()["grammar"]
    )


def _corrected_direct_diagnostic(
    sealed: Mapping[str, Any], completion: Mapping[str, Any]
) -> JsonDict:
    """Diagnose the explicit symbol map without changing the historical reducer."""

    parsed = completion.get("parsed_completion")
    symbol = parsed.get("decision") if isinstance(parsed, Mapping) else None
    authorized = _direct_mapping_authorized(sealed)
    decision = DIRECT_SYMBOLS.get(str(symbol)) if authorized else None
    return {
        "call_order": sealed["call_order"],
        "call_id": sealed["call_id"],
        "unit_id": sealed["unit_id"],
        "symbol": symbol,
        "mapping_authorized_by_request": authorized,
        "historical_parse_valid": completion.get("parse_valid") is True,
        "historical_usable": completion.get("usable") is True,
        "diagnostic_decision": decision,
        "diagnostic_only": True,
    }


def _historical_arm_reduction(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep the three denominators separate so 63-of-192 cannot mix arms."""

    result: JsonDict = {}
    for arm in prior.ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        result[arm] = {
            "rows": len(arm_rows),
            "correct": sum(row.get("decision_correct") is True for row in arm_rows),
            "invalid": sum(row.get("error") is not None for row in arm_rows),
            "abstained": sum(row.get("abstention") is True for row in arm_rows),
        }
    return result


def load_and_replay(root: Path) -> JsonDict:
    """Run the independent byte-to-semantics replay over the frozen denominator."""

    source = json.loads((root / SOURCE_ARTIFACT_PATH).read_text(encoding="utf-8"))
    schedule_payload = json.loads((root / SOURCE_SCHEDULE_PATH).read_text(encoding="utf-8"))
    schedule = list(schedule_payload["schedule"])
    public_rows, authority_rows = prior.load_held_out_manifests(
        root / PUBLIC_PATH, root / AUTHORITY_PATH
    )
    reconstruction_errors = prior.schedule_errors(schedule, public_rows, authority_rows)
    if source.get("schedule") != schedule:
        reconstruction_errors.append("source_schedule_mismatch")
    retained_rows: list[JsonDict] = []
    call_replay_rows: list[JsonDict] = []
    divergences: list[JsonDict] = []
    diagnostics: list[JsonDict] = []
    for index, sealed in enumerate(schedule):
        call_path = root / SOURCE_RAW_DIR / f"call_{index:02d}.json"
        call_payload = json.loads(call_path.read_text(encoding="utf-8"))
        retained = dict(call_payload["completion"])
        retained_rows.append(retained)
        if call_payload.get("schedule") != sealed:
            reconstruction_errors.append(f"call_{index}:schedule")
        expected_payload, expected_request_bytes = prior.request_payload(sealed)
        try:
            request_bytes = _decode_b64(retained.get("raw_request_bytes_b64"))
            response_bytes = _decode_b64(retained.get("raw_response_bytes_b64"))
            decoded_response, response_content = _response_content(response_bytes)
        except ValueError as exc:  # pragma: no cover - authenticated evidence is fixed.
            reconstruction_errors.append(f"call_{index}:{exc}")
            request_bytes = b""
            response_bytes = b""
            decoded_response = {}
            response_content = ""
        rebuilt = capture.replay_completion_rows([sealed], [retained])[0]
        reconstruction_match = rebuilt == retained
        request_match = request_bytes == expected_request_bytes
        response_match = (
            decoded_response == retained.get("raw_response")
            if "raw_response" in retained
            else response_content == retained.get("raw_completion")
        )
        if not reconstruction_match:
            reconstruction_errors.append(f"call_{index}:historical_reconstruction")
        if not request_match or retained.get("actual_parameters") != expected_payload:
            reconstruction_errors.append(f"call_{index}:request_reconstruction")
        if not response_match:
            reconstruction_errors.append(f"call_{index}:response_reconstruction")
        call_replay_rows.append(
            {
                "call_order": index,
                "call_id": sealed["call_id"],
                "unit_id": sealed["unit_id"],
                "arm": sealed["arm"],
                "call_type": sealed["call_type"],
                "seed": sealed["seed"],
                "request_bytes_sha256": sha256_bytes(request_bytes),
                "expected_request_bytes_sha256": sha256_bytes(expected_request_bytes),
                "request_bytes_match": request_match,
                "grammar_forwarded_exactly": expected_payload["grammar"]
                == retained.get("actual_parameters", {}).get("grammar"),
                "decoding_settings_match": all(
                    retained.get("actual_parameters", {}).get(key) == value
                    for key, value in sealed["decoding_parameters"].items()
                ),
                "response_bytes_sha256": sha256_bytes(response_bytes),
                "response_bytes_match": response_match,
                "historical_row_sha256": retained.get("row_sha256"),
                "rebuilt_row_sha256": rebuilt.get("row_sha256"),
                "historical_reconstruction_match": reconstruction_match,
                "error": None
                if reconstruction_match and request_match and response_match
                else "replay_mismatch",
                "abstention": retained.get("explicit_unknown") is True,
                "cost": {
                    "prompt_tokens": retained.get("prompt_tokens", 0),
                    "completion_tokens": retained.get("completion_tokens", 0),
                    "latency_s": retained.get("latency_s", 0.0),
                },
                "metric": int(reconstruction_match and request_match and response_match),
                "censored": False,
            }
        )
        if sealed.get("arm") == "direct_judge":
            diagnostic = _corrected_direct_diagnostic(sealed, retained)
            diagnostics.append(diagnostic)
            divergences.append(
                {
                    "call_order": index,
                    "call_id": sealed["call_id"],
                    "unit_id": sealed["unit_id"],
                    "arm": sealed["arm"],
                    "field": "decision",
                    "expected": list(DIRECT_DECISIONS),
                    "observed": diagnostic["symbol"],
                    "producer_function": "carnot.experiment_7265_v639_mention_heldout._direct_grammar",
                    "consumer_function": "carnot.experiment_7238_v637_mention_capture._direct_shape_valid",
                    "cause": "contract_mismatch",
                    "historical_repair_applied": False,
                }
            )
        else:
            replay_shape = deepcopy(rebuilt)
            replay_shape["request_started_at_utc"] = retained.get("request_started_at_utc")
            replay_shape["response_observed_at_utc"] = retained.get("response_observed_at_utc")
            replay_shape["row_sha256"] = capture._row_hash(replay_shape)
            divergences.append(
                {
                    "call_order": index,
                    "call_id": sealed["call_id"],
                    "unit_id": sealed["unit_id"],
                    "arm": sealed["arm"],
                    "field": "row_sha256",
                    "expected": retained.get("row_sha256"),
                    "observed": replay_shape["row_sha256"],
                    "producer_function": "carnot.experiment_7238_v637_mention_capture.build_completion_row",
                    "consumer_function": "carnot.experiment_7265_v639_mention_heldout.independent_replay",
                    "cause": "contract_mismatch",
                    "historical_repair_applied": False,
                }
            )
    semantic_rows = capture.score_semantics(schedule, retained_rows, public_rows, authority_rows)
    stored_semantic_rows = list(source.get("rows") or [])
    if semantic_rows != stored_semantic_rows:
        reconstruction_errors.append("semantic_rows_mismatch")
    historical_identity = deepcopy(dict(source.get("model_identity_receipt") or {}))
    source_hashes = {
        relative.as_posix(): {
            "sha256": sha256_file(root / relative),
            "retired": False,
            "quarantined": False,
        }
        for relative in PINNED_INPUT_HASHES
    }
    for relative in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH):
        path = root / relative
        if path.is_file():
            source_hashes[relative.as_posix()] = {
                "sha256": sha256_file(path),
                "retired": False,
                "quarantined": False,
            }
    semantic_cause_rows = [
        {
            "unit_id": row["unit_id"],
            "arm": row["arm"],
            "cause": "real_model_error",
            "expected": row.get("expected_decision"),
            "observed": row.get("predicted_decision"),
        }
        for row in semantic_rows
        if row.get("arm") != "direct_judge" and row.get("decision_correct") is not True
    ]
    return {
        "root": root,
        "schedule": schedule,
        "retained_rows": retained_rows,
        "call_replay_rows": call_replay_rows,
        "semantic_rows": semantic_rows,
        "stored_semantic_rows": stored_semantic_rows,
        "reconstruction_errors": reconstruction_errors,
        "first_divergence_rows": divergences,
        "corrected_direct_diagnostics": diagnostics,
        "historical_arm_reduction": _historical_arm_reduction(semantic_rows),
        "historical_identity": historical_identity,
        "source_artifact_hashes": source_hashes,
        "semantic_cause_rows": semantic_cause_rows,
        "cause_classification_counts": {
            **{
                name: 0
                for name in (
                    "corruption",
                    "contract_mismatch",
                    "parser_rejection",
                    "real_model_error",
                    "unknown_cause",
                )
            },
            "contract_mismatch": len(divergences),
            "real_model_error": len(semantic_cause_rows),
        },
    }


def build_native_request(
    prompt: str, grammar: str, seed: int, max_tokens: int
) -> tuple[JsonDict, bytes]:
    """Build the next comparator's exact native request from public fields."""

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": seed,
        "cache_prompt": False,
        "max_tokens": max_tokens,
        "stream": False,
        "grammar": grammar,
    }
    return payload, canonical_json(payload).encode("utf-8")


def _reduce_direct_fixture(
    expected_request: bytes,
    actual_request: bytes,
    raw_completion: str,
    finish_reason: str,
) -> JsonDict:
    """Apply the independent full-label parser and reject transport drift first."""

    if actual_request != expected_request:
        return {"accepted": False, "decision": None, "classification": "corruption"}
    if finish_reason in {"length", "max_tokens"}:
        return {"accepted": False, "decision": None, "classification": "parser_rejection"}
    try:
        parsed = json.loads(raw_completion)
    except json.JSONDecodeError:
        return {"accepted": False, "decision": None, "classification": "parser_rejection"}
    valid = (
        isinstance(parsed, dict)
        and set(parsed) == {"decision"}
        and parsed.get("decision") in DIRECT_DECISIONS
    )
    if not valid:
        return {"accepted": False, "decision": None, "classification": "parser_rejection"}
    return {
        "accepted": True,
        "decision": parsed["decision"],
        "classification": "valid",
    }


def _relation_fixture(
    mentions: Sequence[Mapping[str, Any]], raw_completion: str, expected: Mapping[str, Any]
) -> JsonDict:
    """Resolve relation pointers only when public mention IDs are unique."""

    ids = [str(row.get("mention_id")) for row in mentions]
    if len(ids) != len(set(ids)):
        return {"accepted": False, "decision": None, "classification": "parser_rejection"}
    try:
        parsed = json.loads(raw_completion)
    except json.JSONDecodeError:
        return {"accepted": False, "decision": None, "classification": "parser_rejection"}
    if parsed == {"outcome": "unknown", "relations": []}:
        return {"accepted": True, "decision": "unknown", "classification": "valid"}
    relations = parsed.get("relations") if isinstance(parsed, dict) else None
    if not isinstance(relations, list) or len(relations) != 1:
        return {"accepted": False, "decision": None, "classification": "parser_rejection"}
    relation = relations[0]
    if (
        not isinstance(relation, dict)
        or relation.get("subject_pointer") not in ids
        or relation.get("object_pointer") not in ids
    ):
        return {"accepted": False, "decision": None, "classification": "parser_rejection"}
    correct = all(relation.get(key) == value for key, value in expected.items())
    return {
        "accepted": True,
        "decision": "supported" if correct else "contradicted",
        "classification": "valid" if correct else "real_model_error",
    }


def run_fixture_matrix() -> JsonDict:
    """Exercise valid boundaries and corrupt controls through native bytes."""

    prompt = DIRECT_PROMPT.format(source="Alpha precedes Beta.", claim="Alpha precedes Beta.")
    _payload, request_bytes = build_native_request(prompt, DIRECT_GRAMMAR, RANDOM_SEED, 512)
    rows: list[JsonDict] = []
    for decision in DIRECT_DECISIONS:
        reduced = _reduce_direct_fixture(
            request_bytes,
            request_bytes,
            canonical_json({"decision": decision}),
            "stop",
        )
        rows.append(
            {
                "fixture_id": f"direct_{decision}",
                **reduced,
                "request_bytes_match": True,
                "corrupted_control": False,
                "passed": reduced["accepted"] and reduced["decision"] == decision,
            }
        )
    malformed = _reduce_direct_fixture(request_bytes, request_bytes, "{bad", "stop")
    rows.append(
        {
            "fixture_id": "malformed_output",
            **malformed,
            "request_bytes_match": True,
            "corrupted_control": True,
            "passed": not malformed["accepted"]
            and malformed["classification"] == "parser_rejection",
        }
    )
    mentions = [
        {"mention_id": "m000", "surface_text": "Alpha"},
        {"mention_id": "m001", "surface_text": "Beta"},
    ]
    expected_relation = {
        "subject_pointer": "m000",
        "object_pointer": "m001",
        "predicate": "precedes",
        "polarity": "positive",
    }
    unknown = _relation_fixture(mentions, '{"outcome":"unknown","relations":[]}', expected_relation)
    rows.append(
        {
            "fixture_id": "explicit_unknown",
            **unknown,
            "request_bytes_match": True,
            "corrupted_control": False,
            "passed": unknown["accepted"] and unknown["decision"] == "unknown",
        }
    )
    unicode_prompt = DIRECT_PROMPT.format(
        source="Álpha precedes βeta.", claim="βeta follows Álpha."
    )
    _unicode_payload, unicode_bytes = build_native_request(
        unicode_prompt, DIRECT_GRAMMAR, RANDOM_SEED + 1, 512
    )
    unicode_reduced = _reduce_direct_fixture(
        unicode_bytes, unicode_bytes, '{"decision":"supported"}', "stop"
    )
    rows.append(
        {
            "fixture_id": "unicode_request",
            **unicode_reduced,
            "request_bytes_match": unicode_bytes.decode("utf-8").encode("utf-8") == unicode_bytes,
            "corrupted_control": False,
            "passed": unicode_reduced["accepted"] and "Álpha" in unicode_bytes.decode("utf-8"),
        }
    )
    duplicate = _relation_fixture(
        [mentions[0], deepcopy(mentions[0])],
        canonical_json({"outcome": "known", "relations": [expected_relation]}),
        expected_relation,
    )
    rows.append(
        {
            "fixture_id": "duplicate_mentions",
            **duplicate,
            "request_bytes_match": True,
            "corrupted_control": True,
            "passed": not duplicate["accepted"],
        }
    )
    reversed_relation = dict(expected_relation)
    reversed_relation.update({"subject_pointer": "m001", "object_pointer": "m000"})
    reversed_result = _relation_fixture(
        mentions,
        canonical_json({"outcome": "known", "relations": [reversed_relation]}),
        expected_relation,
    )
    rows.append(
        {
            "fixture_id": "reversed_relation",
            **reversed_result,
            "request_bytes_match": True,
            "corrupted_control": False,
            "passed": reversed_result["classification"] == "real_model_error",
        }
    )
    changed = _reduce_direct_fixture(
        request_bytes, request_bytes + b" ", '{"decision":"supported"}', "stop"
    )
    rows.append(
        {
            "fixture_id": "changed_request_bytes",
            **changed,
            "request_bytes_match": False,
            "corrupted_control": True,
            "passed": not changed["accepted"] and changed["classification"] == "corruption",
        }
    )
    truncated = _reduce_direct_fixture(
        request_bytes, request_bytes, '{"decision":"supported"}', "length"
    )
    rows.append(
        {
            "fixture_id": "token_truncation",
            **truncated,
            "request_bytes_match": True,
            "corrupted_control": True,
            "passed": not truncated["accepted"]
            and truncated["classification"] == "parser_rejection",
        }
    )
    corrupted = [row for row in rows if row["corrupted_control"]]
    return {
        "schema": "carnot.exp7275.reducer_fixtures.v1",
        "rows": rows,
        "corrupted_controls_total": len(corrupted),
        "corrupted_controls_rejected": sum(
            row["accepted"] is False and row["passed"] is True for row in corrupted
        ),
        "unknown_cause_count": sum(row["classification"] == "unknown_cause" for row in rows),
        "fixture_sha256": sha256_bytes(canonical_json(rows).encode("utf-8")),
    }


def reduce_two_draws(first: str, second: str) -> str:
    """Return the shared full decision, with deterministic disagreement abstention."""

    if first not in DIRECT_DECISIONS or second not in DIRECT_DECISIONS:
        raise ValueError("decision must use the sealed full-string schema")
    return first if first == second else "unknown"


def build_comparator_contract(historical_identity: Mapping[str, Any]) -> JsonDict:
    """Seal the public request contract for the next bounded canary."""

    contract: JsonDict = {
        "schema": "carnot.exp7275.comparator_contract.v1",
        "measurement_kind": "direct_comparator_measurement_repair",
        "decision_values": list(DIRECT_DECISIONS),
        "decision_schema": {
            "type": "object",
            "required": ["decision"],
            "additionalProperties": False,
        },
        "prompt_template": DIRECT_PROMPT,
        "public_only_prompts": True,
        "system_prompt": SYSTEM_PROMPT,
        "grammar": DIRECT_GRAMMAR,
        "grammar_sha256": sha256_bytes(DIRECT_GRAMMAR.encode("utf-8")),
        "grammar_forwarding": "exact_requested_bytes",
        "draw_count": 2,
        "draw_token_budgets": [512, 512],
        "draw_seeds": [RANDOM_SEED, RANDOM_SEED + 1],
        "decoding_parameters": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "cache_prompt": False,
            "stream": False,
        },
        "tie_rule": "unknown",
        "embedded_chat_template": {
            "required": True,
            "sha256": historical_identity.get("embedded_chat_template_sha256"),
            "source": "authenticated_historical_identity_sidecar",
        },
        "mention_method": {
            "changed": False,
            "source_schedule_function": "carnot.experiment_7265_v639_mention_heldout.build_schedule",
            "mention_arm": "mention_pointer",
            "schedule_sha256": prior.sha256_json(
                [
                    row
                    for row in prior.build_schedule(
                        *prior.load_held_out_manifests(
                            find_repo_root(start=__file__) / PUBLIC_PATH,
                            find_repo_root(start=__file__) / AUTHORITY_PATH,
                        )
                    )
                    if row["arm"] == "mention_pointer"
                ]
            ),
        },
        "finite_id_answer_channel": False,
        "schema_supported_semantic_reprompt": False,
    }
    contract["contract_sha256"] = sha256_bytes(canonical_json(contract).encode("utf-8"))
    return contract


def _sidecar_receipt(path: Path) -> JsonDict:
    """Return a content receipt without moving sidecar data into the artifact."""

    try:
        shown = path.relative_to(find_repo_root(start=__file__)).as_posix()
    except ValueError:
        shown = path.as_posix()
    return {"path": shown, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def write_sidecars(raw_dir: Path, replay: Mapping[str, Any]) -> JsonDict:
    """Keep historical identity, fixtures, and corrected diagnosis in hashed files."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    fixture = run_fixture_matrix()
    comparator = build_comparator_contract(replay["historical_identity"])
    values = {
        "historical_identity": {
            "schema": "carnot.exp7275.historical_identity.v1",
            "current_invocation": {
                "MODEL_SPECS": [],
                "model_invoked": False,
                "invocation_counts": ZERO_INVOCATION_COUNTS,
            },
            "historical_source": SOURCE_ARTIFACT_PATH.as_posix(),
            "historical_identity": replay["historical_identity"],
        },
        "fixture_matrix": fixture,
        "comparator_contract": comparator,
        "corrected_diagnostic": {
            "schema": "carnot.exp7275.corrected_direct_diagnostic.v1",
            "historical_rows_changed": False,
            "rows": replay["corrected_direct_diagnostics"],
        },
        "independent_reduction": {
            "schema": "carnot.exp7275.independent_reduction.v1",
            "call_replay_rows": replay["call_replay_rows"],
            "semantic_rows_sha256": sha256_bytes(
                canonical_json(replay["semantic_rows"]).encode("utf-8")
            ),
            "reconstruction_errors": replay["reconstruction_errors"],
        },
    }
    receipts: JsonDict = {}
    for name, value in values.items():
        path = raw_dir / f"{name}.json"
        atomic_write_json(path, value, allow_override=False, sort_keys=True)
        receipts[name] = _sidecar_receipt(path)
    return receipts


def base_artifact(run_date: str) -> JsonDict:
    """Create a complete schema before any fallible external precondition."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "partial",
        "run_date": run_date,
        "timestamps": {"started_at_utc": utc_now(), "completed_at_utc": None},
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_raw_calls": 320,
            "attempted_raw_calls": 0,
            "completed_raw_calls": 0,
            "censored_raw_calls": 0,
            "planned_semantic_rows": 192,
            "completed_semantic_rows": 0,
            "stopping_rule": "replay all frozen calls once; no accuracy stop",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": _gate_summary([]),
        "verifier_is_oracle": True,
        "honest_verdict": "partial_semantic_replay_not_started",
        "verdict_class": "partial",
        "validation_receipts": [],
        "semantic_replay_ready_score": 0,
        "first_divergence_rows": [],
        "comparator_contract_path": {},
        "historical_arm_reduction": {},
        "call_replay_rows": [],
        "cause_classification_counts": {},
        "semantic_cause_rows": [],
        "sidecar_receipts": {},
    }


def finalize_blocked_artifact(
    artifact: JsonDict, checks: Sequence[Mapping[str, Any]], duration_s: float
) -> JsonDict:
    """Finish an external block without describing unfinished task work."""

    artifact.update(
        {
            "status": "blocked",
            "preconditions_checked": deepcopy(list(checks)),
            "duration_s": duration_s,
            "phase_spans": [{"phase": "preconditions", "duration_s": duration_s}],
            "gate_check_summary": _gate_summary(checks),
            "honest_verdict": "blocked_unauthenticated_or_unavailable_exp7265_evidence",
            "verdict_class": "blocked",
            "timestamps": {
                "started_at_utc": artifact["timestamps"]["started_at_utc"],
                "completed_at_utc": utc_now(),
            },
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _acceptance_row(name: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Give each readiness criterion both a value and its governing reason."""

    return {
        "criterion": name,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "principle": GATE_PRINCIPLES[name],
    }


def _validations_complete(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require every named command exactly once and require every command to pass."""

    names = [row.get("name") for row in receipts]
    return Counter(names) == Counter(REQUIRED_VALIDATION_NAMES) and all(
        row.get("passed") is True and row.get("exit_code") == 0 for row in receipts
    )


def finalize_measured_artifact(
    artifact: JsonDict,
    checks: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    sidecars: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    started_at_utc: str,
    completed_at_utc: str,
    phase_spans: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Finish exact replay while retaining the earlier scientific null."""

    fixture = run_fixture_matrix()
    controls_ok = fixture["corrupted_controls_total"] == fixture["corrupted_controls_rejected"]
    reconstruction_errors = list(replay["reconstruction_errors"])
    unknown_causes = int(fixture["unknown_cause_count"]) + int(
        replay["cause_classification_counts"].get("unknown_cause", 0)
    )
    inputs_ok = all(row.get("passed") is True for row in checks)
    validation_ok = _validations_complete(validation_receipts)
    readiness = int(inputs_ok and not reconstruction_errors and controls_ok and unknown_causes == 0)
    gates = [
        _acceptance_row("authenticated_inputs", True, inputs_ok, inputs_ok),
        _acceptance_row(
            "raw_reconstruction", 0, len(reconstruction_errors), not reconstruction_errors
        ),
        _acceptance_row(
            "semantic_reconstruction",
            192,
            len(replay["semantic_rows"])
            if replay["semantic_rows"] == replay["stored_semantic_rows"]
            else 0,
            replay["semantic_rows"] == replay["stored_semantic_rows"],
        ),
        _acceptance_row(
            "corrupted_controls",
            fixture["corrupted_controls_total"],
            fixture["corrupted_controls_rejected"],
            controls_ok,
        ),
        _acceptance_row("known_root_cause", 0, unknown_causes, unknown_causes == 0),
        _acceptance_row("validation", True, validation_ok, validation_ok),
    ]
    artifact.update(
        {
            "status": "complete",
            "timestamps": {
                "started_at_utc": started_at_utc,
                "completed_at_utc": completed_at_utc,
            },
            "preconditions_checked": deepcopy(list(checks)),
            "duration_s": duration_s,
            "phase_spans": deepcopy(
                list(phase_spans)
                if phase_spans is not None
                else [{"phase": "semantic_replay", "duration_s": duration_s}]
            ),
            "source_artifact_hashes": deepcopy(replay["source_artifact_hashes"]),
            "rows": deepcopy(replay["semantic_rows"]),
            "sample_size_budget": {
                "planned_raw_calls": 320,
                "attempted_raw_calls": 320,
                "completed_raw_calls": len(replay["call_replay_rows"]),
                "censored_raw_calls": 0,
                "planned_semantic_rows": 192,
                "completed_semantic_rows": len(replay["semantic_rows"]),
                "stopping_rule": "replay all frozen calls once; no accuracy stop",
            },
            "acceptance_gate_results": gates,
            "gate_check_summary": _gate_summary(checks),
            "honest_verdict": "complete_null_transport_replayed_contract_defects_preserve_exp7265_null",
            "verdict_class": "null",
            "validation_receipts": deepcopy(list(validation_receipts)),
            "semantic_replay_ready_score": readiness,
            "first_divergence_rows": deepcopy(replay["first_divergence_rows"]),
            "comparator_contract_path": deepcopy(sidecars["comparator_contract"]),
            "historical_arm_reduction": deepcopy(replay["historical_arm_reduction"]),
            "call_replay_rows": deepcopy(replay["call_replay_rows"]),
            "cause_classification_counts": deepcopy(replay["cause_classification_counts"]),
            "semantic_cause_rows": deepcopy(replay["semantic_cause_rows"]),
            "sidecar_receipts": deepcopy(dict(sidecars)),
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Cold-check terminal identity, provenance, denominators, and readiness."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping"]
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in value:
            return [f"missing_required_field:{field}"]
    errors: list[str] = []
    if value.get("schema") != SCHEMA:
        errors.append("schema")
    if (value.get("experiment_id"), value.get("milestone"), value.get("run_date")) != (
        EXPERIMENT_ID,
        MILESTONE,
        RUN_DATE,
    ):
        errors.append("identity")
    if value.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if (
        value.get("MODEL_SPECS") != []
        or value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("no_llm_contract")
    if (
        value.get("inference_substrate") != "cpu_exact_solver_or_simulator"
        or value.get("inference_substrate_class") != "cpu_exact_solver_or_simulator"
        or value.get("execution_venue") != "host"
    ):
        errors.append("execution_contract")
    if value.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    duration = value.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration < 0:
        errors.append("duration_s")
    status = value.get("status")
    if status == "blocked":
        if (
            value.get("verdict_class") != "blocked"
            or not str(value.get("honest_verdict", "")).startswith("blocked_")
            or not value.get("gate_check_summary", {}).get("failed_check")
            or value.get("semantic_replay_ready_score") != 0
        ):
            errors.append("blocked_terminal_state")
    elif status == "complete":
        budget = value.get("sample_size_budget") or {}
        if (
            not isinstance(value.get("rows"), list)
            or len(value["rows"]) != 192
            or not isinstance(value.get("call_replay_rows"), list)
            or len(value["call_replay_rows"]) != 320
            or not isinstance(value.get("first_divergence_rows"), list)
            or len(value["first_divergence_rows"]) != 320
            or budget.get("completed_raw_calls") != 320
            or budget.get("completed_semantic_rows") != 192
        ):
            errors.append("denominators")
        if value.get("semantic_replay_ready_score") != 1:
            errors.append("semantic_replay_ready_score")
        if value.get("verdict_class") != "null" or not str(
            value.get("honest_verdict", "")
        ).startswith("complete_null_"):
            errors.append("verdict")
        if not _validations_complete(value.get("validation_receipts") or []):
            errors.append("validation_receipts")
        gates = value.get("acceptance_gate_results")
        if (
            not isinstance(gates, list)
            or len(gates) != len(GATE_PRINCIPLES)
            or not all(
                isinstance(row, Mapping)
                and {"expected", "observed", "passed", "principle"} <= set(row)
                and row.get("passed") is True
                for row in gates
            )
        ):
            errors.append("acceptance_gate_results")
        contract = value.get("comparator_contract_path") or {}
        if not str(contract.get("sha256", "")).startswith("sha256:"):
            errors.append("comparator_contract_path")
    else:
        errors.append("status")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum")
    return errors


def _progress(phase: int, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every phase boundary so external orchestration sees truthful work."""

    payload = {"phase": phase, "event": event, **details}
    print(f"[exp7275] {canonical_json(payload)}", flush=True)


def _validation_commands(
    root: Path, raw_dir: Path
) -> list[tuple[str, list[str]]]:  # pragma: no cover
    """Return bounded focused checks; no repository-wide pytest command is allowed."""

    python = str(root / ".venv/bin/python")
    test = TEST_PATH.as_posix()
    affected = [
        "tests/python/test_experiment_7238_v637_mention_capture.py",
        "tests/python/test_experiment_7264_v639_mention_canary.py",
        "tests/python/test_experiment_7265_v639_mention_heldout.py",
    ]
    changed = [MODULE_PATH.as_posix(), WRAPPER_PATH.as_posix(), test]
    coverage_file = "/tmp/.coverage-exp7275-v640"
    candidate = (root / RAW_CANDIDATE_PATH).as_posix()
    return [
        (
            "focused_pytest",
            [
                python,
                "-u",
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "-n",
                "0",
                "--basetemp=/tmp/exp7275-focused",
                test,
                "-q",
            ],
        ),
        (
            "affected_suites",
            [
                python,
                "-u",
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "-n",
                "0",
                "--basetemp=/tmp/exp7275-affected",
                *affected,
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
                "--basetemp=/tmp/exp7275-coverage",
                test,
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
        ("mypy", [python, "-u", "-m", "mypy", MODULE_PATH.as_posix()]),
        (
            "scoped_spec_coverage",
            [python, "-u", "scripts/check_spec_coverage.py", test, *affected],
        ),
        (
            "independent_raw_replay",
            [
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--replay-raw",
                raw_dir.as_posix(),
            ],
        ),
        ("adversarial_verify", [python, "-u", "scripts/adversarial_verify.py", candidate]),
        (
            "verdict_row_consistency",
            [python, "-u", "scripts/verdict_row_consistency_lint.py", candidate],
        ),
    ]


def _run_validations(root: Path, raw_dir: Path) -> list[JsonDict]:  # pragma: no cover
    """Stream every child and retain its exit code, duration, and exact log hash."""

    validation_dir = raw_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = f"{root / 'python'}:{root}"
    receipts: list[JsonDict] = []
    commands = _validation_commands(root, raw_dir)
    for index, (name, command) in enumerate(commands, start=1):
        _progress(
            6,
            "subprocess_start",
            operation=name,
            completed_units=index - 1,
            total_units=len(commands),
        )
        started = time.monotonic()
        process = subprocess.Popen(  # noqa: S603 - argv contains fixed local commands.
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines: list[str] = []
        assert process.stdout is not None
        with prior.live_runtime._heartbeat(6, name, lambda: index - 1, len(commands)):
            for line in process.stdout:
                lines.append(line)
                print(f"[exp7275:{name}] {line.rstrip()}", flush=True)
            returncode = process.wait()
        log_path = validation_dir / f"{name}.log"
        log_path.write_text("".join(lines), encoding="utf-8")
        receipts.append(
            {
                "name": name,
                "command": shlex.join(command),
                "exit_code": returncode,
                "passed": returncode == 0,
                "timed_out": False,
                "duration_s": time.monotonic() - started,
                "log_path": log_path.relative_to(root).as_posix(),
                "log_sha256": sha256_file(log_path),
            }
        )
        _progress(
            6,
            "subprocess_end",
            operation=name,
            exit_code=returncode,
            completed_units=index,
            total_units=len(commands),
        )
    return receipts


def _pending_receipts() -> list[JsonDict]:  # pragma: no cover
    """Make pending validation visible in the raw candidate instead of fabricating it."""

    return [
        {
            "name": name,
            "command": "pending",
            "exit_code": None,
            "passed": False,
            "timed_out": False,
            "duration_s": 0.0,
            "log_path": f"results/raw/experiment_7275/validation/{name}.log",
            "log_sha256": "pending",
        }
        for name in REQUIRED_VALIDATION_NAMES
    ]


def _checkpoint(path: Path, artifact: Mapping[str, Any]) -> None:  # pragma: no cover
    """Write unfinished or failed current work only to the checkpoint path."""

    value = deepcopy(dict(artifact))
    if value.get("status") not in {"complete", "blocked"}:
        value["status"] = "partial"
        value["verdict_class"] = "partial"
        value["honest_verdict"] = "partial_semantic_replay_validation_unfinished"
    value["reproducibility_checksum"] = artifact_checksum(value)
    atomic_write_json(path, value, allow_override=True, sort_keys=True)


def run_experiment(
    root: Path | None = None,
    run_date: str = RUN_DATE,
    *,
    validation_runner: Callable[[Path, Path], list[JsonDict]] = _run_validations,
) -> JsonDict:  # pragma: no cover - exercised by the required native entrypoint.
    """Authenticate, replay, validate, then atomically publish one terminal result."""

    root = root or find_repo_root(start=__file__)
    started = time.monotonic()
    started_utc = utc_now()
    artifact = base_artifact(run_date)
    artifact["timestamps"]["started_at_utc"] = started_utc
    spans: list[JsonDict] = []
    _progress(1, "phase_start", operation="preconditions")
    phase = time.monotonic()
    checks = authenticate_inputs(root)
    spans.append({"phase": "preconditions", "duration_s": time.monotonic() - phase})
    _progress(
        1,
        "phase_end",
        operation="preconditions",
        failed=sum(row["passed"] is not True for row in checks),
    )
    if any(row["passed"] is not True for row in checks):
        blocked = finalize_blocked_artifact(artifact, checks, time.monotonic() - started)
        _progress(7, "write_start", path=str(root / RESULT_PATH))
        atomic_write_json(root / RESULT_PATH, blocked, allow_override=False, sort_keys=True)
        _progress(7, "write_end", path=str(root / RESULT_PATH))
        return blocked
    _progress(2, "benchmark_start", operation="independent_raw_replay", total_units=320)
    phase = time.monotonic()
    replay = load_and_replay(root)
    spans.append({"phase": "independent_raw_replay", "duration_s": time.monotonic() - phase})
    _progress(
        2,
        "benchmark_end",
        operation="independent_raw_replay",
        completed_units=len(replay["call_replay_rows"]),
        errors=len(replay["reconstruction_errors"]),
    )
    _progress(3, "benchmark_start", operation="fixture_matrix", total_units=10)
    phase = time.monotonic()
    fixture = run_fixture_matrix()
    spans.append({"phase": "fixture_matrix", "duration_s": time.monotonic() - phase})
    _progress(
        3,
        "benchmark_end",
        operation="fixture_matrix",
        completed_units=len(fixture["rows"]),
        corrupted_rejected=fixture["corrupted_controls_rejected"],
    )
    _progress(4, "phase_start", operation="sidecar_sealing")
    phase = time.monotonic()
    raw_dir = root / RAW_DIR
    sidecars = write_sidecars(raw_dir, replay)
    spans.append({"phase": "sidecar_sealing", "duration_s": time.monotonic() - phase})
    _progress(4, "phase_end", operation="sidecar_sealing", sidecars=len(sidecars))
    candidate = finalize_measured_artifact(
        artifact,
        checks,
        replay,
        sidecars,
        _pending_receipts(),
        duration_s=time.monotonic() - started,
        started_at_utc=started_utc,
        completed_at_utc=utc_now(),
        phase_spans=spans,
    )
    _progress(5, "write_start", path=str(root / RAW_CANDIDATE_PATH))
    atomic_write_json(root / RAW_CANDIDATE_PATH, candidate, allow_override=True, sort_keys=True)
    _progress(5, "write_end", path=str(root / RAW_CANDIDATE_PATH))
    _checkpoint(root / CHECKPOINT_PATH, candidate)
    _progress(6, "phase_start", operation="focused_validation")
    phase = time.monotonic()
    receipts = validation_runner(root, raw_dir)
    spans.append({"phase": "focused_validation", "duration_s": time.monotonic() - phase})
    _progress(
        6,
        "phase_end",
        operation="focused_validation",
        passed=sum(row["passed"] is True for row in receipts),
        total=len(receipts),
    )
    terminal = finalize_measured_artifact(
        artifact,
        checks,
        replay,
        sidecars,
        receipts,
        duration_s=time.monotonic() - started,
        started_at_utc=started_utc,
        completed_at_utc=utc_now(),
        phase_spans=spans,
    )
    if not _validations_complete(receipts):
        terminal["status"] = "partial"
        terminal["verdict_class"] = "partial"
        terminal["honest_verdict"] = "partial_semantic_replay_validation_failed"
        terminal["reproducibility_checksum"] = artifact_checksum(terminal)
        _checkpoint(root / CHECKPOINT_PATH, terminal)
        raise RuntimeError("focused validation failed; terminal artifact not published")
    errors = validate_artifact(terminal)
    if errors:
        _checkpoint(root / CHECKPOINT_PATH, terminal)
        raise ValueError(f"invalid Exp7275 artifact: {errors}")
    atomic_write_json(root / RAW_CANDIDATE_PATH, terminal, allow_override=True, sort_keys=True)
    _progress(7, "write_start", path=str(root / RESULT_PATH))
    atomic_write_json(root / RESULT_PATH, terminal, allow_override=False, sort_keys=True)
    _progress(7, "write_end", path=str(root / RESULT_PATH))
    return terminal


def _date_argument(value: str) -> str:
    """Accept only the execution date fixed by the V640 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run the replay or cold-check task-owned raw evidence without inference."""

    print("[exp7275] startup", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--replay-raw", type=Path)
    args = parser.parse_args(argv)
    root = find_repo_root(start=__file__)
    if args.replay_raw is not None:
        _progress(1, "benchmark_start", operation="independent_raw_replay", total_units=320)
        checks = authenticate_inputs(root)
        replay = load_and_replay(root) if all(row["passed"] for row in checks) else None
        errors = [] if replay is not None else ["precondition_failure"]
        if replay is not None:
            errors.extend(replay["reconstruction_errors"])
        print(canonical_json({"replay_errors": errors}), flush=True)
        _progress(1, "benchmark_end", operation="independent_raw_replay", errors=len(errors))
        return int(bool(errors))
    artifact = run_experiment(root, args.date)
    print(
        f"[exp7275] terminal verdict={artifact['honest_verdict']} readiness={artifact['semantic_replay_ready_score']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
