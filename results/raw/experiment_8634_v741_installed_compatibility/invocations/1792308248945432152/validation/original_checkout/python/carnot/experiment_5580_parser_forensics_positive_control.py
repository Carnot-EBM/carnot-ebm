"""Exp5580 cached parser forensics and positive controls.

Spec refs: REQ-VERIFY-5580, SCENARIO-VERIFY-5580.

This experiment repairs the instrument, not the model outputs.  The parser
cascade below broadens only syntax handling: strict schema first, then fenced
JSON, then one balanced object from wrapper text, then documented aliases and
one-element list shapes.  It never changes a candidate program or upgrades an
unknown verifier word into a correctness label.  Direct solves still go through
the Exp5566 exact validator before they count as accepted.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from carnot import experiment_5566_exact_asp_fsm_near_miss_corpus as corpus5566
from carnot import experiment_5567_local_sota_solve_verify_asymmetry as exp5567


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5580_parser_forensics_positive_control.json")
EXP5567_RELATIVE_PATH = exp5567.RESULT_RELATIVE_PATH
EXP5566_RELATIVE_PATH = corpus5566.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5580.parser_forensics_positive_control.v505"
EXPERIMENT = 5580
EXPERIMENT_ID = "exp5580-parser-forensics-positive-control"
MILESTONE = "2026.07.505"
RUN_DATE = "2026-07-14"
RANDOM_SEED = 5580
INFERENCE_SUBSTRATE = "cached_exp5567_responses_no_llm"
SPEC_REFS = ("REQ-VERIFY-5580", "SCENARIO-VERIFY-5580", "REQ-VERIFY-5567")
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
MAX_PARSE_CHARS = 65536
MAX_OBJECT_CANDIDATES = 8
CACHED_SAMPLES_REQUIRED_PER_MODEL_FAMILY = 30

FAILURE_TAXONOMY_KEYS = (
    "wrapper_text",
    "fenced_json",
    "field_alias",
    "numeric_list_shape",
    "truncation",
    "semantic_invalidity",
    "other",
)
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "cached_rows_audited",
    "failure_taxonomy",
    "parser_stage_counts",
    "parsed_positive_control_rate",
    "semantic_false_accept_count",
    "per_model_cached_parse_rate",
    "tests_run",
    "parser_repair_ready",
    "inference_substrate",
    "honest_verdict",
)
FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Keeps every required field annotated by its evidence boundary.",
    "cached_rows_audited": "Diagnosis covers the failed denominator.",
    "failure_taxonomy": "Fixes target observed causes.",
    "parser_stage_counts": "Permissive fallbacks stay visible.",
    "parsed_positive_control_rate": "A parser needs known-valid recall.",
    "semantic_false_accept_count": "Syntax repair cannot invent correctness.",
    "per_model_cached_parse_rate": "One family cannot mask another.",
    "tests_run": "Behavior is reproducible.",
    "parser_repair_ready": "Downstream inference runs only after controlled readiness.",
    "inference_substrate": "No new model evidence is claimed.",
    "honest_verdict": "Terminal status names readiness or block.",
}

_SOLVE_KIND_ALIASES = ("candidate_kind", "kind", "candidate_type", "type")
_SOLVE_CANDIDATE_ALIASES = ("candidate", "program", "machine", "payload")
_VERDICT_ALIASES = ("verdict", "label", "decision", "answer")
_SCORE_ALIASES = ("score", "validity_score", "confidence")
_WRAPPER_ALIASES = ("answer", "response", "result", "output")


def canonical_json(value: Any) -> str:
    """Serialize JSON in the stable form used for checksums and comparisons."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking the self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return hashlib.sha256(canonical_json(stable).encode("utf-8")).hexdigest()


def sample_pairs_for_forensics(
    rows: Sequence[Mapping[str, Any]],
    *,
    n: int = exp5567.MIN_INDEPENDENT_INSTANCES,
) -> list[JsonDict]:
    """Return the same valid/near-miss independent pairs used by Exp5567."""

    return exp5567.sample_independent_pairs(rows, n=n)


def parse_solve_candidate(text: str, pair: Mapping[str, Any]) -> JsonDict:
    """Parse one solve candidate and score it only with the Exp5566 oracle."""

    parsed = _parse_payload(text, schema="solve", arm="")
    response_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if parsed["payload"] is None:
        return {
            "parser_ok": False,
            "exact_accepted": False,
            "parser_stage": parsed["stage"],
            "response_hash": response_hash,
            "error_type": parsed["error"],
        }

    payload = parsed["payload"]
    assert isinstance(payload, Mapping)
    valid_row = dict(pair["valid_row"])
    row = {
        "candidate_kind": str(payload["candidate_kind"]),
        "candidate": dict(payload["candidate"]),
        "expected_signature_sha256": valid_row["expected_signature_sha256"],
    }
    try:
        validation = corpus5566.exact_validate_corpus_row(row)
    except Exception as exc:  # noqa: BLE001
        return {
            "parser_ok": True,
            "exact_accepted": False,
            "parser_stage": parsed["stage"],
            "response_hash": response_hash,
            "candidate_kind": row["candidate_kind"],
            "error_type": f"solve_exact_validation_error:{type(exc).__name__}",
        }
    accepted = bool(validation["accepted"])
    return {
        "parser_ok": True,
        "exact_accepted": accepted,
        "parser_stage": parsed["stage"],
        "response_hash": response_hash,
        "candidate_kind": row["candidate_kind"],
        "actual_signature_sha256": validation["actual_signature_sha256"],
        "error_type": "" if accepted else "solve_exact_rejected",
    }


def parse_verifier_label(text: str, arm: str) -> JsonDict:
    """Parse one verifier response into a valid/invalid label or fail closed."""

    parsed = _parse_payload(text, schema="verifier", arm=arm)
    if parsed["payload"] is None:
        return {
            "parser_ok": False,
            "label": None,
            "parser_stage": parsed["stage"],
            "error_type": parsed["error"],
        }
    label = _label_from_payload(parsed["payload"], arm)
    if label is None:
        return {
            "parser_ok": False,
            "label": None,
            "parser_stage": parsed["stage"],
            "error_type": "semantic_invalidity",
        }
    return {
        "parser_ok": True,
        "label": label,
        "parser_stage": parsed["stage"],
        "error_type": "",
    }


def run_positive_controls(repo_root: Path = REPO_ROOT) -> JsonDict:
    """Run deterministic parser positives and negatives without model calls."""

    rows = _load_json(repo_root / EXP5566_RELATIVE_PATH).get("corpus_rows", [])
    pairs = sample_pairs_for_forensics(rows, n=4)
    if not pairs:
        raise ValueError("positive_control_pairs")
    pair = pairs[0]
    valid_row = dict(pair["valid_row"])
    invalid_row = dict(pair["invalid_row"])
    valid_solve = {
        "candidate_kind": valid_row["candidate_kind"],
        "candidate": valid_row["candidate"],
    }
    alias_solve = {"kind": valid_row["candidate_kind"], "program": valid_row["candidate"]}
    positives = [
        ("solve_strict", parse_solve_candidate(canonical_json(valid_solve), pair), "valid"),
        (
            "solve_fenced_alias",
            parse_solve_candidate(f"```json\n{canonical_json(alias_solve)}\n```", pair),
            "valid",
        ),
        ("verify_strict", parse_verifier_label('{"verdict":"valid"}', "discrete_verdict"), "valid"),
        (
            "verify_fenced",
            parse_verifier_label('```json\n{"verdict":"invalid"}\n```', "discrete_verdict"),
            "invalid",
        ),
        (
            "verify_wrapper_alias",
            parse_verifier_label('answer follows {"decision":"accepted"}', "discrete_verdict"),
            "valid",
        ),
        (
            "verify_list_shape",
            parse_verifier_label('[{"verdict":"rejected"}]', "discrete_verdict"),
            "invalid",
        ),
        (
            "verify_score_list",
            parse_verifier_label('{"score":[81]}', "granular_score"),
            "valid",
        ),
        (
            "verify_criteria",
            parse_verifier_label(
                '{"criteria":{"schema":true,"constraints":true}}', "criteria_decomposition"
            ),
            "valid",
        ),
    ]
    positive_passes = sum(1 for _, result, expected in positives if _positive_ok(result, expected))
    stage_counts = Counter(str(result.get("parser_stage", "unknown")) for _, result, _ in positives)

    negatives = [
        parse_verifier_label("not-json", "discrete_verdict"),
        parse_verifier_label('{"verdict":"valid"', "discrete_verdict"),
        parse_verifier_label('{"verdict":"valid"} {"verdict":"invalid"}', "discrete_verdict"),
        parse_verifier_label('{"verdict":"maybe"}', "discrete_verdict"),
        parse_verifier_label('{"score":101}', "granular_score"),
        parse_solve_candidate(
            canonical_json(
                {
                    "candidate_kind": invalid_row["candidate_kind"],
                    "candidate": invalid_row["candidate"],
                }
            ),
            pair,
        ),
    ]
    false_accepts = sum(1 for result in negatives if _negative_false_accept(result))
    return {
        "positive_total": len(positives),
        "positive_passes": positive_passes,
        "parsed_positive_control_rate": round(positive_passes / len(positives), 6),
        "semantic_false_accept_count": false_accepts,
        "parser_stage_counts": dict(sorted(stage_counts.items())),
        "negative_total": len(negatives),
    }


def diagnose_exp5567_failures(exp5567_artifact: Mapping[str, Any]) -> JsonDict:
    """Diagnose the preserved Exp5567 denominator without inventing raw text."""

    per_model = _per_model_cached_parse_rate(exp5567_artifact)
    cached_rows_audited = int(exp5567_artifact.get("parser_failure_count", 0) or 0)
    truncation = _empty_response_candidate_lower_bound(exp5567_artifact)
    other = max(0, cached_rows_audited - truncation)
    taxonomy = {key: 0 for key in FAILURE_TAXONOMY_KEYS}
    taxonomy["truncation"] = truncation
    taxonomy["other"] = other
    return {
        "cached_rows_audited": cached_rows_audited,
        "failure_taxonomy": taxonomy,
        "per_model_cached_parse_rate": per_model,
        "raw_response_text_available": _has_raw_response_text(exp5567_artifact),
        "failure_taxonomy_notes": {
            "truncation": (
                "Lower bound from response hashes equal to the SHA-256 digest of empty text; "
                "Qwen is entirely empty and Gemma batches 02-03 are empty."
            ),
            "other": (
                "Non-empty Exp5567 response bodies are not preserved in the checked-in artifact, "
                "so wrapper/fence/alias/list causes cannot be claimed from hashes alone."
            ),
        },
    }


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5580 readiness or blocked-forensics artifact."""

    exp5567_artifact = _load_json(repo_root / EXP5567_RELATIVE_PATH)
    controls = run_positive_controls(repo_root)
    forensics = diagnose_exp5567_failures(exp5567_artifact)
    cached_sample_audit = _cached_sample_audit(forensics["per_model_cached_parse_rate"])
    ready = bool(
        controls["parsed_positive_control_rate"] >= 0.95
        and controls["semantic_false_accept_count"] == 0
        and forensics["raw_response_text_available"] is True
        and cached_sample_audit["raw_samples_available"] is True
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
        "field_principles": dict(FIELD_PRINCIPLES),
        "cached_rows_audited": forensics["cached_rows_audited"],
        "failure_taxonomy": forensics["failure_taxonomy"],
        "parser_stage_counts": controls["parser_stage_counts"],
        "parsed_positive_control_rate": controls["parsed_positive_control_rate"],
        "semantic_false_accept_count": controls["semantic_false_accept_count"],
        "per_model_cached_parse_rate": forensics["per_model_cached_parse_rate"],
        "tests_run": [dict(row) for row in tests_run],
        "parser_repair_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready, forensics["raw_response_text_available"]),
        "cached_sample_audit": cached_sample_audit,
        "raw_response_text_available": forensics["raw_response_text_available"],
        "failure_taxonomy_notes": forensics["failure_taxonomy_notes"],
        "positive_control_summary": {
            "positive_total": controls["positive_total"],
            "positive_passes": controls["positive_passes"],
            "negative_total": controls["negative_total"],
        },
        "llm_invoked": False,
        "research_conductor_modified": False,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the artifact and fail closed on readiness overclaims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(
        set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"]), "field_principles"
    )
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("llm_invoked") is False, "llm_invoked")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    expected_ready = bool(
        float(artifact.get("parsed_positive_control_rate", 0.0)) >= 0.95
        and int(artifact.get("semantic_false_accept_count", 1)) == 0
        and artifact.get("raw_response_text_available") is True
        and artifact.get("cached_sample_audit", {}).get("raw_samples_available") is True
    )
    _require(artifact.get("parser_repair_ready") is expected_ready, "parser_repair_ready")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")
    taxonomy = artifact.get("failure_taxonomy")
    _require(isinstance(taxonomy, Mapping), "failure_taxonomy")
    _require(set(taxonomy) == set(FAILURE_TAXONOMY_KEYS), "failure_taxonomy")
    _require(
        sum(int(value) for value in taxonomy.values()) == artifact["cached_rows_audited"],
        "cached_rows_audited",
    )
    verdict = str(artifact.get("honest_verdict", ""))
    if expected_ready:
        _require(verdict.startswith("complete:"), "honest_verdict")
    else:
        _require(verdict.startswith("blocked_"), "honest_verdict")


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    repo_root: Path = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5580 artifact without invoking an LLM."""

    artifact = build_artifact(repo_root=repo_root, tests_run=tests_run)
    output = Path(result_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _parse_payload(text: str, *, schema: str, arm: str) -> JsonDict:
    if len(text) > MAX_PARSE_CHARS:
        return {"payload": None, "stage": "too_long", "error": "input_too_long"}
    candidates = _json_candidates(text)
    if not candidates:
        return {
            "payload": None,
            "stage": "none",
            "error": "truncation" if _looks_truncated(text) else "json_parse_failure",
        }
    normalized: list[tuple[JsonDict, str]] = []
    for value, source_stage in candidates:
        payload, transform_stage = _normalize_payload(value, schema=schema, arm=arm)
        if payload is None:
            continue
        stage = transform_stage or source_stage
        normalized.append((payload, stage))
    if not normalized:
        return {"payload": None, "stage": "schema", "error": "schema_mismatch"}
    unique = {canonical_json(payload): (payload, stage) for payload, stage in normalized}
    if len(unique) > 1:
        return {"payload": None, "stage": "wrapper_text", "error": "ambiguous_json_objects"}
    payload, stage = next(iter(unique.values()))
    return {"payload": payload, "stage": stage, "error": ""}


def _json_candidates(text: str) -> list[tuple[Any, str]]:
    cleaned = text.strip()
    out: list[tuple[Any, str]] = []
    try:
        out.append((json.loads(cleaned), "strict_schema"))
        return out
    except json.JSONDecodeError:
        pass
    for match in re.finditer(
        r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL
    ):
        _append_json_candidate(out, match.group(1).strip(), "fenced_json")
        if len(out) >= MAX_OBJECT_CANDIDATES:
            return out
    for snippet in _balanced_object_snippets(cleaned):
        _append_json_candidate(out, snippet, "wrapper_text")
        if len(out) >= MAX_OBJECT_CANDIDATES:
            break
    return out


def _append_json_candidate(out: list[tuple[Any, str]], snippet: str, stage: str) -> None:
    try:
        out.append((json.loads(snippet), stage))
    except json.JSONDecodeError:
        return


def _balanced_object_snippets(text: str) -> list[str]:
    snippets: list[str] = []
    starts = [index for index, char in enumerate(text) if char == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for offset, char in enumerate(text[start:], start=start):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    snippets.append(text[start : offset + 1])
                    break
        if len(snippets) >= MAX_OBJECT_CANDIDATES:
            break
    return snippets


def _normalize_payload(value: Any, *, schema: str, arm: str) -> tuple[JsonDict | None, str]:
    value, list_stage = _unwrap_one_item_list(value)
    value, wrapper_stage = _unwrap_mapping_wrapper(value)
    if schema == "solve":
        payload, alias_stage = _normalize_solve_payload(value)
    else:
        payload, alias_stage = _normalize_verifier_payload(value, arm)
    stage = list_stage or wrapper_stage or alias_stage
    return payload, stage


def _unwrap_one_item_list(value: Any) -> tuple[Any, str]:
    if isinstance(value, list) and len(value) == 1:
        return value[0], "numeric_list_shape"
    return value, ""


def _unwrap_mapping_wrapper(value: Any) -> tuple[Any, str]:
    if not isinstance(value, Mapping):
        return value, ""
    for key in _WRAPPER_ALIASES:
        nested = value.get(key)
        if isinstance(nested, Mapping):
            return nested, "field_alias"
    return value, ""


def _normalize_solve_payload(value: Any) -> tuple[JsonDict | None, str]:
    if not isinstance(value, Mapping):
        return None, ""
    kind, kind_stage = _first_alias_value(value, _SOLVE_KIND_ALIASES)
    candidate, candidate_stage = _first_alias_value(value, _SOLVE_CANDIDATE_ALIASES)
    candidate, list_stage = _unwrap_one_item_list(candidate)
    if kind is None or not isinstance(candidate, Mapping):
        return None, ""
    stage = (
        "numeric_list_shape"
        if list_stage
        else ("field_alias" if kind_stage or candidate_stage else "")
    )
    return {"candidate_kind": str(kind), "candidate": dict(candidate)}, stage


def _normalize_verifier_payload(value: Any, arm: str) -> tuple[JsonDict | None, str]:
    if not isinstance(value, Mapping):
        return None, ""
    if arm == "granular_score":
        score, score_stage = _first_alias_value(value, _SCORE_ALIASES)
        score, list_stage = _unwrap_one_item_list(score)
        if score is None:
            return None, ""
        stage = "numeric_list_shape" if list_stage else ("field_alias" if score_stage else "")
        return {"score": score}, stage
    if arm == "criteria_decomposition" and isinstance(value.get("criteria"), Mapping):
        return {"criteria": dict(value["criteria"])}, ""
    verdict, verdict_stage = _first_alias_value(value, _VERDICT_ALIASES)
    verdict, list_stage = _unwrap_one_item_list(verdict)
    if verdict is None:
        return None, ""
    stage = "numeric_list_shape" if list_stage else ("field_alias" if verdict_stage else "")
    return {"verdict": verdict}, stage


def _first_alias_value(value: Mapping[str, Any], aliases: Sequence[str]) -> tuple[Any, str]:
    for index, key in enumerate(aliases):
        if key in value:
            return value[key], "" if index == 0 else "field_alias"
    return None, ""


def _label_from_payload(payload: Mapping[str, Any], arm: str) -> str | None:
    if arm == "granular_score":
        try:
            score = float(payload.get("score"))
        except (TypeError, ValueError):
            return None
        if not 0.0 <= score <= 100.0:
            return None
        return "valid" if score >= 50.0 else "invalid"
    criteria = payload.get("criteria")
    if arm == "criteria_decomposition" and isinstance(criteria, Mapping) and criteria:
        return "valid" if all(bool(item) for item in criteria.values()) else "invalid"
    return _normalize_label(payload.get("verdict"))


def _normalize_label(value: Any) -> str | None:
    text = str(value).strip().lower()
    if text in {"valid", "accept", "accepted", "true", "yes", "pass", "passed"}:
        return "valid"
    if text in {"invalid", "reject", "rejected", "false", "no", "fail", "failed", "near_miss"}:
        return "invalid"
    return None


def _positive_ok(result: Mapping[str, Any], expected: str) -> bool:
    if "exact_accepted" in result:
        return result.get("parser_ok") is True and result.get("exact_accepted") is True
    return result.get("parser_ok") is True and result.get("label") == expected


def _negative_false_accept(result: Mapping[str, Any]) -> bool:
    if "exact_accepted" in result:
        return result.get("exact_accepted") is True
    return result.get("parser_ok") is True and result.get("label") == "valid"


def _per_model_cached_parse_rate(exp5567_artifact: Mapping[str, Any]) -> dict[str, JsonDict]:
    out: dict[str, JsonDict] = {}
    solve = exp5567_artifact.get("solve_accuracy_by_model", {})
    verify = exp5567_artifact.get("verifier_metrics_by_model_and_arm", {})
    model_ids = sorted(set(solve) | set(verify))
    for model_id in model_ids:
        solve_row = solve.get(model_id, {}) if isinstance(solve, Mapping) else {}
        verify_rows = verify.get(model_id, {}) if isinstance(verify, Mapping) else {}
        solve_n = int(solve_row.get("n", 0) or 0)
        solve_failures = int(solve_row.get("parser_failures", 0) or 0)
        verifier_n = sum(int(row.get("n_candidates", 0) or 0) for row in verify_rows.values())
        verifier_failures = sum(
            int(row.get("parser_failures", 0) or 0) for row in verify_rows.values()
        )
        denominator = solve_n + verifier_n
        failures = solve_failures + verifier_failures
        out[str(model_id)] = {
            "candidate_denominator": denominator,
            "candidate_failures": failures,
            "original_parse_rate": round((denominator - failures) / denominator, 6)
            if denominator
            else 0.0,
            "repaired_parse_rate": None,
            "raw_response_text_available": False,
        }
    return out


def _empty_response_candidate_lower_bound(exp5567_artifact: Mapping[str, Any]) -> int:
    raw_hash = exp5567_artifact.get("raw_response_hash", {})
    sampled = exp5567_artifact.get("sampled_instance_ids", [])
    if not isinstance(raw_hash, Mapping) or not isinstance(sampled, list):
        return 0
    batch_sizes = [
        len(sampled[index : index + exp5567.LIVE_BATCH_PAIR_COUNT])
        for index in range(0, len(sampled), exp5567.LIVE_BATCH_PAIR_COUNT)
    ]
    model_ids = sorted({str(key).split(":", 1)[0] for key in raw_hash})
    total = 0
    for model_id in model_ids:
        for batch_index, batch_size in enumerate(batch_sizes):
            batch_id = f"batch{batch_index:02d}"
            if raw_hash.get(f"{model_id}:solve_batch::{batch_id}") == EMPTY_SHA256:
                total += batch_size
            for arm in exp5567.ARMS:
                repeats = 3 if arm == "repeated_verdict_3x" else 1
                keys = [
                    f"{model_id}:verify_batch::{batch_id}::{arm}::{repeat}"
                    for repeat in range(repeats)
                ]
                if keys and all(raw_hash.get(key) == EMPTY_SHA256 for key in keys):
                    total += batch_size * 2
    return total


def _cached_sample_audit(per_model: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    return {
        "required_samples_per_model_family": CACHED_SAMPLES_REQUIRED_PER_MODEL_FAMILY,
        "raw_samples_available": False,
        "hand_checked_samples_per_model_family": {model_id: 0 for model_id in sorted(per_model)},
        "block_reason": "exp5567_artifact_preserves_hashes_not_raw_response_text",
    }


def _has_raw_response_text(exp5567_artifact: Mapping[str, Any]) -> bool:
    for key in ("raw_responses", "cached_responses", "response_texts"):
        value = exp5567_artifact.get(key)
        if isinstance(value, Mapping | list) and value:
            return True
    return False


def _looks_truncated(text: str) -> bool:
    cleaned = text.strip()
    return bool(cleaned and cleaned.count("{") > cleaned.count("}"))


def _honest_verdict(ready: bool, raw_available: bool) -> str:
    if ready:
        return "complete: parser repair ready on cached responses with zero semantic false accepts"
    if not raw_available:
        return "blocked_cached_raw_responses_unavailable_hash_only_forensics"
    return "blocked_parser_repair_gate_failed"


def _load_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, field: str) -> None:
    if not condition:
        raise ValueError(field)
