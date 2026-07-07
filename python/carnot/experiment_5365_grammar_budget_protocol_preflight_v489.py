"""Exp 5365 deterministic grammar-budget preflight for structured output.

Spec refs: REQ-VERIFY-5365, SCENARIO-VERIFY-5365.

The preflight answers the operational question that Exp 5351 left open: before
spending live GGUF time again, can Carnot prove that the schema fields, final
JSON token budget, truncation classifier, and tool/action markers are reachable
under a deterministic protocol fixture? It intentionally makes no model-quality
claim and reuses the Exp 5351 parser contract for regression evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import time
from typing import Any

from carnot import experiment_5351_trigger_constrain_structured_protocol_v488 as exp5351


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5365_grammar_budget_protocol_preflight_v489"
MILESTONE = "2026.07.489"
RESULT_RELATIVE_PATH = Path("results/experiment_5365_grammar_budget_protocol_preflight_v489.json")
SCHEMA = "carnot.experiment_5365.grammar_budget_protocol_preflight.v489"
SPEC_REFS = ("REQ-VERIFY-5365", "SCENARIO-VERIFY-5365")
RANDOM_SEED = 5365
INFERENCE_SUBSTRATE = "deterministic_structured_output_preflight_no_llm"
DEFAULT_TOOL_ACTION_MARKERS = ("TOOL_ACTION:", "END_TOOL_ACTION", "FINAL_JSON:", "END_FINAL_JSON")
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")
TERMINAL_PREFIXES = ("complete:", "blocked_")

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "grammar_budget_protocol_ready",
    "schema_reachability_cases",
    "required_field_reachability_rate",
    "completion_slack_min_tokens",
    "truncation_failure_count",
    "schema_failure_count",
    "tool_action_token_reachability_rate",
    "methodology_duration_s",
    "tests_run",
    "active_roadmap_modified",
    "conductor_modified",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "complete only if the deterministic preflight runs and reports all required metrics.",
    "grammar_budget_protocol_ready": (
        "boolean gate for the live GGUF rerun; true only when schema reachability "
        "and completion slack are both measurable."
    ),
    "schema_reachability_cases": "integer count of schema cases checked.",
    "required_field_reachability_rate": (
        "fraction of required fields reachable by the compiled/derived grammar."
    ),
    "completion_slack_min_tokens": (
        "minimum estimated token slack remaining after completing a valid JSON "
        "response across fixtures."
    ),
    "truncation_failure_count": "count of cases classified as truncation risks.",
    "schema_failure_count": (
        "count of cases classified as schema risks independent of truncation."
    ),
    "tool_action_token_reachability_rate": (
        "fraction of tool/action markers reachable under the protocol fixture."
    ),
    "methodology_duration_s": (
        "wall-clock duration for the preflight, not including planning prose."
    ),
    "tests_run": "list of commands run for changed code or an explicit no-code-change explanation.",
    "active_roadmap_modified": "must be false.",
    "conductor_modified": "must be false.",
    "honest_verdict": "one-line summary of readiness or blocking reason.",
}


def estimate_tokens(text: str) -> int:
    """Estimate JSON completion size with a punctuation-aware deterministic proxy."""

    return len(TOKEN_RE.findall(str(text or "")))


def derive_schema_grammar_summary(schema: Mapping[str, Any]) -> JsonDict:
    """Summarize the JSON schema as a deterministic grammar surface.

    This is not a native llguidance or XGrammar compile. It is the smaller
    preflight needed here: the required keys, their primitive emitters, and the
    literal tokens that a grammar-backed rerun must make reachable.
    """

    required = [str(field) for field in schema.get("required", ())]
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        properties = {}
    property_types = {
        field: str(properties.get(field, {}).get("type", "any"))
        for field in required
        if isinstance(properties.get(field, {}), Mapping)
    }
    grammar_rules = {
        field: f'"{field}":<{property_types.get(field, "any")}>'
        for field in required
    }
    return {
        "grammar_backend": "deterministic_json_schema_summary",
        "schema_type": str(schema.get("type", "object")),
        "required_fields": required,
        "property_types": property_types,
        "grammar_rules": grammar_rules,
        "literal_tokens": ["{", "}", ":", ",", *[f'"{field}"' for field in required]],
        "grammar_state_count": len(grammar_rules) + 4,
    }


def build_schema_reachability_cases(
    prompts: Sequence[Mapping[str, Any]],
    variant: Mapping[str, Any],
    grammar_summary: Mapping[str, Any],
) -> list[JsonDict]:
    """Build one minimal valid completion case per calibration prompt."""

    cases: list[JsonDict] = []
    grammar_fields = set(grammar_summary.get("required_fields") or ())
    grammar_rules = grammar_summary.get("grammar_rules") or {}
    for prompt in prompts:
        schema = prompt.get("schema")
        active_schema = schema if isinstance(schema, Mapping) else exp5351.DEFAULT_SCHEMA
        required = [str(field) for field in active_schema.get("required", ())]
        payload = minimal_valid_payload(prompt, active_schema)
        json_text = _canonical_json(payload)
        schema_error_rows = exp5351.schema_errors(payload, active_schema)
        reachable = [
            field
            for field in required
            if field in grammar_fields and field in grammar_rules and field in payload
        ]
        completion_tokens = estimate_tokens(json_text)
        max_tokens = int(variant.get("n_predict") or 0)
        cases.append(
            {
                "prompt_id": str(prompt.get("prompt_id")),
                "required_fields": required,
                "reachable_required_fields": reachable if not schema_error_rows else [],
                "minimal_valid_json": json_text,
                "completion_tokens": completion_tokens,
                "max_tokens": max_tokens,
                "completion_slack_tokens": max_tokens - completion_tokens,
                "schema_valid": not schema_error_rows,
                "schema_errors": schema_error_rows,
            }
        )
    return cases


def minimal_valid_payload(
    prompt: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> JsonDict:
    """Create a smallest schema-valid payload from the prompt's target object."""

    target = prompt.get("target_final_object")
    target_payload = target if isinstance(target, Mapping) else {}
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        properties = {}
    payload: JsonDict = {}
    for field in [str(item) for item in schema.get("required", ())]:
        subschema = properties.get(field, {})
        expected_type = str(subschema.get("type", "string")) if isinstance(subschema, Mapping) else "string"
        candidate = target_payload.get(field)
        payload[field] = candidate if _value_matches_type(candidate, expected_type) else _default_value(expected_type)
    return payload


def required_field_reachability_rate(cases: Sequence[Mapping[str, Any]]) -> float:
    """Return the fraction of required fields reachable across all cases."""

    denominator = sum(len(case.get("required_fields") or ()) for case in cases)
    if denominator == 0:
        return 0.0
    numerator = sum(len(case.get("reachable_required_fields") or ()) for case in cases)
    return numerator / denominator


def completion_slack_min_tokens(cases: Sequence[Mapping[str, Any]]) -> int:
    """Return the minimum remaining token budget after valid JSON completion."""

    return min((int(case.get("completion_slack_tokens") or 0) for case in cases), default=-1)


def extract_exp5351_generation_receipts(exp5351_artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Extract cached Exp 5351 generation receipts from its wrapped artifact field."""

    variants = exp5351_artifact.get("protocol_variants", {})
    rows = variants.get("value") if isinstance(variants, Mapping) else variants
    if not isinstance(rows, list) or not rows:
        return []
    receipts = rows[0].get("generation_receipts", [])
    return list(receipts) if isinstance(receipts, list) else []


def classify_protocol_failure(
    receipt: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> str:
    """Classify one cached protocol receipt without collapsing truncation into schema."""

    score = receipt.get("score", {})
    score = score if isinstance(score, Mapping) else {}
    if score.get("accepted_for_protocol") is True:
        return "accepted"
    if _looks_like_truncation(receipt, variant):
        return "truncation"
    if score.get("parse_success") is True and score.get("schema_success") is False:
        return "schema"
    return "parse"


def classify_failure_rows(
    receipts: Sequence[Mapping[str, Any]],
    variant: Mapping[str, Any],
) -> list[JsonDict]:
    """Classify every cached Exp 5351 receipt into the preflight risk taxonomy."""

    rows: list[JsonDict] = []
    for receipt in receipts:
        score = receipt.get("score", {})
        score = score if isinstance(score, Mapping) else {}
        failure_class = classify_protocol_failure(receipt, variant)
        rows.append(
            {
                "prompt_id": str(receipt.get("prompt_id") or score.get("prompt_id")),
                "failure_class": failure_class,
                "schema_errors": list(score.get("schema_errors") or ()),
                "parse_success": bool(score.get("parse_success")),
                "schema_success": bool(score.get("schema_success")),
                "truncation_risk": failure_class == "truncation",
                "schema_risk_independent_of_truncation": failure_class == "schema",
            }
        )
    return rows


def failure_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count accepted, parse, schema, and truncation outcomes for the artifact."""

    return {
        "accepted_count": sum(1 for row in rows if row.get("failure_class") == "accepted"),
        "parse_failure_count": sum(1 for row in rows if row.get("failure_class") == "parse"),
        "schema_failure_count": sum(1 for row in rows if row.get("failure_class") == "schema"),
        "truncation_failure_count": sum(1 for row in rows if row.get("failure_class") == "truncation"),
    }


def build_tool_action_protocol_fixture(
    payload: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> str:
    """Return a deterministic tool/action envelope followed by final JSON."""

    return (
        "TOOL_ACTION: emit_final_json END_TOOL_ACTION\n"
        f"{variant['sentinel']} {_canonical_json(dict(payload))} {variant['end_sentinel']}"
    )


def tool_action_token_reachability(
    fixture_text: str,
    markers: Sequence[str] = DEFAULT_TOOL_ACTION_MARKERS,
) -> JsonDict:
    """Measure which protocol marker literals are present in the fixture."""

    rows = [
        {"marker": marker, "reachable": marker in fixture_text, "index": fixture_text.find(marker)}
        for marker in markers
    ]
    return {
        "rate": 0.0 if not rows else sum(1 for row in rows if row["reachable"]) / len(rows),
        "rows": rows,
    }


def field_provenance() -> dict[str, JsonDict]:
    """Return principle annotations for all required artifact fields."""

    return {
        field: {
            "principle": principle,
            "satisfied_by": "deterministic Exp 5365 preflight computation",
        }
        for field, principle in FIELD_PRINCIPLES.items()
    }


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    exp5351_artifact: Mapping[str, Any] | None = None,
    exp5351_path: Path | str | None = None,
    tests_run: Sequence[str] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Build the deterministic preflight artifact without writing it."""

    start = time.perf_counter() if started_s is None else float(started_s)
    root_path = Path(root)
    source_path = Path(exp5351_path) if exp5351_path is not None else root_path / exp5351.RESULT_RELATIVE_PATH
    prior = dict(exp5351_artifact or _load_json(source_path))
    variant = exp5351.DEFAULT_PROTOCOL_VARIANTS[0]
    grammar = derive_schema_grammar_summary(exp5351.DEFAULT_SCHEMA)
    reachability_cases = build_schema_reachability_cases(
        exp5351.DEFAULT_CALIBRATION_PROMPTS,
        variant,
        grammar,
    )
    reachability_rate = required_field_reachability_rate(reachability_cases)
    slack_min = completion_slack_min_tokens(reachability_cases)
    receipts = extract_exp5351_generation_receipts(prior)
    classification_rows = classify_failure_rows(receipts, variant)
    counts = failure_counts(classification_rows)
    tool_fixture = build_tool_action_protocol_fixture(
        exp5351.DEFAULT_CALIBRATION_PROMPTS[2]["target_final_object"],
        variant,
    )
    tool_reachability = tool_action_token_reachability(tool_fixture)
    ready = bool(
        reachability_cases
        and reachability_rate == 1.0
        and slack_min >= 0
        and tool_reachability["rate"] == 1.0
    )
    elapsed = (time.perf_counter() if now_s is None else float(now_s)) - start
    status = "complete" if ready else "blocked"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "status": status,
        "grammar_budget_protocol_ready": ready,
        "schema_reachability_cases": len(reachability_cases),
        "required_field_reachability_rate": reachability_rate,
        "completion_slack_min_tokens": slack_min,
        "truncation_failure_count": counts["truncation_failure_count"],
        "schema_failure_count": counts["schema_failure_count"],
        "tool_action_token_reachability_rate": tool_reachability["rate"],
        "methodology_duration_s": round(max(0.0, elapsed), 6),
        "tests_run": list(tests_run or []),
        "active_roadmap_modified": False,
        "conductor_modified": False,
        "honest_verdict": _honest_verdict(ready=ready, slack_min=slack_min),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "live_llm_inference_run": False,
        "schema_grammar_summary": grammar,
        "completion_budget_cases": reachability_cases,
        "exp5351_failure_classification_rows": classification_rows,
        "parse_failure_count": counts["parse_failure_count"],
        "accepted_fixture_count": counts["accepted_count"],
        "tool_action_protocol_fixture": tool_fixture,
        "tool_action_token_reachability_rows": tool_reachability["rows"],
        "source_artifacts": [
            {
                "path": exp5351.RESULT_RELATIVE_PATH.as_posix(),
                "used": True,
                "purpose": "cached .488 failed output shape regression",
            }
        ],
        "random_seed": RANDOM_SEED,
        "field_provenance": field_provenance(),
    }
    artifact["reproducibility_checksum"] = exp5351._sha16(
        _canonical_json(
            {
                "experiment_id": EXPERIMENT_ID,
                "reachability_cases": reachability_cases,
                "classification_rows": classification_rows,
                "ready": ready,
                "seed": RANDOM_SEED,
            }
        )
    )
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    artifact_path: Path | None = None,
    exp5351_path: Path | str | None = None,
    tests_run: Sequence[str] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the Exp 5365 preflight artifact."""

    root_path = Path(root)
    destination = artifact_path or root_path / RESULT_RELATIVE_PATH
    if not destination.is_absolute():
        destination = root_path / destination
    artifact = build_artifact(
        root=root_path,
        exp5351_path=exp5351_path,
        tests_run=tests_run,
        started_s=started_s,
        now_s=now_s,
    )
    if write:
        _write_json(destination, artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return all schema errors that would make the preflight artifact unusable."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("status") not in {"complete", "blocked"}:
        errors.append("status must be complete or blocked")
    if not isinstance(artifact.get("grammar_budget_protocol_ready"), bool):
        errors.append("grammar_budget_protocol_ready must be boolean")
    if not isinstance(artifact.get("schema_reachability_cases"), int):
        errors.append("schema_reachability_cases must be integer")
    if not _rate_is_valid(artifact.get("required_field_reachability_rate")):
        errors.append("required_field_reachability_rate must be in [0, 1]")
    if not isinstance(artifact.get("completion_slack_min_tokens"), int):
        errors.append("completion_slack_min_tokens must be integer")
    for field in ("truncation_failure_count", "schema_failure_count"):
        if not _non_negative_int(artifact.get(field)):
            errors.append(f"{field} must be non-negative integer")
    if not _rate_is_valid(artifact.get("tool_action_token_reachability_rate")):
        errors.append("tool_action_token_reachability_rate must be in [0, 1]")
    if not isinstance(artifact.get("methodology_duration_s"), int | float):
        errors.append("methodology_duration_s must be numeric")
    if not isinstance(artifact.get("tests_run"), list):
        errors.append("tests_run must be list")
    if artifact.get("active_roadmap_modified") is not False:
        errors.append("active_roadmap_modified must be false")
    if artifact.get("conductor_modified") is not False:
        errors.append("conductor_modified must be false")
    honest = artifact.get("honest_verdict")
    if not isinstance(honest, str) or not honest.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete: or blocked_")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping) or any(field not in provenance for field in REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if artifact.get("grammar_budget_protocol_ready") is True:
        if not isinstance(artifact.get("completion_slack_min_tokens"), int) or artifact["completion_slack_min_tokens"] < 0:
            errors.append("ready preflight requires non-negative completion slack")
    if artifact.get("status") == "complete" and artifact.get("grammar_budget_protocol_ready") is not True:
        errors.append("complete status requires grammar_budget_protocol_ready")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed when a .489 preflight artifact drifts from its contract."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise AssertionError("; ".join(errors))


def _value_matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, Mapping)
    return isinstance(value, str)


def _default_value(expected_type: str) -> Any:
    defaults: JsonDict = {
        "array": [],
        "integer": 0,
        "number": 0.0,
        "boolean": False,
        "object": {},
    }
    return defaults.get(expected_type, "")


def _looks_like_truncation(receipt: Mapping[str, Any], variant: Mapping[str, Any]) -> bool:
    if receipt.get("timed_out") is True:
        return True
    text = str(receipt.get("stdout_tail") or "")
    sentinel = str(variant.get("sentinel") or "FINAL_JSON:")
    end_sentinel = str(variant.get("end_sentinel") or "END_FINAL_JSON")
    sentinel_index = text.rfind(sentinel)
    if sentinel_index < 0:
        return False
    segment = text[sentinel_index + len(sentinel) :]
    if end_sentinel in segment:
        return False
    return "{" in segment


def _rate_is_valid(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and 0.0 <= float(value) <= 1.0


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _honest_verdict(*, ready: bool, slack_min: int) -> str:
    if ready:
        return "complete: grammar-budget preflight ready; .488 truncation and schema risks separated"
    return f"blocked_grammar_budget_protocol_not_ready: completion_slack_min_tokens={slack_min}"


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--exp5351", type=Path, default=REPO_ROOT / exp5351.RESULT_RELATIVE_PATH)
    parser.add_argument("--tests-run-json", default="[]")
    args = parser.parse_args(argv)
    artifact = run(
        artifact_path=args.out,
        exp5351_path=args.exp5351,
        tests_run=json.loads(args.tests_run_json),
        write=True,
    )
    print(
        f"[exp5365] status={artifact['status']} "
        f"ready={artifact['grammar_budget_protocol_ready']} out={args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
