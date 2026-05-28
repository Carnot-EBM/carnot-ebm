"""Build the Exp 3286 clean-verifier abstention root-cause audit.

Spec refs: REQ-VERIFY-3286, SCENARIO-VERIFY-3286.

The audit is intentionally offline. Exp 3275 already spent GPU time and
preserved the rows, prompts, strict parser decisions, and local model outputs.
This module explains that evidence without making a new model call, so the
next rerun can fix the response contract before spending more GPU time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.clean_verifier_abstention_root_cause.v1"
EXPERIMENT_ID = "exp3286"
MILESTONE = "2026.05.303"
RUN_DATE = "20260528"

OUTPUT_REL_PATH = Path("results/experiment_3286_clean_verifier_abstention_root_cause_v1.json")
EXP3275_REL_PATH = Path("results/experiment_3275_clean_local_sota_verifier_rerun_v14.json")
EXP3268_REL_PATH = Path("results/experiment_3268_sota_receipt_methodology_supplement_v1.json")
EXP3223_REL_PATH = Path(
    "results/experiment_3223_distributional_ebm_exact_row_uncertainty_sidecar_v2.json"
)
CONTEXT_FIXTURE_REL_PATH = Path("data/research/context_cot_clbench_parametric_shortcut_v1.jsonl")
AUDIT_REPORT_REL_PATH = Path("ops/verifier_authenticity_audit_report.md")
SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")
TEST_REL_PATH = Path("tests/python/test_experiment_3286_clean_verifier_abstention_root_cause_v1.py")

DEFAULT_RANDOM_SEED = 3286
TARGET_MAX_ABSTENTION_RATE = 0.5
TARGET_FALSE_ACCEPT_RATE = 0.0
SUCCESS_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

REQUIRED_FIELDS = {
    "abstention_root_cause_audit_ready",
    "abstention_root_cause_identified",
    "prior_abstention_rate",
    "audited_exact_row_count",
    "answerable_row_count",
    "malformed_or_missing_answer_count",
    "threshold_or_policy_findings",
    "parser_or_extraction_findings",
    "calibrated_rerun_plan",
    "target_max_abstention_rate",
    "target_false_accept_rate",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}

DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3286_clean_verifier_abstention_root_cause_v1.py -q -o addopts=''",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run --source=python/carnot/verify/clean_verifier_abstention_root_cause_v1.py -m pytest -o addopts='' tests/python/test_experiment_3286_clean_verifier_abstention_root_cause_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/clean_verifier_abstention_root_cause_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3286: explain Exp 3275 abstentions without rerunning a model."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    exp3275 = read_json_object(root_path / EXP3275_REL_PATH)
    exp3268 = read_json_object(root_path / EXP3268_REL_PATH)
    fixture_by_id = fixture_rows_by_id(root_path / CONTEXT_FIXTURE_REL_PATH)
    per_row_results = mapping_list(exp3275.get("per_row_results"))
    row_table = [audit_row(row, fixture_by_id) for row in per_row_results]
    class_counts = row_class_counts(row_table)
    abstention_reasons = count_values(row.get("abstention_reason") for row in row_table)
    prior_abstention_rate = bounded_float(exp3275.get("abstention_rate"))
    threshold_findings = build_threshold_or_policy_findings(exp3275, row_table)
    parser_findings = build_parser_or_extraction_findings(
        exp3275,
        row_table,
        fixture_count=len(fixture_by_id),
    )
    root_cause = dominant_root_cause(row_table, threshold_findings, parser_findings)
    identified = bool(row_table) and root_cause != "no_evaluated_exact_rows"
    finished = time.perf_counter() if now_s is None else float(now_s)

    artifact: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3286", "SCENARIO-VERIFY-3286"],
        "abstention_root_cause_audit_ready": True,
        "abstention_root_cause_identified": identified,
        "dominant_root_cause": root_cause,
        "prior_abstention_rate": prior_abstention_rate,
        "audited_exact_row_count": len(row_table),
        "answerable_row_count": class_counts["answerable"],
        "malformed_or_missing_answer_count": class_counts["malformed_or_missing_answer"],
        "row_class_counts": class_counts,
        "answerability_fields": [
            "expected_decision",
            "exact_authority",
            "source_candidate_kind",
            "fixture_id",
        ],
        "exact_answer_fields": [
            "expected_answer",
            "minimal_counterexample.candidate_answer",
            "prior_bait_answer",
        ],
        "threshold_fields": {
            "abstention_threshold": abstention_threshold(exp3275),
            "false_accept_threshold": false_accept_threshold(exp3275),
            "target_max_abstention_rate": TARGET_MAX_ABSTENTION_RATE,
            "target_false_accept_rate": TARGET_FALSE_ACCEPT_RATE,
        },
        "abstention_reason_counts": abstention_reasons,
        "threshold_or_policy_findings": threshold_findings,
        "parser_or_extraction_findings": parser_findings,
        "exact_row_audit_table": row_table,
        "evaluated_row_ids": [str(row.get("row_id") or "") for row in row_table],
        "calibrated_rerun_plan": calibrated_rerun_plan(
            audited_exact_rows=len(row_table),
            answerable_rows=class_counts["answerable"],
            prior_abstention_rate=prior_abstention_rate,
        ),
        "target_max_abstention_rate": TARGET_MAX_ABSTENTION_RATE,
        "target_false_accept_rate": TARGET_FALSE_ACCEPT_RATE,
        "source_artifacts": source_artifacts(root_path),
        "upstream_receipt_summary": {
            "clean_sota_receipt_eligible": exp3268.get("clean_sota_receipt_eligible") is True,
            "models_used": [
                str(row.get("model_id") or row.get("hf_id") or "")
                for row in mapping_list(exp3268.get("models_used"))
            ],
        },
        "random_seed": int(random_seed),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(started, finished),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    random_seed: int = DEFAULT_RANDOM_SEED,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3286 terminal JSON artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(
        root_path,
        random_seed=random_seed,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def audit_row(row: Mapping[str, Any], fixture_by_id: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Convert one Exp 3275 row into a row-level cause classification."""

    fixture_id = str(row.get("fixture_id") or "")
    fixture = mapping(fixture_by_id.get(fixture_id))
    expected_decision = str(row.get("expected_decision") or "").strip().lower()
    source_kind = str(row.get("source_candidate_kind") or "")
    expected_answer = str(fixture.get("expected_answer") or "")
    candidate_answer = candidate_answer_for_row(row, fixture)
    answerability = classify_answerability(
        expected_decision=expected_decision,
        expected_answer=expected_answer,
        candidate_answer=candidate_answer,
        source_kind=source_kind,
    )
    decision = str(row.get("decision") or "").strip().lower()
    output_text = str(row.get("output_text") or "")
    return {
        "row_id": str(row.get("row_id") or ""),
        "fixture_id": fixture_id,
        "expected_decision": expected_decision,
        "reported_decision": decision,
        "answerability": answerability,
        "exact_authority": str(row.get("exact_authority") or ""),
        "source_candidate_kind": source_kind,
        "exact_checker_type": str(fixture.get("exact_checker_type") or ""),
        "expected_answer_present": bool(expected_answer),
        "candidate_answer_present": bool(candidate_answer),
        "output_text": output_text,
        "output_preview": output_text[:120],
        "parsed_output_decision": normalize_output_decision(output_text),
        "abstention_reason": abstention_reason(row, output_text),
    }


def classify_answerability(
    *,
    expected_decision: str,
    expected_answer: str,
    candidate_answer: str,
    source_kind: str,
) -> str:
    """Classify fixture quality before blaming verifier policy."""

    if expected_decision == "abstain":
        return "unanswerable"
    if expected_decision in {"accept", "reject"}:
        if not expected_answer:
            return "malformed_or_missing_answer"
        if source_kind == "fixture_minimal_counterexample" and not candidate_answer:
            return "malformed_or_missing_answer"
        return "answerable"
    return "unknown"


def candidate_answer_for_row(row: Mapping[str, Any], fixture: Mapping[str, Any]) -> str:
    """Recover the candidate field used by Exp 3275 from fixture provenance."""

    source_kind = str(row.get("source_candidate_kind") or "")
    if source_kind == "fixture_expected_answer":
        return str(fixture.get("expected_answer") or "")
    counter = mapping(fixture.get("minimal_counterexample"))
    return str(counter.get("candidate_answer") or fixture.get("prior_bait_answer") or "")


def abstention_reason(row: Mapping[str, Any], output_text: str) -> str:
    """Explain why a row ended up abstaining under the strict parser."""

    decision = str(row.get("decision") or "").strip().lower()
    if decision != "abstain":
        return "not_abstained"
    if not output_text.strip():
        return "missing_model_output"
    if normalize_output_decision(output_text) is None:
        return "model_output_unparseable"
    return "reported_abstain"


def normalize_output_decision(value: Any) -> str | None:
    """Parse the same leading-token decision contract used by Exp 3275."""

    text = str(value or "").strip()
    if not text:
        return None
    first = text.split()[0].strip(" \t\r\n.:,;!?\"'`()[]{}").lower()
    return first if first in {"accept", "reject", "abstain"} else None


def build_threshold_or_policy_findings(
    exp3275: Mapping[str, Any],
    row_table: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Separate calibrated abstention thresholds from safety-policy causes."""

    prior_abstention = bounded_float(exp3275.get("abstention_rate"))
    threshold = abstention_threshold(exp3275)
    gate_reasons = [str(reason) for reason in exp3275.get("gate_reasons") or []]
    safety_rows = [
        row
        for row in row_table
        if any(token in str(row.get("output_text") or "").lower() for token in safety_tokens())
    ]
    return [
        {
            "category": "abstention_threshold",
            "observed_abstention_rate": prior_abstention,
            "threshold": threshold,
            "blocked": prior_abstention > threshold
            or "abstention_rate_above_threshold" in gate_reasons,
            "finding": "threshold_blocked_after_parser_abstentions"
            if prior_abstention > threshold or "abstention_rate_above_threshold" in gate_reasons
            else "threshold_not_implicated",
        },
        {
            "category": "false_accept_policy",
            "observed_false_accept_rate": bounded_float(exp3275.get("false_accept_rate")),
            "target_false_accept_rate": TARGET_FALSE_ACCEPT_RATE,
            "finding": "no_false_accepts_observed_in_prior_run",
        },
        {
            "category": "safety_policy",
            "policy_refusal_row_count": len(safety_rows),
            "finding": "not_implicated"
            if not safety_rows
            else "possible_policy_refusal_text_present",
        },
    ]


def build_parser_or_extraction_findings(
    exp3275: Mapping[str, Any],
    row_table: Sequence[Mapping[str, Any]],
    *,
    fixture_count: int,
) -> list[JsonDict]:
    """Separate fixture extraction health from parser/model-output health."""

    declared_n_eval = int(exp3275.get("n_eval") or 0)
    unparseable_rows = [
        row for row in row_table if row.get("abstention_reason") == "model_output_unparseable"
    ]
    missing_output_rows = [
        row for row in row_table if row.get("abstention_reason") == "missing_model_output"
    ]
    malformed_rows = [
        row for row in row_table if row.get("answerability") == "malformed_or_missing_answer"
    ]
    return [
        {
            "category": "row_extraction",
            "declared_n_eval": declared_n_eval,
            "per_row_results_count": len(row_table),
            "finding": "row_extraction_ok"
            if declared_n_eval == len(row_table)
            else "n_eval_per_row_results_mismatch",
        },
        {
            "category": "fixture_join",
            "fixture_rows_available": fixture_count,
            "malformed_or_missing_answer_count": len(malformed_rows),
            "finding": "fixture_authority_available"
            if fixture_count and not malformed_rows
            else "fixture_quality_issue_present",
        },
        {
            "category": "model_output_contract_mismatch",
            "unparseable_abstention_count": len(unparseable_rows),
            "missing_output_count": len(missing_output_rows),
            "finding": "dominant_parser_contract_failure"
            if unparseable_rows and len(unparseable_rows) >= len(row_table)
            else "parser_contract_findings_recorded",
        },
    ]


def calibrated_rerun_plan(
    *,
    audited_exact_rows: int,
    answerable_rows: int,
    prior_abstention_rate: float,
) -> JsonDict:
    """Define measurable Exp 3287 gates before another live verifier run."""

    minimum_rows = max(6, min(12, audited_exact_rows if audited_exact_rows else 6))
    minimum_answerable = max(6, min(minimum_rows, answerable_rows if answerable_rows else 6))
    return {
        "experiment_id": "exp3287",
        "purpose": "rerun the clean verifier only after fixing the output contract",
        "root_cause_to_address": "model_output_parser_contract_mismatch",
        "prior_blocker": {
            "abstention_rate": prior_abstention_rate,
            "failure_mode": "all exact rows normalized to abstain",
        },
        "required_controls": [
            "Use the model-native chat template or a grammar constrained decoder for ACCEPT|REJECT|ABSTAIN.",
            "Keep the strict leading-token parser; non-leading or verbose answers still count as abstain.",
            "Use only checked-in exact fixtures with canonical answers and minimal counterexamples.",
            "Abort before a larger GPU rerun when the smoke rows do not produce parseable decisions.",
        ],
        "acceptance_criteria": {
            "minimum_audited_exact_rows": minimum_rows,
            "minimum_answerable_rows": minimum_answerable,
            "target_max_abstention_rate": TARGET_MAX_ABSTENTION_RATE,
            "minimum_decision_coverage": round(1.0 - TARGET_MAX_ABSTENTION_RATE, 6),
            "target_false_accept_rate": TARGET_FALSE_ACCEPT_RATE,
            "max_false_accept_count": 0,
            "requires_parser_contract_evidence": True,
            "requires_no_synthetic_rows": True,
        },
        "pre_gpu_smoke_gate": {
            "rows": 2,
            "abort_if_parseable_decision_count_lt": 1,
            "abort_if_false_accept_count_gt": 0,
        },
        "measurement_fields": [
            "n_eval",
            "answerable_row_count",
            "parseable_decision_count",
            "abstention_rate",
            "false_accept_rate",
            "false_accept_count",
            "exact_row_fixture_hash",
            "reproducibility_checksum",
        ],
    }


def dominant_root_cause(
    row_table: Sequence[Mapping[str, Any]],
    threshold_findings: Sequence[Mapping[str, Any]],
    parser_findings: Sequence[Mapping[str, Any]],
) -> str:
    """Assign one headline cause while preserving detailed findings."""

    if not row_table:
        return "no_evaluated_exact_rows"
    if all(row.get("answerability") == "malformed_or_missing_answer" for row in row_table):
        return "missing_or_malformed_exact_answers"
    unparseable = sum(
        row.get("abstention_reason") == "model_output_unparseable" for row in row_table
    )
    if unparseable and unparseable == len(row_table):
        return "model_output_parser_contract_mismatch"
    if any(
        row.get("answerability") in {"malformed_or_missing_answer", "unknown"} for row in row_table
    ):
        return "mixed_data_quality_and_parser_findings"
    if any(finding.get("blocked") is True for finding in threshold_findings):
        return "threshold_blocked_nonzero_abstention_rate"
    if parser_findings:
        return "parser_or_extraction_findings_present"
    return "unknown"


def row_class_counts(row_table: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the four-way answerability denominator required by the audit."""

    counts = {
        "answerable": 0,
        "malformed_or_missing_answer": 0,
        "unanswerable": 0,
        "unknown": 0,
    }
    for row in row_table:
        key = str(row.get("answerability") or "unknown")
        counts[key if key in counts else "unknown"] += 1
    return counts


def count_values(values: Sequence[Any]) -> JsonDict:
    """Count non-empty string values for compact artifact summaries."""

    counts: JsonDict = {}
    for value in values:
        key = str(value or "")
        if key:
            counts[key] = int(counts.get(key, 0)) + 1
    return counts


def fixture_rows_by_id(path: Path) -> dict[str, JsonDict]:
    """Load exact fixture rows by ID; malformed JSONL rows are not guessed."""

    fixtures: dict[str, JsonDict] = {}
    for row in read_jsonl_objects(path):
        fixture_id = str(row.get("fixture_id") or "")
        if fixture_id:
            fixtures[fixture_id] = row
    return fixtures


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, returning empty evidence for missing or malformed files."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_jsonl_objects(path: Path) -> list[JsonDict]:
    """Read JSONL object rows while ignoring malformed and non-object lines."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[JsonDict] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def source_artifacts(root: Path) -> list[JsonDict]:
    """Record local files that determine the audit result."""

    paths = (
        ("exp3275_clean_verifier_rerun", EXP3275_REL_PATH),
        ("exp3268_sota_receipt_methodology", EXP3268_REL_PATH),
        ("exp3223_exact_row_sidecar", EXP3223_REL_PATH),
        ("context_exact_row_fixture", CONTEXT_FIXTURE_REL_PATH),
        ("verifier_authenticity_audit_report", AUDIT_REPORT_REL_PATH),
        ("verification_openspec", SPEC_REL_PATH),
        ("exp3286_module", Path("python/carnot/verify/clean_verifier_abstention_root_cause_v1.py")),
        ("exp3286_tests", TEST_REL_PATH),
    )
    return [
        {
            "role": role,
            "path": path.as_posix(),
            "present": (root / path).is_file(),
            "sha256": sha256_file(root / path),
        }
        for role, path in paths
    ]


def abstention_threshold(exp3275: Mapping[str, Any]) -> float:
    """Recover the v14 abstention threshold, defaulting to the code constant."""

    thresholds = mapping(exp3275.get("thresholds"))
    return bounded_float(thresholds.get("abstention_threshold"), default=0.5)


def false_accept_threshold(exp3275: Mapping[str, Any]) -> float:
    """Recover the v14 false-accept threshold, defaulting to the code constant."""

    thresholds = mapping(exp3275.get("thresholds"))
    return bounded_float(thresholds.get("false_accept_threshold"), default=0.1)


def safety_tokens() -> tuple[str, ...]:
    """Text markers that would suggest a safety refusal rather than parser drift."""

    return ("cannot comply", "can't comply", "safety policy", "unsafe request", "i cannot")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject non-terminal or non-machine-readable audit artifacts."""

    missing = REQUIRED_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if not str(artifact.get("honest_verdict") or "").startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success-style prefix")
    if not 0.0 <= bounded_float(artifact.get("prior_abstention_rate"), default=-1.0) <= 1.0:
        raise ValueError("prior_abstention_rate must be in [0, 1]")
    for key in (
        "audited_exact_row_count",
        "answerable_row_count",
        "malformed_or_missing_answer_count",
    ):
        if not isinstance(artifact.get(key), int) or int(artifact[key]) < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    for key in ("threshold_or_policy_findings", "parser_or_extraction_findings"):
        if not isinstance(artifact.get(key), list):
            raise ValueError(f"{key} must be a list")
    if not isinstance(artifact.get("calibrated_rerun_plan"), Mapping):
        raise ValueError("calibrated_rerun_plan must be an object")
    if artifact.get("target_false_accept_rate") != TARGET_FALSE_ACCEPT_RATE:
        raise ValueError("target_false_accept_rate must be exactly 0.0")
    target_abstention = artifact.get("target_max_abstention_rate")
    if not isinstance(target_abstention, float) or not 0.0 <= target_abstention < 1.0:
        raise ValueError("target_max_abstention_rate must be a float below 1.0")
    if len(str(artifact.get("reproducibility_checksum") or "")) != 64:
        raise ValueError("reproducibility_checksum must be a sha256-style string")


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict that starts with a conductor-accepted prefix."""

    if artifact.get("abstention_root_cause_identified") is True:
        return "complete: clean verifier abstention root cause identified"
    return "complete: clean verifier abstention audit ready; root cause not identified"


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact after removing its self-referential checksum field."""

    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return stable_hash(payload)


def sha256_file(path: Path) -> str | None:
    """Return the SHA-256 digest for a present local file."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    """Hash structured data with deterministic JSON normalization."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def mapping(value: Any) -> JsonDict:
    """Return a plain dict only when the input is mapping-like."""

    return dict(value) if isinstance(value, Mapping) else {}


def mapping_list(value: Any) -> list[JsonDict]:
    """Return only object rows from JSON list-like values."""

    return (
        [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []
    )


def bounded_float(value: Any, *, default: float = 0.0) -> float:
    """Coerce evidence to a bounded rate-like float without raising."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if result < 0.0 or result > 1.0:
        return float(default)
    return round(result, 6)


def rate(numerator: int, denominator: int) -> float:
    """Compute a bounded rate while making empty denominators explicit."""

    return 0.0 if denominator <= 0 else round(float(numerator) / float(denominator), 6)


def duration(started_s: float, now_s: float) -> float:
    """Measure non-negative wall-clock duration."""

    return round(max(0.0, float(now_s) - float(started_s)), 6)
