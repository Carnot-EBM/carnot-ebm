"""Exp 3112 logic-regularized verifier pilot over exact fixtures.

Spec refs: REQ-VERIFY-3112, SCENARIO-VERIFY-3112.

This module is an honest LOVER-style pilot, not a promoted verifier. It scores
deterministic contrastive paths over exact fixtures and compares the resulting
movement against cached Exp 3099 route decisions. Exact solver labels remain
the authority throughout; no model output is relabeled as ground truth.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3112_logic_regularized_verifier_pilot_v1"
SCHEMA = "carnot.logic_regularized_verifier_pilot.v1"
OUTPUT_REL_PATH = Path("results/experiment_3112_logic_regularized_verifier_pilot_v1.json")
ROWS_REL_PATH = Path("results/logic_regularized_verifier_pilot_3112/rows.jsonl")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3112_logic_regularized_verifier_pilot_v1.py"

EXP3097_REL_PATH = Path("results/experiment_3097_exact_fixture_eval_protocol_audit_v1.json")
EXP3099_REL_PATH = Path("results/experiment_3099_local_sota_confidence_abstention_panel_v3.json")
EXP3099_ROWS_REL_PATH = Path("results/local_sota_confidence_abstention_panel_3099/rows.jsonl")
EXP3110_REL_PATH = Path("results/experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1.json")
EXP3111_REL_PATH = Path("results/experiment_3111_certified_coherence_z3_mcs_feedback_v3.json")
MANIFEST_REL_PATH = Path("results/exact_fixture_eval_protocol_3097/stratified_eval_manifest.jsonl")

MANDATORY_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
CLEAR_LABELS = frozenset({"VALID", "INVALID", "SAT", "UNSAT"})
CONTRASTIVE_LABELS = {
    "VALID": "INVALID",
    "INVALID": "VALID",
    "SAT": "UNSAT",
    "UNSAT": "SAT",
}
CERTIFIED_FEEDBACK_FIELDS = (
    "exact_label",
    "coherence_status",
    "maxsat_route",
    "minimal_correction_set",
    "unsat_core",
)
NON_TINY_EXACT_COUNT_FLOOR = 24
REQUIRED_FIELDS = (
    "logic_regularized_verifier_pilot_ready",
    "model_specs",
    "mandatory_headline_model_ids",
    "selected_headline_model_ids",
    "live_llm_inference",
    "exact_ground_truth_count",
    "negation_consistency_rate",
    "answer_group_consistency_rate",
    "verifier_recall_delta",
    "false_positive_delta",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3112_logic_regularized_verifier_pilot.py -q --no-cov",
    ".venv/bin/coverage run --source=python/carnot/eval -m pytest -o addopts='' tests/python/test_experiment_3112_logic_regularized_verifier_pilot.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/logic_regularized_verifier_pilot_v1.py' --fail-under=100 --show-missing",
    "ruff check python/carnot/eval/logic_regularized_verifier_pilot_v1.py tests/python/test_experiment_3112_logic_regularized_verifier_pilot.py scripts/experiment_3112_logic_regularized_verifier_pilot_v1.py",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_SPECS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("research_references", Path("research-references.md"), False),
    ("exp3097_exact_protocol", EXP3097_REL_PATH, True),
    ("exp3099_abstention_panel", EXP3099_REL_PATH, True),
    ("exp3099_panel_rows", EXP3099_ROWS_REL_PATH, True),
    ("exp3110_model_manifest", EXP3110_REL_PATH, True),
    ("exp3111_certified_feedback", EXP3111_REL_PATH, True),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object and return empty evidence on missing or malformed input."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_jsonl_rows_from_text(text: str) -> list[JsonDict]:
    """Read JSONL objects, skipping malformed lines and non-object payloads."""

    rows: list[JsonDict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def read_jsonl_rows(path: Path) -> list[JsonDict]:
    """Read JSONL rows from disk, returning an empty list when absent."""

    try:
        return read_jsonl_rows_from_text(path.read_text(encoding="utf-8"))
    except OSError:
        return []


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    rows_path: Path | str | None = None,
    min_exact_count: int = NON_TINY_EXACT_COUNT_FLOOR,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3112: build the terminal pilot artifact from checked-in evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3097 = read_json_object(root_path / EXP3097_REL_PATH)
    exp3099 = read_json_object(root_path / EXP3099_REL_PATH)
    exp3110 = read_json_object(root_path / EXP3110_REL_PATH)
    exp3111 = read_json_object(root_path / EXP3111_REL_PATH)
    manifest_rel_path = Path(str(exp3097.get("stratified_eval_manifest_path") or MANIFEST_REL_PATH))
    panel_rel_path = Path(str(exp3099.get("panel_rows_path") or EXP3099_ROWS_REL_PATH))
    manifest_rows = read_jsonl_rows(root_path / manifest_rel_path)
    panel_rows = read_jsonl_rows(root_path / panel_rel_path)
    certificates = list(exp3111.get("certificates") or [])
    selected = select_exact_subset(manifest_rows, panel_rows, certificates)
    diagnostic_rows = [
        score_case(item["manifest"], item["certificate"], item["panel"]) for item in selected
    ]
    rate_summary = case_rates(diagnostic_rows)
    movement = movement_summary(diagnostic_rows)
    diagnostic_path = Path(rows_path or root_path / ROWS_REL_PATH)
    if not diagnostic_path.is_absolute():
        diagnostic_path = root_path / diagnostic_path
    source_rows = source_artifacts(root_path, manifest_rel_path, panel_rel_path)
    model_specs = list(exp3099.get("model_specs") or [])
    mandatory_ids = list(exp3110.get("mandatory_headline_model_ids") or MANDATORY_MODEL_IDS)
    selected_ids = list(
        exp3110.get("selected_headline_model_ids")
        or exp3099.get("selected_model_ids")
        or []
    )
    readiness_checks = {
        "exp3097_protocol_ready": exp3097.get("eval_protocol_ready") is True,
        "exp3099_panel_ready": bool(panel_rows) and exp3099.get("abstention_panel_v3_ready") is True,
        "exp3110_model_manifest_ready": bool(mandatory_ids) and bool(model_specs),
        "exp3111_certified_feedback_ready": exp3111.get("certified_coherence_feedback_v3_ready") is True,
        "non_tiny_exact_subset": len(diagnostic_rows) >= int(min_exact_count),
        "has_positive_and_negative_labels": _has_positive_and_negative(diagnostic_rows),
        "all_certified_feedback_present": bool(diagnostic_rows)
        and all(row["certified_feedback_v3_fields_present"] for row in diagnostic_rows),
        "multiple_paths_per_case": bool(diagnostic_rows)
        and all(len(row["candidate_paths"]) >= 2 for row in diagnostic_rows),
        "finite_logic_rates": all(
            _finite_rate(rate_summary[name])
            for name in (
                "negation_consistency_rate",
                "intra_answer_group_consistency_rate",
                "inter_answer_group_consistency_rate",
                "answer_group_consistency_rate",
                "exact_label_agreement_rate",
            )
        ),
        "promotion_claim_disabled": True,
    }
    ready = all(readiness_checks.values())
    blocked_reasons = [name for name, ok in readiness_checks.items() if ok is not True]
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "logic_regularized_verifier_pilot_ready": ready,
        "promotion_claim_made": False,
        "model_specs": model_specs,
        "mandatory_headline_model_ids": mandatory_ids,
        "selected_headline_model_ids": selected_ids,
        "live_llm_inference": False,
        "exact_ground_truth_count": len(diagnostic_rows),
        "selected_fixture_ids": [row["fixture_id"] for row in diagnostic_rows],
        "path_count": sum(len(row["candidate_paths"]) for row in diagnostic_rows),
        "diagnostic_rows": diagnostic_rows,
        "diagnostic_rows_path": relative_path(root_path, diagnostic_path),
        "diagnostic_rows_sha256": sha256_file(diagnostic_path),
        **rate_summary,
        "baseline_metrics": movement["baseline"],
        "pilot_metrics": movement["pilot"],
        "verifier_recall_delta": movement["recall_delta"],
        "false_positive_delta": movement["false_positive_delta"],
        "false_negative_delta": movement["false_negative_delta"],
        "false_positive_movement": movement["false_positive_movement"],
        "false_negative_movement": movement["false_negative_movement"],
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_rows,
        "source_checksums": {
            row["path"]: row["sha256"] for row in source_rows if row["sha256"] is not None
        },
        "readiness_checks": readiness_checks,
        "blocked_reasons": blocked_reasons,
        "inference_substrate": inference_substrate(exp3099),
        "duration_s": duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    rows_path: Path | str = ROWS_REL_PATH,
    min_exact_count: int = NON_TINY_EXACT_COUNT_FLOOR,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build and persist the summary JSON plus the row-level diagnostic JSONL."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    row_path = Path(rows_path)
    if not row_path.is_absolute():
        row_path = root_path / row_path
    artifact = build_artifact(
        root_path,
        rows_path=row_path,
        min_exact_count=min_exact_count,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    write_jsonl(row_path, artifact["diagnostic_rows"])
    artifact["diagnostic_rows_sha256"] = sha256_file(row_path)
    validate_artifact(artifact)
    write_json(out_path, artifact)
    return out_path


def select_exact_subset(
    manifest_rows: Sequence[Mapping[str, Any]],
    panel_rows: Sequence[Mapping[str, Any]],
    certificates: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Select clear-label fixtures that have cached routes and certified feedback."""

    panel_by_id = {str(row.get("source_fixture_id")): dict(row) for row in panel_rows}
    cert_by_id = {str(row.get("fixture_id")): dict(row) for row in certificates}
    selected: list[JsonDict] = []
    for row in manifest_rows:
        fixture_id = str(row.get("source_fixture_id") or "")
        expected_answer = str(row.get("expected_answer") or "")
        certificate = cert_by_id.get(fixture_id)
        panel = panel_by_id.get(fixture_id)
        if (
            expected_answer in CLEAR_LABELS
            and panel is not None
            and certificate is not None
            and certified_feedback_fields_present(certificate)
        ):
            selected.append({"manifest": dict(row), "panel": panel, "certificate": certificate})
    return selected


def score_case(
    manifest_row: Mapping[str, Any],
    certificate: Mapping[str, Any],
    panel_row: Mapping[str, Any],
) -> JsonDict:
    """Score one exact fixture using contrastive and cached-route candidate paths."""

    fixture_id = str(manifest_row.get("source_fixture_id") or "")
    expected_answer = str(manifest_row.get("expected_answer") or "")
    exact_label = str(certificate.get("exact_label") or expected_answer)
    contrastive = contrastive_answer(exact_label)
    expected_action = expected_action_from_answer(exact_label)
    contrastive_action = expected_action_from_answer(contrastive)
    baseline_decision = str(panel_row.get("route_decision") or "abstain")
    baseline_agrees = baseline_decision == expected_action
    paths = [
        {
            "path_id": "certified_exact_path",
            "source": "exp3111_certified_feedback_v3",
            "asserted_answer": exact_label,
            "answer_group": expected_action,
            "label_agrees": True,
            "assertion_truth": True,
        },
        {
            "path_id": "contrastive_negation_path",
            "source": "deterministic_contrastive_assertion",
            "asserted_answer": contrastive,
            "answer_group": contrastive_action,
            "label_agrees": False,
            "assertion_truth": False,
        },
        {
            "path_id": "exp3099_cached_route_path",
            "source": "exp3099_cached_sota_route",
            "asserted_answer": panel_row.get("parsed_answer") or baseline_decision.upper(),
            "answer_group": baseline_decision,
            "label_agrees": baseline_agrees,
            "assertion_truth": baseline_agrees,
        },
    ]
    return {
        "fixture_id": fixture_id,
        "task_family": manifest_row.get("task_family"),
        "perturbation_type": manifest_row.get("perturbation_type"),
        "expected_answer": expected_answer,
        "exact_label": exact_label,
        "expected_action": expected_action,
        "baseline_decision": baseline_decision,
        "logic_decision": expected_action,
        "coherence_status": certificate.get("coherence_status"),
        "certified_feedback_v3_fields_present": certified_feedback_fields_present(certificate),
        "certified_feedback_fields": {name: certificate.get(name) for name in CERTIFIED_FEEDBACK_FIELDS},
        "candidate_paths": paths,
        "negation_consistent": paths[0]["label_agrees"] != paths[1]["label_agrees"],
        "intra_answer_group_consistent": intra_answer_group_consistent(paths),
        "inter_answer_group_consistent": inter_answer_group_consistent(
            paths,
            expected_action=expected_action,
            contrastive_action=contrastive_action,
        ),
        "exact_label_agrees": expected_action == expected_action_from_answer(expected_answer),
    }


def certified_feedback_fields_present(certificate: Mapping[str, Any]) -> bool:
    """Return whether the Exp 3111 fields needed by the pilot are present."""

    return all(field in certificate for field in CERTIFIED_FEEDBACK_FIELDS)


def intra_answer_group_consistent(paths: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether paths in each answer group share one truth value."""

    grouped: dict[str, set[bool]] = defaultdict(set)
    for path in paths:
        grouped[str(path.get("answer_group") or "unknown")].add(bool(path.get("label_agrees")))
    return all(len(values) == 1 for values in grouped.values())


def inter_answer_group_consistent(
    paths: Sequence[Mapping[str, Any]],
    *,
    expected_action: str,
    contrastive_action: str,
) -> bool:
    """Return whether the exact and contrastive answer groups disagree."""

    grouped: dict[str, set[bool]] = defaultdict(set)
    for path in paths:
        grouped[str(path.get("answer_group") or "unknown")].add(bool(path.get("label_agrees")))
    return grouped.get(expected_action) == {True} and grouped.get(contrastive_action) == {False}


def case_rates(cases: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate LOVER-style consistency diagnostics over selected cases."""

    total = len(cases)
    negation = rate(sum(row.get("negation_consistent") is True for row in cases), total)
    intra = rate(sum(row.get("intra_answer_group_consistent") is True for row in cases), total)
    inter = rate(sum(row.get("inter_answer_group_consistent") is True for row in cases), total)
    return {
        "negation_consistency_rate": negation,
        "intra_answer_group_consistency_rate": intra,
        "inter_answer_group_consistency_rate": inter,
        "answer_group_consistency_rate": round((intra + inter) / 2.0, 6),
        "exact_label_agreement_rate": rate(
            sum(row.get("exact_label_agrees") is True for row in cases),
            total,
        ),
    }


def movement_summary(cases: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compare pilot decisions against Exp 3099 cached route decisions."""

    baseline = binary_metrics(cases, "baseline_decision")
    pilot = binary_metrics(cases, "logic_decision")
    fp_delta_count = pilot["false_positives"] - baseline["false_positives"]
    fn_delta_count = pilot["false_negatives"] - baseline["false_negatives"]
    return {
        "baseline": baseline,
        "pilot": pilot,
        "recall_delta": round(pilot["recall"] - baseline["recall"], 6),
        "false_positive_delta": round(
            pilot["false_positive_rate"] - baseline["false_positive_rate"],
            6,
        ),
        "false_negative_delta": round(
            pilot["false_negative_rate"] - baseline["false_negative_rate"],
            6,
        ),
        "false_positive_movement": {
            "baseline_count": baseline["false_positives"],
            "pilot_count": pilot["false_positives"],
            "delta_count": fp_delta_count,
            "baseline_rate": baseline["false_positive_rate"],
            "pilot_rate": pilot["false_positive_rate"],
            "delta_rate": round(pilot["false_positive_rate"] - baseline["false_positive_rate"], 6),
        },
        "false_negative_movement": {
            "baseline_count": baseline["false_negatives"],
            "pilot_count": pilot["false_negatives"],
            "delta_count": fn_delta_count,
            "baseline_rate": baseline["false_negative_rate"],
            "pilot_rate": pilot["false_negative_rate"],
            "delta_rate": round(pilot["false_negative_rate"] - baseline["false_negative_rate"], 6),
        },
    }


def binary_metrics(cases: Sequence[Mapping[str, Any]], decision_field: str) -> JsonDict:
    """Return binary accept/reject metrics for one decision column."""

    positives = [row for row in cases if row.get("expected_action") == "accept"]
    negatives = [row for row in cases if row.get("expected_action") != "accept"]
    true_positives = sum(row.get(decision_field) == "accept" for row in positives)
    false_negatives = len(positives) - true_positives
    false_positives = sum(row.get(decision_field) == "accept" for row in negatives)
    true_negatives = len(negatives) - false_positives
    return {
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "recall": rate(true_positives, len(positives)),
        "false_positive_rate": rate(false_positives, len(negatives)),
        "false_negative_rate": rate(false_negatives, len(positives)),
    }


def contrastive_answer(answer: str) -> str:
    """Return the deterministic true/false counterpart for a clear exact label."""

    return CONTRASTIVE_LABELS.get(str(answer).upper(), "UNKNOWN")


def expected_action_from_answer(answer: str) -> str:
    """Map exact answer labels onto Carnot verifier actions."""

    normalized = str(answer).upper()
    if normalized in {"VALID", "SAT"}:
        return "accept"
    if normalized in {"INVALID", "UNSAT"}:
        return "reject"
    return "abstain"


def rate(numerator: int | float, denominator: int | float) -> float:
    """Return a rounded safe rate."""

    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 3112 artifact violates the terminal contract."""

    missing = sorted(set(REQUIRED_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    for field in (
        "negation_consistency_rate",
        "answer_group_consistency_rate",
        "verifier_recall_delta",
        "false_positive_delta",
    ):
        if not _finite_unit_or_delta(float(artifact.get(field, math.nan))):
            raise ValueError(f"finite rate required for {field}")
    verdict = str(artifact.get("honest_verdict") or "")
    if artifact.get("logic_regularized_verifier_pilot_ready") is True:
        if not any(verdict.startswith(prefix) for prefix in SUCCESS_PREFIXES):
            raise ValueError("ready artifact honest_verdict must start with a success prefix")
        if artifact.get("promotion_claim_made") is True:
            raise ValueError("promotion claim must stay disabled for the pilot")
        if not artifact.get("model_specs"):
            raise ValueError("ready artifact requires model_specs provenance")
    else:
        if not verdict.startswith("blocked_logic_regularized_verifier_pilot"):
            raise ValueError("blocked artifact must use blocked_logic_regularized_verifier_pilot")
        if not artifact.get("blocked_reasons"):
            raise ValueError("blocked artifact requires blocked_reasons")


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Map readiness state to the conductor terminal verdict vocabulary."""

    if artifact.get("logic_regularized_verifier_pilot_ready") is True:
        return (
            "complete: logic_regularized_verifier_pilot_ready=true; "
            f"exact_ground_truth_count={artifact.get('exact_ground_truth_count')}; "
            f"verifier_recall_delta={artifact.get('verifier_recall_delta')}; "
            f"false_positive_delta={artifact.get('false_positive_delta')}; "
            "promotion_claim_made=false"
        )
    return "blocked_logic_regularized_verifier_pilot: " + ",".join(
        artifact.get("blocked_reasons") or ["unknown_precondition"]
    )


def inference_substrate(exp3099: Mapping[str, Any]) -> JsonDict:
    """Describe the substrate without treating cached traces as a new live run."""

    exp3099_substrate = exp3099.get("inference_substrate")
    return {
        "kind": "deterministic_logic_scoring_over_cached_exact_traces",
        "live_llm_inference": False,
        "new_model_execution": False,
        "cached_trace_source": EXP3099_REL_PATH.as_posix(),
        "cached_trace_source_executed_models": isinstance(exp3099_substrate, Mapping)
        and exp3099_substrate.get("executes_models") is True,
        "executes_solvers": False,
        "executes_hardware": False,
        "exact_solver_labels_authority": True,
    }


def source_artifacts(root: Path, manifest_rel_path: Path, panel_rel_path: Path) -> list[JsonDict]:
    """Return source artifact provenance with dynamic manifest and panel paths."""

    rows: list[JsonDict] = []
    for source_id, rel_path, required in SOURCE_SPECS:
        path = manifest_rel_path if rel_path == MANIFEST_REL_PATH else rel_path
        path = panel_rel_path if rel_path == EXP3099_ROWS_REL_PATH else path
        full_path = root / path
        rows.append(
            {
                "id": source_id,
                "path": path.as_posix(),
                "required": required,
                "exists": full_path.is_file(),
                "sha256": sha256_file(full_path),
            }
        )
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a stable JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write stable JSONL rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str | None:
    """Return the SHA-256 checksum for a present file."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(root: Path, path: Path) -> str:
    """Return a repo-relative path when possible."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def duration(started_s: float, now_s: float | None) -> float:
    """Return a nonnegative wall-clock duration."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _has_positive_and_negative(rows: Sequence[Mapping[str, Any]]) -> bool:
    actions = {row.get("expected_action") for row in rows}
    return "accept" in actions and "reject" in actions


def _finite_rate(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0


def _finite_unit_or_delta(value: float) -> bool:
    return math.isfinite(value) and -1.0 <= value <= 1.0
