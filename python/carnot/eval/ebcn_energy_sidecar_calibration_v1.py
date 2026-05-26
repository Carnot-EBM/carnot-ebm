"""Exp 3158 EBCN-style energy sidecar calibration.

Spec refs: REQ-VERIFY-3158, SCENARIO-VERIFY-3158.

This module is an offline diagnostic. It does not implement an Energy-Based
Constraint Network, call a live verifier, or wire energy into generation.
Instead, it uses the existing exact-labeled sidecar rows to ask whether a
bounded scalar energy and row-level localization evidence would have separated
known false accepts from clean exact accepts. The scalar score deliberately
uses only label-blind sidecar fields so the calibration cannot smuggle exact
labels into the energy it is evaluating.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3158_ebcn_energy_sidecar_calibration_v1"
SCHEMA = "carnot.ebcn_energy_sidecar_calibration.v1"
OUTPUT_REL_PATH = Path("results/experiment_3158_ebcn_energy_sidecar_calibration_v1.json")
EXP3144_REL_PATH = Path("results/experiment_3144_ebt_arm_false_accept_calibration_boundary_v3.json")
EXP3136_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
EXP3137_REL_PATH = Path("results/experiment_3137_exact_safe_accept_abstain_contract_v1.json")
EXP3138_REL_PATH = Path("results/experiment_3138_canonical_answer_vericot_grounding_pilot_v1.json")
SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
REQUIRED_FIELDS = (
    "ebcn_energy_sidecar_calibration_v1_ready",
    "exact_labeled_row_count",
    "known_false_accept_rows_scored",
    "scalar_energy_auc",
    "violation_localization_coverage",
    "scale_compatibility_notes",
    "live_integration_claim_allowed",
    "residual_blockers",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)
SCALAR_SCORE_INPUTS = (
    "deterministic_constraint_penalty",
    "final_energy_proxy",
    "uncertainty_proxy",
)
LABEL_AWARE_EXCLUDED_FIELDS = (
    "exact_label",
    "exact_outcome",
    "false_accept",
    "known_false_accept",
    "approximation_gap_to_exact_binary",
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3158_ebcn_energy_sidecar_calibration_v1.py -q --no-cov",
    ".venv/bin/coverage run --source=python/carnot/eval -m pytest -o addopts='' tests/python/test_experiment_3158_ebcn_energy_sidecar_calibration_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/ebcn_energy_sidecar_calibration_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_3158_ebcn_energy_sidecar_calibration_v1.py",
    ".venv/bin/pytest tests/python -q",
)


def read_json_object(path: Path) -> JsonDict:
    """Read one checked-in JSON object and fail closed to empty evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3158: build the offline EBCN-style sidecar calibration."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    exp3144 = read_json_object(root_path / EXP3144_REL_PATH)
    exp3136 = read_json_object(root_path / EXP3136_REL_PATH)
    exp3137 = read_json_object(root_path / EXP3137_REL_PATH)
    exp3138 = read_json_object(root_path / EXP3138_REL_PATH)
    source_rows = source_artifacts(root_path)
    false_ids = set(string_list(exp3136.get("false_accept_row_ids")))
    exact_rows = exact_labeled_rows(exp3144, exp3136, exp3137, exp3138, false_ids)
    scored_rows = score_rows(exact_rows)
    clean_scores = [
        row["scalar_energy"] for row in scored_rows if "clean_accept" in row["categories"]
    ]
    false_scores = [row["scalar_energy"] for row in scored_rows if row["known_false_accept"]]
    scalar_auc = auc(false_scores, clean_scores)
    localization_coverage = violation_localization_coverage(scored_rows)
    false_scored = sum(1 for row in scored_rows if row["row_id"] in false_ids)
    notes = scale_compatibility_notes(scored_rows)
    blockers = residual_blockers(scored_rows, false_ids, exp3144)
    checks = readiness_checks(
        source_rows, scored_rows, false_ids, scalar_auc, localization_coverage
    )
    ready = all(checks.values())
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3158", "SCENARIO-VERIFY-3158"],
        "ebcn_energy_sidecar_calibration_v1_ready": ready,
        "exact_labeled_row_count": len(scored_rows),
        "known_false_accept_rows_scored": false_scored,
        "scalar_energy_auc": scalar_auc,
        "violation_localization_coverage": localization_coverage,
        "scale_compatibility_notes": notes,
        "live_integration_claim_allowed": False,
        "residual_blockers": blockers,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_rows,
        "source_checksums": {
            source["path"]: source["sha256"] for source in source_rows if source["sha256"]
        },
        "inference_substrate": inference_substrate(exp3144),
        "scalar_energy_definition": scalar_energy_definition(),
        "label_leakage_audit": label_leakage_audit(scored_rows),
        "row_category_counts": row_category_counts(scored_rows),
        "calibration_rows": scored_rows,
        "readiness_checks": checks,
        "blocked_reasons": [name for name, ok in checks.items() if ok is not True],
        "duration_s": duration(started, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3158 terminal JSON artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(out_path, artifact)
    return out_path


def exact_labeled_rows(
    exp3144: Mapping[str, Any],
    exp3136: Mapping[str, Any],
    exp3137: Mapping[str, Any],
    exp3138: Mapping[str, Any],
    false_ids: set[str],
) -> list[JsonDict]:
    """Join exact labels, monitor events, and sidecar fields by row ID."""

    verifier_by_id = first_rows_by_id(mapping_rows(exp3136.get("verifier_rows")))
    contract_by_id = first_rows_by_id(mapping_rows(exp3137.get("replay_rows")))
    grounding_by_id = first_rows_by_id(mapping_rows(exp3138.get("regression_row_replay")))
    rows: list[JsonDict] = []
    for row in mapping_rows(exp3144.get("calibration_rows")):
        row_id = str(row.get("row_id") or "")
        exact_label = str(row.get("exact_label") or "")
        if not row_id or not exact_label:
            continue
        verifier = verifier_by_id.get(row_id, {})
        joined = {
            "row_id": row_id,
            "exact_label": exact_label,
            "exact_outcome": str(row.get("exact_outcome") or ""),
            "expected_action": str(row.get("expected_action") or ""),
            "live_decision": str(row.get("live_decision") or ""),
            "known_false_accept": row_id in false_ids or row.get("false_accept") is True,
            "fixture_family": str(
                row.get("fixture_family") or verifier.get("fixture_family") or ""
            ),
            "difficulty_buckets": string_list(verifier.get("difficulty_buckets")),
            "failure_mechanism": str(verifier.get("failure_mechanism_from_exp3124") or ""),
            "monitor_events": mapping_rows(verifier.get("monitor_events")),
            "contract_decision": str(contract_by_id.get(row_id, {}).get("decision") or ""),
            "grounding_blocks": string_list(grounding_by_id.get(row_id, {}).get("blocked_by")),
            "deterministic_constraint_penalty": as_float(
                row.get("deterministic_constraint_penalty")
            ),
            "final_energy_proxy": as_float(row.get("final_energy_proxy")),
            "quality_proxy": as_float(row.get("quality_proxy")),
            "uncertainty_proxy": as_float(row.get("uncertainty_proxy")),
            "uses_exact_label_reference_for_score": bool(
                row.get("uses_exact_label_reference_for_score")
            ),
        }
        joined["categories"] = row_categories(joined)
        rows.append(joined)
    return sorted(rows, key=lambda item: item["row_id"])


def mapping_rows(value: Any) -> list[JsonDict]:
    """Keep only JSON object rows from untrusted artifact lists."""

    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def first_rows_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Return first row per `row_id`, preferring rows with monitor evidence."""

    indexed: dict[str, JsonDict] = {}
    for row in rows:
        row_id = str(row.get("row_id") or "")
        if row_id and (
            row_id not in indexed
            or len(mapping_rows(row.get("monitor_events")))
            > len(mapping_rows(indexed[row_id].get("monitor_events")))
        ):
            indexed[row_id] = dict(row)
    return indexed


def row_categories(row: Mapping[str, Any]) -> list[str]:
    """Name the exact-label strata used by the calibration panel."""

    categories: list[str] = []
    buckets = set(string_list(row.get("difficulty_buckets")))
    if row.get("known_false_accept") is True:
        categories.append("known_false_accept")
    if row.get("exact_outcome") == "accepted" and row.get("live_decision") == "accept":
        categories.append("clean_accept")
    if "contradiction" in buckets or row.get("failure_mechanism") == "contradiction":
        categories.append("contradiction")
    if "satisfiable_drift" in buckets:
        categories.append("satisfiable_drift")
    if row.get("exact_outcome") == "repairable":
        categories.append("repairable_drift")
    return categories or ["exact_labeled"]


def score_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Attach bounded scalar energy and violation localization to each row."""

    scored: list[JsonDict] = []
    for row in rows:
        penalty = as_float(row.get("deterministic_constraint_penalty"))
        final_energy = as_float(row.get("final_energy_proxy"))
        uncertainty = min(max(as_float(row.get("uncertainty_proxy")), 0.0), 1.0)
        branches = [
            {"name": "structural_constraint", "value": bounded_unit(penalty), "weight": 0.45},
            {"name": "sidecar_global_energy", "value": bounded_unit(final_energy), "weight": 0.35},
            {"name": "answer_uncertainty", "value": uncertainty, "weight": 0.20},
        ]
        scalar = round(sum(branch["value"] * branch["weight"] for branch in branches), 6)
        enriched = dict(row)
        enriched["energy_branches"] = branches
        enriched["scalar_energy"] = scalar
        enriched["violation_expected"] = (
            row.get("exact_outcome") != "accepted" or row.get("known_false_accept") is True
        )
        enriched["violation_localization"] = violation_localization_for(enriched)
        enriched["localization_covered"] = not enriched["violation_expected"] or bool(
            enriched["violation_localization"]
        )
        scored.append(enriched)
    return scored


def violation_localization_for(row: Mapping[str, Any]) -> list[JsonDict]:
    """Localize violations to monitor constraints or deterministic row-level fallbacks."""

    entries: list[JsonDict] = []
    for event in mapping_rows(row.get("monitor_events")):
        payload = event.get("payload")
        payload_map = payload if isinstance(payload, Mapping) else {}
        if event.get("event_type") == "constraint_ledger":
            for constraint in mapping_rows(payload_map.get("constraints")):
                if constraint.get("status") == "fail":
                    entries.append(
                        {
                            "position": str(constraint.get("constraint_id") or "constraint"),
                            "branch": "structural_constraint",
                            "source": "monitor_constraint_ledger",
                            "severity": bounded_unit(
                                as_float(row.get("deterministic_constraint_penalty"))
                            ),
                        }
                    )
        if event.get("event_type") == "candidate_final_answer" and (
            payload_map.get("final_answer_consistent_with_exact") is False
            or payload_map.get("final_answer_consistent_with_ledger") is False
        ):
            entries.append(
                {
                    "position": "candidate_final_answer",
                    "branch": "answer_consistency",
                    "source": "monitor_candidate_final_answer",
                    "severity": 1.0,
                }
            )
    if not entries and as_float(row.get("deterministic_constraint_penalty")) > 0.0:
        entries.append(
            {
                "position": f"{row.get('row_id')}:sidecar_constraint_penalty",
                "branch": "structural_constraint",
                "source": "deterministic_sidecar_fallback",
                "severity": bounded_unit(as_float(row.get("deterministic_constraint_penalty"))),
            }
        )
    return entries


def scalar_energy_definition() -> JsonDict:
    """Describe the bounded diagnostic score and its no-label-leakage boundary."""

    return {
        "score_inputs": list(SCALAR_SCORE_INPUTS),
        "branch_weights": {
            "structural_constraint": 0.45,
            "sidecar_global_energy": 0.35,
            "answer_uncertainty": 0.20,
        },
        "bounded_transform": "x -> max(0, x) / (1 + max(0, x)); uncertainty is already [0,1]",
        "excluded_label_aware_fields": list(LABEL_AWARE_EXCLUDED_FIELDS),
    }


def label_leakage_audit(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Confirm exact-label fields are excluded from the scalar score."""

    return {
        "uses_exact_label_for_scalar_energy": False,
        "excluded_fields": list(LABEL_AWARE_EXCLUDED_FIELDS),
        "row_count_with_exact_label_reference_flag": sum(
            1 for row in rows if row.get("uses_exact_label_reference_for_score") is True
        ),
    }


def row_category_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count calibration strata without treating categories as score inputs."""

    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(string_list(row.get("categories")))
    return dict(sorted(counts.items()))


def auc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """Compute pairwise AUROC, returning 0.0 when either class is absent."""

    if not positive_scores or not negative_scores:
        return 0.0
    wins = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return round(wins / (len(positive_scores) * len(negative_scores)), 6)


def violation_localization_coverage(rows: Sequence[Mapping[str, Any]]) -> float:
    """Measure coverage over rows that require localization evidence."""

    required = [row for row in rows if row.get("violation_expected") is True]
    covered = [row for row in required if row.get("violation_localization")]
    return rate(len(covered), len(required))


def scale_compatibility_notes(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Record why these branch energies are comparable only for diagnostics."""

    max_penalty = max(
        [as_float(row.get("deterministic_constraint_penalty")) for row in rows] or [0.0]
    )
    max_energy = max([as_float(row.get("final_energy_proxy")) for row in rows] or [0.0])
    return [
        "bounded branch energies are mapped to [0,1] before composition; exact labels are not scalar score inputs",
        "weights are fixed diagnostic assumptions: structural_constraint=0.45, sidecar_global_energy=0.35, answer_uncertainty=0.20",
        f"raw deterministic_constraint_penalty max={round(max_penalty, 6)} and final_energy_proxy max={round(max_energy, 6)} are not directly composable without calibration",
        "localization uses monitor ledger positions or deterministic row-level fallbacks, not token-level live EBCN hidden-state energy",
    ]


def residual_blockers(
    rows: Sequence[Mapping[str, Any]], false_ids: set[str], exp3144: Mapping[str, Any]
) -> list[str]:
    """Name the work needed before diagnostic energy could become verifier evidence."""

    blockers = [
        "no live verifier integration implemented or exercised",
        "EBCN branch scales are hand-bounded from sidecar proxies, not learned comparable energies",
        "per-position localization is row/constraint-event diagnostic, not token-level live energy",
        "exact-safe thresholds are not validated on held-out live rows",
    ]
    confound = exp3144.get("model_identity_confound_audit")
    confound_map = confound if isinstance(confound, Mapping) else {}
    if confound_map.get("single_model_trace_only") is True:
        blockers.append("single selected-model trace confound remains")
    if len(rows) < 20:
        blockers.append("calibration panel is too small for live threshold promotion")
    if false_ids and sum(1 for row in rows if row.get("row_id") in false_ids) != len(false_ids):
        blockers.append("not all known false accepts were scored")
    return blockers


def readiness_checks(
    source_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    false_ids: set[str],
    scalar_auc: float,
    localization_coverage: float,
) -> JsonDict:
    """Collect the explicit gates behind the ready boolean."""

    required_sources = [row for row in source_rows if row.get("required") is True]
    return {
        "required_sources_present": all(row.get("exists") is True for row in required_sources),
        "exact_labeled_rows_present": bool(rows),
        "known_false_accept_rows_complete": bool(false_ids)
        and sum(1 for row in rows if row.get("row_id") in false_ids) == len(false_ids),
        "clean_accept_rows_present": any(
            "clean_accept" in row.get("categories", []) for row in rows
        ),
        "scalar_energy_auc_finite": math.isfinite(scalar_auc) and 0.0 <= scalar_auc <= 1.0,
        "violation_localization_complete": localization_coverage == 1.0,
        "no_label_leakage_into_scalar_energy": all(
            row.get("uses_exact_label_reference_for_score") is not True for row in rows
        ),
        "live_integration_claim_blocked": True,
    }


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return source provenance for the offline calibration."""

    specs = (
        ("agents_repo_instructions", Path("AGENTS.md"), False),
        ("codex_repo_workflow", Path("CODEX.md"), False),
        ("claude_authenticity_rules", Path("CLAUDE.md"), False),
        ("research_references", Path("research-references.md"), False),
        ("verification_openspec", SPEC_REL_PATH, False),
        ("exp3158_module", Path("python/carnot/eval/ebcn_energy_sidecar_calibration_v1.py"), False),
        (
            "exp3158_tests",
            Path("tests/python/test_experiment_3158_ebcn_energy_sidecar_calibration_v1.py"),
            False,
        ),
        ("exp3144_false_accept_sidecar_calibration", EXP3144_REL_PATH, True),
        ("exp3136_false_accept_autopsy", EXP3136_REL_PATH, True),
        ("exp3137_exact_safe_contract", EXP3137_REL_PATH, True),
        ("exp3138_canonical_grounding", EXP3138_REL_PATH, True),
    )
    rows: list[JsonDict] = []
    for source_id, rel_path, required in specs:
        path = root / rel_path
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "exists": path.is_file(),
                "required": required,
                "sha256": file_sha256(path),
            }
        )
    return rows


def inference_substrate(exp3144: Mapping[str, Any]) -> JsonDict:
    """Declare that this run only reads checked-in artifacts."""

    upstream = exp3144.get("inference_substrate")
    upstream_map = upstream if isinstance(upstream, Mapping) else {}
    return {
        "kind": "checked_in_artifact_ebcn_energy_sidecar_calibration",
        "executes_models": False,
        "loads_model_weights": False,
        "generation_performed": False,
        "training_performed": False,
        "live_integration": False,
        "new_live_model_calls": 0,
        "offline_diagnostic_only": True,
        "upstream_live_trace_count": int(as_float(exp3144.get("live_call_count"))),
        "upstream_exp3144_new_live_model_calls": upstream_map.get("new_live_model_calls"),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject artifacts that omit required fields or overclaim live integration."""

    missing = sorted(set(REQUIRED_FIELDS) - set(artifact))
    _require(not missing, f"missing required fields: {missing}")
    _require(
        artifact.get("live_integration_claim_allowed") is False,
        "live_integration_claim_allowed must be false",
    )
    substrate = artifact.get("inference_substrate")
    _require(isinstance(substrate, Mapping), "inference_substrate must be an object")
    _require(substrate.get("new_live_model_calls") == 0, "new_live_model_calls must be 0")
    _require(bool(artifact.get("residual_blockers")), "residual_blockers must be non-empty")
    _require(
        0.0 <= as_float(artifact.get("scalar_energy_auc"), -1.0) <= 1.0,
        "scalar_energy_auc must be in [0,1]",
    )
    _require(
        0.0 <= as_float(artifact.get("violation_localization_coverage"), -1.0) <= 1.0,
        "violation_localization_coverage must be in [0,1]",
    )
    verdict = str(artifact.get("honest_verdict", ""))
    _require(
        verdict.startswith(SUCCESS_PREFIXES) or verdict.startswith("blocked_"),
        "honest_verdict must start with success or blocked prefix",
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the terminal verdict without implying live verifier deployment."""

    exact_count = int(artifact.get("exact_labeled_row_count") or 0)
    false_count = int(artifact.get("known_false_accept_rows_scored") or 0)
    auc_value = as_float(artifact.get("scalar_energy_auc"))
    if artifact.get("ebcn_energy_sidecar_calibration_v1_ready") is True:
        return (
            "complete: ebcn_energy_sidecar_calibration_v1_ready=true; "
            f"exact_labeled_row_count={exact_count}; known_false_accept_rows_scored={false_count}; "
            f"scalar_energy_auc={auc_value}; live_integration_claim_allowed=false"
        )
    reasons = artifact.get("blocked_reasons")
    reason_text = ",".join(str(reason) for reason in reasons) if isinstance(reasons, list) else ""
    if exact_count == 0:
        return f"blocked_missing_exact_evidence: exact_labeled_row_count=0; {reason_text}"
    return f"blocked_incomplete_calibration: {reason_text}"


def bounded_unit(value: float) -> float:
    """Map non-negative diagnostic magnitudes into a stable [0,1] interval."""

    numeric = as_float(value)
    clipped = max(0.0, numeric)
    return round(clipped / (1.0 + clipped), 6)


def rate(numerator: float, denominator: float) -> float:
    """Return a rounded rate with a deterministic zero denominator fallback."""

    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def as_float(value: Any, default: float = 0.0) -> float:
    """Convert artifact scalars into finite floats."""

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return float(default)
    return converted if math.isfinite(converted) else float(default)


def string_list(value: Any) -> list[str]:
    """Return string members from a JSON list."""

    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def file_sha256(path: Path) -> str | None:
    """Hash a source artifact when it exists."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def relative_path(root: Path, path: Path) -> str:
    """Return a stable repository-relative path when possible."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist deterministic JSON so result diffs remain reviewable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def duration(started_s: float, now_s: float | None) -> float:
    """Return rounded wall-clock duration for the artifact."""

    now = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, now - started_s), 6)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
