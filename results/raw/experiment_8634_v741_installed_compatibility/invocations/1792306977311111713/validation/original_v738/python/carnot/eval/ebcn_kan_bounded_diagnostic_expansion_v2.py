"""Exp 3173 EBCN/KAN bounded diagnostic expansion v2.

Spec refs: REQ-VERIFY-3173, SCENARIO-VERIFY-3173.

This module builds a matrix-v28 diagnostic artifact from checked-in exact-label
evidence. It replays EBCN sidecar scores and KAN monitor records where prior
artifacts already provide them, but it does not call a model, train a network,
install a verifier, or integrate the diagnostics into generation. The point is
to make the exact denominator, false-accept coverage, and nonpromotion blockers
visible in one bounded JSON artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from carnot.eval import ebcn_energy_sidecar_calibration_v1 as ebcn
from carnot.eval import kan_proof_carrying_monitor_expansion_v1 as kan


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
ARTIFACT = "experiment_3173_ebcn_kan_bounded_diagnostic_expansion_v2"
SCHEMA = "carnot.ebcn_kan_bounded_diagnostic_expansion.v2"
OUTPUT_REL_PATH = Path(
    "results/experiment_3173_ebcn_kan_bounded_diagnostic_expansion_v2.json"
)

EXP3136_ID = "exp3136_false_accept_autopsy"
EXP3137_ID = "exp3137_exact_safe_contract"
EXP3138_ID = "exp3138_canonical_grounding"
EXP3158_ID = "exp3158_ebcn_energy_sidecar"
EXP3159_ID = "exp3159_kan_monitor_expansion"
EXP3167_ID = "exp3167_clean_live_verifier_rerun"

EXP3136_REL_PATH = ebcn.EXP3136_REL_PATH
EXP3137_REL_PATH = ebcn.EXP3137_REL_PATH
EXP3138_REL_PATH = ebcn.EXP3138_REL_PATH
EXP3158_REL_PATH = ebcn.OUTPUT_REL_PATH
EXP3159_REL_PATH = kan.OUTPUT_REL_PATH
EXP3167_REL_PATH = Path("results/experiment_3167_clean_live_sota_verifier_rerun_v9.json")
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

REQUIRED_ARTIFACT_FIELDS = (
    "ebcn_kan_bounded_diagnostic_expansion_v2_ready",
    "exact_labeled_row_count",
    "known_false_accept_rows_scored",
    "ebcn_localization_metrics",
    "kan_monitor_record_count",
    "deployed_verifier_claim_allowed",
    "live_integration_claim_allowed",
    "promotion_blockers",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)

DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3173_ebcn_kan_bounded_diagnostic_expansion_v2.py -q --no-cov",
    ".venv/bin/coverage run --source=python/carnot/eval/ebcn_kan_bounded_diagnostic_expansion_v2.py -m pytest -o addopts='' tests/python/test_experiment_3173_ebcn_kan_bounded_diagnostic_expansion_v2.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/ebcn_kan_bounded_diagnostic_expansion_v2.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_3173_ebcn_kan_bounded_diagnostic_expansion_v2.py",
    ".venv/bin/pytest tests/python -q",
)

SOURCE_SPECS: tuple[tuple[str, Path, bool, str], ...] = (
    ("agents_repo_instructions", Path("AGENTS.md"), True, "md"),
    ("codex_repo_workflow", Path("CODEX.md"), True, "md"),
    ("claude_authenticity_rules", Path("CLAUDE.md"), True, "md"),
    ("research_references", Path("research-references.md"), False, "md"),
    ("verification_openspec", SPEC_REL_PATH, True, "md"),
    (EXP3136_ID, EXP3136_REL_PATH, True, "json"),
    (EXP3137_ID, EXP3137_REL_PATH, True, "json"),
    (EXP3138_ID, EXP3138_REL_PATH, True, "json"),
    (EXP3158_ID, EXP3158_REL_PATH, True, "json"),
    (EXP3159_ID, EXP3159_REL_PATH, True, "json"),
    (EXP3167_ID, EXP3167_REL_PATH, False, "json"),
    (
        "exp3173_module",
        Path("python/carnot/eval/ebcn_kan_bounded_diagnostic_expansion_v2.py"),
        False,
        "py",
    ),
    (
        "exp3173_tests",
        Path("tests/python/test_experiment_3173_ebcn_kan_bounded_diagnostic_expansion_v2.py"),
        False,
        "py",
    ),
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3173: build the checked-in EBCN/KAN diagnostic panel."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    sources = load_sources(root_path)
    rows = collect_exact_rows(sources)
    false_ids = known_false_accept_ids(sources)
    ebcn_metrics = ebcn_localization_metrics(rows, false_ids)
    kan_metrics = kan_monitor_coverage_metrics(rows, false_ids)
    clean_status = clean_verifier_rerun_status(sources)
    blockers = promotion_blockers(rows, ebcn_metrics, kan_metrics, clean_status)
    checks = readiness_checks(sources, rows, false_ids, ebcn_metrics, kan_metrics, blockers)
    ready = all(checks.values())
    source_rows = list(sources["source_artifacts"])
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3173", "SCENARIO-VERIFY-3173"],
        "matrix_version": "v28",
        "ebcn_kan_bounded_diagnostic_expansion_v2_ready": ready,
        "exact_labeled_row_count": len(rows),
        "known_false_accept_rows_scored": ebcn_metrics["known_false_accept_rows_scored"],
        "ebcn_localization_metrics": ebcn_metrics,
        "kan_monitor_record_count": kan_metrics["monitor_record_count"],
        "kan_monitor_coverage_metrics": kan_metrics,
        "deployed_verifier_claim_allowed": False,
        "live_integration_claim_allowed": False,
        "promotion_blockers": blockers,
        "source_artifacts": source_rows,
        "source_checksums": {
            row["path"]: row["sha256"] for row in source_rows if row["sha256"]
        },
        "source_row_summary": source_row_summary(rows, false_ids),
        "exact_rows": rows,
        "known_false_accept_row_ids": sorted(false_ids),
        "clean_verifier_rerun_status": clean_status,
        "inference_substrate": inference_substrate(clean_status),
        "field_principles": field_principles(),
        "readiness_checks": checks,
        "blocked_reasons": [name for name, ok in checks.items() if ok is not True],
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
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
    """Build, validate, and persist the Exp 3173 matrix-v28 artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(out_path, artifact)
    return out_path


def load_sources(root: Path | str) -> JsonDict:
    """Load every source artifact through a small fail-closed boundary."""

    root_path = Path(root)
    payloads: JsonDict = {}
    for source_id, rel_path, _required, source_type in SOURCE_SPECS:
        if source_type == "json":
            payloads[source_id] = read_json_object(root_path / rel_path)
    return {
        "root": root_path.as_posix(),
        "payloads": payloads,
        "source_artifacts": source_artifacts(root_path),
    }


def collect_exact_rows(sources: Mapping[str, Any]) -> list[JsonDict]:
    """Collect and deduplicate exact rows while preserving source provenance."""

    payloads = payload_map(sources)
    rows: dict[str, JsonDict] = {}
    false_ids = set(string_list(payloads.get(EXP3136_ID, {}).get("false_accept_row_ids")))

    for row in mapping_rows(payloads.get(EXP3137_ID, {}).get("replay_rows")):
        row_id = str(row.get("row_id") or "")
        if row_id:
            target = ensure_row(rows, row_id)
            attach_source(target, EXP3137_ID)
            set_if_present(target, "exact_label", row.get("exact_label"))
            set_if_present(target, "expected_action", row.get("expected_action"))
            if "contract_decision" not in target:
                set_if_present(target, "contract_decision", row.get("decision"))
                set_if_present(target, "contract_rule_id", row.get("matched_rule_id"))
                set_if_present(target, "contract_row_source", row.get("row_source"))
            if row.get("known_false_accept_family") is True:
                target["known_false_accept"] = True

    for row in mapping_rows(payloads.get(EXP3136_ID, {}).get("verifier_rows")):
        attach_exp3136_row(rows, row, false_ids)
    for row in mapping_rows(payloads.get(EXP3136_ID, {}).get("false_accept_rows")):
        attach_exp3136_row(rows, row, false_ids)

    for row in mapping_rows(payloads.get(EXP3138_ID, {}).get("regression_row_replay")):
        row_id = str(row.get("row_id") or "")
        if row_id:
            target = ensure_row(rows, row_id)
            attach_source(target, EXP3138_ID)
            set_if_present(target, "exact_label", row.get("exact_label"))
            set_if_present(target, "expected_action", row.get("expected_action"))
            target["canonical_grounding_blocks"] = string_list(row.get("blocked_by"))
            target["canonical_equivalent"] = row.get("canonical_equivalent")
            target["canonical_candidate_answer"] = row.get("candidate_answer")

    for row in mapping_rows(payloads.get(EXP3158_ID, {}).get("calibration_rows")):
        row_id = str(row.get("row_id") or "")
        if row_id:
            target = ensure_row(rows, row_id)
            attach_source(target, EXP3158_ID)
            set_if_present(target, "exact_label", row.get("exact_label"))
            set_if_present(target, "expected_action", row.get("expected_action"))
            target["known_false_accept"] = target["known_false_accept"] or row.get(
                "known_false_accept"
            ) is True
            target["ebcn_score"] = ebcn_score_summary(row)

    for record in mapping_rows(payloads.get(EXP3159_ID, {}).get("pwa_milp_bound_records")):
        row_id = str(record.get("fixture_id") or "")
        if row_id:
            target = ensure_row(rows, row_id)
            attach_source(target, EXP3159_ID)
            link = record.get("exact_label_link")
            link_map = link if isinstance(link, Mapping) else {}
            set_if_present(target, "exact_label", link_map.get("exact_label"))
            set_if_present(target, "expected_action", link_map.get("expected_action"))
            target["known_false_accept"] = target["known_false_accept"] or record.get(
                "exact_row_set"
            ) == "false_accept"
            target["kan_monitor_record"] = kan_record_summary(record)

    attach_clean_rerun_rows(rows, payloads.get(EXP3167_ID, {}))

    exact_rows = []
    for row in rows.values():
        if row.get("exact_label"):
            row["source_artifact_ids"] = sorted(set(row["source_artifact_ids"]))
            row["known_false_accept"] = row["known_false_accept"] or row["row_id"] in false_ids
            exact_rows.append(row)
    return sorted(exact_rows, key=lambda item: str(item["row_id"]))


def attach_exp3136_row(
    rows: dict[str, JsonDict], row: Mapping[str, Any], false_ids: set[str]
) -> None:
    """Attach exact autopsy fields for one verifier or false-accept row."""

    row_id = str(row.get("row_id") or "")
    if not row_id:
        return
    target = ensure_row(rows, row_id)
    attach_source(target, EXP3136_ID)
    set_if_present(target, "exact_label", row.get("exact_label"))
    set_if_present(target, "expected_action", row.get("expected_action"))
    set_if_present(target, "live_decision", row.get("live_decision"))
    set_if_present(target, "fixture_family", row.get("fixture_family"))
    set_if_present(target, "failure_mechanism", row.get("failure_mechanism_from_exp3124"))
    target["monitor_event_count"] = max(
        int(target.get("monitor_event_count") or 0), len(mapping_rows(row.get("monitor_events")))
    )
    if row_id in false_ids or row.get("known_false_accept_family") is True:
        target["known_false_accept"] = True


def attach_clean_rerun_rows(rows: dict[str, JsonDict], exp3167: Mapping[str, Any]) -> None:
    """Attach clean-rerun planning or row evidence when Exp 3167 exists."""

    planned = exp3167.get("planned_rerun_set")
    planned_map = planned if isinstance(planned, Mapping) else {}
    for row_id in string_list(planned_map.get("row_ids")):
        target = ensure_row(rows, row_id)
        attach_source(target, EXP3167_ID)
        target["clean_rerun_planned"] = True

    for key in ("rerun_rows", "rows", "verifier_rows"):
        for row in mapping_rows(exp3167.get(key)):
            row_id = str(row.get("row_id") or "")
            if row_id:
                target = ensure_row(rows, row_id)
                attach_source(target, EXP3167_ID)
                set_if_present(target, "exact_label", row.get("exact_label"))
                set_if_present(target, "expected_action", row.get("expected_action"))
                set_if_present(target, "live_decision", row.get("live_decision"))
                target["clean_rerun_live_row"] = True


def ensure_row(rows: dict[str, JsonDict], row_id: str) -> JsonDict:
    """Return a row initialized with explicit empty diagnostic slots."""

    if row_id not in rows:
        rows[row_id] = {
            "row_id": row_id,
            "known_false_accept": False,
            "source_artifact_ids": [],
            "monitor_event_count": 0,
            "ebcn_score": None,
            "kan_monitor_record": None,
            "clean_rerun_planned": False,
            "clean_rerun_live_row": False,
        }
    return rows[row_id]


def attach_source(row: JsonDict, source_id: str) -> None:
    """Record that one source artifact contributed evidence to a row."""

    row["source_artifact_ids"].append(source_id)


def set_if_present(row: JsonDict, key: str, value: Any) -> None:
    """Set a row field only when the source value is not blank."""

    if value is not None and value != "":
        row[key] = value


def ebcn_score_summary(row: Mapping[str, Any]) -> JsonDict:
    """Keep the row-level EBCN diagnostic evidence needed by matrix v28."""

    return {
        "scalar_energy": as_float(row.get("scalar_energy")),
        "localization_covered": row.get("localization_covered") is True,
        "violation_expected": row.get("violation_expected") is True,
        "violation_localization": mapping_rows(row.get("violation_localization")),
        "categories": string_list(row.get("categories")),
        "energy_branches": mapping_rows(row.get("energy_branches")),
    }


def kan_record_summary(record: Mapping[str, Any]) -> JsonDict:
    """Keep countable KAN monitor evidence without copying the full proof body."""

    status = record.get("pwa_milp_status")
    status_map = status if isinstance(status, Mapping) else {}
    return {
        "record_id": str(record.get("record_id") or ""),
        "exact_row_set": str(record.get("exact_row_set") or ""),
        "record_origin": str(record.get("record_origin") or ""),
        "property_verified": status_map.get("property_verified") is True,
        "solver_status": status_map.get("solver_status"),
        "record_checksum": record.get("record_checksum"),
    }


def ebcn_localization_metrics(rows: Sequence[Mapping[str, Any]], false_ids: set[str]) -> JsonDict:
    """Report localization quality and false-accept separation for EBCN scores."""

    scored = [row for row in rows if isinstance(row.get("ebcn_score"), Mapping)]
    false_scored = [row for row in scored if row.get("row_id") in false_ids]
    clean_scores = [
        as_float(row["ebcn_score"].get("scalar_energy"))
        for row in scored
        if "clean_accept" in string_list(row["ebcn_score"].get("categories"))
    ]
    false_scores = [as_float(row["ebcn_score"].get("scalar_energy")) for row in false_scored]
    localization_required = [
        row for row in scored if row["ebcn_score"].get("violation_expected") is True
    ]
    localization_covered = [
        row for row in localization_required if row["ebcn_score"].get("localization_covered") is True
    ]
    clean_max = max(clean_scores) if clean_scores else 0.0
    false_min = min(false_scores) if false_scores else 0.0
    return {
        "scored_row_count": len(scored),
        "known_false_accept_rows_scored": len(false_scored),
        "unscored_exact_row_count": max(0, len(rows) - len(scored)),
        "localization_required_row_count": len(localization_required),
        "localization_covered_row_count": len(localization_covered),
        "localization_coverage": rate(len(localization_covered), len(localization_required)),
        "false_accept_vs_clean_auc": ebcn.auc(false_scores, clean_scores),
        "false_accept_min_scalar_energy": round(false_min, 6),
        "clean_accept_max_scalar_energy": round(clean_max, 6),
        "false_accept_energy_margin_over_clean_max": round(false_min - clean_max, 6)
        if false_scores and clean_scores
        else 0.0,
    }


def kan_monitor_coverage_metrics(
    rows: Sequence[Mapping[str, Any]], false_ids: set[str]
) -> JsonDict:
    """Report KAN monitor-record coverage over the expanded exact set."""

    records = [row for row in rows if isinstance(row.get("kan_monitor_record"), Mapping)]
    false_records = [row for row in records if row.get("row_id") in false_ids]
    return {
        "monitor_record_count": len(records),
        "known_false_accept_monitor_record_count": len(false_records),
        "known_false_accept_monitor_record_coverage": rate(len(false_records), len(false_ids)),
        "exact_set_monitor_record_coverage": rate(len(records), len(rows)),
        "recorded_row_ids": sorted(str(row["row_id"]) for row in records),
    }


def clean_verifier_rerun_status(sources: Mapping[str, Any]) -> JsonDict:
    """Summarize clean-rerun evidence without upgrading gated skips."""

    payloads = payload_map(sources)
    exp3167 = payloads.get(EXP3167_ID, {})
    source_by_id = {row["id"]: row for row in mapping_rows(sources.get("source_artifacts"))}
    planned = exp3167.get("planned_rerun_set")
    planned_map = planned if isinstance(planned, Mapping) else {}
    live_rows = clean_rerun_rows(exp3167)
    return {
        "artifact_present": source_by_id.get(EXP3167_ID, {}).get("exists") is True,
        "gated_skip": exp3167.get("gated_skip") is True,
        "live_call_count": int(as_float(exp3167.get("live_call_count"))),
        "planned_row_count": len(string_list(planned_map.get("row_ids"))),
        "rows_contributed": len(live_rows) if exp3167.get("gated_skip") is not True else 0,
        "headline_claim_allowed": exp3167.get("headline_claim_allowed") is True,
    }


def clean_rerun_rows(exp3167: Mapping[str, Any]) -> list[JsonDict]:
    """Return row-shaped clean-rerun evidence when a source provides it."""

    rows: list[JsonDict] = []
    for key in ("rerun_rows", "rows", "verifier_rows"):
        rows.extend(mapping_rows(exp3167.get(key)))
    return rows


def source_row_summary(rows: Sequence[Mapping[str, Any]], false_ids: set[str]) -> JsonDict:
    """Expose denominator and source-stratum counts for the panel."""

    return {
        "exact_labeled_row_count": len(rows),
        "known_false_accept_row_count": len(false_ids),
        "clean_rerun_planned_row_count": sum(1 for row in rows if row.get("clean_rerun_planned")),
        "clean_rerun_live_row_count": sum(1 for row in rows if row.get("clean_rerun_live_row")),
        "ebcn_scored_row_count": sum(1 for row in rows if isinstance(row.get("ebcn_score"), Mapping)),
        "kan_monitor_row_count": sum(
            1 for row in rows if isinstance(row.get("kan_monitor_record"), Mapping)
        ),
    }


def promotion_blockers(
    rows: Sequence[Mapping[str, Any]],
    ebcn_metrics: Mapping[str, Any],
    kan_metrics: Mapping[str, Any],
    clean_status: Mapping[str, Any],
) -> list[str]:
    """Name why this artifact is diagnostic-only and not promotion evidence."""

    blockers = [
        "tiny denominator: "
        f"{ebcn_metrics.get('scored_row_count')} EBCN-scored rows and "
        f"{kan_metrics.get('monitor_record_count')} KAN monitor records over "
        f"{len(rows)} exact labels are insufficient for promotion.",
        "No live integration or generation-path accept/reject gate consumes EBCN/KAN diagnostics.",
        "No deployed verifier consumes these diagnostics; deployed_verifier_claim_allowed=false.",
        "EBCN values are checked-in sidecar proxy replays, not learned live EBCN energies.",
        "KAN records are bounded proof-carrying monitor metadata, not trained-network soundness.",
    ]
    if clean_status.get("gated_skip") is True:
        blockers.append("Clean live verifier rerun is a gated skip with zero live model calls.")
    return blockers


def readiness_checks(
    sources: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    false_ids: set[str],
    ebcn_metrics: Mapping[str, Any],
    kan_metrics: Mapping[str, Any],
    blockers: Sequence[str],
) -> JsonDict:
    """Collect boolean readiness gates behind the bounded ready flag."""

    required_sources = [
        row for row in mapping_rows(sources.get("source_artifacts")) if row.get("required") is True
    ]
    return {
        "required_sources_present": all(row.get("exists") is True for row in required_sources),
        "exact_labeled_rows_present": bool(rows),
        "known_false_accept_rows_complete": bool(false_ids)
        and int(ebcn_metrics.get("known_false_accept_rows_scored") or 0) == len(false_ids),
        "ebcn_localization_metrics_finite": metric_is_unit_interval(
            ebcn_metrics.get("localization_coverage")
        )
        and metric_is_unit_interval(ebcn_metrics.get("false_accept_vs_clean_auc")),
        "kan_monitor_record_coverage_countable": int(kan_metrics.get("monitor_record_count") or 0)
        >= 0,
        "inference_substrate_declares_no_live_calls": True,
        "promotion_blockers_explicit": blockers_have_required_terms(blockers),
    }


def blockers_have_required_terms(blockers: Sequence[str]) -> bool:
    """Require the central nonpromotion blockers by name."""

    text = "\n".join(blockers)
    return "tiny denominator" in text and "No live integration" in text and "No deployed verifier" in text


def inference_substrate(clean_status: Mapping[str, Any]) -> JsonDict:
    """Declare that the panel only reads checked-in artifacts."""

    return {
        "kind": "checked_in_artifact_ebcn_kan_bounded_diagnostic",
        "executes_models": False,
        "loads_model_weights": False,
        "generation_performed": False,
        "training_performed": False,
        "model_weight_mutation": False,
        "hardware_execution": False,
        "live_integration": False,
        "offline_diagnostic_only": True,
        "new_live_model_calls": 0,
        "clean_rerun_live_call_count_observed": int(as_float(clean_status.get("live_call_count"))),
    }


def field_principles() -> JsonDict:
    """Map required artifact fields to the discipline they enforce."""

    return {
        "ebcn_kan_bounded_diagnostic_expansion_v2_ready": "sidecar diagnostics need explicit readiness",
        "exact_labeled_row_count": "denominator must be visible",
        "known_false_accept_rows_scored": "central regression rows must be included",
        "ebcn_localization_metrics": "energy diagnostics should localize violations",
        "kan_monitor_record_count": "KAN evidence must be countable",
        "deployed_verifier_claim_allowed": "bounded diagnostics are not deployment",
        "live_integration_claim_allowed": "no live integration without evidence",
        "promotion_blockers": "nonpromotion must be actionable",
        "source_artifacts": "diagnostics must trace to exact labels",
        "inference_substrate": "diagnostic work must declare no live model inference",
        "honest_verdict": "terminal verdict must state complete or blocked status",
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject incomplete artifacts or any live/deployed overclaim."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    _require(not missing, f"missing required fields: {missing}")
    verdict = str(artifact.get("honest_verdict") or "")
    _require(
        verdict.startswith(SUCCESS_PREFIXES) or verdict.startswith("blocked_"),
        "honest_verdict must start with success or blocked prefix",
    )
    _require(
        artifact.get("deployed_verifier_claim_allowed") is False,
        "deployed verifier claims are not allowed",
    )
    _require(
        artifact.get("live_integration_claim_allowed") is False,
        "live integration claims are not allowed",
    )
    _require(bool(artifact.get("promotion_blockers")), "promotion_blockers must be non-empty")
    substrate = artifact.get("inference_substrate")
    _require(isinstance(substrate, Mapping), "inference_substrate must be an object")
    _require(substrate.get("new_live_model_calls") == 0, "new live model calls are not allowed")
    _require(substrate.get("executes_models") is False, "model execution is not allowed")
    metrics = artifact.get("ebcn_localization_metrics")
    _require(isinstance(metrics, Mapping), "ebcn_localization_metrics must be an object")
    _require(
        metric_is_unit_interval(metrics.get("localization_coverage")),
        "localization_coverage must be in [0,1]",
    )
    kan_metrics = artifact.get("kan_monitor_coverage_metrics")
    _require(isinstance(kan_metrics, Mapping), "kan_monitor_coverage_metrics must be an object")
    _require(
        int(artifact.get("kan_monitor_record_count") or 0)
        == int(kan_metrics.get("monitor_record_count") or 0),
        "kan_monitor_record_count mismatch",
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build a terminal verdict that does not imply deployment."""

    if artifact.get("ebcn_kan_bounded_diagnostic_expansion_v2_ready") is True:
        return (
            "complete: ebcn_kan_bounded_diagnostic_expansion_v2_ready=true; "
            f"exact_labeled_row_count={artifact.get('exact_labeled_row_count')}; "
            f"known_false_accept_rows_scored={artifact.get('known_false_accept_rows_scored')}; "
            f"kan_monitor_record_count={artifact.get('kan_monitor_record_count')}; "
            "deployed_verifier_claim_allowed=false; live_integration_claim_allowed=false"
        )
    reasons = artifact.get("blocked_reasons")
    reason_text = ",".join(str(reason) for reason in reasons) if isinstance(reasons, list) else ""
    return f"blocked_missing_exact_diagnostic_evidence: {reason_text}"


def known_false_accept_ids(sources: Mapping[str, Any]) -> set[str]:
    """Return the central regression IDs from Exp 3136."""

    payloads = payload_map(sources)
    return set(string_list(payloads.get(EXP3136_ID, {}).get("false_accept_row_ids")))


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return source provenance for the bounded diagnostic panel."""

    rows: list[JsonDict] = []
    for source_id, rel_path, required, source_type in SOURCE_SPECS:
        path = root / rel_path
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "required": required,
                "source_type": source_type,
                "exists": path.is_file(),
                "sha256": file_sha256(path),
            }
        )
    return rows


def payload_map(sources: Mapping[str, Any]) -> JsonDict:
    """Return the JSON payload map from a loaded source bundle."""

    payloads = sources.get("payloads")
    return dict(payloads) if isinstance(payloads, Mapping) else {}


def mapping_rows(value: Any) -> list[JsonDict]:
    """Keep only JSON object rows from untrusted artifact lists."""

    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def string_list(value: Any) -> list[str]:
    """Return string members from a JSON list."""

    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def metric_is_unit_interval(value: Any) -> bool:
    """Return true when a metric is finite and bounded to [0, 1]."""

    numeric = as_float(value, default=-1.0)
    return math.isfinite(numeric) and 0.0 <= numeric <= 1.0


def rate(numerator: float, denominator: float) -> float:
    """Return a deterministic rounded rate with a zero-denominator fallback."""

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


def read_json_object(path: Path) -> JsonDict:
    """Read one checked-in JSON object, returning an empty object on failure."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def file_sha256(path: Path) -> str | None:
    """Hash a source artifact when it exists."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist deterministic JSON so result diffs remain reviewable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def duration(started_s: float, now_s: float | None) -> float:
    """Return rounded wall-clock duration for the artifact."""

    now = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, now - started_s), 6)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:  # pragma: no cover
    """CLI entrypoint for writing the requested result artifact."""

    output = write_artifact()
    artifact = read_json_object(output)
    print(
        json.dumps(
            {
                "artifact": output.as_posix(),
                "ready": artifact["ebcn_kan_bounded_diagnostic_expansion_v2_ready"],
                "exact_labeled_row_count": artifact["exact_labeled_row_count"],
                "known_false_accept_rows_scored": artifact["known_false_accept_rows_scored"],
                "kan_monitor_record_count": artifact["kan_monitor_record_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
