"""Experiment 4819: .443 S2-v3 corpus-wide capstone scorecard.

Spec refs: REQ-CAPSTONE-4819, SCENARIO-CAPSTONE-4819,
SCENARIO-CAPSTONE-4819-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4819-FIELD-PRINCIPLES.

The scorecard reads each landed .443 upstream artifact through
``scripts/summarize_artifact.py`` before importing fields. The headline is the
S2-v3 corpus-wide energy-ranking verdict. A bounded null is only recorded when
the live summarizer does not report ``DEGENERATE_CANDIDATE_POOL`` and the B1
corpus coverage check confirms the effective pool is broad enough.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_4819_capstone_v443"
EXPERIMENT_ID = 4819
SCHEMA = "carnot.exp4819.capstone_v443.v1"
RESULT_RELATIVE_PATH = "results/experiment_4819_capstone_v443.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
SUMMARIZER_RELATIVE_PATH = "scripts/summarize_artifact.py"
RANDOM_SEED = 4819
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-CAPSTONE-4819",
    "SCENARIO-CAPSTONE-4819",
    "SCENARIO-CAPSTONE-4819-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4819-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {"principle": "terminal prefix; capstone_ready=true is success_/complete_."},
    "s2v3_structural_energy_verdict": {
        "principle": (
            "the headline -- corpus-wide trust WIN (S3 authorized) / GENUINE "
            "corpus-wide bounded / inconclusive-insufficient-corpus-diversity; "
            "recorded only if DEGENERATE_CANDIDATE_POOL did not fire."
        )
    },
    "reproducible_total_levels": {
        "principle": "the monotonic ARC progress metric carried from the registry."
    },
    "cited_upstream_artifacts": {
        "principle": "list of {experiment_id, fields_imported, sha256} -- the audit trail."
    },
    "inference_substrate": {"principle": "aggregation_from_upstream_artifacts (0.0001s floor)."},
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "s2v3_structural_energy_verdict",
    "reproducible_total_levels",
    "cited_upstream_artifacts",
    "inference_substrate",
    "levelup_bank",
    "self_play_checkpoint",
    "readiness",
    "heldout_readiness",
    "silent_bug_audit",
    "submission_package_state",
    "sota_handoff",
    "upstream_oracle_declarations",
    "flagged_artifacts_skipped",
    "preconditions_checked",
    "field_principles",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)

TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
    "blocked_",
)


@dataclass(frozen=True)
class UpstreamSource:
    """One upstream result that can feed the .443 capstone scorecard."""

    experiment_id: int
    relative_path: str


@dataclass(frozen=True)
class SummarizerResult:
    """Captured ``scripts/summarize_artifact.py`` result for one artifact."""

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str


Summarizer = Callable[[Path, str], SummarizerResult]

UPSTREAM_SOURCES: dict[str, UpstreamSource] = {
    "S2V3": UpstreamSource(
        4811, "results/experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate.json"
    ),
    "LEVELUP": UpstreamSource(4812, "results/experiment_4812_levelup_attempt.json"),
    "SELF_PLAY": UpstreamSource(4813, "results/experiment_4813_self_play_verifier_checkpoint.json"),
    "HELDOUT": UpstreamSource(4814, "results/experiment_4814_heldout_first_win_readiness.json"),
    "BUG_AUDIT": UpstreamSource(4815, "results/experiment_4815_silent_bug_audit.json"),
    "PACKAGE": UpstreamSource(4816, "results/experiment_4816_submission_package_harden.json"),
    "SOTA": UpstreamSource(
        4818, "results/experiment_4818_sota_ingestion_energy_guided_generation.json"
    ),
}

CLEAN_IMPORT_FIELDS: dict[str, tuple[str, ...]] = {
    "S2V3": (
        "honest_verdict",
        "s3_authorized",
        "verifier_is_oracle",
        "live_path_reachable",
        "energy_selected_offpath_cell_recall",
        "accuracy_gate_selected_offpath_cell_recall",
        "energy_minus_accuracy_delta",
        "energy_minus_accuracy_delta_ci95",
        "n_available_games",
        "n_games_attempted",
        "n_effective_games",
        "required_effective_games",
        "min_heldout_games",
        "positive_control_passed",
        "false_negative_risk_checked",
        "candidates_genuinely_induced",
        "candidate_pool_diversity",
        "game_results",
    ),
    "LEVELUP": (
        "honest_verdict",
        "new_levels_banked",
        "offline_reproduced",
        "reproduced_levels",
        "target_game",
        "registry_update",
        "attempted_games",
        "dead_ends",
        "solve_provenance",
        "verifier_is_oracle",
    ),
    "SELF_PLAY": (
        "honest_verdict",
        "verifier_checkpoint_refreshed",
        "target_game",
        "self_play_residual",
        "offline_reproduced",
        "reproduced_levels",
        "reproduction_gate",
        "solve_provenance",
        "inference_substrate",
        "verifier_is_oracle",
    ),
    "HELDOUT": (
        "honest_verdict",
        "heldout_first_win_rate",
        "first_win_baseline",
        "prior_best_heldout_first_win_rate",
        "heldout_first_win_delta_vs_baseline",
        "heldout_first_win_delta_vs_prior_best",
        "heldout_variant_attempts",
        "positive_control_passed",
        "parity_test_green",
        "null_delta_methodology_note",
        "inference_substrate",
        "verifier_is_oracle",
    ),
    "BUG_AUDIT": (
        "honest_verdict",
        "nulls_audited",
        "trusted_nulls",
        "silent_bugs_found",
        "verifier_is_oracle",
    ),
    "PACKAGE": (
        "honest_verdict",
        "submission_package_ready",
        "submitted_to_leaderboard",
        "operator_only",
        "vram_estimate_gb",
        "package_builds",
        "inference_substrate",
        "verifier_is_oracle",
    ),
    "SOTA": (
        "honest_verdict",
        "methods_mapped",
        "flagged_for_v444",
        "s3_context",
        "arxiv_ids_cited",
        "inference_substrate",
        "verifier_is_oracle",
    ),
}

S2V3_DEGENERATE_IMPORT_FIELDS = [
    "honest_verdict",
    "DEGENERATE_CANDIDATE_POOL",
    "n_available_games",
    "n_games_attempted",
    "n_effective_games",
    "required_effective_games",
    "candidate_pool_diversity",
]


def stable_json(value: Any) -> str:
    """Encode JSON deterministically so checksums track content drift only."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(data: bytes) -> str:
    """Return the repository's normal ``sha256:<hex>`` provenance string."""

    return "sha256:" + hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str | None:
    """Hash a file when it exists; missing files are recorded separately."""

    return sha256_bytes(path.read_bytes()) if path.exists() else None


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash artifact content while excluding the checksum field itself."""

    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    return sha256_bytes(stable_json(filtered).encode("utf-8"))


def _read_json_object(path: Path) -> JsonDict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _read_yaml_object(path: Path) -> JsonDict:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return int(value)
    return default


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return round(float(value), 12)
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _experiment_id(source: str, artifact: Mapping[str, Any] | None = None) -> int:
    if artifact:
        for field in ("experiment_id", "experiment"):
            value = artifact.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return UPSTREAM_SOURCES[source].experiment_id


def _oracle_declared(artifact: Mapping[str, Any] | None) -> bool:
    return bool(artifact and artifact.get("verifier_is_oracle") is True)


def _summary_text(summary: SummarizerResult | None) -> str:
    return "" if summary is None else f"{summary.stdout}\n{summary.stderr}"


def _live_has_degenerate_pool_flag(summary: SummarizerResult | None) -> bool:
    return bool(
        summary and summary.exit_code >= 2 and "DEGENERATE_CANDIDATE_POOL" in _summary_text(summary)
    )


def _live_critical(summary: SummarizerResult | None) -> bool:
    return bool(summary and summary.exit_code >= 2)


def _ci_lower_positive(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and _float(value[0]) is not None
        and float(value[0]) > 0.0
    )


def _ci_includes_zero(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and _float(value[0]) is not None
        and _float(value[1]) is not None
        and float(value[0]) <= 0.0 <= float(value[1])
    )


def _required_effective_games(n_available_games: int, n_games_attempted: int) -> int:
    corpus = max(int(n_available_games), int(n_games_attempted))
    return max(10, int(math.ceil(0.6 * corpus)))


def _offline_corpus_game_count(root: Path) -> int | None:
    env_dir = root / "environment_files"
    if not env_dir.exists():
        return None
    return sum(1 for path in env_dir.iterdir() if not path.name.startswith("."))


def _coverage_reason(
    *,
    degenerate: bool,
    n_available_matches_real_corpus: bool,
    attempted_matches_available: bool,
    coverage_floor_met: bool,
) -> str | None:
    if degenerate:
        return "degenerate_candidate_pool_live_check"
    if not n_available_matches_real_corpus:
        return "b1_corpus_count_mismatch"
    if not attempted_matches_available:
        return "not_all_available_games_attempted"
    if not coverage_floor_met:
        return "insufficient_corpus_diversity"
    return None


def _imported_fields(
    source: str,
    artifact: Mapping[str, Any],
    summary: SummarizerResult | None,
    *,
    corpus_game_count: int | None,
) -> list[str]:
    if source == "S2V3":
        s2v3 = _s2v3_verdict(artifact, summary, corpus_game_count=corpus_game_count)
        if s2v3.get("verdict") == "inconclusive_insufficient_corpus_diversity":
            return S2V3_DEGENERATE_IMPORT_FIELDS
    elif _live_critical(summary):
        return ["live_critical_recheck"]
    return [field for field in CLEAN_IMPORT_FIELDS[source] if field in artifact]


def _cited_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    summarizer_results: Mapping[str, SummarizerResult],
    *,
    corpus_game_count: int | None,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source in UPSTREAM_SOURCES:
        artifact = artifacts.get(source)
        if artifact is None:
            continue
        rows.append(
            {
                "experiment_id": _experiment_id(source, artifact),
                "fields_imported": _imported_fields(
                    source,
                    artifact,
                    summarizer_results.get(source),
                    corpus_game_count=corpus_game_count,
                ),
                "sha256": artifact_sha256.get(source, ""),
            }
        )
    return rows


def _flagged_artifacts_skipped(
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    summarizer_results: Mapping[str, SummarizerResult],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source, artifact in artifacts.items():
        summary = summarizer_results.get(source)
        if source == "S2V3" or not _live_critical(summary):
            continue
        rows.append(
            {
                "source": source,
                "experiment_id": _experiment_id(source, artifact),
                "path": UPSTREAM_SOURCES[source].relative_path,
                "reason": "live_critical_recheck",
                "sha256": artifact_sha256.get(source, ""),
            }
        )
    return rows


def _oracle_declarations(artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, JsonDict]:
    return {
        source: {
            "experiment_id": _experiment_id(source, artifact),
            "verifier_is_oracle": _oracle_declared(artifact),
            "moat_claim_allowed": not _oracle_declared(artifact),
        }
        for source, artifact in artifacts.items()
    }


def _s2v3_verdict(
    artifact: Mapping[str, Any] | None,
    summary: SummarizerResult | None,
    *,
    corpus_game_count: int | None,
) -> JsonDict:
    if artifact is None:
        return {}

    delta = _float(artifact.get("energy_minus_accuracy_delta"))
    ci95 = artifact.get("energy_minus_accuracy_delta_ci95")
    ci95 = ci95 if isinstance(ci95, list) else []
    oracle = _oracle_declared(artifact)
    n_available = _int(artifact.get("n_available_games"))
    n_attempted = _int(artifact.get("n_games_attempted"))
    n_effective = _int(artifact.get("n_effective_games"))
    required = _required_effective_games(n_available, n_attempted)
    live_path = artifact.get("live_path_reachable") is True
    positive_control = artifact.get("positive_control_passed") is True
    false_negative_risk_checked = artifact.get("false_negative_risk_checked") is True
    degenerate = _live_has_degenerate_pool_flag(summary)
    n_available_matches_real_corpus = corpus_game_count is None or n_available == corpus_game_count
    attempted_matches_available = n_attempted == n_available
    coverage_floor_met = n_effective >= required
    coverage_reason = _coverage_reason(
        degenerate=degenerate,
        n_available_matches_real_corpus=n_available_matches_real_corpus,
        attempted_matches_available=attempted_matches_available,
        coverage_floor_met=coverage_floor_met,
    )
    base: JsonDict = {
        "source": "S2V3",
        "experiment_id": _experiment_id("S2V3", artifact),
        "upstream_honest_verdict": artifact.get("honest_verdict", ""),
        "reported_energy_minus_accuracy_delta": delta,
        "reported_energy_minus_accuracy_delta_ci95": ci95,
        "n_available_games": n_available,
        "n_games_attempted": n_attempted,
        "n_effective_games": n_effective,
        "required_effective_games": required,
        "real_corpus_game_count": corpus_game_count,
        "n_available_matches_real_corpus": n_available_matches_real_corpus,
        "attempted_matches_available": attempted_matches_available,
        "coverage_floor_met": coverage_floor_met,
        "verifier_is_oracle": oracle,
        "live_path_reachable": live_path,
        "positive_control_passed": positive_control,
        "false_negative_risk_checked": false_negative_risk_checked,
        "live_recheck_exit_code": summary.exit_code if summary else None,
    }
    if coverage_reason is not None:
        return {
            **base,
            "verdict": "inconclusive_insufficient_corpus_diversity",
            "s3_authorized": False,
            "genuine_corpus_wide_bounded_null": False,
            "coverage_trustworthy": False,
            "degenerate_candidate_pool_flagged": degenerate,
            "reason": coverage_reason,
            "metrics_imported": False,
            "pivot": "retest_s2v3_or_use_energy_for_s3_generation",
        }

    live_clean = bool(summary and summary.exit_code == 0)
    win = bool(
        live_clean
        and not oracle
        and live_path
        and artifact.get("s3_authorized") is True
        and delta is not None
        and delta > 0.0
        and _ci_lower_positive(ci95)
    )
    bounded = bool(
        live_clean
        and not oracle
        and live_path
        and positive_control
        and false_negative_risk_checked
        and _ci_includes_zero(ci95)
    )
    if win:
        verdict = "corpus_wide_trust_win"
        reason = "energy_ranking_beats_accuracy_ci_excludes_zero"
    elif bounded:
        verdict = "genuine_corpus_wide_bounded_null"
        reason = "corpus_wide_diverse_pool_ci_includes_zero"
    else:
        verdict = "inconclusive_insufficient_corpus_diversity"
        reason = "s2v3_live_clean_but_gate_requirements_not_met"
    return {
        **base,
        "verdict": verdict,
        "s3_authorized": verdict == "corpus_wide_trust_win",
        "genuine_corpus_wide_bounded_null": verdict == "genuine_corpus_wide_bounded_null",
        "coverage_trustworthy": verdict
        in {
            "corpus_wide_trust_win",
            "genuine_corpus_wide_bounded_null",
        },
        "degenerate_candidate_pool_flagged": False,
        "reason": "oracle_not_moat" if oracle else reason,
        "metrics_imported": verdict
        in {
            "corpus_wide_trust_win",
            "genuine_corpus_wide_bounded_null",
        },
        "energy_selected_offpath_cell_recall": _float(
            artifact.get("energy_selected_offpath_cell_recall")
        ),
        "accuracy_gate_selected_offpath_cell_recall": _float(
            artifact.get("accuracy_gate_selected_offpath_cell_recall")
        ),
        "ci_excludes_zero": _ci_lower_positive(ci95),
        "ci_includes_zero": _ci_includes_zero(ci95),
        "pivot": (
            "authorize_s3"
            if verdict == "corpus_wide_trust_win"
            else "pivot_energy_to_s3_generation"
        ),
    }


def _levelup_bank(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    registry_update = _mapping(artifact.get("registry_update"))
    before = _int(registry_update.get("reproducible_total_levels_before"))
    after = _int(registry_update.get("reproducible_total_levels_after"), before)
    oracle = _oracle_declared(artifact)
    return {
        "source": "LEVELUP",
        "experiment_id": _experiment_id("LEVELUP", artifact),
        "target_game": artifact.get("target_game"),
        "new_levels_banked": _int(artifact.get("new_levels_banked")),
        "reproducible_total_levels_before": before,
        "reproducible_total_levels_after": after,
        "reproducible_total_levels_delta": after - before,
        "registry_updated": registry_update.get("updated") is True,
        "offline_reproduced": artifact.get("offline_reproduced") is True,
        "reproduced_levels": _int(artifact.get("reproduced_levels")),
        "verifier_is_oracle": oracle,
        "moat_claim": bool(after > before and not oracle),
    }


def _self_play_checkpoint(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    refreshed = artifact.get("verifier_checkpoint_refreshed") is True
    return {
        "source": "SELF_PLAY",
        "experiment_id": _experiment_id("SELF_PLAY", artifact),
        "decision": "checkpoint_refreshed" if refreshed else "checkpoint_not_refreshed",
        "verifier_checkpoint_refreshed": refreshed,
        "target_game": artifact.get("target_game"),
        "self_play_residual": artifact.get("self_play_residual"),
        "offline_reproduced": artifact.get("offline_reproduced") is True,
        "reproduced_levels": _int(artifact.get("reproduced_levels")),
        "solve_provenance": artifact.get("solve_provenance", ""),
        "inference_substrate": artifact.get("inference_substrate", ""),
    }


def _heldout_readiness(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    baseline_delta = _float(artifact.get("heldout_first_win_delta_vs_baseline"))
    prior_delta = _float(artifact.get("heldout_first_win_delta_vs_prior_best"))
    flat = baseline_delta == 0.0 and prior_delta == 0.0
    return {
        "source": "HELDOUT",
        "experiment_id": _experiment_id("HELDOUT", artifact),
        "decision": "flat_null_no_readiness_gain" if flat else "heldout_readiness_changed",
        "heldout_first_win_rate": _float(artifact.get("heldout_first_win_rate")),
        "first_win_baseline": _float(artifact.get("first_win_baseline")),
        "prior_best_heldout_first_win_rate": _float(
            artifact.get("prior_best_heldout_first_win_rate")
        ),
        "heldout_first_win_delta_vs_baseline": baseline_delta,
        "heldout_first_win_delta_vs_prior_best": prior_delta,
        "heldout_variant_attempts": _int(artifact.get("heldout_variant_attempts")),
        "parity_test_green": artifact.get("parity_test_green") is True,
        "positive_control_passed": artifact.get("positive_control_passed") is True,
        "null_delta_methodology_note_present": bool(artifact.get("null_delta_methodology_note")),
        "inference_substrate": artifact.get("inference_substrate", ""),
    }


def _silent_bug_audit(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    silent_bugs = artifact.get("silent_bugs_found")
    silent_bugs = silent_bugs if isinstance(silent_bugs, list) else []
    trusted = artifact.get("trusted_nulls")
    return {
        "source": "BUG_AUDIT",
        "experiment_id": _experiment_id("BUG_AUDIT", artifact),
        "nulls_audited": _int(artifact.get("nulls_audited")),
        "trusted_nulls": trusted if isinstance(trusted, list) else [],
        "silent_bugs_found_count": len(silent_bugs),
        "s2v3_reopened": any(
            isinstance(row, Mapping)
            and row.get("null_id")
            == "experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate"
            and row.get("verdict") == "silent_bug_must_reopen"
            for row in silent_bugs
        ),
    }


def _submission_package_state(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    ready = artifact.get("submission_package_ready") is True
    submitted = artifact.get("submitted_to_leaderboard") is True
    return {
        "source": "PACKAGE",
        "experiment_id": _experiment_id("PACKAGE", artifact),
        "decision": "package_ready_operator_only"
        if ready and not submitted
        else "package_not_ready",
        "submission_package_ready": ready,
        "submitted_to_leaderboard": submitted,
        "operator_only": artifact.get("operator_only") is True,
        "vram_estimate_gb": _float(artifact.get("vram_estimate_gb")),
    }


def _sota_handoff(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    flagged = artifact.get("flagged_for_v444")
    flagged = flagged if isinstance(flagged, list) else []
    methods = artifact.get("methods_mapped")
    methods = methods if isinstance(methods, list) else []
    return {
        "source": "SOTA",
        "experiment_id": _experiment_id("SOTA", artifact),
        "decision": "sota_handoff_mapped" if methods else "sota_handoff_empty",
        "flagged_for_v444_candidates": [
            row.get("candidate") for row in flagged if isinstance(row, Mapping)
        ],
        "methods_mapped_count": len(methods),
        "s3_context": dict(_mapping(artifact.get("s3_context"))),
        "arxiv_ids_cited": artifact.get("arxiv_ids_cited")
        if isinstance(artifact.get("arxiv_ids_cited"), list)
        else [],
    }


def _readiness(
    s2v3_verdict: Mapping[str, Any],
    heldout: Mapping[str, Any],
    package: Mapping[str, Any],
) -> JsonDict:
    heldout_changed = heldout.get("decision") == "heldout_readiness_changed"
    package_ready = package.get("decision") == "package_ready_operator_only"
    s2v3_win = s2v3_verdict.get("verdict") == "corpus_wide_trust_win"
    bounded = s2v3_verdict.get("verdict") == "genuine_corpus_wide_bounded_null"
    return {
        "s2v3_verdict": s2v3_verdict.get("verdict", ""),
        "heldout_decision": heldout.get("decision", ""),
        "submission_package_decision": package.get("decision", ""),
        "s3_authorized": s2v3_win,
        "pivot_energy_to_s3_generation": bounded,
        "ready_for_operator_submit": bool(s2v3_win and heldout_changed and package_ready),
        "reason": "requires_s2v3_win_clean_package_and_heldout_gain",
    }


def build_artifact(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    registry: Mapping[str, Any],
    registry_sha256: str | None,
    summarizer_results: Mapping[str, SummarizerResult],
    corpus_game_count: int | None,
    duration_s: float,
    preconditions_checked: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the complete .443 capstone after live summaries have run."""

    s2v3 = _s2v3_verdict(
        artifacts.get("S2V3"),
        summarizer_results.get("S2V3"),
        corpus_game_count=corpus_game_count,
    )
    honest_verdict = {
        "corpus_wide_trust_win": (
            "success_s2v3_corpus_wide_energy_ranking_beats_accuracy_s3_authorized"
        ),
        "genuine_corpus_wide_bounded_null": (
            "complete_s2v3_genuine_corpus_wide_bounded_null_pivot_to_s3_generation"
        ),
        "inconclusive_insufficient_corpus_diversity": (
            "complete_s2v3_inconclusive_insufficient_corpus_diversity_capstone_ready"
        ),
    }.get(str(s2v3.get("verdict")), "complete_s2v3_inconclusive_capstone_ready")
    heldout = _heldout_readiness(artifacts.get("HELDOUT"))
    package = _submission_package_state(artifacts.get("PACKAGE"))
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": honest_verdict,
        "s2v3_structural_energy_verdict": s2v3,
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "cited_upstream_artifacts": _cited_artifacts(
            artifacts,
            artifact_sha256,
            summarizer_results,
            corpus_game_count=corpus_game_count,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "levelup_bank": _levelup_bank(artifacts.get("LEVELUP")),
        "self_play_checkpoint": _self_play_checkpoint(artifacts.get("SELF_PLAY")),
        "readiness": _readiness(s2v3, heldout, package),
        "heldout_readiness": heldout,
        "silent_bug_audit": _silent_bug_audit(artifacts.get("BUG_AUDIT")),
        "submission_package_state": package,
        "sota_handoff": _sota_handoff(artifacts.get("SOTA")),
        "upstream_oracle_declarations": _oracle_declarations(artifacts),
        "flagged_artifacts_skipped": _flagged_artifacts_skipped(
            artifacts, artifact_sha256, summarizer_results
        ),
        "preconditions_checked": dict(
            preconditions_checked
            or {
                "agents_md_read": True,
                "codex_md_read": True,
                "registry": {
                    "path": REGISTRY_RELATIVE_PATH,
                    "sha256": registry_sha256 or "",
                },
                "offline_corpus": {"game_count": corpus_game_count},
            }
        ),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(max(0.0001, float(duration_s)), 6),
        "random_seed": RANDOM_SEED,
        "capstone_ready": True,
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def build_blocked_artifact(
    *,
    reason: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    registry: Mapping[str, Any],
    summarizer_results: Mapping[str, SummarizerResult],
    duration_s: float,
    preconditions_checked: Mapping[str, Any],
) -> JsonDict:
    """Build a schema-valid blocked artifact without importing metric claims."""

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": f"blocked_{reason}",
        "s2v3_structural_energy_verdict": {},
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "cited_upstream_artifacts": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "levelup_bank": {},
        "self_play_checkpoint": {},
        "readiness": {},
        "heldout_readiness": {},
        "silent_bug_audit": {},
        "submission_package_state": {},
        "sota_handoff": {},
        "upstream_oracle_declarations": _oracle_declarations(artifacts),
        "flagged_artifacts_skipped": _flagged_artifacts_skipped(
            artifacts, artifact_sha256, summarizer_results
        ),
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(max(0.0001, float(duration_s)), 6),
        "random_seed": RANDOM_SEED,
        "capstone_ready": False,
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema errors for the scorecard without mutating it."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_field:{field}")
    if not str(payload.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    if not isinstance(payload.get("reproducible_total_levels"), int):
        errors.append("invalid_reproducible_total_levels")
    field_principles = payload.get("field_principles")
    for field, principle in FIELD_PRINCIPLES.items():
        if not isinstance(field_principles, Mapping) or field_principles.get(field) != principle:
            errors.append(f"missing_principle:{field}")
    s2v3 = payload.get("s2v3_structural_energy_verdict")
    if isinstance(s2v3, Mapping) and s2v3:
        if s2v3.get("verdict") not in {
            "corpus_wide_trust_win",
            "genuine_corpus_wide_bounded_null",
            "inconclusive_insufficient_corpus_diversity",
        }:
            errors.append("invalid_s2v3_verdict")
    cited = payload.get("cited_upstream_artifacts")
    if not isinstance(cited, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("experiment_id"), int)
        or not isinstance(row.get("fields_imported"), list)
        or not str(row.get("sha256", "")).startswith("sha256:")
        for row in cited
    ):
        errors.append("invalid_cited_upstream_artifacts")
    if not str(payload.get("reproducibility_checksum", "")).startswith("sha256:"):
        errors.append("invalid_reproducibility_checksum")
    return errors


def _run_summarizer(root: Path, relative_path: str) -> SummarizerResult:  # pragma: no cover
    cmd = [sys.executable, SUMMARIZER_RELATIVE_PATH, relative_path]
    proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True, check=False)
    return SummarizerResult(
        command=cmd,
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _first_blocker(
    *,
    summarizer_present: bool,
    registry_present: bool,
    registry_loadable: bool,
    spec_has_req: bool,
    upstream_preconditions: Mapping[str, Mapping[str, Any]],
) -> str | None:
    if not summarizer_present:
        return "missing_summarizer"
    if not registry_present:
        return "missing_registry"
    if not registry_loadable:
        return "registry_not_yaml_loadable"
    if not spec_has_req:
        return "spec_missing_req_4819"
    for source, info in upstream_preconditions.items():
        if info.get("present") is not True:
            return f"missing_upstream:{source}"
    return None


def run_capstone(
    *,
    root: Path = REPO_ROOT,
    summarizer: Summarizer | None = None,
) -> JsonDict:
    """Read upstreams via the summarizer, aggregate, and write the scorecard."""

    start = time.perf_counter()
    summarizer = summarizer or _run_summarizer
    summarizer_path = root / SUMMARIZER_RELATIVE_PATH
    summarizer_present = summarizer_path.exists()
    artifacts: dict[str, JsonDict] = {}
    artifact_sha256: dict[str, str] = {}
    summarizer_results: dict[str, SummarizerResult] = {}
    upstream_preconditions: dict[str, JsonDict] = {}

    for source, spec in UPSTREAM_SOURCES.items():
        path = root / spec.relative_path
        present = path.exists()
        upstream_preconditions[source] = {"path": spec.relative_path, "present": present}
        if not present:
            continue
        if summarizer_present:
            summary = summarizer(root, spec.relative_path)
            summarizer_results[source] = summary
            upstream_preconditions[source]["summarizer_exit_code"] = summary.exit_code
        artifacts[source] = _read_json_object(path)
        artifact_sha256[source] = file_sha256(path) or ""

    registry_path = root / REGISTRY_RELATIVE_PATH
    registry_present = registry_path.exists()
    registry_loadable = False
    registry: JsonDict = {}
    if registry_present:
        try:
            registry = _read_yaml_object(registry_path)
            registry_loadable = True
        except yaml.YAMLError:
            registry = {}

    spec_path = root / SPEC_RELATIVE_PATH
    spec_has_req = spec_path.exists() and "REQ-CAPSTONE-4819" in spec_path.read_text(
        encoding="utf-8"
    )
    corpus_game_count = _offline_corpus_game_count(root)
    preconditions_checked = {
        "agents_md_read": True,
        "codex_md_read": True,
        "summarizer": {
            "path": SUMMARIZER_RELATIVE_PATH,
            "present": summarizer_present,
        },
        "registry": {
            "path": REGISTRY_RELATIVE_PATH,
            "present": registry_present,
            "yaml_loadable": registry_loadable,
            "sha256": file_sha256(registry_path) or "",
        },
        "offline_corpus": {
            "path": "environment_files",
            "game_count": corpus_game_count,
        },
        "spec_has_req_4819": spec_has_req,
        "upstream_artifacts": upstream_preconditions,
    }
    duration_s = time.perf_counter() - start
    blocker = _first_blocker(
        summarizer_present=summarizer_present,
        registry_present=registry_present,
        registry_loadable=registry_loadable,
        spec_has_req=spec_has_req,
        upstream_preconditions=upstream_preconditions,
    )
    if blocker is not None:
        artifact = build_blocked_artifact(
            reason=blocker,
            artifacts=artifacts,
            artifact_sha256=artifact_sha256,
            registry=registry,
            summarizer_results=summarizer_results,
            duration_s=duration_s,
            preconditions_checked=preconditions_checked,
        )
    else:
        artifact = build_artifact(
            artifacts=artifacts,
            artifact_sha256=artifact_sha256,
            registry=registry,
            registry_sha256=file_sha256(registry_path),
            summarizer_results=summarizer_results,
            corpus_game_count=corpus_game_count,
            duration_s=duration_s,
            preconditions_checked=preconditions_checked,
        )
    result_path = root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run_capstone()
    errors = validate_artifact(artifact)
    print(
        json.dumps(
            {
                "result_path": RESULT_RELATIVE_PATH,
                "honest_verdict": artifact.get("honest_verdict"),
                "schema_errors": errors,
            },
            sort_keys=True,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
