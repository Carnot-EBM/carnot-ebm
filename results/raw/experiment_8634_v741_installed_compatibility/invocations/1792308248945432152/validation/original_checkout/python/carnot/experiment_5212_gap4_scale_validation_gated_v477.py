"""Exp 5212: GAP-4 expanded-pool scale validation.

Spec refs: REQ-REPORT-5212, SCENARIO-REPORT-5212,
SCENARIO-REPORT-5212-BLOCKED-PROTOCOL-METADATA.

This module validates Exp 5211's expanded candidate pool against the unchanged
GAP-4 scoring protocol from Exp 5161/5177/5197. Rows that do not already expose
the registered vote-vs-gated pass@2 labels are excluded instead of being
relabeled after the fact.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5177_gap4_scaleup_decentralization_tier_v474 as exp5177
from carnot import experiment_5197_gap4_scaleup_real_checkpoint_v476 as exp5197
from carnot import experiment_5211_gap4_sota_local_candidate_expansion_v477 as exp5211


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5212_gap4_scale_validation_gated_v477"
EXPERIMENT_ID = 5212
SCHEMA = "carnot.gap4_scale_validation_gated_5212.v1"
RESULT_RELATIVE_PATH = "results/experiment_5212_gap4_scale_validation_gated_v477.json"
EXP5211_RELATIVE_PATH = exp5211.RESULT_RELATIVE_PATH
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "blocked_")
STATUS_RECOMMENDATIONS = {
    "filled",
    "scale_up_recommended",
    "retire_local_generation_path",
    "blocked",
}
SPEC_REFS = [
    "REQ-REPORT-5212",
    "SCENARIO-REPORT-5212",
    "SCENARIO-REPORT-5212-BLOCKED-PROTOCOL-METADATA",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "n_scored": {
        "principle": (
            "Rows scored only after Exp 5211 feasibility/leakage metadata and the "
            "established pass@2 protocol labels are present."
        )
    },
    "exact_test_discordant_wins": {
        "principle": (
            "Actual gated-over-vote pass@2 wins under the unchanged Exp 5161/5177/5197 "
            "protocol."
        )
    },
    "exact_test_discordant_losses": {
        "principle": (
            "Actual vote-over-gated pass@2 losses under the unchanged Exp 5161/5177/5197 "
            "protocol."
        )
    },
    "exact_test_p_value_two_sided": {
        "principle": "Two-sided scipy.stats.binomtest p-value over discordant pass@2 pairs."
    },
    "exact_test_passes_min6_rule": {
        "principle": (
            "The GAP-4 floor remains at least six discordant wins, zero discordant losses, "
            "and p < 0.05."
        )
    },
    "cluster_bootstrap_delta_ci95": {
        "principle": (
            "Cluster bootstrap CI over gated pass@2 minus vote pass@2, using the existing "
            "Exp 5177 bootstrap implementation."
        )
    },
    "gap4_status_recommendation": {
        "principle": (
            "filled only if the exact min-6 rule passes; blocked or scale-up/retirement is "
            "reported otherwise."
        )
    },
    "excluded_rows": {
        "principle": (
            "Rows excluded before statistics because they are malformed, leak-prone, or lack "
            "unchanged-protocol scoring labels."
        )
    },
    "tests_run": {
        "principle": "Commands actually run for this artifact, with pass/fail status."
    },
    "inference_substrate": {
        "principle": "Must be verifier_ensemble_against_cached_candidates."
    },
    "honest_verdict": {
        "principle": (
            "Must start with complete:/complete_/success:/success_ and must not claim the "
            "floor crossed unless exact_test_passes_min6_rule is true."
        )
    },
}

REQUIRED_FIELDS = (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "result_path",
    "exp5211_result_path",
    "exp5211_candidate_pool_n",
    "exp5211_gap4_expansion_usable",
    "n_scored",
    "exact_test_discordant_wins",
    "exact_test_discordant_losses",
    "exact_test_p_value_two_sided",
    "exact_test_passes_min6_rule",
    "cluster_bootstrap_delta_ci95",
    "gap4_status_recommendation",
    "excluded_rows",
    "tests_run",
    "inference_substrate",
    "honest_verdict",
    "failure_mode",
    "exclusion_summary",
    "excluded_row_examples",
    "arc1_slice_result",
    "arc2_heldout_slice_result",
    "scoring_protocol",
    "scored_rows",
    "source_artifacts",
    "field_principles",
    "duration_s",
    "reproducibility_checksum",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _principled(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]["principle"]}


def _wrapped_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = json.loads(json.dumps(dict(artifact), sort_keys=True, default=str))
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_exp5211_artifact(root: Path | str = REPO_ROOT) -> JsonDict:
    payload = _read_json(Path(root) / EXP5211_RELATIVE_PATH)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _leakage_or_metadata_reason(
    row: Mapping[str, Any],
    *,
    exp5197_task_ids: set[str],
) -> str | None:
    if row.get("accepted") is not True:
        return "not_accepted"
    if row.get("demo_perfect") is not True:
        return "demo_perfect_missing_or_false"
    if row.get("output_shape_matches") is not True:
        return "output_shape_missing_or_false"
    if row.get("guard_status") != "accepted":
        return "guard_status_not_accepted"
    code = row.get("code")
    if not isinstance(code, str) or not code.strip():
        return "missing_code"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "code_parse_error"
    lowered_code = code.lower()
    if any(token in lowered_code for token in exp5211._LEAKAGE_TOKENS):
        return "leakage_token_in_code"
    if exp5211._forbidden_ast_reason(tree) is not None:
        return "forbidden_ast"
    raw = json.dumps(dict(row), sort_keys=True, default=str).lower()
    if "test_output" in raw or "target_hash" in raw:
        return "leakage_metadata"
    if str(row.get("task_id")) in exp5197_task_ids:
        return "exp5197_task_leakage"
    return None


def _protocol_label_reason(row: Mapping[str, Any]) -> str | None:
    if not isinstance(row.get("vote_top2"), bool) or not isinstance(row.get("gated_top2"), bool):
        return "missing_protocol_pass2_fields"
    return None


def _row_domain(row: Mapping[str, Any]) -> str:
    domain = str(row.get("domain") or "")
    if domain in {"arc1", "arc2", "heldout", "arc2_heldout"}:
        return "arc2" if domain in {"heldout", "arc2_heldout"} else domain
    text = f"{row.get('source', '')}:{row.get('task_id', '')}".lower()
    if "arc2" in text or "heldout" in text:
        return "arc2"
    if "arc1" in text:
        return "arc1"
    return "unlabeled"


def _scoring_row(row: Mapping[str, Any], index: int) -> JsonDict:
    task = str(row.get("task") or row.get("task_id") or f"row_{index}")
    domain = _row_domain(row)
    return {
        **dict(row),
        "pilot_key": str(row.get("pilot_key") or f"{domain}:{index}:{task}"),
        "cluster_id": str(row.get("cluster_id") or f"{domain}:{task}"),
        "domain": domain,
        "task": task,
        "vote_top2": row.get("vote_top2") is True,
        "gated_top2": row.get("gated_top2") is True,
    }


def score_expanded_pool(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    exp5197_task_ids: set[str],
) -> tuple[list[JsonDict], list[JsonDict]]:
    scored: list[JsonDict] = []
    excluded: list[JsonDict] = []
    for index, row in enumerate(candidate_rows):
        metadata_reason = _leakage_or_metadata_reason(row, exp5197_task_ids=exp5197_task_ids)
        protocol_reason = metadata_reason or _protocol_label_reason(row)
        if protocol_reason is not None:
            excluded.append(
                {
                    "index": index,
                    "task_id": row.get("task_id") or row.get("task"),
                    "reason": protocol_reason,
                }
            )
            continue
        scored.append(_scoring_row(row, index))
    return scored, excluded


def _stats(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    if not rows:
        return {
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "p_value_two_sided": 1.0,
            "passes_min6_rule": False,
        }
    return exp5197.exact_test(rows)


def _bootstrap_ci(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    if not rows:
        return [0.0, 0.0]
    ci = exp5177.cluster_bootstrap_delta_ci(rows, seed=exp5197.RANDOM_SEED)
    return [0.0, 0.0] if ci is None else ci


def _slice_result(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    if not rows:
        return {"labels_available": False, "n_entries": 0, "reason": "no_labeled_rows"}
    return {"labels_available": True, **exp5177._slice_result(rows)}


def _recommendation(stats: Mapping[str, Any], *, n_scored: int) -> str:
    if n_scored <= 0:
        return "blocked"
    if stats.get("passes_min6_rule") is True:
        return "filled"
    if int(stats.get("losses") or 0) > int(stats.get("wins") or 0):
        return "retire_local_generation_path"
    return "scale_up_recommended"


def _failure_mode(
    *,
    stats: Mapping[str, Any],
    n_scored: int,
    exp5211_usable: bool,
    exclusions: Sequence[Mapping[str, Any]],
) -> str:
    if not exp5211_usable:
        return "exp5211_pool_not_usable"
    if n_scored == 0 and exclusions:
        return str(Counter(str(row.get("reason")) for row in exclusions).most_common(1)[0][0])
    if n_scored == 0:
        return "no_scored_rows"
    if int(stats.get("losses") or 0) > 0:
        return "discordant_losses_present"
    if int(stats.get("wins") or 0) < 6:
        return "wins_below_min6"
    if float(stats.get("p_value_two_sided") or 1.0) >= 0.05:  # pragma: no cover
        return "p_value_not_below_0.05"
    return "floor_crossed"


def _verdict(*, n_scored: int, recommendation: str, failure_mode: str) -> str:
    if recommendation == "filled":
        return f"success_gap4_scale_validation_v477_n{n_scored}_floor_crossed_filled"
    return f"complete_gap4_scale_validation_v477_n{n_scored}_{failure_mode}_{recommendation}"


def build_artifact(
    *,
    scored_rows: Sequence[Mapping[str, Any]],
    exclusions: Sequence[Mapping[str, Any]],
    exp5211_candidate_pool_n: int,
    exp5211_gap4_expansion_usable: bool,
    source_artifacts: Sequence[Mapping[str, Any]],
    duration_s: float,
    tests_run: Sequence[str],
) -> JsonDict:
    rows = [dict(row) for row in scored_rows]
    excluded = [dict(row) for row in exclusions]
    stats = _stats(rows)
    recommendation = _recommendation(stats, n_scored=len(rows))
    failure_mode = _failure_mode(
        stats=stats,
        n_scored=len(rows),
        exp5211_usable=bool(exp5211_gap4_expansion_usable),
        exclusions=excluded,
    )
    summary = dict(Counter(str(row.get("reason")) for row in excluded))
    arc1_rows = [row for row in rows if row.get("domain") == "arc1"]
    arc2_rows = [row for row in rows if row.get("domain") == "arc2"]
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "exp5211_result_path": EXP5211_RELATIVE_PATH,
        "exp5211_candidate_pool_n": int(exp5211_candidate_pool_n),
        "exp5211_gap4_expansion_usable": bool(exp5211_gap4_expansion_usable),
        "n_scored": _principled("n_scored", len(rows)),
        "exact_test_discordant_wins": _principled("exact_test_discordant_wins", stats["wins"]),
        "exact_test_discordant_losses": _principled(
            "exact_test_discordant_losses", stats["losses"]
        ),
        "exact_test_p_value_two_sided": _principled(
            "exact_test_p_value_two_sided", stats["p_value_two_sided"]
        ),
        "exact_test_passes_min6_rule": _principled(
            "exact_test_passes_min6_rule", stats["passes_min6_rule"]
        ),
        "cluster_bootstrap_delta_ci95": _principled("cluster_bootstrap_delta_ci95", _bootstrap_ci(rows)),
        "gap4_status_recommendation": _principled("gap4_status_recommendation", recommendation),
        "excluded_rows": _principled("excluded_rows", len(excluded)),
        "tests_run": _principled("tests_run", list(tests_run)),
        "inference_substrate": _principled("inference_substrate", INFERENCE_SUBSTRATE),
        "honest_verdict": _verdict(
            n_scored=len(rows),
            recommendation=recommendation,
            failure_mode=failure_mode,
        ),
        "failure_mode": failure_mode,
        "exclusion_summary": summary,
        "excluded_row_examples": excluded[:10],
        "arc1_slice_result": _slice_result(arc1_rows),
        "arc2_heldout_slice_result": _slice_result(arc2_rows),
        "scoring_protocol": {
            "lineage": ["experiment_5161", "experiment_5177", "experiment_5197"],
            "exact_test": "experiment_5197.exact_test scipy.stats.binomtest",
            "bootstrap": "experiment_5177.cluster_bootstrap_delta_ci",
            "method_changed_mid_test": False,
        },
        "scored_rows": rows,
        "source_artifacts": [dict(row) for row in source_artifacts],
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": max(0.0, round(float(duration_s), 6)),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _wrapped_int(value: Any) -> int | None:
    value = _wrapped_value(value)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _wrapped_bool(value: Any) -> bool | None:
    value = _wrapped_value(value)
    return value if isinstance(value, bool) else None


def _wrapped_number(value: Any) -> float | None:
    value = _wrapped_value(value)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")

    rows = [dict(row) for row in artifact.get("scored_rows", []) if isinstance(row, Mapping)]
    stats = _stats(rows)
    n_scored = _wrapped_int(artifact.get("n_scored"))
    if n_scored != len(rows):
        errors.append("n_scored")
    if _wrapped_int(artifact.get("exact_test_discordant_wins")) != stats["wins"]:
        errors.append("exact_test_discordant_wins")
    if _wrapped_int(artifact.get("exact_test_discordant_losses")) != stats["losses"]:
        errors.append("exact_test_discordant_losses")
    p_value = _wrapped_number(artifact.get("exact_test_p_value_two_sided"))
    if p_value is None or p_value != float(stats["p_value_two_sided"]):
        errors.append("exact_test_p_value_two_sided")
    if _wrapped_bool(artifact.get("exact_test_passes_min6_rule")) is not bool(
        stats["passes_min6_rule"]
    ):
        errors.append("exact_test_passes_min6_rule")

    ci = _wrapped_value(artifact.get("cluster_bootstrap_delta_ci95"))
    if (
        not isinstance(ci, list)
        or len(ci) != 2
        or any(isinstance(value, bool) or not isinstance(value, int | float) for value in ci)
    ):
        errors.append("cluster_bootstrap_delta_ci95")

    recommendation = _wrapped_value(artifact.get("gap4_status_recommendation"))
    expected_recommendation = _recommendation(stats, n_scored=len(rows))
    if recommendation not in STATUS_RECOMMENDATIONS or recommendation != expected_recommendation:
        errors.append("gap4_status_recommendation")
    excluded_rows = _wrapped_int(artifact.get("excluded_rows"))
    if excluded_rows is None or excluded_rows < len(artifact.get("excluded_row_examples", [])):
        errors.append("excluded_rows")
    if not isinstance(_wrapped_value(artifact.get("tests_run")), list):
        errors.append("tests_run")
    if _wrapped_value(artifact.get("inference_substrate")) != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")

    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if stats["passes_min6_rule"] is not True and (
        "floor_crossed" in verdict or recommendation == "filled"
    ):
        errors.append("honest_verdict_floor_overclaim")

    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return sorted(dict.fromkeys(errors))


def write_artifact(root: Path | str, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    _write_json_atomic(path, artifact)
    return path


def describe_source_artifacts(root: Path | str = REPO_ROOT) -> list[JsonDict]:
    path = Path(root) / EXP5211_RELATIVE_PATH
    return [{"path": EXP5211_RELATIVE_PATH, "exists": path.exists()}]


def run(
    *,
    root: Path | str = REPO_ROOT,
    exp5211_loader: Callable[[Path | str], JsonDict] = load_exp5211_artifact,
    exp5197_task_loader: Callable[[Path | str], set[str]] = exp5211.load_exp5197_scored_task_ids,
    source_artifact_loader: Callable[[Path | str], list[JsonDict]] = describe_source_artifacts,
    tests_run: Sequence[str] = (),
    now: Callable[[], float] = time.time,
) -> JsonDict:
    root_path = Path(root)
    started = float(now())
    exp5211_payload = exp5211_loader(root_path)
    rows = exp5211_payload.get("candidate_rows")
    candidate_rows = [dict(row) for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []
    exp5197_task_ids = exp5197_task_loader(root_path)
    scored, exclusions = score_expanded_pool(candidate_rows, exp5197_task_ids=exp5197_task_ids)
    artifact = build_artifact(
        scored_rows=scored,
        exclusions=exclusions,
        exp5211_candidate_pool_n=int(exp5211_payload.get("candidate_pool_n") or 0),
        exp5211_gap4_expansion_usable=(
            exp5211_payload.get("gap4_expansion_usable") is True
            and int(exp5211_payload.get("candidate_pool_n") or 0) >= exp5211.CANDIDATE_POOL_TARGET_N
        ),
        source_artifacts=source_artifact_loader(root_path),
        duration_s=float(now()) - started,
        tests_run=tests_run,
    )
    write_artifact(root_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run()
    print(f"wrote {RESULT_RELATIVE_PATH}")
    print(artifact["honest_verdict"])
    print(f"n_scored={artifact['n_scored']['value']}")
    print(f"excluded_rows={artifact['excluded_rows']['value']}")
    print(f"gap4_status_recommendation={artifact['gap4_status_recommendation']['value']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
