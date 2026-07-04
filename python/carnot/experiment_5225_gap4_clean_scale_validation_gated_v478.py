"""Exp 5225: clean GAP-4 validation over the canonical pool.

Spec refs: REQ-REPORT-5225, SCENARIO-REPORT-5225-CLEAN-NULL,
SCENARIO-REPORT-5225-BLOCKED-OR-EXCLUDED.

This module makes the validation decision from Exp 5224's canonical rows
without inventing new labels. It re-runs the canonical row linter, drops
row-level flagged or protocol-empty records, then applies the unchanged GAP-4
paired exact-test rule to the remaining vote-vs-gated pass@2 fields.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_5197_gap4_scaleup_real_checkpoint_v476 as exp5197
from carnot import experiment_5223_gap4_flagged_pool_authenticity_audit_v478 as exp5223
from carnot import experiment_5224_gap4_canonical_pool_builder_v478 as exp5224


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5225_gap4_clean_scale_validation_gated_v478"
EXPERIMENT_ID = 5225
SCHEMA = "carnot.gap4_clean_scale_validation_5225.v1"
RESULT_RELATIVE_PATH = "results/experiment_5225_gap4_clean_scale_validation_gated_v478.json"
EXP5224_RELATIVE_PATH = exp5224.RESULT_RELATIVE_PATH
CANONICAL_POOL_MIN_N = exp5224.CANONICAL_POOL_TARGET_N
RANDOM_SEED = 5225
INFERENCE_SUBSTRATE = "deterministic_validation_over_canonical_pool"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
EFFECT_DIRECTIONS = {"positive", "null", "negative", "blocked"}
SPEC_REFS = [
    "REQ-REPORT-5225",
    "SCENARIO-REPORT-5225-CLEAN-NULL",
    "SCENARIO-REPORT-5225-BLOCKED-OR-EXCLUDED",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "gap4_clean_validation_complete": {
        "principle": (
            "True only when Exp 5224 gates pass, at least one canonical row is scored, "
            "row-level exclusions are applied before metrics, and adversarial "
            "verification of the Exp 5225 artifact passes."
        )
    },
    "n_scored": {
        "principle": (
            "Rows scored after canonical schema lint, protocol pass@2 presence, and "
            "row-level adversarial exclusion."
        )
    },
    "wins": {
        "principle": (
            "Discordant canonical rows where gated pass@2 succeeds and vote pass@2 "
            "does not, under the unchanged GAP-4 protocol."
        )
    },
    "losses": {
        "principle": (
            "Discordant canonical rows where vote pass@2 succeeds and gated pass@2 "
            "does not, under the unchanged GAP-4 protocol."
        )
    },
    "ties": {
        "principle": "Canonical scored rows where vote and gated pass@2 agree; ties are real rows, not missing data."
    },
    "exact_test_p_value": {
        "principle": "Two-sided Exp 5197 binomial/sign-test p-value over discordant canonical rows."
    },
    "exact_test_passes_min6_rule": {
        "principle": "The unchanged GAP-4 floor: at least six wins, zero losses, and exact p < 0.05."
    },
    "effect_direction": {
        "principle": "Single clean decision label: positive, null, negative, or blocked."
    },
    "canonical_pool_path": {
        "principle": "Path to the exact canonical pool artifact read for validation."
    },
    "adversarial_verify_passed": {
        "principle": "True only after scripts/adversarial_verify.py reports no flags for the Exp 5225 artifact."
    },
    "inference_substrate": {
        "principle": "Must be deterministic_validation_over_canonical_pool; no fresh LLM inference is invoked."
    },
    "honest_verdict": {
        "principle": (
            "Must start with complete:/complete_/success:/success_ and state the clean "
            "GAP-4 validation decision."
        )
    },
}

REQUIRED_FIELDS = (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "result_path",
    "gap4_clean_validation_complete",
    "n_scored",
    "wins",
    "losses",
    "ties",
    "exact_test_p_value",
    "exact_test_passes_min6_rule",
    "effect_direction",
    "canonical_pool_path",
    "adversarial_verify_passed",
    "inference_substrate",
    "honest_verdict",
    "random_seed",
    "reproducibility_checksum",
    "field_principles",
    "duration_s",
    "tests_run",
    "precondition_errors",
    "excluded_rows",
    "exclusion_summary",
    "excluded_row_examples",
    "scored_rows",
    "canonical_pool_n",
    "schema_linter_passed",
    "scoring_protocol",
    "source_artifacts",
    "adversarial_verify_summary",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


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


def load_canonical_pool(root: Path | str = REPO_ROOT) -> JsonDict:
    payload = _read_json(Path(root) / EXP5224_RELATIVE_PATH)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _candidate_rows(pool: Mapping[str, Any]) -> list[JsonDict]:
    rows = pool.get("candidate_rows")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def precondition_errors(canonical_pool: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if canonical_pool.get("gap4_canonical_pool_usable") is not True:
        errors.append("exp5224_gate_not_usable")
    canonical_n = canonical_pool.get("canonical_pool_n")
    if isinstance(canonical_n, bool) or not isinstance(canonical_n, int):
        errors.append("canonical_pool_n_not_int")
    elif canonical_n < CANONICAL_POOL_MIN_N:
        errors.append("canonical_pool_n_below_120")
    rows = _candidate_rows(canonical_pool)
    if not rows:
        errors.append("canonical_pool_rows_missing")
    elif isinstance(canonical_n, int) and not isinstance(canonical_n, bool) and len(rows) < canonical_n:
        errors.append("candidate_rows_below_canonical_pool_n")
    return sorted(dict.fromkeys(errors))


def _row_flag_reason(row: Mapping[str, Any]) -> str | None:
    if row.get("flagged_adversarial") is True or row.get("flagged") is True:
        return "row_flagged_adversarial"
    flags = row.get("adversarial_flags")
    if isinstance(flags, list) and flags:
        return "row_adversarial_flags"
    corrigendum = row.get("corrigendum_pending")
    if isinstance(corrigendum, list) and corrigendum:
        return "row_corrigendum_pending"
    return None


def _protocol_reason(row: Mapping[str, Any]) -> str | None:
    fields = row.get("pass_at_2_fields")
    if not isinstance(fields, Mapping):
        return "protocol_empty_pass2_fields"
    if not isinstance(fields.get("vote_top2"), bool) or not isinstance(
        fields.get("gated_top2"), bool
    ):
        return "protocol_empty_pass2_fields"
    if not _nonempty_string(fields.get("scoring_protocol")):
        return "protocol_empty_pass2_fields"
    return None


def row_exclusion_reason(row: Mapping[str, Any]) -> str | None:
    protocol_reason = _protocol_reason(row)
    if protocol_reason is not None:
        return protocol_reason
    schema_errors = exp5223.canonical_candidate_record_errors(row)
    if schema_errors:
        return "schema:" + schema_errors[0]
    flag_reason = _row_flag_reason(row)
    if flag_reason is not None:
        return flag_reason
    return None


def _scored_row(row: Mapping[str, Any], index: int) -> JsonDict:
    pass2 = row["pass_at_2_fields"]
    protocol = row.get("decoding_protocol")
    pass_fields_mode = (
        protocol.get("pass_fields_mode")
        if isinstance(protocol, Mapping) and protocol.get("pass_fields_mode") is not None
        else "unspecified"
    )
    candidate_id = str(row.get("candidate_id") or f"row_{index}")
    source_task_id = str(row.get("source_task_id") or candidate_id)
    return {
        "candidate_id": candidate_id,
        "source_task_id": source_task_id,
        "pilot_key": candidate_id,
        "cluster_id": source_task_id,
        "vote_top2": pass2.get("vote_top2") is True,
        "gated_top2": pass2.get("gated_top2") is True,
        "pass_at_2_scoring_protocol": str(pass2.get("scoring_protocol")),
        "pass_fields_mode": str(pass_fields_mode),
    }


def score_canonical_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], list[JsonDict]]:
    scored: list[JsonDict] = []
    excluded: list[JsonDict] = []
    for index, row in enumerate(rows):
        reason = row_exclusion_reason(row)
        if reason is not None:
            excluded.append(
                {
                    "index": index,
                    "candidate_id": row.get("candidate_id"),
                    "source_task_id": row.get("source_task_id"),
                    "reason": reason,
                }
            )
            continue
        scored.append(_scored_row(row, index))
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


def _effect_direction(
    *,
    stats: Mapping[str, Any],
    n_scored: int,
    precondition_errors: Sequence[str],
    adversarial_verify_passed: bool,
) -> str:
    if precondition_errors or n_scored <= 0 or not adversarial_verify_passed:
        return "blocked"
    if stats.get("passes_min6_rule") is True:
        return "positive"
    if int(stats.get("losses") or 0) > int(stats.get("wins") or 0):
        return "negative"
    return "null"


def _complete(effect_direction: str) -> bool:
    return effect_direction != "blocked"


def _verdict(effect_direction: str, n_scored: int, stats: Mapping[str, Any]) -> str:
    if effect_direction == "positive":
        return (
            f"success: clean GAP-4 validation positive decision with n={n_scored}, "
            f"wins={stats['wins']}, losses={stats['losses']}; min-six rule crossed"
        )
    if effect_direction == "negative":
        return (
            f"complete: clean GAP-4 validation negative decision with n={n_scored}, "
            f"wins={stats['wins']}, losses={stats['losses']}; min-six rule not crossed"
        )
    if effect_direction == "null":
        return (
            f"complete: clean GAP-4 validation null decision with n={n_scored}, "
            f"wins={stats['wins']}, losses={stats['losses']}, ties={stats['ties']}; "
            "min-six rule not crossed"
        )
    return (
        f"complete: clean GAP-4 validation blocked with n={n_scored}; "
        "precondition, row validity, or adversarial verification blocked decision"
    )


def build_artifact(
    *,
    canonical_pool: Mapping[str, Any],
    canonical_pool_path: str,
    scored_rows: Sequence[Mapping[str, Any]],
    exclusions: Sequence[Mapping[str, Any]],
    precondition_errors: Sequence[str],
    duration_s: float,
    tests_run: Sequence[str],
    adversarial_verify_passed: bool,
    adversarial_verify_summary: Mapping[str, Any] | None = None,
) -> JsonDict:
    rows = [dict(row) for row in scored_rows]
    excluded = [dict(row) for row in exclusions]
    stats = _stats(rows)
    effect = _effect_direction(
        stats=stats,
        n_scored=len(rows),
        precondition_errors=precondition_errors,
        adversarial_verify_passed=adversarial_verify_passed,
    )
    schema_linter_passed = not any(
        str(item.get("reason", "")).startswith(("schema:", "protocol_")) for item in excluded
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "gap4_clean_validation_complete": _complete(effect),
        "n_scored": len(rows),
        "wins": int(stats["wins"]),
        "losses": int(stats["losses"]),
        "ties": int(stats["ties"]),
        "exact_test_p_value": float(stats["p_value_two_sided"]),
        "exact_test_passes_min6_rule": bool(stats["passes_min6_rule"]),
        "effect_direction": effect,
        "canonical_pool_path": canonical_pool_path,
        "adversarial_verify_passed": bool(adversarial_verify_passed),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _verdict(effect, len(rows), stats),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": max(0.0, round(float(duration_s), 6)),
        "tests_run": list(tests_run),
        "precondition_errors": list(precondition_errors),
        "excluded_rows": len(excluded),
        "exclusion_summary": dict(Counter(str(row.get("reason")) for row in excluded)),
        "excluded_row_examples": excluded[:10],
        "scored_rows": rows,
        "canonical_pool_n": canonical_pool.get("canonical_pool_n"),
        "schema_linter_passed": schema_linter_passed,
        "scoring_protocol": {
            "lineage": ["experiment_5161", "experiment_5177", "experiment_5197"],
            "exact_test": "experiment_5197.exact_test scipy.stats.binomtest",
            "method_changed_mid_test": False,
            "row_labels_source": "canonical_pool_pass_at_2_fields",
            "row_payload_minimized_for_adversarial_verify": True,
        },
        "source_artifacts": [
            {
                "path": canonical_pool_path,
                "experiment": canonical_pool.get("experiment"),
                "experiment_id": canonical_pool.get("experiment_id"),
                "gap4_canonical_pool_usable": canonical_pool.get("gap4_canonical_pool_usable"),
                "canonical_pool_n": canonical_pool.get("canonical_pool_n"),
            }
        ],
        "adversarial_verify_summary": dict(adversarial_verify_summary or {}),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")

    rows = [dict(row) for row in artifact.get("scored_rows", []) if isinstance(row, Mapping)]
    stats = _stats(rows)
    if artifact.get("gap4_clean_validation_complete") is not _complete(
        str(artifact.get("effect_direction"))
    ):
        errors.append("gap4_clean_validation_complete")
    if not _is_int(artifact.get("n_scored")) or artifact.get("n_scored") != len(rows):
        errors.append("n_scored")
    for field in ("wins", "losses", "ties"):
        if not _is_int(artifact.get(field)) or artifact.get(field) != stats[field]:
            errors.append(field)
    p_value = artifact.get("exact_test_p_value")
    if (
        isinstance(p_value, bool)
        or not isinstance(p_value, int | float)
        or float(p_value) != float(stats["p_value_two_sided"])
    ):
        errors.append("exact_test_p_value")
    if artifact.get("exact_test_passes_min6_rule") is not bool(stats["passes_min6_rule"]):
        errors.append("exact_test_passes_min6_rule")
    expected_effect = _effect_direction(
        stats=stats,
        n_scored=len(rows),
        precondition_errors=[
            str(item) for item in artifact.get("precondition_errors", []) if isinstance(item, str)
        ],
        adversarial_verify_passed=artifact.get("adversarial_verify_passed") is True,
    )
    if (
        artifact.get("effect_direction") not in EFFECT_DIRECTIONS
        or artifact.get("effect_direction") != expected_effect
    ):
        errors.append("effect_direction")
    if not _nonempty_string(artifact.get("canonical_pool_path")):
        errors.append("canonical_pool_path")
    if not isinstance(artifact.get("adversarial_verify_passed"), bool):
        errors.append("adversarial_verify_passed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if str(artifact.get("effect_direction")) not in verdict:
        errors.append("honest_verdict_decision")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    if not isinstance(artifact.get("tests_run"), list):
        errors.append("tests_run")
    if not isinstance(artifact.get("precondition_errors"), list):
        errors.append("precondition_errors")
    if not _is_int(artifact.get("excluded_rows")) or artifact.get("excluded_rows") < len(
        artifact.get("excluded_row_examples", [])
    ):
        errors.append("excluded_rows")
    if not isinstance(artifact.get("schema_linter_passed"), bool):
        errors.append("schema_linter_passed")
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


def run_adversarial_verify(path: Path) -> JsonDict:  # pragma: no cover - subprocess wrapper.
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "adversarial_verify.py"), str(path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    return {
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _adversarial_passed(summary: Mapping[str, Any]) -> bool:
    if summary.get("passed") is True or summary.get("returncode") == 0:
        return True
    reports = summary.get("reports")
    return isinstance(reports, list) and all(
        isinstance(report, Mapping) and int(report.get("flag_count") or 0) == 0
        for report in reports
    )


def run(
    *,
    root: Path | str = REPO_ROOT,
    canonical_pool_loader: Callable[[Path | str], JsonDict] = load_canonical_pool,
    adversarial_verify_runner: Callable[[Path], Mapping[str, Any]] = run_adversarial_verify,
    tests_run: Sequence[str] = (),
    now: Callable[[], float] = time.time,
) -> JsonDict:
    root_path = Path(root)
    started = float(now())
    canonical_pool = canonical_pool_loader(root_path)
    rows = _candidate_rows(canonical_pool)
    preconditions = precondition_errors(canonical_pool)
    scored, exclusions = score_canonical_rows(rows) if not preconditions else ([], [])
    duration_s = float(now()) - started

    provisional = build_artifact(
        canonical_pool=canonical_pool,
        canonical_pool_path=EXP5224_RELATIVE_PATH,
        scored_rows=scored,
        exclusions=exclusions,
        precondition_errors=preconditions,
        duration_s=duration_s,
        tests_run=tests_run,
        adversarial_verify_passed=True,
    )
    path = write_artifact(root_path, provisional)
    adversarial = dict(adversarial_verify_runner(path))
    passed = _adversarial_passed(adversarial)
    final = build_artifact(
        canonical_pool=canonical_pool,
        canonical_pool_path=EXP5224_RELATIVE_PATH,
        scored_rows=scored,
        exclusions=exclusions,
        precondition_errors=preconditions,
        duration_s=duration_s,
        tests_run=tests_run,
        adversarial_verify_passed=passed,
        adversarial_verify_summary=adversarial,
    )
    write_artifact(root_path, final)
    return final


def main() -> int:  # pragma: no cover
    artifact = run()
    print(f"wrote {RESULT_RELATIVE_PATH}")
    print(artifact["honest_verdict"])
    print(f"n_scored={artifact['n_scored']}")
    print(f"wins={artifact['wins']} losses={artifact['losses']} ties={artifact['ties']}")
    print(f"exact_test_p_value={artifact['exact_test_p_value']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
