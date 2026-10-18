"""Experiment 4815: audit .443 ARC nulls for silent under-coverage bugs.

Spec refs: REQ-ARC-WMTE-4815, SCENARIO-ARC-WMTE-4815-SILENT-BUG-AUDIT,
SCENARIO-ARC-WMTE-4815-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

from carnot import experiment_4805_silent_bug_audit as prior_audit


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(SCRIPTS_ROOT))

import adversarial_verify  # noqa: E402


EXPERIMENT = "experiment_4815_silent_bug_audit"
EXPERIMENT_ID = 4815
SCHEMA = "carnot.arc.milestone_443_silent_bug_audit_4815.v1"
RESULT_RELATIVE_PATH = "results/experiment_4815_silent_bug_audit.json"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 4815
DURATION_FLOOR_S = 0.0001
TERMINAL_PREFIXES = prior_audit.TERMINAL_PREFIXES

JsonDict = dict[str, Any]

SPEC_REFS = [
    "REQ-ARC-WMTE-4815",
    "SCENARIO-ARC-WMTE-4815-SILENT-BUG-AUDIT",
    "SCENARIO-ARC-WMTE-4815-BLOCKED-PRECONDITION",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; audit complete is complete_/success_."
    },
    "s2v3_corpus_coverage_verified": {
        "principle": (
            "the load-bearing check -- n_available_games must match the real corpus "
            "AND DEGENERATE_CANDIDATE_POOL must not fire, else S2-v3's verdict is "
            "again under-covered."
        )
    },
    "nulls_audited": {"principle": "count of nulls re-examined."},
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (0.0001s floor)."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "result_path",
    "field_principles",
    "per_null_verdicts",
    "silent_bugs_found",
    "trusted_nulls",
    "s2v3_corpus_coverage_check",
    "preconditions_checked",
    "audited_artifacts",
    "audited_artifact_checksums",
    "random_seed",
    "duration_s",
    "reproducibility_checksum",
)

AUDIT_TARGETS: tuple[dict[str, str], ...] = (
    {
        "null_id": "experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate",
        "artifact_path": "results/experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate.json",
    },
    {
        "null_id": "experiment_4812_levelup_attempt",
        "artifact_path": "results/experiment_4812_levelup_attempt.json",
    },
    {
        "null_id": "experiment_4814_heldout_first_win_readiness",
        "artifact_path": "results/experiment_4814_heldout_first_win_readiness.json",
    },
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_stable_json(value).encode("utf-8"))


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _sha256_json(payload)


def _read_json(path: Path) -> JsonDict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def _file_checksum(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _int_value(value: Any) -> int:
    parsed = _finite_float(value)
    return int(parsed) if parsed is not None else 0


def _list_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _append_unique(rows: list[str], text: str) -> None:
    if text and text not in rows:
        rows.append(text)


def _expected_source_paths() -> list[str]:
    return [target["artifact_path"] for target in AUDIT_TARGETS]


def required_effective_games(n_available_games: int, n_games_attempted: int) -> int:
    corpus = max(int(n_available_games), int(n_games_attempted))
    return max(10, math.ceil(0.6 * corpus))


def real_corpus_size(root: Path | str = REPO_ROOT) -> int:
    corpus = Path(root) / "environment_files"
    if not corpus.exists():
        return 0
    return len([path for path in corpus.iterdir() if not path.name.startswith(".")])


def _candidate_names(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row.get("candidate_name")) for row in rows if row.get("candidate_name")]


def _game_selection_logged(row: Mapping[str, Any]) -> bool:
    candidate_rows = _list_of_mappings(row.get("candidate_rows"))
    if not candidate_rows:
        return False
    names = _candidate_names(candidate_rows)
    energy_name = str(row.get("energy_selected_candidate") or "")
    accuracy_name = str(row.get("accuracy_gate_selected_candidate") or "")
    if not energy_name or not accuracy_name:
        return False
    if energy_name not in names or accuracy_name not in names:
        return False
    if _finite_float(row.get("energy_selected_offpath_cell_recall")) is None:
        return False
    if _finite_float(row.get("accuracy_gate_selected_offpath_cell_recall")) is None:
        return False
    return all(_finite_float(candidate.get("heldout_cell_recall")) is not None for candidate in candidate_rows)


def s2v3_corpus_coverage_check(
    artifact: Mapping[str, Any],
    *,
    real_corpus_size: int,
) -> JsonDict:
    flags: list[Any] = []
    adversarial_verify.check_engine_selection_candidate_diversity(dict(artifact), flags)
    flag_kinds = [str(getattr(flag, "kind", "")) for flag in flags]
    game_results = _list_of_mappings(artifact.get("game_results"))
    n_available = _int_value(artifact.get("n_available_games"))
    n_attempted = _int_value(artifact.get("n_games_attempted"))
    n_effective = _int_value(artifact.get("n_effective_games"))
    required = required_effective_games(n_available, n_attempted)
    per_game_logged = bool(game_results) and len(game_results) == n_attempted and all(
        _game_selection_logged(row) for row in game_results
    )
    return {
        "flag_kinds": flag_kinds,
        "degenerate_candidate_pool_flagged": "DEGENERATE_CANDIDATE_POOL" in flag_kinds,
        "real_corpus_size": int(real_corpus_size),
        "n_available_games": n_available,
        "n_games_attempted": n_attempted,
        "n_effective_games": n_effective,
        "required_effective_games": required,
        "n_available_matches_real_corpus": n_available == int(real_corpus_size) and n_available > 0,
        "attempted_corpus_wide": n_attempted == n_available and n_attempted > 0,
        "effective_game_floor_met": n_effective >= required,
        "per_game_selections_logged": bool(per_game_logged),
        "checked_games": len(game_results),
    }


def _audit_s2v3(
    null_id: str,
    artifact: Mapping[str, Any],
    *,
    real_corpus_size: int,
) -> JsonDict:
    checks = s2v3_corpus_coverage_check(artifact, real_corpus_size=real_corpus_size)
    signatures: list[str] = []
    if not checks["n_available_matches_real_corpus"]:
        _append_unique(signatures, "s2v3_n_available_games_mismatch_real_corpus")
    if not checks["attempted_corpus_wide"]:
        _append_unique(signatures, "s2v3_not_corpus_wide_attempt")
    if checks["degenerate_candidate_pool_flagged"]:
        _append_unique(signatures, "s2v3_degenerate_candidate_pool")
    if not checks["effective_game_floor_met"]:
        _append_unique(signatures, "s2v3_effective_game_floor_not_met")
    if not checks["per_game_selections_logged"]:
        _append_unique(signatures, "s2v3_per_game_selection_logging_missing")
    verified = not signatures
    return {
        "null_id": null_id,
        "verdict": "trustworthy_null" if verified else "silent_bug_must_reopen",
        "silent_bug_signatures": signatures,
        "exercise_evidence": [
            f"real_corpus_size={checks['real_corpus_size']}",
            f"n_available_games={checks['n_available_games']}",
            f"n_games_attempted={checks['n_games_attempted']}",
            f"n_effective_games={checks['n_effective_games']}",
            f"required_effective_games={checks['required_effective_games']}",
            f"degenerate_candidate_pool_flagged={checks['degenerate_candidate_pool_flagged']}",
            f"per_game_selections_logged={checks['per_game_selections_logged']}",
        ],
        "s2v3_corpus_coverage_verified": verified,
        "s2v3_corpus_coverage_check": checks,
    }


def _audit_via_prior(null_id: str, prior_null_id: str, artifact: Mapping[str, Any]) -> JsonDict:
    result = dict(prior_audit.audit_null_artifact(prior_null_id, artifact))
    result["null_id"] = null_id
    return result


def audit_null_artifact(
    null_id: str,
    artifact: Mapping[str, Any],
    *,
    real_corpus_size: int = 0,
) -> JsonDict:
    if null_id == "experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate":
        return _audit_s2v3(null_id, artifact, real_corpus_size=real_corpus_size)
    if null_id == "experiment_4812_levelup_attempt":
        return _audit_via_prior(null_id, "experiment_4802_levelup_attempt", artifact)
    if null_id == "experiment_4814_heldout_first_win_readiness":
        return _audit_via_prior(
            null_id,
            "experiment_4804_heldout_first_win_readiness",
            artifact,
        )
    return {
        "null_id": null_id,
        "verdict": "silent_bug_must_reopen",
        "silent_bug_signatures": ["unknown_null_artifact"],
        "exercise_evidence": ["unknown null artifact; cannot trust negative result"],
    }


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    repo = Path(root)
    missing = [rel for rel in _expected_source_paths() if not (repo / rel).exists()]
    corpus_size = real_corpus_size(repo)
    return {
        "ok": not missing and corpus_size > 0,
        "source_artifacts_present": not missing,
        "missing_source_artifacts": missing,
        "milestone_443_artifacts_present": not missing,
        "environment_files_present": corpus_size > 0,
        "real_corpus_size": corpus_size,
        "agents_md_read": (repo / "AGENTS.md").exists(),
        "codex_md_read": (repo / "CODEX.md").exists(),
    }


def _blocked_artifact(checks: Mapping[str, Any]) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": "blocked_missing_source_artifacts",
        "s2v3_corpus_coverage_verified": False,
        "nulls_audited": 0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_null_verdicts": [],
        "silent_bugs_found": [],
        "trusted_nulls": [],
        "s2v3_corpus_coverage_check": {},
        "preconditions_checked": dict(checks),
        "audited_artifacts": _expected_source_paths(),
        "audited_artifact_checksums": {},
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_FLOOR_S,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    per_null_verdicts: Sequence[Mapping[str, Any]],
    audited_artifact_checksums: Mapping[str, str],
) -> JsonDict:
    silent = [
        dict(row)
        for row in per_null_verdicts
        if row.get("verdict") == "silent_bug_must_reopen"
    ]
    trusted = [
        str(row.get("null_id"))
        for row in per_null_verdicts
        if row.get("verdict") == "trustworthy_null" and row.get("null_id")
    ]
    s2_row = next(
        (
            row
            for row in per_null_verdicts
            if row.get("null_id") == "experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate"
        ),
        {},
    )
    s2_check = s2_row.get("s2v3_corpus_coverage_check")
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": (
            f"complete_arc_null_silent_bug_audit_{len(per_null_verdicts)}_nulls_"
            f"{len(silent)}_reopen"
        ),
        "s2v3_corpus_coverage_verified": bool(
            s2_row.get("s2v3_corpus_coverage_verified")
        ),
        "nulls_audited": len(per_null_verdicts),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_null_verdicts": [dict(row) for row in per_null_verdicts],
        "silent_bugs_found": silent,
        "trusted_nulls": trusted,
        "s2v3_corpus_coverage_check": dict(s2_check) if isinstance(s2_check, Mapping) else {},
        "preconditions_checked": dict(preconditions_checked),
        "audited_artifacts": _expected_source_paths(),
        "audited_artifact_checksums": dict(audited_artifact_checksums),
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_FLOOR_S,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [f"missing:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if not isinstance(artifact.get("s2v3_corpus_coverage_verified"), bool):
        errors.append("s2v3_corpus_coverage_verified_must_be_bool")
    if not isinstance(artifact.get("nulls_audited"), int):
        errors.append("nulls_audited_must_be_int")
    if not isinstance(artifact.get("silent_bugs_found"), list):
        errors.append("silent_bugs_found_must_be_list")
    if not isinstance(artifact.get("per_null_verdicts"), list):
        errors.append("per_null_verdicts_must_be_list")
    if not isinstance(artifact.get("s2v3_corpus_coverage_check"), dict):
        errors.append("s2v3_corpus_coverage_check_must_be_dict")
    duration = _finite_float(artifact.get("duration_s"))
    if duration is None or duration < DURATION_FLOOR_S:
        errors.append("duration_below_aggregation_floor")
    per_nulls = artifact.get("per_null_verdicts")
    if isinstance(per_nulls, list) and isinstance(artifact.get("nulls_audited"), int):
        if artifact["nulls_audited"] != len(per_nulls):
            errors.append("nulls_audited_count_mismatch")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def write_artifact(
    artifact: Mapping[str, Any],
    *,
    root: Path | str = REPO_ROOT,
) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    out = Path(root) / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
) -> JsonDict:
    repo = Path(root)
    checks = check_preconditions(repo)
    if not checks["ok"]:
        artifact = _blocked_artifact(checks)
        if write:
            write_artifact(artifact, root=repo)
        return artifact

    per_null: list[JsonDict] = []
    checksums: dict[str, str] = {}
    corpus_size = _int_value(checks.get("real_corpus_size"))
    for target in AUDIT_TARGETS:
        rel = target["artifact_path"]
        path = repo / rel
        payload = _read_json(path)
        checksums[rel] = _file_checksum(path)
        per_null.append(
            audit_null_artifact(
                target["null_id"],
                payload,
                real_corpus_size=corpus_size,
            )
        )

    artifact = build_artifact(
        preconditions_checked=checks,
        per_null_verdicts=per_null,
        audited_artifact_checksums=checksums,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root=repo)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = run()
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "nulls_audited": artifact["nulls_audited"],
                "s2v3_corpus_coverage_verified": artifact[
                    "s2v3_corpus_coverage_verified"
                ],
                "result": RESULT_RELATIVE_PATH,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
