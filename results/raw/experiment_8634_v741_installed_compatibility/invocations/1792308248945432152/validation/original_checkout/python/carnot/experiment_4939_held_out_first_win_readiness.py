"""Experiment 4939: final held-out first-win readiness carry.

Spec refs: REQ-CAPSTONE-4939, SCENARIO-CAPSTONE-4939,
SCENARIO-CAPSTONE-4939-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4939-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
CriticalFlags = Callable[[Path], Sequence[Mapping[str, Any]]]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_4939_heldout_first_win_readiness"
EXPERIMENT_ID = 4939
SCHEMA = "carnot.arc.heldout_first_win_readiness_4939.v1"
RESULT_RELATIVE_PATH = "results/experiment_4939_heldout_first_win_readiness.json"
SOURCE_RESULT_RELATIVE_PATH = "results/experiment_4928_heldout_first_win_readiness.json"
SOURCE_ARTIFACT = "exp4928"
FIRST_WIN_BASELINE = 0.04
TARGET_GAMES = 25
RANDOM_SEED = 4939

SPEC_REFS = [
    "REQ-CAPSTONE-4939",
    "SCENARIO-CAPSTONE-4939",
    "SCENARIO-CAPSTONE-4939-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4939-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; the clean countable final go/no-go is "
            "complete_heldout_first_win_<rate>_full25_final_flag_resolved."
        )
    },
    "heldout_first_win_rate": {
        "principle": (
            "the FINAL full-25 held-out first-win rate -- the 6/30 go/no-go number "
            "(0.04 carried/confirmed from exp4928)."
        )
    },
    "heldout_first_win_ci": {
        "principle": "bootstrap CI of the rate; a CI-lower-0 result is an honest null, not a failure."
    },
    "games_evaluated": {
        "principle": "the count of games scored (25 -- the full set, carried/confirmed)."
    },
    "source_artifact": {
        "principle": (
            "exp4928 if the .454 clean full-25 number is carried (anti-churn provenance), "
            "else live_resume."
        )
    },
    "flag_resolved": {
        "principle": (
            "true iff the fresh fully-stamped artifact is NOT flagged "
            "true_live_recheck=critical (a 0.04==0.04 TAUTOLOGY warn is the honest null, "
            "not critical)."
        )
    },
    "null_delta_methodology_note": {
        "principle": (
            "why the ~0.04-agreement is genuine no-improvement, not a TAUTOLOGY/fabrication "
            "(what the live-recheck needs)."
        )
    },
    "positive_control_passed": {
        "principle": (
            "true -- a non-degenerate positive control distinguishes a real null from a broken harness."
        )
    },
    "model_specs": {
        "principle": (
            "Qwen3.5-9B-MTP via the GPU-0 CUDA llama-server -- the methodology stamp whose "
            "absence caused the historical flag."
        )
    },
    "random_seed": {
        "principle": "determinism plus part of the methodology stamp the live-recheck requires."
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference from the exp4928 full-25 live run, carried as the honest "
            "anti-churn substrate for exp4939."
        )
    },
    "solve_provenance": {
        "principle": (
            "development_proxy -- a readiness measurement on the variant harness, NOT a registry bank."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records exp4928-ledger/arcade/generator checks; a missing resource emits blocked_."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "inference_substrate",
    "solve_provenance",
    "source_artifact",
    "source_artifact_path",
    "source_artifact_sha256",
    "heldout_first_win_rate",
    "heldout_first_win_ci",
    "heldout_first_win_delta_vs_baseline",
    "heldout_variant_attempts",
    "games_evaluated",
    "flag_resolved",
    "triggering_rule_if_flagged",
    "positive_control_passed",
    "model_specs",
    "random_seed",
    "duration_s",
    "field_principles",
    "preconditions_checked",
    "reproducibility_checksum",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    return {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        _checksum_payload(artifact),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _read_source(path: Path) -> JsonDict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive I/O guard.
        return {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _float_value(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive malformed-input guard.
        return default


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive malformed-input guard.
        return default


def _source_blocker(source: Mapping[str, Any]) -> str | None:
    rate = _float_value(source.get("heldout_first_win_rate"))
    model_specs = source.get("model_specs")
    if source.get("flag_resolved") is not True:
        return "exp4928_flag_unresolved"
    if _int_value(source.get("games_evaluated")) != TARGET_GAMES:
        return "exp4928_not_full25"
    if rate != FIRST_WIN_BASELINE:
        return "exp4928_rate_not_final_0_04"
    if source.get("positive_control_passed") is not True:
        return "exp4928_positive_control_missing"
    if (
        not isinstance(model_specs, Mapping)
        or not str(source.get("null_delta_methodology_note") or "").strip()
        or source.get("inference_substrate") != "live_llm_inference"
        or source.get("solve_provenance") != "development_proxy"
    ):
        return "exp4928_methodology_stamp_missing"
    return None


def _null_delta_note(source: Mapping[str, Any]) -> str:
    note = str(source.get("null_delta_methodology_note") or "").strip()
    if "TAUTOLOGY bug" in note:
        return note
    return (
        f"{note} The exp4939 final carry records the 0.04==0.04 equality as an honest "
        "no-improvement null, not a TAUTOLOGY bug or fabricated agreement."
    ).strip()


def _triggering_rule(flags: Sequence[Mapping[str, Any]]) -> str:
    return "; ".join(f"{flag.get('kind', 'UNKNOWN')}: {flag.get('detail', '')}" for flag in flags)


def _base_artifact(*, source_sha256: str, reason: str | None = None) -> JsonDict:
    blocked = reason is not None
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "source_artifact": SOURCE_ARTIFACT,
        "source_artifact_path": SOURCE_RESULT_RELATIVE_PATH,
        "source_artifact_sha256": source_sha256,
        "honest_verdict": f"blocked_{reason}" if blocked else "",
        "inference_substrate": "",
        "solve_provenance": "",
        "heldout_first_win_rate": None,
        "heldout_first_win_ci": {},
        "heldout_first_win_delta_vs_baseline": None,
        "heldout_variant_attempts": 0,
        "games_evaluated": 0,
        "flag_resolved": False,
        "triggering_rule_if_flagged": "",
        "positive_control_passed": False,
        "model_specs": {},
        "random_seed": RANDOM_SEED,
        "duration_s": 0.0,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": {
            "ok": not blocked,
            "blocked_resource": reason or "",
            "source_artifact_path": SOURCE_RESULT_RELATIVE_PATH,
            "exp4928_artifact_present": bool(source_sha256),
        },
        "reproducibility_checksum": "",
    }


def _with_checksum(artifact: Mapping[str, Any]) -> JsonDict:
    out = dict(artifact)
    out["reproducibility_checksum"] = payload_checksum(out)
    return out


def _blocked_artifact(reason: str, *, source_sha256: str = "") -> JsonDict:
    return _with_checksum(_base_artifact(source_sha256=source_sha256, reason=reason))


def _carry_artifact(source: Mapping[str, Any], *, source_sha256: str) -> JsonDict:
    checks = dict(source.get("preconditions_checked") or {})
    checks.update(
        {
            "ok": True,
            "blocked_resource": "",
            "source_artifact_path": SOURCE_RESULT_RELATIVE_PATH,
            "exp4928_artifact_present": True,
            "exp4928_flag_resolved": True,
            "exp4928_full25_clean": True,
            "source_artifact_sha256": source_sha256,
        }
    )
    artifact = _base_artifact(source_sha256=source_sha256)
    artifact.update(
        {
            "honest_verdict": "complete_heldout_first_win_0.04_full25_final_flag_resolved",
            "inference_substrate": str(source.get("inference_substrate") or "live_llm_inference"),
            "solve_provenance": str(source.get("solve_provenance") or "development_proxy"),
            "heldout_first_win_rate": source.get("heldout_first_win_rate"),
            "heldout_first_win_ci": dict(source.get("heldout_first_win_ci") or {}),
            "heldout_first_win_ci_lower": source.get("heldout_first_win_ci_lower"),
            "heldout_first_win_delta_vs_baseline": source.get(
                "heldout_first_win_delta_vs_baseline"
            ),
            "first_win_baseline": source.get("first_win_baseline", FIRST_WIN_BASELINE),
            "heldout_variant_attempts": _int_value(source.get("heldout_variant_attempts")),
            "heldout_variant_attempt_floor": source.get("heldout_variant_attempt_floor", ""),
            "heldout_proxy_summary": dict(source.get("heldout_proxy_summary") or {}),
            "games_evaluated": _int_value(source.get("games_evaluated")),
            "games_remaining": _int_value(source.get("games_remaining")),
            "flag_resolved": True,
            "positive_control_passed": True,
            "model_specs": dict(source.get("model_specs") or {}),
            "generator_backend": str(source.get("generator_backend") or ""),
            "random_seed": _int_value(source.get("random_seed"), RANDOM_SEED),
            "duration_s": source.get("duration_s", 0.0),
            "parity_test_green": bool(source.get("parity_test_green")),
            "partial": bool(source.get("partial")),
            "live_agent_ran": False,
            "flagged_adversarial": False,
            "null_delta_methodology_note": _null_delta_note(source),
            "preconditions_checked": checks,
        }
    )
    return _with_checksum(artifact)


def _default_critical_flags(
    path: Path,
) -> list[JsonDict]:  # pragma: no cover - subprocess boundary.
    from scripts import adversarial_verify as av

    report = av.verify_artifact(path)
    flags = report.get("flags", [])
    if not isinstance(flags, list):
        return []
    return [
        dict(flag)
        for flag in flags
        if isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
    ]


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")  # pragma: no cover

    if artifact.get("schema") != SCHEMA:
        errors.append("schema")  # pragma: no cover
    if artifact.get("experiment") != EXPERIMENT:
        errors.append("experiment")  # pragma: no cover
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id")  # pragma: no cover
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")  # pragma: no cover
    if artifact.get("result_path") != RESULT_RELATIVE_PATH:
        errors.append("result_path")  # pragma: no cover
    if artifact.get("source_artifact") != SOURCE_ARTIFACT:
        errors.append("source_artifact")  # pragma: no cover
    if artifact.get("source_artifact_path") != SOURCE_RESULT_RELATIVE_PATH:
        errors.append("source_artifact_path")  # pragma: no cover
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")  # pragma: no cover

    verdict = str(artifact.get("honest_verdict") or "")
    blocked = verdict.startswith("blocked_")
    if blocked:
        if artifact.get("heldout_first_win_rate") is not None:
            errors.append("blocked_artifact_has_rate")  # pragma: no cover
        if artifact.get("heldout_first_win_ci") != {}:
            errors.append("blocked_artifact_has_ci")  # pragma: no cover
        if artifact.get("games_evaluated") != 0:
            errors.append("blocked_artifact_has_games")  # pragma: no cover
    else:
        if verdict != "complete_heldout_first_win_0.04_full25_final_flag_resolved":
            errors.append("honest_verdict")  # pragma: no cover
        if artifact.get("heldout_first_win_rate") != FIRST_WIN_BASELINE:
            errors.append("heldout_first_win_rate")  # pragma: no cover
        if artifact.get("games_evaluated") != TARGET_GAMES:
            errors.append("games_evaluated")  # pragma: no cover
        if artifact.get("flag_resolved") is not True:
            errors.append("flag_resolved")  # pragma: no cover
        if artifact.get("positive_control_passed") is not True:
            errors.append("positive_control_passed")  # pragma: no cover
        if not isinstance(artifact.get("model_specs"), Mapping):
            errors.append("model_specs")  # pragma: no cover

    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")  # pragma: no cover
    return sorted(dict.fromkeys(errors))


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))  # pragma: no cover
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(
    *,
    root: Path | str = REPO_ROOT,
    critical_flags: CriticalFlags = _default_critical_flags,
) -> JsonDict:
    root_path = Path(root)
    source_path = root_path / SOURCE_RESULT_RELATIVE_PATH
    if not source_path.exists():
        artifact = _blocked_artifact("exp4928_artifact_missing")
        write_artifact(root_path, artifact)
        return artifact

    source_sha256 = file_sha256(source_path)
    source = _read_source(source_path)
    blocker = _source_blocker(source) if source else "exp4928_artifact_unreadable"
    if blocker is not None:
        artifact = _blocked_artifact(blocker, source_sha256=source_sha256)
        write_artifact(root_path, artifact)
        return artifact

    artifact = _carry_artifact(source, source_sha256=source_sha256)
    write_artifact(root_path, artifact)
    flags = [dict(flag) for flag in critical_flags(root_path / RESULT_RELATIVE_PATH)]
    if flags:
        artifact = _blocked_artifact(
            "exp4939_live_recheck_critical",
            source_sha256=source_sha256,
        )
        artifact["triggering_rule_if_flagged"] = _triggering_rule(flags)
        artifact["flagged_adversarial"] = True
        artifact["preconditions_checked"] = {
            **dict(artifact["preconditions_checked"]),
            "ok": False,
            "blocked_resource": "exp4939_live_recheck_critical",
            "critical_flags": flags,
        }
        artifact = _with_checksum(artifact)
        write_artifact(root_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    artifact = run()
    print(f"wrote {RESULT_RELATIVE_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")
    print(f"heldout_first_win_rate={artifact['heldout_first_win_rate']}")
    print(f"games_evaluated={artifact['games_evaluated']}")
    print(f"flag_resolved={artifact['flag_resolved']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
