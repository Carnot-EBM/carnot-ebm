"""Exp 1341 HalluGuard-inspired certificate failure split replay.

Spec: REQ-VERIFY-1341,
      SCENARIO-VERIFY-1341
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping


DEFAULT_RUN_DATE = "20260505"
DEFAULT_OUTPUT_PATH = Path("results/experiment_1341_halluguard_certificate_failure_split.json")
DEFAULT_EXP1323_PATH = Path(
    "results/experiment_1323_sota_gguf_token_health_prompt_runtime_diagnostic.json"
)
DEFAULT_EXP1324_PATH = Path(
    "results/experiment_1324_certificate_failure_taxonomy_formalizer_reality_check.json"
)
DEFAULT_EXP1340_PATH = Path(
    "results/experiment_1340_trigger_before_constrain_certificate_v6_sota.json"
)
ARTIFACT_NAME = "experiment_1341_halluguard_certificate_failure_split"
SCHEMA_VERSION = 1
EXP1340_LIMITATION = "exp1340_absent_or_unreadable_fallback_to_exp1324"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "source_cases_available",
    "data_driven_risk_proxy",
    "reasoning_driven_risk_proxy",
    "parser_schema_risk_count",
    "undergeneration_risk_count",
    "semantic_invalidity_count",
    "unknown_mishandling_count",
    "repair_policy_by_failure_type",
    "universal_detector_claim_allowed",
    "honest_verdict",
)

WriteObserver = Callable[[Path, dict[str, Any]], None]


def build_halluguard_certificate_failure_split_artifact(
    *,
    exp1323_artifact: Mapping[str, Any],
    exp1324_artifact: Mapping[str, Any],
    exp1340_artifact: Mapping[str, Any] | None,
    exp1340_limitation: str | None = None,
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Build the replay-only split from already-written certificate evidence.

    HalluGuard separates failures that look like data/context problems from
    failures that look like reasoning instability. Exp 1324 does not contain
    model-internal HalluGuard scores, so this artifact reports bounded proxies
    and leaves parser/runtime/schema failures as separate engineering risks.
    """
    failure_counts = _failure_mode_counts(exp1324_artifact)
    parser_schema_count = _int(exp1324_artifact.get("parser_failure_count"))
    undergeneration_count = _int(exp1324_artifact.get("undergeneration_failure_count"))
    semantic_count = _int(exp1324_artifact.get("semantic_failure_count"))
    unknown_count = _int(exp1324_artifact.get("unknown_state_mishandling_count"))
    leakage_count = _int(exp1324_artifact.get("possible_hardcoded_solution_leakage_count"))
    solver_disagreement_count = failure_counts.get("solver_disagreement", 0)
    reasoning_failure_types = {
        "semantic_invalidity": semantic_count,
        "solver_disagreement": solver_disagreement_count,
        "unknown_state_mishandling": unknown_count,
    }
    limitations = [exp1340_limitation] if exp1340_limitation else []

    artifact = _base_artifact(
        project_root=Path(project_root),
        run_date=run_date,
        status="complete",
    )
    artifact.update(
        {
            "source_cases_available": {
                "exp1323": bool(exp1323_artifact),
                "exp1324": bool(exp1324_artifact),
                "exp1340": exp1340_artifact is not None,
                "limitations": limitations,
                "source_failure_classes": sorted(failure_counts),
                "exp1323_token_health_context": {
                    "min_tokens_recovered": exp1323_artifact.get("min_tokens_recovered"),
                    "empty_or_one_token_rate": exp1323_artifact.get("empty_or_one_token_rate"),
                    "entropy_production_rate_available": exp1323_artifact.get(
                        "entropy_production_rate_available"
                    ),
                    "topk_logprob_available": exp1323_artifact.get("topk_logprob_available"),
                },
                "exp1324_source_metrics": exp1324_artifact.get("source_metrics", {}),
            },
            "data_driven_risk_proxy": {
                "proxy_name": "hardcoded_solution_leakage_path_proxy",
                "count": leakage_count,
                "rate": exp1324_artifact.get("hardcoded_solution_leakage_rate"),
                "failure_types": {
                    "possible_hardcoded_solution_leakage": leakage_count,
                },
                "interpretation": (
                    "Counts verifier-label repair or compact-encoding paths as a "
                    "data-driven shortcut/leakage proxy, not as proof of an "
                    "independent formalizer success."
                ),
            },
            "reasoning_driven_risk_proxy": {
                "proxy_name": "semantic_solver_unknown_instability_proxy",
                "count": sum(reasoning_failure_types.values()),
                "failure_types": reasoning_failure_types,
                "solver_vs_certificate_delta": exp1324_artifact.get("solver_vs_certificate_delta"),
                "interpretation": (
                    "Counts parseable-but-untruthful certificates, solver disagreement, "
                    "and UNKNOWN collapse as reasoning/decision instability proxies."
                ),
            },
            "parser_schema_risk_count": parser_schema_count,
            "undergeneration_risk_count": undergeneration_count,
            "semantic_invalidity_count": semantic_count,
            "unknown_mishandling_count": unknown_count,
            "repair_policy_by_failure_type": _repair_policy_by_failure_type(
                minimum_parseable_attempts=exp1324_artifact.get(
                    "minimum_parseable_attempts_to_recover"
                )
            ),
            "universal_detector_claim_allowed": False,
            "honest_verdict": _honest_verdict(exp1340_artifact is None),
            "source_honest_verdicts": {
                "exp1323": exp1323_artifact.get("honest_verdict"),
                "exp1324": exp1324_artifact.get("honest_verdict"),
                "exp1340": exp1340_artifact.get("honest_verdict")
                if exp1340_artifact is not None
                else None,
            },
            "measurement_note": (
                "Replay/audit artifact only. No fresh SOTA model generation was run."
            ),
        }
    )
    return artifact


def run_experiment(
    *,
    project_root: str | Path = ".",
    run_date: str = DEFAULT_RUN_DATE,
    exp1323_path: str | Path = DEFAULT_EXP1323_PATH,
    exp1324_path: str | Path = DEFAULT_EXP1324_PATH,
    exp1340_path: str | Path = DEFAULT_EXP1340_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Write in-progress first, then write the replay-only completed artifact."""
    root = Path(project_root)
    output = _resolve(root, output_path)
    _write_json(
        output,
        _base_artifact(project_root=root, run_date=run_date, status="in_progress"),
        write_observer=write_observer,
    )
    exp1340_artifact, exp1340_limitation = _read_optional_json(_resolve(root, exp1340_path))
    artifact = build_halluguard_certificate_failure_split_artifact(
        exp1323_artifact=_read_json(_resolve(root, exp1323_path)),
        exp1324_artifact=_read_json(_resolve(root, exp1324_path)),
        exp1340_artifact=exp1340_artifact,
        exp1340_limitation=exp1340_limitation,
        run_date=run_date,
        project_root=root,
    )
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def _base_artifact(*, project_root: Path, run_date: str, status: str) -> dict[str, Any]:
    return {
        "artifact": ARTIFACT_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "status": status,
        "source_cases_available": {},
        "data_driven_risk_proxy": {},
        "reasoning_driven_risk_proxy": {},
        "parser_schema_risk_count": None,
        "undergeneration_risk_count": None,
        "semantic_invalidity_count": None,
        "unknown_mishandling_count": None,
        "repair_policy_by_failure_type": {},
        "universal_detector_claim_allowed": False,
        "honest_verdict": "in_progress" if status == "in_progress" else "not_run",
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "source_experiments": ["1323", "1324", "1340"],
        },
    }


def _failure_mode_counts(exp1324_artifact: Mapping[str, Any]) -> dict[str, int]:
    modes = exp1324_artifact.get("formalizer_failure_modes", [])
    counts: dict[str, int] = {}
    if not isinstance(modes, list):
        return counts
    for mode in modes:
        if isinstance(mode, Mapping):
            failure_class = str(mode.get("class") or "")
            if failure_class:
                counts[failure_class] = _int(mode.get("count"))
    return counts


def _repair_policy_by_failure_type(*, minimum_parseable_attempts: Any) -> dict[str, dict[str, Any]]:
    parse_target = _int(minimum_parseable_attempts)
    return {
        "undergeneration": {
            "risk_axis": "runtime_and_prompt_underproduction",
            "next_actions": [
                "prompt retrieval: inject the certificate schema before generation",
                "reasoning budget: remove premature newline stop strings and allocate enough max_tokens for a certificate tail",
            ],
        },
        "parser_schema_mismatch": {
            "risk_axis": "parser_schema",
            "next_actions": [
                f"grammar branch: recover at least {parse_target} additional parseable attempts before reopening the parse gate",
                "prompt retrieval: retrieve the exact JSON field contract before constrained decoding",
            ],
        },
        "semantic_invalidity": {
            "risk_axis": "reasoning_driven",
            "next_actions": [
                "semantic validator: run solver-backed validation after parsing",
                "reasoning budget: allow trigger-before-constrain reasoning before the certificate tail",
            ],
        },
        "solver_disagreement": {
            "risk_axis": "reasoning_driven",
            "next_actions": [
                "semantic validator: compare solver answer, certificate label, and verifier label as separate columns",
                "reasoning budget: escalate cases with solver disagreement before accepting repaired certificates",
            ],
        },
        "unknown_state_mishandling": {
            "risk_axis": "unknown_preservation",
            "next_actions": [
                "UNKNOWN-preserving fallback: keep UNKNOWN and ABSTAIN as first-class labels",
                "grammar branch: dispatch UNKNOWN cases to a schema branch that forbids forced SAT/UNSAT collapse",
            ],
        },
        "possible_hardcoded_solution_leakage": {
            "risk_axis": "data_driven_proxy",
            "next_actions": [
                "prompt retrieval: separate source facts from verifier labels before formalizer evaluation",
                "semantic validator: exclude verifier-label repair paths from headline formalizer metrics unless derivation evidence is present",
            ],
        },
    }


def _honest_verdict(exp1340_missing: bool) -> str:
    if exp1340_missing:
        return "local_certificate_slice_diagnostic_exp1340_missing_no_universal_detector_claim"
    return "local_certificate_slice_diagnostic_complete_no_universal_detector_claim"


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return _read_json(path), None
    except (FileNotFoundError, json.JSONDecodeError):
        return None, EXP1340_LIMITATION


def _write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    write_observer: WriteObserver | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if write_observer is not None:
        write_observer(path, payload)


def _int(value: Any) -> int:
    return int(value) if isinstance(value, int) else 0


def main() -> None:  # pragma: no cover - CLI wrapper, covered through run_experiment.
    run_experiment(project_root=Path.cwd())


if __name__ == "__main__":  # pragma: no cover
    main()
