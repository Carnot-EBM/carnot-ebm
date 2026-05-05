"""Exp 1324 certificate failure taxonomy diagnostic.

Spec: REQ-VERIFY-1324,
      SCENARIO-VERIFY-1324
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_RUN_DATE = "20260505"
DEFAULT_EXP1311_PATH = Path(
    "results/experiment_1311_sota_constraintbench_satquest_answer_stability.json"
)
DEFAULT_EXP1312_PATH = Path(
    "results/experiment_1312_triggered_certificate_extraction_dccd_gbnf.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "results/experiment_1324_certificate_failure_taxonomy_formalizer_reality_check.json"
)
ARTIFACT_NAME = "experiment_1324_certificate_failure_taxonomy_formalizer_reality_check"
SCHEMA_VERSION = 1
PARSE_GATE = 0.75
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "formalizer_failure_modes",
    "parser_failure_count",
    "semantic_failure_count",
    "undergeneration_failure_count",
    "hardcoded_solution_leakage_rate",
    "solver_vs_certificate_delta",
    "reasoning_token_overhead",
    "parse_recovery_recommendation",
    "minimum_gate_delta_needed",
    "honest_verdict",
)


def build_failure_taxonomy_artifact(
    *,
    exp1311_artifact: Mapping[str, Any],
    exp1312_artifact: Mapping[str, Any],
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Build the diagnostic artifact from already-saved Exp 1311/1312 records.

    This function intentionally performs no fresh model calls. The point of the
    diagnostic is to separate what the existing records can prove from what a
    rerun must still measure with solver-backed certificates.
    """
    attempts = _attempt_records(exp1312_artifact)
    direct_audit = bool(attempts)
    attempt_count = _attempt_count(exp1312_artifact, attempts)
    parse_rate = _number(exp1312_artifact.get("certificate_parse_rate"), 0.0)
    parsed_count = _parsed_count(attempts, attempt_count, parse_rate)

    parser_failure_count = _parser_failure_count(attempts, attempt_count, parse_rate)
    semantic_failure_count = _semantic_failure_count(exp1312_artifact, attempts, parsed_count)
    undergeneration_failure_count = _undergeneration_failure_count(exp1311_artifact)
    source_solver_disagreement_count = _source_solver_disagreement_count(exp1311_artifact)
    unknown_state_mishandling_count = _unknown_state_mishandling_count(
        exp1311_artifact,
        attempts,
    )
    leakage_count = _possible_hardcoded_solution_leakage_count(attempts)
    leakage_rate = _rate(leakage_count, len(attempts)) if direct_audit else None
    solver_vs_certificate_delta = _solver_vs_certificate_delta(
        exp1311_artifact,
        exp1312_artifact,
    )
    minimum_gate_delta_needed = round(max(0.0, PARSE_GATE - parse_rate), 6)
    parseable_to_recover = _minimum_parseable_attempts_to_recover(
        attempt_count,
        parsed_count,
        parse_rate,
    )

    artifact = _base_artifact(
        project_root=Path(project_root),
        run_date=run_date,
        status="complete",
    )
    artifact.update(
        {
            "formalizer_failure_modes": _formalizer_failure_modes(
                undergeneration_failure_count=undergeneration_failure_count,
                parser_failure_count=parser_failure_count,
                semantic_failure_count=semantic_failure_count,
                source_solver_disagreement_count=source_solver_disagreement_count,
                unknown_state_mishandling_count=unknown_state_mishandling_count,
                possible_leakage_count=leakage_count,
                possible_leakage_rate=leakage_rate,
                parseable_to_recover=parseable_to_recover,
            ),
            "parser_failure_count": parser_failure_count,
            "semantic_failure_count": semantic_failure_count,
            "undergeneration_failure_count": undergeneration_failure_count,
            "hardcoded_solution_leakage_rate": leakage_rate,
            "solver_vs_certificate_delta": solver_vs_certificate_delta,
            "reasoning_token_overhead": _reasoning_token_overhead(exp1312_artifact),
            "parse_recovery_recommendation": _parse_recovery_recommendation(
                parseable_to_recover=parseable_to_recover,
                parser_failure_count=parser_failure_count,
            ),
            "minimum_gate_delta_needed": minimum_gate_delta_needed,
            "honest_verdict": _honest_verdict(minimum_gate_delta_needed),
            "artifact_metadata": {
                "project_root": str(project_root),
                "run_date": run_date,
                "source_experiments": ["1311", "1312"],
                "parse_gate": PARSE_GATE,
                "direct_exp1312_attempt_audit": direct_audit,
            },
            "source_data_limitations": _source_data_limitations(direct_audit),
            "literature_reality_check_summary": _literature_reality_check_summary(),
            "source_metrics": {
                "exp1311_answer_stability_score": exp1311_artifact.get(
                    "answer_stability_score"
                ),
                "exp1311_pysat_verified_rate": exp1311_artifact.get("pysat_verified_rate"),
                "exp1311_source_solver_disagreement_count": source_solver_disagreement_count,
                "exp1312_certificate_attempt_count": attempt_count,
                "exp1312_certificate_parse_rate": exp1312_artifact.get(
                    "certificate_parse_rate"
                ),
                "exp1312_certificate_truthfulness_rate": exp1312_artifact.get(
                    "certificate_truthfulness_rate"
                ),
            },
            "minimum_parseable_attempts_to_recover": parseable_to_recover,
            "possible_hardcoded_solution_leakage_count": leakage_count if direct_audit else None,
            "unknown_state_mishandling_count": unknown_state_mishandling_count,
            "per_path_failure_summary": _per_path_failure_summary(attempts),
            "exp1325_fix_priorities": [
                "runtime settings",
                "prompt schema",
                "parser repair",
                "grammar coverage",
                "DCCD compact encoding with hardcoded-solution leakage guard",
            ],
            "measurement_note": (
                "Diagnostic-only artifact built from Exp 1311 and Exp 1312 records; "
                "no fresh SOTA generation run was performed."
            ),
        }
    )
    return artifact


def run_experiment(
    *,
    project_root: str | Path = ".",
    run_date: str = DEFAULT_RUN_DATE,
    exp1311_path: str | Path = DEFAULT_EXP1311_PATH,
    exp1312_path: str | Path = DEFAULT_EXP1312_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Write an in-progress marker, then replace it with the completed artifact."""
    root = Path(project_root)
    output = Path(output_path)
    _write_json(output, _base_artifact(project_root=root, run_date=run_date))
    exp1311_artifact = _load_json(Path(exp1311_path))
    exp1312_artifact = _load_json(Path(exp1312_path))
    artifact = build_failure_taxonomy_artifact(
        exp1311_artifact=exp1311_artifact,
        exp1312_artifact=exp1312_artifact,
        run_date=run_date,
        project_root=root,
    )
    _write_json(output, artifact)
    return artifact


def _base_artifact(*, project_root: Path, run_date: str, status: str = "in_progress") -> dict[str, Any]:
    return {
        "artifact": ARTIFACT_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "status": status,
        "formalizer_failure_modes": [],
        "parser_failure_count": None,
        "semantic_failure_count": None,
        "undergeneration_failure_count": None,
        "hardcoded_solution_leakage_rate": None,
        "solver_vs_certificate_delta": None,
        "reasoning_token_overhead": None,
        "parse_recovery_recommendation": None,
        "minimum_gate_delta_needed": None,
        "honest_verdict": "in_progress" if status == "in_progress" else "not_run",
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "source_experiments": ["1311", "1312"],
        },
    }


def _attempt_records(exp1312_artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    attempts = exp1312_artifact.get("attempts")
    if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
        return []
    return [dict(attempt) for attempt in attempts if isinstance(attempt, Mapping)]


def _attempt_count(exp1312_artifact: Mapping[str, Any], attempts: Sequence[Mapping[str, Any]]) -> int:
    if attempts:
        return len(attempts)
    return int(_number(exp1312_artifact.get("certificate_attempt_count"), 0.0))


def _parsed_count(
    attempts: Sequence[Mapping[str, Any]],
    attempt_count: int,
    parse_rate: float,
) -> int:
    if attempts:
        return sum(1 for attempt in attempts if attempt.get("parseable") is True)
    return int(round(attempt_count * parse_rate))


def _parser_failure_count(
    attempts: Sequence[Mapping[str, Any]],
    attempt_count: int,
    parse_rate: float,
) -> int:
    if attempts:
        return sum(1 for attempt in attempts if attempt.get("parseable") is not True)
    return max(0, attempt_count - int(round(attempt_count * parse_rate)))


def _semantic_failure_count(
    exp1312_artifact: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    parsed_count: int,
) -> int:
    if attempts:
        return sum(
            1
            for attempt in attempts
            if attempt.get("parseable") is True and attempt.get("truthful") is not True
        )
    truthfulness_rate = _number(exp1312_artifact.get("certificate_truthfulness_rate"), 0.0)
    return max(0, int(round(parsed_count * (1.0 - truthfulness_rate))))


def _undergeneration_failure_count(exp1311_artifact: Mapping[str, Any]) -> int:
    rows = _response_rows(exp1311_artifact)
    return sum(
        1
        for row in rows
        if int(_number(row.get("token_count"), 0.0)) <= 1
        or len(str(row.get("raw_output") or "")) == 0
    )


def _source_solver_disagreement_count(exp1311_artifact: Mapping[str, Any]) -> int:
    return sum(1 for row in _response_rows(exp1311_artifact) if row.get("verified") is False)


def _unknown_state_mishandling_count(
    exp1311_artifact: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
) -> int:
    if attempts:
        return sum(
            1
            for attempt in attempts
            if attempt.get("parseable") is True
            and attempt.get("truthful") is not True
            and "unknown" in str(attempt.get("item_id") or "").lower()
        )
    return sum(
        1
        for row in _response_rows(exp1311_artifact)
        if str(row.get("verifier_label") or "").upper() == "UNKNOWN"
        and str(row.get("parsed_label") or "").upper() not in {"UNKNOWN", "ABSTAIN"}
    )


def _possible_hardcoded_solution_leakage_count(attempts: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        1
        for attempt in attempts
        if attempt.get("path") == "repaired_certificate"
        or (attempt.get("path") == "dccd_compact" and attempt.get("compact_encoding") is True)
    )


def _solver_vs_certificate_delta(
    exp1311_artifact: Mapping[str, Any],
    exp1312_artifact: Mapping[str, Any],
) -> float:
    source_verified_rate = _source_verified_rate(exp1311_artifact)
    certificate_truthfulness_rate = _number(
        exp1312_artifact.get("certificate_truthfulness_rate"),
        0.0,
    )
    return round(certificate_truthfulness_rate - source_verified_rate, 6)


def _source_verified_rate(exp1311_artifact: Mapping[str, Any]) -> float:
    pysat_rate = exp1311_artifact.get("pysat_verified_rate")
    if isinstance(pysat_rate, (int, float)):
        return float(pysat_rate)
    rows = _response_rows(exp1311_artifact)
    if rows:
        return _rate(sum(1 for row in rows if row.get("verified") is True), len(rows))
    return 0.0


def _minimum_parseable_attempts_to_recover(
    attempt_count: int,
    parsed_count: int,
    parse_rate: float,
) -> int:
    if attempt_count > 0:
        return max(0, math.ceil(PARSE_GATE * attempt_count) - parsed_count)
    return 1 if parse_rate < PARSE_GATE else 0


def _formalizer_failure_modes(
    *,
    undergeneration_failure_count: int,
    parser_failure_count: int,
    semantic_failure_count: int,
    source_solver_disagreement_count: int,
    unknown_state_mishandling_count: int,
    possible_leakage_count: int,
    possible_leakage_rate: float | None,
    parseable_to_recover: int,
) -> list[dict[str, Any]]:
    return [
        {
            "class": "undergeneration",
            "count": undergeneration_failure_count,
            "evidence": "Exp 1311 source rows with empty raw_output or token_count <= 1.",
            "smallest_repair_needed": (
                "runtime settings: remove premature stop strings and require enough "
                "tokens for a certificate tail before rerunning exp1325."
            ),
        },
        {
            "class": "parser_schema_mismatch",
            "count": parser_failure_count,
            "evidence": "Exp 1312 unparseable attempts, dominated by raw_trigger no_json_object.",
            "smallest_repair_needed": (
                f"parser repair plus prompt schema: recover at least {parseable_to_recover} "
                "additional parseable attempt(s) to clear the 0.75 gate."
            ),
        },
        {
            "class": "semantic_invalidity",
            "count": semantic_failure_count,
            "evidence": "Parseable Exp 1312 certificates whose final answer was not truthful.",
            "smallest_repair_needed": (
                "semantic validator/MUS repair after parsing; parseability alone is not "
                "accepted as a certificate."
            ),
        },
        {
            "class": "solver_disagreement",
            "count": source_solver_disagreement_count,
            "evidence": "Exp 1311 answer rows rejected by the local verifier/solver.",
            "smallest_repair_needed": (
                "compare solver answer, certificate label, and verifier label as separate "
                "columns in exp1325."
            ),
        },
        {
            "class": "unknown_state_mishandling",
            "count": unknown_state_mishandling_count,
            "evidence": "UNKNOWN cases forced into SAT/UNSAT or untruthful certificates.",
            "smallest_repair_needed": (
                "preserve UNKNOWN/ABSTAIN as first-class certificate states in the schema "
                "and grammar."
            ),
        },
        {
            "class": "possible_hardcoded_solution_leakage",
            "count": possible_leakage_count,
            "rate": possible_leakage_rate,
            "evidence": (
                "Repaired certificates and compact DCCD projections can be solver-label "
                "aligned, so they are leakage-risk evidence rather than independent "
                "formalizer success."
            ),
            "smallest_repair_needed": (
                "keep DCCD compact encoding, but exclude verifier-label repair paths from "
                "headline formalizer metrics unless the certificate payload proves the "
                "solution was derived."
            ),
        },
    ]


def _reasoning_token_overhead(exp1312_artifact: Mapping[str, Any]) -> dict[str, Any]:
    tax = exp1312_artifact.get("grammar_projection_tax_proxy")
    tax = tax if isinstance(tax, Mapping) else {}
    return {
        "available": False,
        "proxy": "prompt_chars_divided_by_4_not_model_reasoning_tokens",
        "limitation": (
            "Exp 1312 did not save model reasoning-token traces, so this reports prompt "
            "overhead proxies only."
        ),
        "gbnf_extra_token_proxy": _char_to_token_proxy(
            tax.get("gbnf_mean_extra_prompt_chars")
        ),
        "dccd_extra_token_proxy": _char_to_token_proxy(
            tax.get("dccd_mean_extra_prompt_chars")
        ),
        "repair_extra_token_proxy": _char_to_token_proxy(
            tax.get("repair_mean_extra_prompt_chars")
        ),
    }


def _parse_recovery_recommendation(*, parseable_to_recover: int, parser_failure_count: int) -> str:
    if parseable_to_recover <= 0:
        return (
            "No parse-rate gate delta is required; exp1325 should focus on semantic "
            "validator and leakage guards before claiming formalizer success."
        )
    return (
        "Apply parser repair plus runtime settings and prompt schema cleanup before "
        f"exp1325; recover at least {parseable_to_recover} of "
        f"{parser_failure_count} parser/schema failures while preserving DCCD compact "
        "encoding behind a hardcoded-solution leakage guard."
    )


def _honest_verdict(minimum_gate_delta_needed: float) -> str:
    if minimum_gate_delta_needed > 0:
        return "diagnostic_complete_parse_gate_shortfall_parser_recovery_needed"
    return "diagnostic_complete_parse_gate_met_semantic_leakage_review_needed"


def _source_data_limitations(direct_audit: bool) -> str:
    if direct_audit:
        return (
            "Exp 1312 saved per-attempt records with path, item_id, parseable, truthful, "
            "errors, compact_encoding, and prompt_chars, so parser and semantic counts "
            "are direct. It did not save full certificate payloads, proof text, or "
            "perturbation indices, so hardcoded-solution leakage is a path-based proxy."
        )
    return (
        "Exp 1312 records are aggregate-only for this audit; parser and semantic counts "
        "are reconstructed from certificate_attempt_count, certificate_parse_rate, and "
        "certificate_truthfulness_rate, while leakage cannot be estimated."
    )


def _literature_reality_check_summary() -> str:
    return (
        "The post-.102 Reality Check section reports that LLM formalizers can fail on "
        "real CSPs even when solver-style answers agree, including excessive reasoning "
        "tokens and hard-coded solutions. SatIR reinforces that incomplete inputs need "
        "explicit UNKNOWN-preserving formal constraints, not forced labels. The "
        "orthographic constraint work shows deterministic hard constraints can expose "
        "architecture-specific failures hidden by aggregate answer agreement. Therefore "
        "Carnot must evaluate CSP formalization with solver-backed certificates, not "
        "answer agreement or parseability alone."
    )


def _per_path_failure_summary(attempts: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for attempt in attempts:
        path = str(attempt.get("path") or "unknown")
        current = summary.setdefault(
            path,
            {"attempts": 0, "parser_failures": 0, "semantic_failures": 0},
        )
        current["attempts"] += 1
        if attempt.get("parseable") is not True:
            current["parser_failures"] += 1
        elif attempt.get("truthful") is not True:
            current["semantic_failures"] += 1
    return summary


def _response_rows(exp1311_artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = exp1311_artifact.get("responses")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _char_to_token_proxy(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value) / 4.0, 6)


def _number(value: Any, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover - CLI wrapper, covered through run_experiment.
    run_experiment(project_root=Path.cwd())


if __name__ == "__main__":  # pragma: no cover
    main()
